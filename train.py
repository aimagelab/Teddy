import torch
import argparse
import random
import numpy as np
import os
from model.teddy import Teddy, freeze, unfreeze
from pathlib import Path
from datasets import dataset_factory
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from util import RandCheckpointScheduler, SineCheckpointScheduler, AlternatingScheduler, RandReducingScheduler, OneLinearScheduler, RandomLinearScheduler
from util import WarmupLRScheduler, loss_hinge_dis, loss_hinge_gen, GradSwitch
from util import SquareThresholdMSELoss, NoCudnnCTCLoss, TextSampler
from torchvision.utils import make_grid
from einops import rearrange, repeat
import wandb


def train(rank, args):
    if args.ddp:
        setup(rank=rank, world_size=args.world_size)
        device = f'cuda:{rank}'
    else:
        device = args.device

    dataset = dataset_factory(args.datasets, args.datasets_path, 'train', max_width=args.max_width, channels=args.img_channels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                         collate_fn=dataset.collate_fn, pin_memory=True, drop_last=True)

    ocr_checkpoint_a = torch.load(args.ocr_checkpoint_a, map_location=device)
    # ocr_checkpoint_b = torch.load(args.ocr_checkpoint_b, map_location=device)
    # ocr_checkpoint_c = torch.load(args.ocr_checkpoint_c, map_location=device)
    # assert ocr_checkpoint_a['charset'] == ocr_checkpoint_b['charset'], "OCR checkpoints must have the same charset"

    teddy = Teddy(ocr_checkpoint_a['charset'], dim=args.dim, img_channels=args.img_channels).to(device)
    teddy_ddp = DDP(teddy, device_ids=[rank], find_unused_parameters=True) if args.ddp else teddy  # find_unused_parameters=True

    optimizer_dis = torch.optim.AdamW(teddy.discriminator.parameters(), lr=args.lr_dis)
    # scheduler_dis_warmup = WarmupLRScheduler(optimizer_dis, args.warmup, args.lr_dis)
    # scheduler_dis_exp = torch.optim.lr_scheduler.ExponentialLR(optimizer_dis, gamma=0.9999)
    # scheduler_dis = torch.optim.lr_scheduler.SequentialLR(optimizer_dis, [scheduler_dis_warmup, scheduler_dis_exp], [args.warmup])
    scheduler_dis = torch.optim.lr_scheduler.ExponentialLR(optimizer_dis, gamma=0.9999)

    optimizer_gen = torch.optim.AdamW(teddy.generator.parameters(), lr=args.lr_gen)
    scheduler_gen = torch.optim.lr_scheduler.ExponentialLR(optimizer_gen, gamma=0.9999)

    if args.train_ocr:
        optimizer_ocr = torch.optim.AdamW(teddy.ocr.parameters(), lr=args.lr_ocr)
    else:
        freeze(teddy.ocr)
        # optimizer_ocr = SineCheckpointScheduler(teddy.ocr, ocr_checkpoint_a['model'], ocr_checkpoint_b['model'], period=len(loader))
        # optimizer_ocr = RandCheckpointScheduler(teddy.ocr, ocr_checkpoint_a['model'], ocr_checkpoint_b['model'])
        # optimizer_ocr = AlternatingScheduler(teddy.ocr, ocr_checkpoint_a['model'], ocr_checkpoint_b['model'], ocr_checkpoint_c['model'])
        # optimizer_ocr = RandReducingScheduler(teddy.ocr, ocr_checkpoint_a['model'], ocr_checkpoint_b['model'])
        # optimizer_ocr = OneLinearScheduler(teddy.ocr, ocr_checkpoint_a['model'], ocr_checkpoint_b['model'], period=len(loader) * 20)
        optimizer_ocr = RandomLinearScheduler(teddy.ocr, ocr_checkpoint_a['model'], period=len(loader) * 20)
        optimizer_ocr._step(0)

    scaler = GradScaler()

    if args.resume is not None:
        assert args.resume.exists(), f"Resume path {args.resume} doesn't exist"
        checkpoint = torch.load(args.resume)
        teddy.load_state_dict(checkpoint['model'])
        optimizer_dis.load_state_dict(checkpoint['optimizer_dis'])
        optimizer_gen.load_state_dict(checkpoint['optimizer_gen'])
        scheduler_dis.load_state_dict(checkpoint['scheduler_dis'])
        scheduler_gen.load_state_dict(checkpoint['scheduler_gen'])
        optimizer_ocr.load_state_dict(checkpoint['optimizer_ocr'])

    ctc_criterion = NoCudnnCTCLoss(reduction='mean', zero_infinity=True).to(device)
    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean').to(device)
    cycle_criterion = torch.nn.L1Loss(reduction='mean').to(device)
    tmse_criterion = SquareThresholdMSELoss(threshold=0)

    text_generator = TextSampler(dataset.labels, 6)

    if rank == 0 and args.wandb:
        wandb.init(project='teddy', entity='fomo_aiisdh', config=args)
        wandb.watch(teddy)

    for epoch in range(10 ** 10):
        for idx, batch in enumerate(loader):
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch['gen_texts'] = text_generator.sample(len(batch['style_imgs']))

            preds = teddy_ddp(batch)
            dis_loss_real, dis_loss_fake = loss_hinge_dis(preds['dis_fake_pred'], preds['dis_real_pred'])
            loss_dis = dis_loss_real + dis_loss_fake

            # preds = teddy_ddp(batch)
            gen_loss_fake = loss_hinge_gen(preds['dis_fake_pred'])

            optimizer_dis.zero_grad()
            optimizer_gen.zero_grad()

            with GradSwitch(teddy, teddy.discriminator):
                loss_dis.backward(retain_graph=True)

            with GradSwitch(teddy, teddy.generator):
                gen_loss_fake.backward(retain_graph=True)

            optimizer_dis.step()
            optimizer_gen.step()

            # CTC loss from OCR
            # b, w, _ = preds['fake_text_pred'].shape
            # preds_size = torch.IntTensor([w] * b).to(device)
            # preds['fake_text_pred'] = preds['fake_text_pred'].permute(1, 0, 2).log_softmax(2)
            # fake_ctc_loss = ctc_criterion(preds['fake_text_pred'], preds['enc_gen_texts'], preds_size, preds['enc_gen_texts_len'])
            # fake_ctc_loss *= args.weight_ctc

            # b, w, _ = preds['real_text_pred'].shape
            # preds_size = torch.IntTensor([w] * b).to(device)
            # preds['real_text_pred'] = preds['real_text_pred'].permute(1, 0, 2).log_softmax(2)
            # real_ctc_loss = ctc_criterion(preds['real_text_pred'], preds['enc_style_text'], preds_size, preds['enc_style_text_len'])
            # real_ctc_loss *= args.weight_ctc

            # Binary cross entropy loss
            # real_fake_tgt = torch.ones((4, preds['real_fake_pred'].size(0) // 4, 1), device=device)
            # real_fake_tgt[::2, :, :] = 0  # 0 -> real | 1 -> fake
            # real_fake_loss = bce_criterion(preds['real_fake_pred'], real_fake_tgt.reshape((-1, 1)))
            # real_fake_loss *= args.weight_real_fake

            # same_other_tgt = torch.ones((4, preds['same_other_pred'].size(0) // 4, 1), device=device)
            # same_other_tgt[:2, :, :] = 0  # 0 -> same | 1 -> other
            # same_other_tgt = (same_other_tgt.flatten() * batch['multi_authors'].repeat(4))
            # same_other_loss = bce_criterion(preds['same_other_pred'], same_other_tgt.unsqueeze(-1))
            # same_other_loss *= args.weight_same_other

            # Binary cross entropy loss
            # real_fake_tgt = torch.ones_like(preds['real_fake_pred'])
            # real_fake_tgt[:args.batch_size, :] = 0  # 0 -> real | 1 -> fake
            # real_fake_loss = bce_criterion(preds['real_fake_pred'], real_fake_tgt)
            # real_fake_loss *= args.weight_real_fake

            # Cycle loss
            # src_style_emb = preds['src_style_emb'][:, :2].repeat(teddy.expansion_factor, 1, 1)
            # gen_style_emb = preds['gen_style_emb'][:, :2]
            # cycle_loss = cycle_criterion(src_style_emb, gen_style_emb)
            # cycle_loss *= args.weight_cycle

            # MSE loss
            # mse_loss = tmse_criterion(preds['fakes'])
            # mse_loss *= args.weight_mse

            # Update
            # train_loss = fake_ctc_loss + real_ctc_loss + same_other_loss + real_fake_loss + cycle_loss - mse_loss
            # train_loss = real_fake_loss

            # optimizer_dis.zero_grad()
            # optimizer_gen.zero_grad()

            # scaler.scale(train_loss).backward()

            # scaler.step(optimizer_dis)
            # scaler.step(optimizer_gen)

            # scaler.update()

            # if args.train_ocr:
            #     raise NotImplementedError
            # else:
            #     optimizer_ocr.step()

            if idx % args.log_every == 0 and rank == 0:
                # print(f"Epoch: {epoch} | Iter: {idx:06d} | Real/Fake CTC loss: {real_ctc_loss.item():.4f}/{fake_ctc_loss.item():.4f} | SO loss: {same_other_loss.item():.4f} | RF loss: {real_fake_loss.item():.4f} | Cycle loss: {cycle_loss.item():.4f} | MSE loss: {mse_loss.item():.4f} | Train loss: {train_loss.item():.4f}")
                print(f"Epoch: {epoch} | Iter: {idx:06d} | DReal loss: {dis_loss_real.item():.4f} | DFake loss: {dis_loss_fake.item():.4f} | GFake loss: {gen_loss_fake.item():.4f} | Train loss: {loss_dis.item():.4f}")

                with torch.inference_mode():
                    fakes = rearrange(preds['fakes'], 'b e c h w -> (b e) c h w')
                    img_grid = make_grid(fakes, nrow=1, normalize=True, value_range=(-1, 1))

                    # fake = fakes[0].detach().cpu().permute(1, 2, 0).numpy()
                    # fake_pred = teddy.text_converter.decode_batch(preds['fake_text_pred'])[0]
                    # fake_gt = batch['gen_texts'][0]

                    # real = batch['style_imgs'][0].detach().cpu().permute(1, 2, 0).numpy()
                    # real_pred = teddy.text_converter.decode_batch(preds['real_text_pred'])[0]
                    # real_gt = batch['style_texts'][0]

                    style_imgs = make_grid(batch['style_imgs'], nrow=1, normalize=True, value_range=(-1, 1))
                    same_author_imgs = make_grid(batch['same_author_imgs'], nrow=1, normalize=True, value_range=(-1, 1))
                    other_author_imgs = make_grid(batch['other_author_imgs'], nrow=1, normalize=True, value_range=(-1, 1))

                if args.wandb:
                    wandb.log({
                        # 'fake_ctc_loss': fake_ctc_loss.item(),
                        # 'real_ctc_loss': real_ctc_loss.item(),
                        # 'same_other_loss': same_other_loss.item(),
                        # 'real_fake_loss': real_fake_loss.item(),
                        # 'cycle_loss': cycle_loss.item(),
                        # 'mse_loss': mse_loss.item(),
                        # 'train_loss': train_loss.item(),
                        'dis_loss_real': dis_loss_real.item(),
                        'dis_loss_fake': dis_loss_fake.item(),
                        'gen_loss_fake': gen_loss_fake.item(),
                        'lr_dis': scheduler_dis.get_last_lr()[0],
                        'lr_gen': scheduler_gen.get_last_lr()[0],
                        'alphas': optimizer_ocr.last_alpha,
                        'fakes/all': [wandb.Image(img_grid.detach().cpu().permute(1, 2, 0).numpy())],
                        # 'fakes/sample': [wandb.Image(fake, caption=f"GT: {fake_gt}\nP: {fake_pred}")],
                        # 'reals/sample': [wandb.Image(real, caption=f"GT: {real_gt}\nP: {real_pred}")],
                        'reals/style_imgs': [wandb.Image(style_imgs, caption=f"style_imgs")],
                        'reals/same_author_imgs': [wandb.Image(same_author_imgs, caption=f"same_author_imgs")],
                        'reals/other_author_imgs': [wandb.Image(other_author_imgs, caption=f"other_author_imgs")],
                    })
        scheduler_dis.step()
        scheduler_gen.step()
    cleanup()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def cleanup_on_error(rank, fn, *args, **kwargs):
    try:
        fn(rank, *args, **kwargs)
    except Exception as e:
        cleanup()
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_gen', type=float, default=0.0001)
    parser.add_argument('--lr_dis', type=float, default=0.0001)
    parser.add_argument('--lr_ocr', type=float, default=0.0001)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=742)
    parser.add_argument('--device', type=str, default='cuda:0', help="Device")
    parser.add_argument('--root_path', type=str, default='/mnt/ssd/datasets', help="Root path")
    parser.add_argument('--datasets_path', type=str, nargs='+', default=[
        'IAM',
        # 'Norhand',
        # 'Rimes',
        # 'ICFHR16',
        # 'ICFHR14',
        # 'LAM_msgpack',
        # 'Rodrigo',
        # 'SaintGall',
        # 'Washington',
        # 'LEOPARDI/leopardi',
    ], help="Datasets path")
    parser.add_argument('--datasets', type=str, nargs='+', default=[
        'iam_lines',
        # 'norhand',
        # 'rimes',
        # 'icfhr16',
        # 'icfhr14',
        # 'lam',
        # 'rodrigo',
        # 'saintgall',
        # 'washington',
        # 'leopardi',
    ], help="Datasets")
    parser.add_argument('--max_width', type=int, default=1800, help="Filter images with width > max_width")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers")
    parser.add_argument('--resume', type=Path, default=None, help="Resume path")
    parser.add_argument('--wandb', action='store_true', help="Use wandb")

    # Teddy
    parser.add_argument('--dim', type=int, default=512, help="Model dimension")
    parser.add_argument('--ocr_checkpoint_a', type=Path, default='files/f745_all_datasets/0345000_iter.pth', help="OCR checkpoint a")
    parser.add_argument('--ocr_checkpoint_b', type=Path, default='files/0ea8_all_datasets/0345000_iter.pth', help="OCR checkpoint b")
    parser.add_argument('--ocr_checkpoint_c', type=Path, default='files/0259_all_datasets/0345000_iter.pth', help="OCR checkpoint c")
    parser.add_argument('--weight_ctc', type=float, default=1.0, help="CTC loss weight")
    parser.add_argument('--weight_same_other', type=float, default=6.0, help="Same/other loss weight")
    parser.add_argument('--weight_real_fake', type=float, default=6.0, help="Real/fake loss weight")
    parser.add_argument('--weight_cycle', type=float, default=6.0, help="Cycle loss weight")
    parser.add_argument('--weight_mse', type=float, default=0.0, help="MSE loss weight")
    parser.add_argument('--img_channels', type=int, default=1, help="Image channels")
    parser.add_argument('--ddp', action='store_true', help="Use DDP")
    parser.add_argument('--train_ocr', action='store_true', help="Use DDP")
    args = parser.parse_args()

    args.datasets_path = [Path(args.root_path, path) for path in args.datasets_path]

    set_seed(args.seed)

    if args.ddp:
        args.world_size = torch.cuda.device_count()
        assert args.world_size > 1, "You need at least 2 GPUs to train Teddy"
        mp.spawn(cleanup_on_error, args=(train, args), nprocs=args.world_size, join=True)
    else:
        train(0, args)
