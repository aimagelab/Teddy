import torch
import argparse
import random
import numpy as np
import os
from model.teddy import Teddy
from pathlib import Path
from datasets import dataset_factory
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from util import RandCheckpointScheduler, SineCheckpointScheduler, SquareThresholdMSELoss, TextSampler
from torchvision.utils import make_grid
from einops import rearrange
import wandb


def train(rank, args):
    setup(rank=rank, world_size=args.world_size)

    dataset = dataset_factory(args.datasets, args.datasets_path, 'train', max_width=args.max_width)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=dataset.collate_fn)

    device = f'cuda:{rank}'
    teddy = Teddy(dataset.alphabet, dim=args.dim).to(device)
    teddy = DDP(teddy, device_ids=[rank])

    optimizer_dis = torch.optim.AdamW(teddy.module.discriminator.parameters(), lr=args.lr_dis)
    optimizer_gen = torch.optim.AdamW(teddy.module.generator.parameters(), lr=args.lr_gen)

    scheduler_dis = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_dis, patience=args.patience // 3)
    scheduler_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_gen, patience=args.patience // 3)

    ocr_checkpoint_a = torch.load(args.ocr_checkpoint_a, map_location=device)['model']
    ocr_checkpoint_b = torch.load(args.ocr_checkpoint_b, map_location=device)['model']
    ocr_scheduler = SineCheckpointScheduler(teddy.module.ocr, ocr_checkpoint_a, ocr_checkpoint_b, period=len(loader))
    ocr_scheduler._step(0)

    scaler = GradScaler()

    if args.resume is not None:
        assert args.resume.exists(), f"Resume path {args.resume} doesn't exist"
        checkpoint = torch.load(args.resume)
        teddy.load_state_dict(checkpoint['model'])
        optimizer_dis.load_state_dict(checkpoint['optimizer_dis'])
        optimizer_gen.load_state_dict(checkpoint['optimizer_gen'])
        scheduler_dis.load_state_dict(checkpoint['scheduler_dis'])
        scheduler_gen.load_state_dict(checkpoint['scheduler_gen'])
        ocr_scheduler.load_state_dict(checkpoint['ocr_scheduler'])

    ctc_criterion = torch.nn.CTCLoss(reduction='mean', zero_infinity=True).to(device)
    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean').to(device)
    cycle_criterion = torch.nn.L1Loss(reduction='mean').to(device)
    tmse_criterion = SquareThresholdMSELoss(threshold=0)

    text_generator = TextSampler(dataset.labels, 6)

    if rank == 0 and args.wandb:
        wandb.init(project='teddy', entity='fomo_aiisdh', config=args)
        wandb.watch(teddy)

    for epoch in range(10 ** 10):
        for idx, batch in enumerate(loader):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch['gen_texts'] = text_generator.sample(len(batch['style_imgs']))

                preds = teddy(batch)

                # CTC loss from OCR
                b, w, _ = preds['texts_pred'].shape
                preds_size = torch.IntTensor([w] * b).to(device)
                preds['texts_pred'] = preds['texts_pred'].permute(1, 0, 2).log_softmax(2)
                ctc_loss = ctc_criterion(preds['texts_pred'], preds['enc_gen_texts'], preds_size, preds['enc_gen_texts_len'])
                ctc_loss *= args.weight_ctc

                # Binary cross entropy loss
                real_fake_tgt = torch.ones((4, preds['real_fake_pred'].size(0) // 4, 1), device=device)
                real_fake_tgt[::2, :, :] = 0  # 0 -> real | 1 -> fake
                real_fake_loss = bce_criterion(preds['real_fake_pred'], real_fake_tgt.reshape((-1, 1)))
                real_fake_loss *= args.weight_real_fake

                same_other_tgt = torch.ones((4, preds['same_other_pred'].size(0) // 4, 1), device=device)
                same_other_tgt[:2, :, :] = 0  # 0 -> same | 1 -> other
                same_other_tgt = (same_other_tgt.flatten() * batch['multi_authors'].repeat(4))
                same_other_loss = bce_criterion(preds['same_other_pred'], same_other_tgt.unsqueeze(-1))
                same_other_loss *= args.weight_same_other

                # Cycle loss
                src_style_emb = preds['src_style_emb'][:2].repeat(1, teddy.module.expantion_factor, 1)
                gen_style_emb = preds['gen_style_emb'][:2]
                cycle_loss = cycle_criterion(src_style_emb, gen_style_emb)
                cycle_loss *= args.weight_cycle

                # MSE loss
                mse_loss = tmse_criterion(preds['fakes'])
                mse_loss *= args.weight_mse

                # Update
                train_loss = ctc_loss + same_other_loss + real_fake_loss + cycle_loss - mse_loss

            optimizer_dis.zero_grad()
            optimizer_gen.zero_grad()

            scaler.scale(train_loss).backward()

            scaler.step(optimizer_dis)
            scaler.step(optimizer_gen)

            scaler.update()

            ocr_scheduler.step()

            if idx % 1000 == 0 and rank == 0:
                print(f"Epoch: {epoch} | Iter: {idx:06d} | CTC loss: {ctc_loss.item():.4f} | SO loss: {same_other_loss.item():.4f} | RF loss: {real_fake_loss.item():.4f} | Cycle loss: {cycle_loss.item():.4f} | MSE loss: {mse_loss.item():.4f} | Train loss: {train_loss.item():.4f}")

                if args.wandb:
                    fakes = rearrange(preds['fakes'], 'b e c h w -> (b e) c h w')
                    img_grid = make_grid(fakes, nrow=1, normalize=True, value_range=(-1, 1))

                    wandb.log({
                        'ctc_loss': ctc_loss.item(),
                        'same_other_loss': same_other_loss.item(),
                        'real_fake_loss': real_fake_loss.item(),
                        'cycle_loss': cycle_loss.item(),
                        'mse_loss': mse_loss.item(),
                        'train_loss': train_loss.item(),
                        'lr_dis': scheduler_dis.optimizer.param_groups[0]['lr'],
                        'lr_gen': scheduler_gen.optimizer.param_groups[0]['lr'],
                        'alphas': ocr_scheduler.alpha,
                        'fakes': [wandb.Image(img_grid.detach().cpu().permute(1, 2, 0).numpy())],
                    })

        scheduler_dis.step(train_loss)
        scheduler_gen.step(train_loss)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_dis', type=float, default=0.0001)
    parser.add_argument('--lr_gen', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=742)
    parser.add_argument('--root_path', type=str, default='/mnt/ssd/datasets', help="Root path")
    parser.add_argument('--datasets_path', type=str, nargs='+', default=[
        'IAM',
        'Norhand',
        'Rimes',
        'ICFHR16',
        'ICFHR14',
        'LAM_msgpack',
        'Rodrigo',
        'SaintGall',
        'Washington',
        'LEOPARDI/leopardi',
    ], help="Datasets path")
    parser.add_argument('--datasets', type=str, nargs='+', default=[
        'iam_lines',
        'norhand',
        'rimes',
        'icfhr16',
        'icfhr14',
        'lam',
        'rodrigo',
        'saintgall',
        'washington',
        'leopardi',
    ], help="Datasets")
    parser.add_argument('--max_width', type=int, default=1200, help="Filter images with width > max_width")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers")
    parser.add_argument('--resume', type=Path, default=None, help="Resume path")
    parser.add_argument('--wandb', action='store_true', help="Use wandb")

    # Teddy
    parser.add_argument('--dim', type=int, default=512, help="Model dimension")
    parser.add_argument('--ocr_checkpoint_a', type=Path, default='files/f745_all_datasets/0155000_iter.pth', help="OCR checkpoint a")
    parser.add_argument('--ocr_checkpoint_b', type=Path, default='files/f745_all_datasets/0315000_iter.pth', help="OCR checkpoint b")
    parser.add_argument('--weight_ctc', type=float, default=1.0, help="CTC loss weight")
    parser.add_argument('--weight_same_other', type=float, default=0.0, help="Same/other loss weight")
    parser.add_argument('--weight_real_fake', type=float, default=1.0, help="Real/fake loss weight")
    parser.add_argument('--weight_cycle', type=float, default=1.0, help="Cycle loss weight")
    parser.add_argument('--weight_mse', type=float, default=0.0, help="MSE loss weight")
    args = parser.parse_args()

    args.datasets_path = [Path(args.root_path, path) for path in args.datasets_path]

    set_seed(args.seed)

    args.world_size = torch.cuda.device_count()
    assert args.world_size > 1, "You need at least 2 GPUs to train Teddy"
    mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)
