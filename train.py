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
from util.ocr_scheduler import RandCheckpointScheduler, SineCheckpointScheduler, AlternatingScheduler, RandReducingScheduler, OneLinearScheduler, RandomLinearScheduler
from util.losses import SquareThresholdMSELoss, NoCudnnCTCLoss, AdversarialHingeLoss
from util.functional import TextSampler, GradSwitch, MetricCollector, Clock
from torchvision.utils import make_grid
from einops import rearrange, repeat
import wandb
from tqdm import tqdm
import time
from torch.profiler import tensorboard_trace_handler


def free_mem_percent():
    return [torch.cuda.mem_get_info(i)[0] / torch.cuda.mem_get_info(i)[1] for i in range(torch.cuda.device_count())]


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
    ocr_checkpoint_b = torch.load(args.ocr_checkpoint_b, map_location=device)
    ocr_checkpoint_c = torch.load(args.ocr_checkpoint_c, map_location=device)
    assert ocr_checkpoint_a['charset'] == ocr_checkpoint_b['charset'], "OCR checkpoints must have the same charset"

    teddy = Teddy(ocr_checkpoint_a['charset'], dim=args.dim, img_channels=args.img_channels).to(device)
    teddy_ddp = DDP(teddy, device_ids=[rank], find_unused_parameters=True) if args.ddp else teddy  # find_unused_parameters=True

    optimizer_dis = torch.optim.AdamW(teddy.discriminator.parameters(), lr=args.lr_dis)
    # scheduler_dis = torch.optim.lr_scheduler.ExponentialLR(optimizer_dis, gamma=0.9999)
    scheduler_dis = torch.optim.lr_scheduler.ConstantLR(optimizer_dis, args.lr_dis)

    optimizer_gen = torch.optim.AdamW(teddy.generator.parameters(), lr=args.lr_gen)
    # scheduler_gen = torch.optim.lr_scheduler.ExponentialLR(optimizer_gen, gamma=0.9999)
    scheduler_gen = torch.optim.lr_scheduler.ConstantLR(optimizer_gen, args.lr_gen)

    match args.ocr_scheduler:
        case 'sine': optimizer_ocr = SineCheckpointScheduler(teddy.ocr, ocr_checkpoint_a['model'], ocr_checkpoint_b['model'], period=len(loader))
        case 'alt': optimizer_ocr = AlternatingScheduler(teddy.ocr, ocr_checkpoint_a['model'], ocr_checkpoint_b['model'], ocr_checkpoint_c['model'])
        case 'line': optimizer_ocr = OneLinearScheduler(teddy.ocr, ocr_checkpoint_a['model'], ocr_checkpoint_b['model'], period=len(loader) * 20)
        case 'rand': optimizer_ocr = RandCheckpointScheduler(teddy.ocr, ocr_checkpoint_a['model'], ocr_checkpoint_b['model'])
        case 'rand_line': optimizer_ocr = RandomLinearScheduler(teddy.ocr, ocr_checkpoint_a['model'], period=len(loader) * 20)
        case 'rand_reduce': optimizer_ocr = RandReducingScheduler(teddy.ocr, ocr_checkpoint_a['model'], ocr_checkpoint_b['model'])
        case 'fixed': optimizer_ocr = AlternatingScheduler(teddy.ocr, ocr_checkpoint_a['model'])
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
    style_criterion = torch.nn.TripletMarginLoss()
    tmse_criterion = SquareThresholdMSELoss(threshold=0)
    hinge_criterion = AdversarialHingeLoss()

    text_generator = TextSampler(dataset.labels, max_len=args.max_width // 16)

    if rank == 0 and args.wandb:
        wandb.init(project='teddy', entity='fomo_aiisdh', config=args)
        wandb.watch(teddy)

    collector = MetricCollector()

    for epoch in range(args.epochs):
        teddy.train()
        teddy_ddp.train()
        epoch_start_time = time.time()

        clock_verbose = False
        clock = Clock(collector, 'time/data_load', clock_verbose)
        clock.start()

        for idx, batch in tqdm(enumerate(loader), total=len(loader), disable=rank != 0):
            batch['last_batch'] = idx == len(loader) - 1
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                clock.stop()  # time/data_load

                with Clock(collector, 'time/data_gpu', clock_verbose):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                with Clock(collector, 'time/sample_text', clock_verbose):
                    batch['gen_texts'] = text_generator.sample(len(batch['style_imgs']))

                with Clock(collector, 'time/forward', clock_verbose):
                    preds = teddy_ddp(batch)

                if idx % args.dis_critic_num == 0:
                    # Discriminator loss
                    with Clock(collector, 'time/discriminator_loss', clock_verbose):
                        dis_loss_real, dis_loss_fake = hinge_criterion.discriminator(preds['dis_fake_pred'], preds['dis_real_pred'])
                        loss_dis = (dis_loss_real + dis_loss_fake) * args.weight_dis
                        collector['loss_dis', 'dis_loss_real', 'dis_loss_fake'] = loss_dis, dis_loss_real, dis_loss_fake

                # Generator loss
                with Clock(collector, 'time/generator_loss', clock_verbose):
                    gen_loss_fake = hinge_criterion.generator(preds['dis_fake_pred'])

                    preds_size = torch.IntTensor([preds['ocr_fake_pred'].size(1)] * args.batch_size).to(device)
                    preds['ocr_fake_pred'] = preds['ocr_fake_pred'].permute(1, 0, 2).log_softmax(2)
                    ocr_loss_fake = ctc_criterion(preds['ocr_fake_pred'], preds['enc_gen_text'], preds_size, preds['enc_gen_text_len'])

                    style_loss = style_criterion(preds['fake_style_emb'], preds['real_style_emb'], preds['other_style_emb'])

                    loss_gen = gen_loss_fake * args.weight_gen + ocr_loss_fake * args.weight_ocr + style_loss * args.weight_style
                    collector['loss_gen', 'gen_loss_fake', 'ocr_loss_fake', 'style_loss'] = loss_gen, gen_loss_fake, ocr_loss_fake, style_loss

                clock.start()  # time/data_load

            with Clock(collector, 'time/backward', clock_verbose):
                if idx % args.dis_critic_num == 0:
                    optimizer_dis.zero_grad()
                optimizer_gen.zero_grad()

                if idx % args.dis_critic_num == 0:
                    with GradSwitch(teddy, teddy.discriminator):
                        scaler.scale(loss_dis).backward(retain_graph=True)
                with GradSwitch(teddy, teddy.generator):
                    scaler.scale(loss_gen).backward(retain_graph=True)

                if args.clip_grad_norm > 0:
                    if idx % args.dis_critic_num == 0:
                        scaler.unscale_(optimizer_dis)
                    scaler.unscale_(optimizer_gen)

                    if idx % args.dis_critic_num == 0:
                        torch.nn.utils.clip_grad_norm_(teddy.discriminator.parameters(), args.clip_grad_norm)
                    torch.nn.utils.clip_grad_norm_(teddy.generator.parameters(), args.clip_grad_norm)

                if idx % args.dis_critic_num == 0:
                    scaler.step(optimizer_dis)
                scaler.step(optimizer_gen)

                scaler.update()

            with Clock(collector, 'time/ocr_step', clock_verbose):
                optimizer_ocr.step()

        collector['time/epoch_train'] = time.time() - epoch_start_time
        collector['time/iter_train'] = (time.time() - epoch_start_time) / len(dataset)

        if rank == 0:
            epoch_start_time = time.time()
            if args.wandb:
                with torch.inference_mode():
                    fakes = rearrange(preds['fakes'], 'b e c h w -> (b e) c h w')
                    img_grid = make_grid(fakes, nrow=1, normalize=True, value_range=(-1, 1))

                    fake = fakes[0].detach().cpu().permute(1, 2, 0).numpy()
                    fake_pred = teddy.text_converter.decode_batch(preds['ocr_fake_pred'])[0]
                    fake_gt = batch['gen_texts'][0]

                    preds_size = torch.IntTensor([preds['ocr_real_pred'].size(1)] * args.batch_size).to(device)
                    preds['ocr_real_pred'] = preds['ocr_real_pred'].permute(1, 0, 2).log_softmax(2)
                    ocr_loss_real = ctc_criterion(preds['ocr_real_pred'], preds['enc_style_text'], preds_size, preds['enc_style_text_len'])

                    real = batch['style_imgs'][0].detach().cpu().permute(1, 2, 0).numpy()
                    real_pred = teddy.text_converter.decode_batch(preds['ocr_real_pred'])[0]
                    real_gt = batch['style_texts'][0]

                    style_imgs = make_grid(batch['style_imgs'], nrow=1, normalize=True, value_range=(-1, 1))
                    # same_author_imgs = make_grid(batch['same_author_imgs'], nrow=1, normalize=True, value_range=(-1, 1))
                    # other_author_imgs = make_grid(batch['other_author_imgs'], nrow=1, normalize=True, value_range=(-1, 1))

                collector['time/epoch_inference'] = time.time() - epoch_start_time
                collector['ocr_loss_real'] = ocr_loss_real
            collector['lr_dis', 'lr_gen'] = scheduler_dis.get_last_lr()[0], scheduler_gen.get_last_lr()[0]
            collector += teddy.collector
            collector.print(f'Epoch {epoch} | ')

            if args.wandb:
                wandb.log({
                    'alphas': optimizer_ocr.last_alpha,
                    'fakes/all': [wandb.Image(img_grid.detach().cpu().permute(1, 2, 0).numpy())],
                    'fakes/sample': [wandb.Image(fake, caption=f"GT: {fake_gt}\nP: {fake_pred}")],
                    'reals/sample': [wandb.Image(real, caption=f"GT: {real_gt}\nP: {real_pred}")],
                    'reals/style_imgs': [wandb.Image(style_imgs, caption=f"style_imgs")],
                    # 'reals/same_author_imgs': [wandb.Image(same_author_imgs, caption=f"same_author_imgs")],
                    # 'reals/other_author_imgs': [wandb.Image(other_author_imgs, caption=f"other_author_imgs")],
                } | collector.dict())

        collector.reset()
        teddy.collector.reset()

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
    parser.add_argument('--lr_gen', type=float, default=0.00001)
    parser.add_argument('--lr_dis', type=float, default=0.00001)
    parser.add_argument('--lr_ocr', type=float, default=0.0001)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=742)
    parser.add_argument('--device', type=str, default='auto', help="Device")
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
    parser.add_argument('--max_width', type=int, default=600, help="Filter images with width > max_width")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers")
    parser.add_argument('--resume', type=Path, default=None, help="Resume path")
    parser.add_argument('--wandb', action='store_true', help="Use wandb")

    # Teddy
    parser.add_argument('--dim', type=int, default=512, help="Model dimension")
    parser.add_argument('--ocr_checkpoint_a', type=Path, default='files/f745_all_datasets/0345000_iter.pth', help="OCR checkpoint a")
    parser.add_argument('--ocr_checkpoint_b', type=Path, default='files/0ea8_all_datasets/0345000_iter.pth', help="OCR checkpoint b")
    parser.add_argument('--ocr_checkpoint_c', type=Path, default='files/0259_all_datasets/0345000_iter.pth', help="OCR checkpoint c")
    parser.add_argument('--ocr_scheduler', type=str, default='alt', help="OCR scheduler")
    parser.add_argument('--weight_ocr', type=float, default=1.0, help="OCR loss weight")
    parser.add_argument('--weight_dis', type=float, default=1.0, help="Discriminator loss weight")
    parser.add_argument('--weight_gen', type=float, default=1.0, help="Generator loss weight")
    parser.add_argument('--weight_style', type=float, default=1.0, help="Style loss weight")
    parser.add_argument('--weight_mse', type=float, default=0.0, help="MSE loss weight")
    parser.add_argument('--img_channels', type=int, default=1, help="Image channels")
    parser.add_argument('--ddp', action='store_true', help="Use DDP")
    parser.add_argument('--train_ocr', action='store_true', help="Use DDP")
    parser.add_argument('--dis_critic_num', type=int, default=2, help="Discriminator critic num")
    parser.add_argument('--clip_grad_norm', type=float, default=-1, help="Clip grad norm")
    parser.add_argument('--epochs', type=int, default=10 ** 9, help="Epochs")
    args = parser.parse_args()

    args.datasets_path = [Path(args.root_path, path) for path in args.datasets_path]

    set_seed(args.seed)

    if args.ddp:
        args.world_size = torch.cuda.device_count()
        assert args.world_size > 1, "You need at least 2 GPUs to train Teddy"
        mp.spawn(cleanup_on_error, args=(train, args), nprocs=args.world_size, join=True)
    else:
        assert torch.cuda.is_available(), "You need a GPU to train Teddy"
        if args.device == 'auto':
            args.device = f'cuda:{np.argmax(free_mem_percent())}'
        train(0, args)
