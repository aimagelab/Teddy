import torch
import argparse
import random
import numpy as np
import os
import requests
import uuid
import wandb
import time
import torch.distributed as dist
import torch.multiprocessing as mp

from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torchvision.utils import make_grid
from einops import rearrange, repeat
from tqdm import tqdm
from torch.profiler import tensorboard_trace_handler

from model.teddy import Teddy, freeze, unfreeze
from model.vae import VariationalAutoencoder
from model.convvae import ConvVAE
from datasets import dataset_factory
from util.ocr_scheduler import RandCheckpointScheduler, SineCheckpointScheduler, AlternatingScheduler, RandReducingScheduler, OneLinearScheduler, RandomLinearScheduler
from util.losses import SquareThresholdMSELoss, NoCudnnCTCLoss, AdversarialHingeLoss
from util.functional import TextSampler, GradSwitch, MetricCollector, Clock, TeddyDataParallel, ChunkLoader
from datasets import transforms as T


def free_mem_percent():
    return [torch.cuda.mem_get_info(i)[0] / torch.cuda.mem_get_info(i)[1] for i in range(torch.cuda.device_count())]


def internet_connection():
    try:
        requests.get("https://www.google.com/", timeout=5)
        return True
    except requests.ConnectionError:
        return False


def gather_collectors(collector):
    metrics = collector.pytorch_tensor()
    dist.reduce(metrics, 0, op=dist.ReduceOp.SUM)
    return collector.load_pytorch_tensor(metrics)


def train(rank, args):
    device = torch.device(rank)

    args.pre_transform = T.Compose([
        T.Convert(args.img_channels),
        T.ResizeFixedHeight(args.img_height),
        T.RandomShrink(0.8, 2.0, min_width=192, max_width=2048, snap_to=16),
        T.ToTensor(),
        T.MedianRemove(),
        T.PadNextDivisible(16),
        T.Normalize((0.5,), (0.5,)),
    ])
    args.post_transform = lambda x: x

    dataset = dataset_factory('train', **args.__dict__)

    for d in dataset.datasets:
        d.batch_keys = ['style']

    print()
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                         collate_fn=dataset.collate_fn, pin_memory=True, drop_last=True)
    loader = ChunkLoader(loader, args.epochs_size)


    # vae = VariationalAutoencoder(args.latent_dim, args.img_channels).to(device)
    vae = ConvVAE(args.latent_dim, args.img_channels).to(device)

    optimizer_vae = torch.optim.AdamW(vae.parameters(), lr=args.lr_vae)
    # scheduler_vae = torch.optim.lr_scheduler.ConstantLR(optimizer_vae, args.lr_vae)

    scaler = GradScaler()

    mse_criterion = torch.nn.MSELoss(reduction='sum')

    if args.wandb and rank == 0 and not args.dryrun:
        name = f"{args.run_id}_{args.tag}"
        wandb.init(project='teddy_vae', entity='fomo_aiisdh', name=name, config=args)
        # wandb.watch(teddy, log="all", log_graph=False)  # raise error on DDP

    collector = MetricCollector()

    for epoch in range(args.start_epochs, args.epochs):
        vae.train()
        epoch_start_time = time.time()

        clock_verbose = False
        clock = Clock(collector, 'time/data_load', clock_verbose)
        clock.start()

        alpha = min(1, epoch / 10)
        # alpha = 0
        errors = 0

        good_preds = None
        for idx, batch in tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}', disable=rank != 0):
            clock.stop()  # time/data_load
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                imgs = batch['style_img'].to(device)

                preds, mu, logvar = vae(imgs)

                loss_mse = mse_criterion(preds, imgs)
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = loss_mse + alpha * kld

            if torch.isnan(loss):
                # print(f'ERROR: loss is NaN - Epoch {epoch} | {idx} | {loss} | {preds.shape} | {imgs.shape} | {imgs.min()} | {imgs.max()} | {preds.min()} | {preds.max()}')
                optimizer_vae.zero_grad()
                errors += 1
                # if errors > 50:
                #     raise ValueError("Too many NaNs in this epoch")
                continue

            collector['loss_mse'] = loss_mse
            collector['kld'] = kld
            collector['loss'] = loss
            good_preds = preds

            optimizer_vae.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer_vae)
            torch.nn.utils.clip_grad_value_(vae.parameters(), 1)

            scaler.step(optimizer_vae)
            scaler.update()
            clock.start()  # time/data_load

        collector['errors'] = errors
        collector['time/epoch_train'] = time.time() - epoch_start_time
        collector['time/iter_train'] = (time.time() - epoch_start_time) / len(dataset)

        epoch_start_time = time.time()
        if args.wandb and rank == 0:
            with torch.inference_mode():
                img_grid = make_grid(imgs[:32], nrow=1, normalize=True, value_range=(-1, 1))
                pred_grid = make_grid(good_preds[:32], nrow=1, normalize=True, value_range=(-1, 1))

            collector['time/epoch_inference'] = time.time() - epoch_start_time

        if args.wandb and rank == 0 and not args.dryrun:
            collector.print(f'Epoch {epoch} | ')
            wandb.log({
                'epoch': epoch,
                'images/all': [wandb.Image(torch.cat([img_grid, pred_grid], dim=2), caption='imgs/preds')],
            } | collector.dict())

        if rank == 0 and epoch % 10 == 0 and not args.dryrun:
            dst = args.checkpoint_path / f'{epoch:06d}_epochs_vae.pth'
            dst.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model': vae.state_dict(),
                'optimizer_vae': optimizer_vae.state_dict(),
                # 'scheduler_vae': scheduler_vae.state_dict(),
            }, dst)

        collector.reset()
        # scheduler_vae.step()
    cleanup()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


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
    

def add_arguments(parser):
    parser.add_argument('--lr_vae', type=float, default=0.00001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=742)
    parser.add_argument('--device', type=str, default='auto', help="Device")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers")
    parser.add_argument('--resume', action='store_true', help="Resume")
    parser.add_argument('--wandb', action='store_true', help="Use wandb")
    parser.add_argument('--start_epochs', type=int, default=0, help="Start epochs")
    parser.add_argument('--epochs', type=int, default=10 ** 9, help="Epochs")
    parser.add_argument('--epochs_size', type=int, default=1000, help="Epochs size")
    parser.add_argument('--world_size', type=int, default=1, help="World size")
    parser.add_argument('--checkpoint_path', type=str, default='files/checkpoints', help="Checkpoint path")
    parser.add_argument('--run_id', type=str, default=uuid.uuid4().hex[:4], help="Run id")
    parser.add_argument('--tag', type=str, default='none', help="Tag")
    parser.add_argument('--dryrun', action='store_true', help="Dryrun")

    # datasets
    parser.add_argument('--root_path', type=str, default='/mnt/scratch/datasets', help="Root path")
    parser.add_argument('--datasets_path', type=str, nargs='+', default=[
                                                    'IAM',
                                                    'Rimes',
                                                    'ICFHR16',
                                                    'ICFHR14',
                                                    'LAM_msgpack',
                                                    'Rodrigo',
                                                    'SaintGall',
                                                    'Washington',
                                                    'LEOPARDI/leopardi',
                                                    'Norhand',
                                                ], help="Datasets path")
    parser.add_argument('--datasets', type=str, nargs='+', default=[
                                                    'iam_lines',
                                                    'rimes',
                                                    'icfhr16',
                                                    'icfhr14',
                                                    'lam',
                                                    'rodrigo',
                                                    'saintgall',
                                                    'washington',
                                                    'leopardi',
                                                    'norhand',
                                                ], help="Datasets")
    parser.add_argument('--db_preload', action='store_true', help="Preload dataset")

    # VAE
    parser.add_argument('--latent_dim', type=int, default=64, help="Image channels")
    parser.add_argument('--img_channels', type=int, default=1, help="Image channels")
    parser.add_argument('--img_height', type=int, default=32, help="Image height")
    parser.add_argument('--img_max_width', type=int, default=1000, help="Image height")
    parser.add_argument('--clip_grad_norm', type=float, default=-1, help="Clip grad norm")

    # VAE loss
    parser.add_argument('--weight_mse', type=float, default=1.0, help="MSE loss weight")
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    args.datasets_path = [Path(args.root_path, path) for path in args.datasets_path]
    args.checkpoint_path = Path(args.checkpoint_path, args.run_id)

    set_seed(args.seed)
    if not internet_connection() and args.wandb:
        os.environ['WANDB_MODE'] = 'offline'
        print("No internet connection, wandb switched to offline mode")

    assert torch.cuda.is_available(), "You need a GPU to train Teddy"
    if args.device == 'auto':
        args.device = f'cuda:{np.argmax(free_mem_percent())}'

    train(0, args)
