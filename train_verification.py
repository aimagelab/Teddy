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
import warnings
from datasets import transforms as T

from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torchvision.utils import make_grid
from einops import rearrange, repeat
from tqdm import tqdm
from torch.profiler import tensorboard_trace_handler
from train import add_arguments, set_seed, cleanup_on_error, cleanup, count_parameters_in_millions, free_mem_percent

from model.teddy import Teddy, TeddyDiscriminator, PatchSampler
from generate_images import setup_loader, generate_images, Evaluator
from datasets import dataset_factory
from util.ocr_scheduler import RandCheckpointScheduler, SineCheckpointScheduler, AlternatingScheduler, RandReducingScheduler, OneLinearScheduler, RandomLinearScheduler
from util.losses import SquareThresholdMSELoss, NoCudnnCTCLoss, AdversarialHingeLoss, MaxMSELoss
from util.functional import TextSampler, GradSwitch, MetricCollector, Clock, TeddyDataParallel, ChunkLoader


def gather_collectors(collector):
    metrics = collector.pytorch_tensor()
    dist.reduce(metrics, 0, op=dist.ReduceOp.SUM)
    return collector.load_pytorch_tensor(metrics)


def train(rank, args):
    device = torch.device(rank)

    post_transform = T.Compose([
        T.RandomShrink(0.8, 1.2, min_width=max(args.style_patch_width, args.dis_patch_width), max_width=args.gen_max_width, snap_to=args.gen_patch_width),
        T.ToTensor(),
        T.MedianRemove(),
        T.RandomCrop((args.img_height, args.style_patch_width)),
        T.Normalize((0.5,), (0.5,))
    ])

    dataset = dataset_factory('train', post_transform=post_transform, **args.__dict__)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                         collate_fn=dataset.collate_fn, pin_memory=True, drop_last=True)
    loader = ChunkLoader(loader, args.epochs_size)

    teddy = TeddyDiscriminator((args.img_height, args.style_patch_width), (args.img_height, args.gen_patch_width), channels=args.img_channels).to(device)
    print(f'Teddy has {count_parameters_in_millions(teddy):.2f} M parameters.')

    optimizer = torch.optim.AdamW(teddy.parameters(), lr=args.lr_dis)
    # scheduler_dis = torch.optim.lr_scheduler.ConstantLR(optimizer_dis, args.lr_dis)

    scaler = GradScaler()

    ctc_criterion = NoCudnnCTCLoss(reduction='mean', zero_infinity=True).to(device)
    style_criterion = torch.nn.TripletMarginLoss()
    # tmse_criterion = SquareThresholdMSELoss(threshold=0)
    hinge_criterion = AdversarialHingeLoss()
    cycle_criterion = MaxMSELoss()
    recon_criterion = torch.nn.MSELoss()

    if args.wandb and rank == 0 and not args.dryrun:
        name = f"{args.run_id}_{args.tag}"
        wandb.init(project='teddy', entity='fomo_aiisdh', name=name, config=args)
        wandb.watch(teddy, log="all")  # raise error on DDP

    collector = MetricCollector()

    for epoch in range(args.start_epochs, args.epochs):
        teddy.train()
        epoch_start_time = time.time()

        clock_verbose = False
        clock = Clock(collector, 'time/data_load', clock_verbose)
        clock.start()

        for idx, batch in tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}', disable=rank != 0):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                clock.stop()  # time/data_load

                style_img = batch['style_img'].to(device)
                same_img = batch['same_img'].to(device)
                other_img = batch['other_img'].to(device)

                dis_real_pred = teddy(torch.cat([style_img, same_img], dim=-1))
                dis_fake_pred = teddy(torch.cat([style_img, other_img], dim=-1))

                dis_loss_real, dis_loss_fake = hinge_criterion.discriminator(dis_fake_pred, dis_real_pred)
                loss_dis = (dis_loss_real + dis_loss_fake) * args.weight_dis
                collector['dis_loss_real', 'dis_loss_fake', 'loss_dis'] = dis_loss_real, dis_loss_fake, loss_dis

                clock.start()  # time/data_load

            optimizer.zero_grad()

            with GradSwitch(teddy, teddy):
                scaler.scale(loss_dis).backward()

            scaler.step(optimizer)
            scaler.update()

        collector['time/epoch_train'] = time.time() - epoch_start_time
        collector['time/iter_train'] = (time.time() - epoch_start_time) / len(dataset)

        epoch_start_time = time.time()

        if args.wandb and rank == 0 and not args.dryrun:
            collector.print(f'Epoch {epoch} | ')
            wandb.log({
                'epoch': epoch,
                'images/same': [wandb.Image(torch.cat([style_img, same_img], dim=-1)[:16])],
                'images/other': [wandb.Image(torch.cat([style_img, other_img], dim=-1)[:16])],
            } | collector.dict())

        collector.reset()

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    args.checkpoint_path = Path(args.checkpoint_path, args.run_id)

    set_seed(args.seed)

    assert torch.cuda.is_available(), "You need a GPU to train Teddy"
    if args.device == 'auto':
        args.device = f'cuda:{np.argmax(free_mem_percent())}'

    if args.ddp:
        mp.spawn(cleanup_on_error, args=(train, args), nprocs=args.world_size, join=True)
    else:
        train(0, args)
