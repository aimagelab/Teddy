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
from util.functional import TextSampler, GradSwitch, MetricCollector, Clock, TeddyDataParallel, ChunkLoader
from torchvision.utils import make_grid
from einops import rearrange, repeat
import wandb
from tqdm import tqdm
import time
from torch.profiler import tensorboard_trace_handler
import requests
import uuid


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

    if args.ddp:
        setup(rank, args.world_size)

    dataset = dataset_factory('train', **args.__dict__)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                         collate_fn=dataset.collate_fn, pin_memory=True, drop_last=True)
    loader = ChunkLoader(loader, args.epochs_size)

    ocr_checkpoint_a = torch.load(args.ocr_checkpoint_a, map_location=device)
    ocr_checkpoint_b = torch.load(args.ocr_checkpoint_b, map_location=device)
    ocr_checkpoint_c = torch.load(args.ocr_checkpoint_c, map_location=device)
    assert ocr_checkpoint_a['charset'] == ocr_checkpoint_b['charset'], "OCR checkpoints must have the same charset"

    teddy = Teddy(ocr_checkpoint_a['charset'], **args.__dict__).to(device)
    teddy_ddp = DDP(teddy, device_ids=[rank], find_unused_parameters=True) if args.ddp else teddy

    optimizer_dis = torch.optim.AdamW(teddy.discriminator.parameters(), lr=args.lr_dis)
    scheduler_dis = torch.optim.lr_scheduler.ConstantLR(optimizer_dis, args.lr_dis)

    optimizer_gen = torch.optim.AdamW(teddy.generator.parameters(), lr=args.lr_gen)
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

    text_min_len = max(args.dis_patch_width, args.style_patch_width) // args.gen_patch_width
    text_generator = TextSampler(dataset.labels, min_len=text_min_len, max_len=args.gen_text_line_len)

    if args.wandb and rank == 0:
        name = f"{args.run_id}_{args.lr_gen}_{args.weight_style}_{args.dis_critic_num}"
        wandb.init(project='teddy', entity='fomo_aiisdh', name=name, config=args)
        # wandb.watch(teddy, log="all", log_graph=False)  # raise error on DDP

    collector = MetricCollector()

    for epoch in range(args.epochs):
        teddy.train()
        epoch_start_time = time.time()

        clock_verbose = False
        clock = Clock(collector, 'time/data_load', clock_verbose)
        clock.start()

        for idx, batch in tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}', disable=rank != 0):
            batch['last_batch'] = idx == len(loader) - 1
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                clock.stop()  # time/data_load

                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch['gen_texts'] = text_generator.sample(len(batch['style_imgs']))

                preds = teddy_ddp(batch)

                # Discriminator loss
                if idx % args.dis_critic_num == 0:
                    dis_glob_loss_real, dis_glob_loss_fake = hinge_criterion.discriminator(preds['dis_glob_fake_pred'], preds['dis_glob_real_pred'])
                    dis_local_loss_real, dis_local_loss_fake = hinge_criterion.discriminator(preds['dis_local_fake_pred'], preds['dis_local_real_pred'])
                    loss_dis = (dis_glob_loss_real + dis_glob_loss_fake + dis_local_loss_fake + dis_local_loss_real) * args.weight_dis
                    collector['loss_dis', 'dis_glob_loss_real', 'dis_glob_loss_fake'] = loss_dis, dis_glob_loss_real, dis_glob_loss_fake
                    collector['dis_local_loss_real', 'dis_local_loss_fake'] = dis_local_loss_real, dis_local_loss_fake

                # Generator loss
                gen_glob_loss_fake = hinge_criterion.generator(preds['dis_glob_fake_pred'])
                gen_local_loss_fake = hinge_criterion.generator(preds['dis_local_fake_pred'])
                collector['gen_glob_loss_fake', 'gen_local_loss_fake'] = gen_glob_loss_fake, gen_local_loss_fake
                loss_gen = (gen_glob_loss_fake + gen_local_loss_fake) * args.weight_gen

                # OCR loss
                preds_size = torch.IntTensor([preds['ocr_fake_pred'].size(1)] * args.batch_size).to(device)
                preds['ocr_fake_pred'] = preds['ocr_fake_pred'].permute(1, 0, 2).log_softmax(2)
                ocr_loss_fake = ctc_criterion(preds['ocr_fake_pred'], preds['enc_gen_text'], preds_size, preds['enc_gen_text_len'])
                collector['ocr_loss_fake'] = ocr_loss_fake
                loss_gen += ocr_loss_fake * args.weight_ocr

                # Style loss
                style_glob_loss = style_criterion(preds['style_glob_fakes'], preds['style_glob_positive'], preds['style_glob_negative'])
                style_local_positive = repeat(preds['style_glob_positive'], 'b d -> (b e) d', e=args.style_patch_count)
                style_local_negative = repeat(preds['style_glob_positive'], 'b d -> (b e) d', e=args.style_patch_count)
                style_local_loss = style_criterion(preds['style_local_fakes'], style_local_positive, style_local_negative)
                collector['style_glob_loss', 'style_local_loss'] = style_glob_loss, style_local_loss
                loss_gen += (style_glob_loss + style_local_loss) * args.weight_style

                collector['loss_gen'] = loss_gen

                clock.start()  # time/data_load

            if idx % args.dis_critic_num == 0:
                optimizer_dis.zero_grad()
            optimizer_gen.zero_grad()

            if idx % args.dis_critic_num == 0:
                with GradSwitch(teddy, teddy.discriminator):
                    scaler.scale(loss_dis).backward(retain_graph=True)
            with GradSwitch(teddy, teddy.generator):
                scaler.scale(loss_gen).backward(retain_graph=True)

            if idx % args.dis_critic_num == 0:
                scaler.step(optimizer_dis)
            scaler.step(optimizer_gen)
            scaler.update()

            optimizer_ocr.step()

        collector['time/epoch_train'] = time.time() - epoch_start_time
        collector['time/iter_train'] = (time.time() - epoch_start_time) / len(dataset)

        epoch_start_time = time.time()
        if args.wandb and rank == 0:
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
        # if args.ddp:
        #     collector = gather_collectors(collector)

        if args.wandb and rank == 0:
            collector.print(f'Epoch {epoch} | ')
            wandb.log({
                'alphas': optimizer_ocr.last_alpha,
                'images/all': [wandb.Image(torch.cat([style_imgs, img_grid], dim=2), caption='real/fake')],
                'images/sample_fake': [wandb.Image(fake, caption=f"GT: {fake_gt}\nP: {fake_pred}")],
                'images/sample_real': [wandb.Image(real, caption=f"GT: {real_gt}\nP: {real_pred}")],
            } | collector.dict())

        if rank == 0 and epoch % 10 == 0:
            dst = Path(args.checkpoint_path, args.run_id, f'{epoch:06d}_epochs.pth')
            dst.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model': teddy.state_dict(),
                'optimizer_dis': optimizer_dis.state_dict(),
                'optimizer_gen': optimizer_gen.state_dict(),
                'scheduler_dis': scheduler_dis.state_dict(),
                'scheduler_gen': scheduler_gen.state_dict(),
                # 'optimizer_ocr': optimizer_ocr.state_dict(),
                'args': vars(args),
            }, dst)

        collector.reset()
        teddy.collector.reset()

        scheduler_dis.step()
        scheduler_gen.step()
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_gen', type=float, default=0.00005)
    parser.add_argument('--lr_dis', type=float, default=0.00005)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=742)
    parser.add_argument('--device', type=str, default='auto', help="Device")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers")
    parser.add_argument('--resume', type=Path, default=None, help="Resume path")
    parser.add_argument('--wandb', action='store_true', help="Use wandb")
    parser.add_argument('--epochs', type=int, default=10 ** 9, help="Epochs")
    parser.add_argument('--epochs_size', type=int, default=1000, help="Epochs size")
    parser.add_argument('--world_size', type=int, default=1, help="World size")
    parser.add_argument('--checkpoint_path', type=str, default='files/checkpoints', help="Checkpoint path")
    parser.add_argument('--run_id', type=str, default=uuid.uuid4().hex[:4], help="Run id")

    # datasets
    parser.add_argument('--root_path', type=str, default='/mnt/scratch/datasets', help="Root path")
    parser.add_argument('--datasets_path', type=str, nargs='+', default=['IAM',], help="Datasets path")
    parser.add_argument('--datasets', type=str, nargs='+', default=['iam_lines_sm',], help="Datasets")
    parser.add_argument('--db_preload', action='store_true', help="Preload dataset")

    # Teddy general
    parser.add_argument('--img_channels', type=int, default=1, help="Image channels")
    parser.add_argument('--img_height', type=int, default=32, help="Image height")
    parser.add_argument('--ddp', action='store_true', help="Use DDP")
    parser.add_argument('--clip_grad_norm', type=float, default=-1, help="Clip grad norm")

    # Teddy ocr
    parser.add_argument('--ocr_checkpoint_a', type=Path, default='files/f745_all_datasets/0345000_iter.pth', help="OCR checkpoint a")
    parser.add_argument('--ocr_checkpoint_b', type=Path, default='files/0ea8_all_datasets/0345000_iter.pth', help="OCR checkpoint b")
    parser.add_argument('--ocr_checkpoint_c', type=Path, default='files/0259_all_datasets/0355000_iter.pth', help="OCR checkpoint c")
    parser.add_argument('--ocr_scheduler', type=str, default='alt', help="OCR scheduler")

    # Teddy loss
    parser.add_argument('--weight_ocr', type=float, default=1.5, help="OCR loss weight")
    parser.add_argument('--weight_dis', type=float, default=1.0, help="Discriminator loss weight")
    parser.add_argument('--weight_gen', type=float, default=1.0, help="Generator loss weight")
    parser.add_argument('--weight_style', type=float, default=2.0, help="Style loss weight")
    parser.add_argument('--weight_mse', type=float, default=0.0, help="MSE loss weight")

    # Teddy generator
    parser.add_argument('--gen_dim', type=int, default=512, help="Model dimension")
    parser.add_argument('--gen_max_width', type=int, default=608, help="Max width")
    parser.add_argument('--gen_patch_width', type=int, default=16, help="Patch width")
    parser.add_argument('--gen_expansion_factor', type=int, default=1, help="Expansion factor")
    parser.add_argument('--gen_text_line_len', type=int, default=32, help="Text line len")

    # Teddy discriminator
    parser.add_argument('--dis_dim', type=int, default=512, help="Model dimension")
    parser.add_argument('--dis_critic_num', type=int, default=2, help="Discriminator critic num")
    parser.add_argument('--dis_patch_width', type=int, default=32, help="Discriminator patch width")
    parser.add_argument('--dis_patch_count', type=int, default=8, help="Discriminator patch count")

    # Teddy style
    parser.add_argument('--style_patch_width', type=int, default=256, help="Style patch width")
    parser.add_argument('--style_patch_count', type=int, default=4, help="Style patch count")
    args = parser.parse_args()

    args.datasets_path = [Path(args.root_path, path) for path in args.datasets_path]

    set_seed(args.seed)
    if not internet_connection() and args.wandb:
        os.environ['WANDB_MODE'] = 'offline'
        print("No internet connection, wandb switched to offline mode")

    assert torch.cuda.is_available(), "You need a GPU to train Teddy"
    if args.device == 'auto':
        args.device = f'cuda:{np.argmax(free_mem_percent())}'

    if args.ddp:
        mp.spawn(cleanup_on_error, args=(train, args), nprocs=args.world_size, join=True)
    else:
        train(0, args)
