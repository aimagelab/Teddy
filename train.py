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

from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torchvision.utils import make_grid
from einops import rearrange, repeat
from tqdm import tqdm
from torch.profiler import tensorboard_trace_handler

from model.teddy import Teddy, freeze, unfreeze
from generate_images import setup_loader, generate_images, Evaluator
from datasets import dataset_factory
from util.ocr_scheduler import RandCheckpointScheduler, SineCheckpointScheduler, AlternatingScheduler, RandReducingScheduler, OneLinearScheduler, RandomLinearScheduler
from util.losses import SquareThresholdMSELoss, NoCudnnCTCLoss, AdversarialHingeLoss, MaxMSELoss
from util.functional import TextSampler, GradSwitch, MetricCollector, Clock, TeddyDataParallel, ChunkLoader


def free_mem_percent():
    return [torch.cuda.mem_get_info(i)[0] / torch.cuda.mem_get_info(i)[1] for i in range(torch.cuda.device_count())]

def count_parameters_in_millions(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

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

    dataset = dataset_factory('train', **args.__dict__)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                         collate_fn=dataset.collate_fn, pin_memory=True, drop_last=True)
    loader = ChunkLoader(loader, args.epochs_size)

    evaluation_loader = None 
    evaluator = Evaluator().to(device)
    evaluator.set_real_dataset(args.eval_real_dataset_path)

    ocr_checkpoint_a = torch.load(args.ocr_checkpoint_a, map_location=device)
    ocr_checkpoint_b = torch.load(args.ocr_checkpoint_b, map_location=device)
    ocr_checkpoint_c = torch.load(args.ocr_checkpoint_c, map_location=device)
    assert ocr_checkpoint_a['charset'] == ocr_checkpoint_b['charset'], "OCR checkpoints must have the same charset"

    teddy = Teddy(ocr_checkpoint_a['charset'], **args.__dict__).to(device)
    print(f'Teddy has {count_parameters_in_millions(teddy):.2f} M parameters.')

    optimizer_dis = torch.optim.AdamW(teddy.discriminator.parameters(), lr=args.lr_dis)
    # scheduler_dis = torch.optim.lr_scheduler.ConstantLR(optimizer_dis, args.lr_dis)

    
    transformer_params = [param for name, param in teddy.generator.named_parameters() if not 'cnn_decoder' in name]
    cnn_decoder_params = [param for name, param in teddy.generator.named_parameters() if 'cnn_decoder' in name]
    optimizer_gen = torch.optim.AdamW([
        {'params': transformer_params, 'lr': args.lr_gen_transformer},
        {'params': cnn_decoder_params, 'lr': args.lr_gen_cnn_decoder}
    ])
    # scheduler_gen = torch.optim.lr_scheduler.ConstantLR(optimizer_gen, args.lr_gen)

    match args.ocr_scheduler:
        case 'train': optimizer_ocr = torch.optim.AdamW(teddy.ocr.parameters(), lr=0.0001)
        case 'sine': optimizer_ocr = SineCheckpointScheduler(teddy.ocr, ocr_checkpoint_a['model'], ocr_checkpoint_b['model'], period=len(loader))
        case 'alt1': optimizer_ocr = AlternatingScheduler(teddy.ocr, ocr_checkpoint_a['model'])
        case 'alt2': optimizer_ocr = AlternatingScheduler(teddy.ocr, ocr_checkpoint_a['model'], ocr_checkpoint_b['model'])
        case 'alt3': optimizer_ocr = AlternatingScheduler(teddy.ocr, ocr_checkpoint_a['model'], ocr_checkpoint_b['model'], ocr_checkpoint_c['model'])
        case 'line': optimizer_ocr = OneLinearScheduler(teddy.ocr, ocr_checkpoint_a['model'], ocr_checkpoint_b['model'], period=len(loader) * 20)
        case 'rand': optimizer_ocr = RandCheckpointScheduler(teddy.ocr, ocr_checkpoint_a['model'], ocr_checkpoint_b['model'])
        case 'rand_line': optimizer_ocr = RandomLinearScheduler(teddy.ocr, ocr_checkpoint_a['model'], period=len(loader) * 20)
        case 'rand_reduce': optimizer_ocr = RandReducingScheduler(teddy.ocr, ocr_checkpoint_a['model'], ocr_checkpoint_b['model'])

    scaler = GradScaler()

    if args.resume and args.checkpoint_path.exists() and len(list(args.checkpoint_path.glob('*_epochs.pth'))) > 0:
        last_checkpoint = sorted(args.checkpoint_path.glob('*_epochs.pth'))[-1]
        checkpoint = torch.load(last_checkpoint)
        missing, unexpeted = teddy.load_state_dict(checkpoint['model'], strict=False)
        if len(keys := missing + unexpeted) > 0:
            if sum([not 'pos_encoding' in k for k in missing + unexpeted]) > 0:
                raise ValueError(f"Model not loaded: {keys}")
            if sum(['pos_encoding' in k for k in missing + unexpeted]) > 0:
                warnings.warn(f"Pos encoding not loaded: {keys}")
                
        optimizer_dis.load_state_dict(checkpoint['optimizer_dis'])
        optimizer_gen.load_state_dict(checkpoint['optimizer_gen'])
        # scheduler_dis.load_state_dict(checkpoint['scheduler_dis'])
        # scheduler_gen.load_state_dict(checkpoint['scheduler_gen'])
        args.start_epochs = int(last_checkpoint.name.split('_')[0]) + 1
        optimizer_ocr.load_state_dict(checkpoint['optimizer_ocr'])

    ctc_criterion = NoCudnnCTCLoss(reduction='mean', zero_infinity=True).to(device)
    style_criterion = torch.nn.TripletMarginLoss()
    # tmse_criterion = SquareThresholdMSELoss(threshold=0)
    hinge_criterion = AdversarialHingeLoss()
    cycle_criterion = MaxMSELoss()

    text_min_len = max(args.dis_patch_width, args.style_patch_width) // args.gen_patch_width
    text_generator = TextSampler(dataset.labels, min_len=text_min_len, max_len=args.gen_text_line_len)

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
            batch['ocr_real_train'] = args.ocr_scheduler == 'train'
            batch['ocr_real_eval'] = idx == len(loader) - 1
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                clock.stop()  # time/data_load

                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                gen_text = text_generator.sample(len(batch['style_img']))
                batch['gen_text'] = [g if random.random() > args.gen_same_text_ratio else s for s, g in zip(batch['style_text'], gen_text)]

                preds = teddy(batch)

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

                # Cycle loss
                cycle_loss = cycle_criterion(preds['real_style_emb'], preds['fake_style_emb'])
                collector['cycle_loss'] = cycle_loss
                loss_gen += cycle_loss * args.weight_cycle

                if batch['ocr_real_train']:
                    preds_size = torch.IntTensor([preds['ocr_real_pred'].size(1)] * args.batch_size).to(device)
                    preds['ocr_real_pred'] = preds['ocr_real_pred'].permute(1, 0, 2).log_softmax(2)
                    ocr_loss_real = ctc_criterion(preds['ocr_real_pred'], preds['enc_style_text'], preds_size, preds['enc_style_text_len'])
                    collector['ocr_loss_real'] = ocr_loss_real
                    loss_ocr = ocr_loss_real * args.weight_ocr

                # Style loss
                style_glob_loss = style_criterion(preds['style_glob_fakes'], preds['style_glob_positive'], preds['style_glob_negative'])
                style_local_loss = style_criterion(preds['style_local_fakes'], preds['style_local_real'], preds['style_local_other'])
                appea_local_loss = style_criterion(preds['appea_local_fakes'], preds['appea_local_real'], preds['appea_local_other'])
                collector['style_glob_loss', 'style_local_loss', 'appea_local_loss'] = style_glob_loss, style_local_loss, appea_local_loss
                loss_gen += (style_glob_loss + style_local_loss) * args.weight_style + appea_local_loss * args.weight_appea

                collector['loss_gen'] = loss_gen

                clock.start()  # time/data_load

            if idx % args.dis_critic_num == 0:
                optimizer_dis.zero_grad()
            optimizer_gen.zero_grad()
            optimizer_ocr.zero_grad()

            if idx % args.dis_critic_num == 0:
                with GradSwitch(teddy, teddy.discriminator):
                    scaler.scale(loss_dis).backward(retain_graph=True)
            with GradSwitch(teddy, teddy.generator):
                scaler.scale(loss_gen).backward(retain_graph=True)
            if batch['ocr_real_train']:
                with GradSwitch(teddy, teddy.ocr):
                    scaler.scale(loss_ocr).backward(retain_graph=True)

            # Check gradient magnitude
            collector['grad/transformer'] = np.array([param.grad.float().abs().mean().item() for param in transformer_params]).mean()
            collector['grad/cnn_decoder'] = np.array([param.grad.float().abs().mean().item() for param in cnn_decoder_params]).mean()

            if idx % args.dis_critic_num == 0:
                scaler.step(optimizer_dis)
            scaler.step(optimizer_gen)
            if batch['ocr_real_train']:
                scaler.step(optimizer_ocr)
            else:
                optimizer_ocr.step()

            scaler.update()

        collector['time/epoch_train'] = time.time() - epoch_start_time
        collector['time/iter_train'] = (time.time() - epoch_start_time) / len(dataset)

        epoch_start_time = time.time()
        if args.wandb and rank == 0:
            with torch.inference_mode():
                img_grid = make_grid(preds['fakes'], nrow=1, normalize=True, value_range=(-1, 1))

                fake = preds['fakes'][0].detach().cpu().permute(1, 2, 0).numpy()
                fake_pred = teddy.text_converter.decode_batch(preds['ocr_fake_pred'])[0]
                fake_gt = batch['gen_text'][0]

                if not batch['ocr_real_train']: 
                    preds_size = torch.IntTensor([preds['ocr_real_pred'].size(1)] * args.batch_size).to(device)
                    preds['ocr_real_pred'] = preds['ocr_real_pred'].permute(1, 0, 2).log_softmax(2)
                    ocr_loss_real = ctc_criterion(preds['ocr_real_pred'], preds['enc_style_text'], preds_size, preds['enc_style_text_len'])

                real = batch['style_img'][0].detach().cpu().permute(1, 2, 0).numpy()
                real_pred = teddy.text_converter.decode_batch(preds['ocr_real_pred'])[0]
                real_gt = batch['style_text'][0]

                style_img = make_grid(batch['style_img'], nrow=1, normalize=True, value_range=(-1, 1))
                # same_author_imgs = make_grid(batch['same_author_imgs'], nrow=1, normalize=True, value_range=(-1, 1))
                # other_author_imgs = make_grid(batch['other_author_imgs'], nrow=1, normalize=True, value_range=(-1, 1))

                eval_page = teddy.generate_eval_page(batch['gen_text'], batch['style_text'], batch['style_img'])

            collector['time/epoch_inference'] = time.time() - epoch_start_time
            collector['ocr_loss_real'] = ocr_loss_real
        # collector['lr_dis', 'lr_gen'] = scheduler_dis.get_last_lr()[0], scheduler_gen.get_last_lr()[0]
        collector['lr_dis', 'lr_gen_transformer', 'lr_gen_cnn_decoder'] = args.lr_dis, args.lr_gen_transformer, args.lr_gen_cnn_decoder
        collector += teddy.collector
        # if args.ddp:
        #     collector = gather_collectors(collector)

        if rank == 0 and epoch % 25 == 0 and epoch > 0 and not args.dryrun:
            dst = args.checkpoint_path / f'{epoch:06d}_epochs.pth'
            dst.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model': teddy.state_dict(),
                'optimizer_dis': optimizer_dis.state_dict(),
                'optimizer_gen': optimizer_gen.state_dict(),
                # 'scheduler_dis': scheduler_dis.state_dict(),
                # 'scheduler_gen': scheduler_gen.state_dict(),
                'optimizer_ocr': optimizer_ocr.state_dict(),
                'args': vars(args),
            }, dst)

            evaluation_loader = setup_loader(rank, args) if evaluation_loader else evaluation_loader
            teddy.eval()
            generate_images(rank, args, teddy, evaluation_loader)
            collector['scores/HWD', 'scores/FID', 'scores/KID'] = evaluator.compute_metrics(dst.parent / 'saved_images' / dst.stem / 'test')

        if args.wandb and rank == 0 and not args.dryrun:
            collector.print(f'Epoch {epoch} | ')
            wandb.log({
                'alphas': optimizer_ocr.last_alpha if hasattr(optimizer_ocr, 'last_alpha') else None,
                'epoch': epoch,
                'images/all': [wandb.Image(torch.cat([style_img, img_grid], dim=2), caption='real/fake')],
                'images/page': [wandb.Image(eval_page, caption='eval')],
                'images/sample_fake': [wandb.Image(fake, caption=f"GT: {fake_gt}\nP: {fake_pred}")],
                'images/sample_real': [wandb.Image(real, caption=f"GT: {real_gt}\nP: {real_pred}")],
            } | collector.dict())

        collector.reset()
        teddy.collector.reset()

        # scheduler_dis.step()
        # scheduler_gen.step()
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
    parser.add_argument('--lr_gen_transformer', type=float, default=0.000001)
    parser.add_argument('--lr_gen_cnn_decoder', type=float, default=0.0005)
    parser.add_argument('--lr_dis', type=float, default=0.00005)
    parser.add_argument('--lr_ocr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=4)
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

    # Evaluation
    parser.add_argument('--eval_real_dataset_path', type=Path, default='files/iam', help="Real dataset path")
    parser.add_argument('--eval_avg_char_width_16', action='store_true')
    parser.add_argument('--eval_epoch', type=int)
    parser.add_argument('--eval_batch_size', type=int, default=64)
                        
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

    # Teddy ocr  files/ocr_checkpoints/0ea8_ocr_b.pth
    parser.add_argument('--ocr_checkpoint_a', type=Path, default='files/ocr_checkpoints/f745_ocr_a.pth', help="OCR checkpoint a")
    parser.add_argument('--ocr_checkpoint_b', type=Path, default='files/ocr_checkpoints/0ea8_ocr_b.pth', help="OCR checkpoint b")
    parser.add_argument('--ocr_checkpoint_c', type=Path, default='files/ocr_checkpoints/0259_ocr_c.pth', help="OCR checkpoint c")
    parser.add_argument('--ocr_scheduler', type=str, default='alt3', help="OCR scheduler")

    # Teddy loss
    parser.add_argument('--weight_ocr', type=float, default=1.5, help="OCR loss weight")
    parser.add_argument('--weight_dis', type=float, default=1.0, help="Discriminator loss weight")
    parser.add_argument('--weight_gen', type=float, default=1.0, help="Generator loss weight")
    parser.add_argument('--weight_style', type=float, default=2.0, help="Style loss weight")
    parser.add_argument('--weight_appea', type=float, default=3.0, help="Appearance loss weight")
    parser.add_argument('--weight_cycle', type=float, default=1.0, help="Cycle loss weight")

    # Teddy generator
    parser.add_argument('--gen_dim', type=int, default=512, help="Model dimension")
    parser.add_argument('--gen_max_width', type=int, default=608, help="Max width")
    parser.add_argument('--gen_patch_width', type=int, default=16, help="Patch width")
    parser.add_argument('--gen_expansion_factor', type=int, default=1, help="Expansion factor")
    parser.add_argument('--gen_text_line_len', type=int, default=24, help="Text line len")
    parser.add_argument('--gen_same_text_ratio', type=float, default=0.5, help="Same text ratio")
    parser.add_argument('--gen_emb_module', type=str, default='UnifontModule', help="Embedding module")
    parser.add_argument('--gen_emb_shift', type=eval, default=(0, 0), help="Embedding shift")
    parser.add_argument('--gen_glob_style_tokens', type=int, default=3, help="Text line len")

    # Teddy vae
    parser.add_argument('--vae_dim', type=int, default=512, help="Model dimension")
    parser.add_argument('--vae_channels', type=int, default=16, help="Patch width")
    parser.add_argument('--vae_checkpoint', type=str, default='files/vae_checkpoints/fe26_vae.pth', help="VAE checkpoint")

    # Teddy discriminator
    parser.add_argument('--dis_dim', type=int, default=512, help="Model dimension")
    parser.add_argument('--dis_critic_num', type=int, default=2, help="Discriminator critic num")
    parser.add_argument('--dis_patch_width', type=int, default=32, help="Discriminator patch width")
    parser.add_argument('--dis_patch_count', type=int, default=8, help="Discriminator patch count")

    # Teddy style
    parser.add_argument('--style_patch_width', type=int, default=256, help="Style patch width")
    parser.add_argument('--style_patch_count', type=int, default=4, help="Style patch count")
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

    if args.ddp:
        mp.spawn(cleanup_on_error, args=(train, args), nprocs=args.world_size, join=True)
    else:
        train(0, args)
