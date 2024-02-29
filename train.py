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
from torchvision.transforms import functional as F

from model.teddy import Teddy, freeze, unfreeze
from generate_images import setup_loader, generate_images, Evaluator
from datasets import dataset_factory
from util.ocr_scheduler import RandCheckpointScheduler, SineCheckpointScheduler, AlternatingScheduler, RandReducingScheduler, OneLinearScheduler, RandomLinearScheduler
from util.losses import SquareThresholdMSELoss, NoCudnnCTCLoss, AdversarialHingeLoss, MaxMSELoss
from util.functional import TextSampler, GradSwitch, MetricCollector, Clock, WeightsScheduler, ChunkLoader, FakeOptimizer


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
    print(f"Dataset has {len(dataset)} samples.")
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

    args.wid_num_authors = len(set(dataset.authors))
    teddy = Teddy(ocr_checkpoint_a['charset'], **args.__dict__).to(device)
    print(f'Teddy has {count_parameters_in_millions(teddy):.2f} M parameters.')

    optimizer_dis = torch.optim.AdamW(teddy.discriminator.parameters(), lr=args.lr_dis)
    optimizer_wid = torch.optim.AdamW(teddy.writer_discriminator.parameters(), lr=args.lr_wid) if args.weight_writer_id > 0 else FakeOptimizer()
    
    transformer_params = [param for name, param in teddy.generator.named_parameters() if not 'cnn_decoder' in name]
    cnn_decoder_params = [param for name, param in teddy.generator.named_parameters() if 'cnn_decoder' in name]
    optimizer_gen = torch.optim.AdamW([
        {'params': transformer_params, 'lr': args.lr_gen_transformer},
        {'params': cnn_decoder_params, 'lr': args.lr_gen_cnn_decoder}
    ])
    # scheduler_gen = torch.optim.lr_scheduler.ConstantLR(optimizer_gen, args.lr_gen)

    weights_scheduler = WeightsScheduler(args, 'weight_ocr', args.scheduler_start, args.scheduler_lenght)

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
                warnings.warn(f"Model not properly loaded: {keys}")
            if sum(['pos_encoding' in k for k in missing + unexpeted]) > 0:
                warnings.warn(f"Pos encoding not loaded: {keys}")
                
        optimizer_dis.load_state_dict(checkpoint['optimizer_dis'])
        optimizer_gen.load_state_dict(checkpoint['optimizer_gen'])
        optimizer_wid.load_state_dict(checkpoint['optimizer_wid'])
        # scheduler_dis.load_state_dict(checkpoint['scheduler_dis'])
        # scheduler_gen.load_state_dict(checkpoint['scheduler_gen'])
        args.start_epochs = int(last_checkpoint.name.split('_')[0]) + 1
        optimizer_ocr.load_state_dict(checkpoint['optimizer_ocr'])

    ctc_criterion = NoCudnnCTCLoss(reduction='mean', zero_infinity=True).to(device)
    style_criterion = torch.nn.TripletMarginLoss()
    # tmse_criterion = SquareThresholdMSELoss(threshold=0)
    hinge_criterion = AdversarialHingeLoss()
    cycle_criterion = MaxMSELoss()
    recon_criterion = torch.nn.MSELoss()
    wid_criterion = torch.nn.CrossEntropyLoss()

    text_min_len = max(args.dis_patch_width, args.style_patch_width) // args.gen_patch_width
    text_generator = TextSampler(dataset.labels, min_len=text_min_len, max_len=None, charset=set(dataset.char_to_idx.keys()))

    if args.wandb and rank == 0 and not args.dryrun:
        name = f"{args.run_id}_{args.tag}"
        wandb.init(project='teddy', entity='fomo_aiisdh', name=name, config=args)
        wandb.watch(teddy, log="all")  # raise error on DDP

    collector = MetricCollector()

    for epoch in range(args.start_epochs, args.epochs):
        teddy.train()
        epoch_start_time = time.time()
        
        weights_scheduler.step(epoch)
        collector.update({k.replace('_', '/', 1):v for k, v in vars(args).items() if k.startswith('weight')})

        clock_verbose = False
        clock = Clock(collector, 'time/data_load', clock_verbose)
        clock.start()

        for idx, batch in tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}', disable=rank != 0):
            clock.stop()  # time/data_load
            batch['ocr_real_train'] = args.ocr_scheduler == 'train'
            batch['ocr_real_eval'] = idx == len(loader) - 1 or args.ocr_real

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                if args.weight_dis_global > 0 or args.weight_dis_local > 0:
                    with Clock(collector, 'time/gen_text', clock_verbose):
                        gen_text = text_generator.sample([len(t.split()) for t in batch['style_text']])
                    batch['gen_text'] = [g if random.random() > args.gen_same_text_ratio else s for s, g in zip(batch['style_text'], gen_text)]
                else:
                    batch['gen_text'] = text_generator.sample([4] * len(batch['style_text']))
                batch['weight'] = {k[7:]: v > 0 for k, v in vars(args).items() if k.startswith('weight')}

                if args.no_style_text:
                    batch['style_text'] = ['' for _ in batch['style_text']]

                with Clock(collector, 'time/forward', clock_verbose):
                    preds = teddy(batch)

                loss_dis, loss_gen, loss_ocr, loss_wid = 0, 0, 0, 0

                # Discriminator loss
                if idx % args.dis_critic_num == 0 and args.weight_dis_global > 0:
                    dis_glob_loss_real, dis_glob_loss_fake = hinge_criterion.discriminator(preds['dis_glob_fake_pred'], preds['dis_glob_real_pred'])
                    collector['dis_glob_loss_real', 'dis_glob_loss_fake'] = dis_glob_loss_real, dis_glob_loss_fake
                    loss_dis += (dis_glob_loss_real + dis_glob_loss_fake) * args.weight_dis_global
                if idx % args.dis_critic_num == 0 and args.weight_dis_local > 0:
                    dis_local_loss_real, dis_local_loss_fake = hinge_criterion.discriminator(preds['dis_local_fake_pred'], preds['dis_local_real_pred'])
                    collector['dis_local_loss_real', 'dis_local_loss_fake'] = dis_local_loss_real, dis_local_loss_fake
                    loss_dis += (dis_local_loss_real + dis_local_loss_fake) * args.weight_dis_local

                # Generator loss
                if args.weight_gen_global > 0:
                    gen_glob_loss_fake = hinge_criterion.generator(preds['dis_glob_fake_pred'])
                    collector['gen_glob_loss_fake'] = gen_glob_loss_fake
                    loss_gen += gen_glob_loss_fake * args.weight_gen_global
                if args.weight_gen_local > 0:
                    gen_local_loss_fake = hinge_criterion.generator(preds['dis_local_fake_pred'])
                    collector['gen_local_loss_fake'] = gen_local_loss_fake
                    loss_gen += gen_local_loss_fake * args.weight_gen_local

                # OCR loss fake
                if args.weight_ocr > 0:
                    preds_size = torch.IntTensor([preds['ocr_fake_pred'].size(1)] * args.batch_size).to(device)
                    preds['ocr_fake_pred'] = preds['ocr_fake_pred'].permute(1, 0, 2).log_softmax(2)
                    ocr_loss_fake = ctc_criterion(preds['ocr_fake_pred'], preds['enc_gen_text'], preds_size, preds['enc_gen_text_len'])
                    collector['ocr_loss_fake'] = ocr_loss_fake
                    loss_gen += ocr_loss_fake * args.weight_ocr

                # OCR loss real
                if batch['ocr_real_train'] and args.weight_ocr > 0:
                    preds_size = torch.IntTensor([preds['ocr_real_pred'].size(1)] * args.batch_size).to(device)
                    preds['ocr_real_pred'] = preds['ocr_real_pred'].permute(1, 0, 2).log_softmax(2)
                    ocr_loss_real = ctc_criterion(preds['ocr_real_pred'], preds['enc_style_text'], preds_size, preds['enc_style_text_len'])
                    collector['ocr_loss_real'] = ocr_loss_real
                    loss_ocr += ocr_loss_real * args.weight_ocr

                # Cycle loss
                if args.weight_cycle > 0:
                    cycle_loss = cycle_criterion(preds['real_style_emb'], preds['fake_style_emb'])
                    collector['cycle_loss'] = cycle_loss
                    loss_gen += cycle_loss * args.weight_cycle

                # Style loss
                if args.weight_style_global > 0:
                    style_glob_loss = style_criterion(preds['style_glob_fakes'], preds['style_glob_positive'], preds['style_glob_negative'])
                    collector['style_glob_loss'] = style_glob_loss
                    loss_gen += style_glob_loss * args.weight_style_global
                if args.weight_style_local > 0:
                    style_local_loss = style_criterion(preds['style_local_fakes'], preds['style_local_real'], preds['style_local_other'])
                    collector['style_local_loss'] = style_local_loss
                    loss_gen += style_local_loss * args.weight_style_local

                # Appearance loss
                if args.weight_appea_local > 0:
                    appea_local_loss = style_criterion(preds['appea_local_fakes'], preds['appea_local_real'], preds['appea_local_other'])
                    loss_gen += appea_local_loss * args.weight_appea
                    collector['appea_local_loss'] = appea_local_loss

                # Reconstruction loss
                if args.weight_recon > 0:
                    if preds['fakes_recon'].size(-1) > batch['style_img'].size(-1):
                        style_img = F.pad(batch['style_img'], (0, 0, preds['fakes_recon'].size(-1) - batch['style_img'].size(-1), 0), fill=1)
                    else:
                        style_img = batch['style_img'][..., :preds['fakes_recon'].size(-1)]
                        
                    recon_loss = recon_criterion(preds['fakes_recon'], style_img)
                    collector['recon_loss'] = recon_loss
                    loss_gen += recon_loss * args.weight_recon

                # Writer id loss
                if args.weight_writer_id > 0:
                    patches = preds['fake_local_writer_id'].size(0) // preds['fake_global_writer_id'].size(0)
                    style_local_author_idx = repeat(batch['style_author_idx'], 'b -> (b p)', p=patches)
                    wid_global_fake_loss = wid_criterion(preds['fake_global_writer_id'], batch['style_author_idx'].long())
                    wid_local_fake_loss = wid_criterion(preds['fake_local_writer_id'], style_local_author_idx.long())
                    collector['wid_global_fake_loss', 'wid_local_fake_loss'] = wid_global_fake_loss, wid_local_fake_loss
                    loss_gen += (wid_global_fake_loss + wid_local_fake_loss) * args.weight_writer_id

                    wid_global_real_loss = wid_criterion(preds['real_global_writer_id'], batch['style_author_idx'].long())
                    wid_local_real_loss = wid_criterion(preds['real_local_writer_id'], style_local_author_idx.long())
                    collector['wid_global_real_loss', 'wid_local_real_loss'] = wid_global_real_loss, wid_local_real_loss
                    loss_wid = (wid_global_real_loss + wid_local_real_loss) * args.weight_writer_id

                if idx % args.dis_critic_num == 0:
                    collector['loss_dis'] = loss_dis
                collector['loss_gen'] = loss_gen
                collector['loss_ocr'] = loss_ocr
                collector['loss_wid'] = loss_wid
                # end autocast

            optimizer_dis.zero_grad()
            optimizer_wid.zero_grad()
            optimizer_gen.zero_grad()
            optimizer_ocr.zero_grad()

            if idx % args.dis_critic_num == 0 and (args.weight_dis_global > 0 or args.weight_dis_local > 0):
                with Clock(collector, 'time/backward_dis', clock_verbose):
                    with GradSwitch(teddy, teddy.discriminator):
                        scaler.scale(loss_dis).backward(retain_graph=True)
            with Clock(collector, 'time/backward_gen', clock_verbose):
                with GradSwitch(teddy, teddy.generator):
                    scaler.scale(loss_gen).backward(retain_graph=True)
            if batch['ocr_real_train']:
                with Clock(collector, 'time/backward_ocr', clock_verbose):
                    with GradSwitch(teddy, teddy.ocr):
                        scaler.scale(loss_ocr).backward(retain_graph=True)
            if args.weight_writer_id > 0:
                with Clock(collector, 'time/backward_wid', clock_verbose):
                    with GradSwitch(teddy, teddy.writer_discriminator):
                        scaler.scale(loss_wid).backward(retain_graph=True)

            if idx % args.dis_critic_num == 0 and (args.weight_dis_global > 0 or args.weight_dis_local > 0):
                scaler.step(optimizer_dis)
            scaler.step(optimizer_gen)
            scaler.step(optimizer_wid) if args.weight_writer_id > 0 else None
            scaler.step(optimizer_ocr) if batch['ocr_real_train'] else optimizer_ocr.step()

            scaler.update()
            clock.start()  # time/data_load

        # end of epoch
        collector['time/epoch_train'] = time.time() - epoch_start_time
        collector['time/iter_train'] = (time.time() - epoch_start_time) / len(dataset)

        epoch_start_time = time.time()
        if args.wandb and rank == 0:
            with torch.inference_mode():
                img_grid = make_grid(preds['fakes'], nrow=1, normalize=True, value_range=(-1, 1))

                if args.weight_recon > 0:
                    img_grid_recon = make_grid(preds['fakes_recon'], nrow=1, normalize=True, value_range=(-1, 1))

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
        # collector['lr_gen_transformer', 'lr_gen_cnn_decoder'] = args.lr_gen_transformer, args.lr_gen_cnn_decoder
        # collector['lr_dis_global', 'lr_dis_local'] = args.lr_dis_global, args.lr_dis_local
        collector += teddy.collector
        # if args.ddp:
        #     collector = gather_collectors(collector)

        if rank == 0 and epoch % args.save_interval == 0 and epoch > 0 and not args.dryrun:
            dst = args.checkpoint_path / f'{epoch:06d}_epochs.pth'
            dst.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model': teddy.state_dict(),
                'optimizer_dis': optimizer_dis.state_dict(),
                'optimizer_gen': optimizer_gen.state_dict(),
                'optimizer_wid': optimizer_wid.state_dict(),
                # 'scheduler_dis': scheduler_dis.state_dict(),
                # 'scheduler_gen': scheduler_gen.state_dict(),
                'optimizer_ocr': optimizer_ocr.state_dict(),
                'args': vars(args),
            }, dst)

            # Evaluation
            try:
                evaluation_loader = setup_loader(rank, args, args.eval_dataset) if evaluation_loader else evaluation_loader
                teddy.eval()
                generate_images(rank, args, teddy, evaluation_loader)
                collector['scores/HWD', 'scores/FID', 'scores/KID'] = evaluator.compute_metrics(dst.parent / 'saved_images' / dst.stem / 'test')
            except Exception as e:
                print(f"Error during evaluation: {e}")
            finally:
                teddy.train()

        if args.wandb and rank == 0 and not args.dryrun:
            collector.print(f'Epoch {epoch} | ')
            wandb.log({
                'alphas': optimizer_ocr.last_alpha if hasattr(optimizer_ocr, 'last_alpha') else None,
                'epoch': epoch,
                'images/all': [wandb.Image(torch.cat([style_img, img_grid], dim=2), caption='real/fake')],
                'images/all_recon': [wandb.Image(torch.cat([style_img, img_grid_recon], dim=2), caption='real/fake')] if args.weight_recon > 0 else None,
                'images/page': [wandb.Image(eval_page, caption='eval')],
                'images/sample_fake': [wandb.Image(fake, caption=f"GT: {fake_gt}\nP: {fake_pred}")],
                'images/sample_real': [wandb.Image(real, caption=f"GT: {real_gt}\nP: {real_pred}")],
            } | collector.dict())

        collector.reset()
        teddy.collector.reset()


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
    parser.add_argument('--lr_wid', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=742)
    parser.add_argument('--device', type=str, default='auto', help="Device")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers")
    parser.add_argument('--resume', action='store_true', help="Resume")
    parser.add_argument('--wandb', action='store_true', help="Use wandb")
    parser.add_argument('--start_epochs', type=int, default=0, help="Start epochs")
    parser.add_argument('--epochs', type=int, default=10 ** 9, help="Epochs")
    parser.add_argument('--epochs_size', type=int, default=1000, help="Epochs size")
    parser.add_argument('--save_interval', type=int, default=10, help="Save interval")
    parser.add_argument('--world_size', type=int, default=1, help="World size")
    parser.add_argument('--checkpoint_path', type=str, default='files/checkpoints', help="Checkpoint path")
    parser.add_argument('--run_id', type=str, default=uuid.uuid4().hex[:4], help="Run id")
    parser.add_argument('--tag', type=str, default='none', help="Tag")
    parser.add_argument('--dryrun', action='store_true', help="Dryrun")
    parser.add_argument('--ocr_real', action='store_true', help="Dryrun")

    # Evaluation
    parser.add_argument('--eval_real_dataset_path', type=Path, default='files/iam', help="Real dataset path")
    parser.add_argument('--eval_dataset', type=str, default='iam_eval', help="Eval dataset")
    parser.add_argument('--eval_avg_char_width_16', action='store_true')
    parser.add_argument('--eval_epoch', type=int)
    parser.add_argument('--eval_batch_size', type=int, default=64)
                        
    # Ablation
    parser.add_argument('--no_style_text', action='store_true', help="No style text")
    parser.add_argument('--single_img_dis', action='store_true', help="Single img discriminator")
    
    # datasets
    parser.add_argument('--root_path', type=str, default='/mnt/scratch/datasets', help="Root path")
    parser.add_argument('--datasets', type=str, nargs='+', default=['iam_lines_xl',], help="Datasets")
    parser.add_argument('--db_preload', action='store_true', help="Preload dataset")

    # Teddy general
    parser.add_argument('--img_channels', type=int, default=1, help="Image channels")
    parser.add_argument('--img_height', type=int, default=32, help="Image height")
    parser.add_argument('--ddp', action='store_true', help="Use DDP")
    parser.add_argument('--clip_grad_norm', type=float, default=-1, help="Clip grad norm")

    # Teddy ocr
    parser.add_argument('--ocr_checkpoint_a', type=Path, default='files/ocr_checkpoints/f745_ocr_a.pth', help="OCR checkpoint a")
    parser.add_argument('--ocr_checkpoint_b', type=Path, default='files/ocr_checkpoints/0ea8_ocr_b.pth', help="OCR checkpoint b")
    parser.add_argument('--ocr_checkpoint_c', type=Path, default='files/ocr_checkpoints/0259_ocr_c.pth', help="OCR checkpoint c")
    parser.add_argument('--ocr_scheduler', type=str, default='alt3', help="OCR scheduler")

    # Teddy loss
    parser.add_argument('--weight_gen_global', '--wgg', type=float, default=1.0, help="Generator global loss weight")
    parser.add_argument('--weight_gen_local', '--wgl', type=float, default=1.0, help="Generator local loss weight")
    parser.add_argument('--weight_dis_global', '--wdg', type=float, default=1.0, help="Discriminator global loss weight")
    parser.add_argument('--weight_dis_local', '--wdl', type=float, default=1.0, help="Discriminator local loss weight")
    parser.add_argument('--weight_ocr', '--wo', type=float, default=1.0, help="OCR loss weight")
    parser.add_argument('--weight_style_global', '--wsg', type=float, default=1.0, help="Style gloabl loss weight")
    parser.add_argument('--weight_style_local', '--wsl', type=float, default=1.0, help="Style local loss weight")
    parser.add_argument('--weight_appea_local', '--wal', type=float, default=0.0, help="Appearance local loss weight")
    parser.add_argument('--weight_cycle', '--wc', type=float, default=1.0, help="Cycle loss weight")
    parser.add_argument('--weight_recon', '--wr', type=float, default=0.0, help="Reconstruction loss weight")
    parser.add_argument('--weight_writer_id', '--wid', type=float, default=0.0, help="Writer id loss weight")
    parser.add_argument('--scheduler_start', type=int, help="Weight scheduler start")
    parser.add_argument('--scheduler_lenght', type=int, default=10, help="Weight scheduler length")
    
    # Teddy generator
    parser.add_argument('--gen_dim', type=int, default=512, help="Model dimension")
    parser.add_argument('--gen_max_width', type=int, default=608, help="Max width")
    parser.add_argument('--gen_patch_width', type=int, default=16, help="Patch width")
    parser.add_argument('--gen_expansion_factor', type=int, default=1, help="Expansion factor")
    parser.add_argument('--gen_text_line_len', type=int, default=24, help="Text line len")
    parser.add_argument('--gen_same_text_ratio', type=float, default=0.5, help="Same text ratio")
    parser.add_argument('--gen_emb_module', type=str, default='OnehotModule', help="Embedding module")
    parser.add_argument('--gen_emb_shift', type=eval, default=(0, 0), help="Embedding shift")
    parser.add_argument('--gen_glob_style_tokens', type=int, default=3, help="Text line len")
    parser.add_argument('--gen_cnn_decoder_width', type=int, default=16, choices=[8, 16], help="CNN decoder width")

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

    args.checkpoint_path = Path(args.checkpoint_path, args.run_id)

    set_seed(args.seed)
    if not internet_connection() and args.wandb:
        os.environ['WANDB_MODE'] = 'offline'
        print("No internet connection, wandb switched to offline mode")

    assert torch.cuda.is_available(), "You need a GPU to train Teddy"
    if args.device == 'auto':
        args.device = f'cuda:{np.argmax(free_mem_percent())}'

    train(0, args)
