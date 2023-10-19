import torch
import argparse
import random
import numpy as np
from model.teddy import Teddy
from pathlib import Path
from datasets import dataset_factory
from util import RandCheckpointScheduler, SineCheckpointScheduler, SquareThresholdMSELoss, TextSampler


def train(args):
    dataset = dataset_factory(args.datasets, args.datasets_path, 'train', max_width=args.max_width)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=dataset.collate_fn)

    teddy = Teddy(dataset.alphabet, dim=args.dim).to(args.device)

    optimizer_dis = torch.optim.AdamW(teddy.discriminator.parameters(), lr=args.lr_dis)
    optimizer_gen = torch.optim.AdamW(teddy.generator.parameters(), lr=args.lr_gen)

    scheduler_dis = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_dis, patience=args.patience // 3)
    scheduler_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_gen, patience=args.patience // 3)

    ocr_checkpoint_a = torch.load(args.ocr_checkpoint_a, map_location=args.device)['model']
    ocr_checkpoint_b = torch.load(args.ocr_checkpoint_b, map_location=args.device)['model']
    ocr_scheduler = SineCheckpointScheduler(teddy.ocr, ocr_checkpoint_a, ocr_checkpoint_b, period=len(loader))
    ocr_scheduler.step()

    if args.resume is not None:
        assert args.resume.exists(), f"Resume path {args.resume} doesn't exist"
        checkpoint = torch.load(args.resume)
        teddy.load_state_dict(checkpoint['model'])
        optimizer_dis.load_state_dict(checkpoint['optimizer_dis'])
        optimizer_gen.load_state_dict(checkpoint['optimizer_gen'])
        scheduler_dis.load_state_dict(checkpoint['scheduler_dis'])
        scheduler_gen.load_state_dict(checkpoint['scheduler_gen'])
        ocr_scheduler.load_state_dict(checkpoint['ocr_scheduler'])

    ctc_criterion = torch.nn.CTCLoss(reduction='mean', zero_infinity=True).to(args.device)
    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean').to(args.device)
    cycle_criterion = torch.nn.L1Loss(reduction='mean').to(args.device)
    tmse_criterion = SquareThresholdMSELoss(0.5, reduction='mean').to(args.device)

    text_generator = TextSampler(dataset.labels, 6)

    for epoch in range(10 ** 10):
        for idx, batch in enumerate(loader):
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch['gen_texts'] = text_generator.sample(len(batch['style_imgs']))

            preds = teddy(batch)

            ctc_loss = ctc_criterion(preds['gen_logits'], batch['gen_texts_len'], preds['gen_texts'], batch['gen_texts_len'])

            print()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_dis', type=float, default=0.001)
    parser.add_argument('--lr_gen', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
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
    parser.add_argument('--max_width', type=int, default=1800, help="Filter images with width > max_width")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers")
    parser.add_argument('--resume', type=Path, default=None, help="Resume path")

    # Teddy
    parser.add_argument('--dim', type=int, default=512, help="Model dimension")
    parser.add_argument('--ocr_checkpoint_a', type=Path, default='files/f745_all_datasets/0155000_iter.pth', help="OCR checkpoint a")
    parser.add_argument('--ocr_checkpoint_b', type=Path, default='files/f745_all_datasets/0315000_iter.pth', help="OCR checkpoint b")
    args = parser.parse_args()

    args.datasets_path = [Path(args.root_path, path) for path in args.datasets_path]

    set_seed(args.seed)
    train(args)

    # import string

    # device = 'cuda'
    # teddy = Teddy(string.ascii_lowercase + ' ' + string.digits, dim=512).to(device)
    # # compute parameters
    # params = sum(p.numel() for p in teddy.parameters() if p.requires_grad)
    # print(f'The model has {params} trainable parameters')

    # style_imgs = torch.cat([torch.ones((args.batch_size, 3, 32, 16)) * i for i in range(57)], dim=-1).to(device)
    # style_imgs_len = torch.randint(30, 57, (args.batch_size, )).to(device) * 16
    # same_author_imgs = torch.cat([torch.ones((args.batch_size, 3, 32, 16)) * i for i in range(100)], dim=-1).to(device)
    # same_author_imgs_len = torch.randint(30, 100, (args.batch_size, )).to(device) * 16
    # other_author_imgs = torch.cat([torch.ones((args.batch_size, 3, 32, 16)) * i for i in range(67)], dim=-1).to(device)
    # other_author_imgs_len = torch.randint(30, 67, (args.batch_size, )).to(device) * 16
    # style_text = [f'style ciao{i**i}' for i in range(args.batch_size)]
    # gen_text = [f'ciao{i**i}' for i in range(args.batch_size)]
    # out = teddy(style_imgs, style_imgs_len, style_text, gen_text, same_author_imgs, same_author_imgs_len, other_author_imgs, other_author_imgs_len)
    # print('No errors')
