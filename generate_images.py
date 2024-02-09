import argparse
import torch
import torch.nn as nn
import numpy as np
import warnings
from pathlib import Path
from train import add_arguments, set_seed, free_mem_percent
from datasets import dataset_factory
from model.teddy import Teddy
from torchvision.utils import save_image
from tqdm import tqdm
from util.functional import ChunkLoader
from datasets import transforms as T
import concurrent.futures


def process_image(fakes, gen_texts, dsts, fakes_path):
    for fake, txt, dst in zip(fakes, gen_texts, dsts):
        fake = fake[:, :, :16 * len(txt)]
        fake_dst = fakes_path / Path(dst)
        fake_dst.parent.mkdir(parents=True, exist_ok=True)
        save_image(fake, fake_dst)


@torch.no_grad()
def generate_images(rank, args):
    device = torch.device(rank)

    args.datasets = ['iam_eval']
    datasets_kwargs = args.__dict__.copy()
    datasets_kwargs['pre_transform'] = T.Compose([
            T.Convert(1),
            T.ResizeFixedHeight(32),
            T.FixedCharWidth(16) if args.avg_char_width_16 else lambda x: x,
            T.ToTensor(),
            T.PadNextDivisible(16),
            T.Normalize((0.5,), (0.5,))
        ])
    datasets_kwargs['post_transform'] = lambda x: x
    dataset = dataset_factory('all', **datasets_kwargs)
    dataset.batch_keys('style')

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                         collate_fn=dataset.collate_fn, pin_memory=True, drop_last=False)

    ocr_checkpoint_a = torch.load(args.ocr_checkpoint_a, map_location=device)
    teddy = Teddy(ocr_checkpoint_a['charset'], **args.__dict__).to(device)
    teddy.eval()

    if args.checkpoint_path.exists() and len(list(args.checkpoint_path.glob('*_epochs.pth'))) > 0:
        last_checkpoint = sorted(args.checkpoint_path.glob('*_epochs.pth'))[-1]
        if args.start_epochs > 0 and Path(args.checkpoint_path, f'{args.start_epochs:06d}_epochs.pth').exists():
            last_checkpoint = Path(args.checkpoint_path, f'{args.start_epochs:06d}_epochs.pth')
        checkpoint = torch.load(last_checkpoint)
        missing, unexpeted = teddy.load_state_dict(checkpoint['model'], strict=False)
        if len(keys := missing + unexpeted) > 0:
            if sum([not 'pos_encoding' in k for k in missing + unexpeted]) > 0:
                raise ValueError(f"Model not loaded: {keys}")
            if sum(['pos_encoding' in k for k in missing + unexpeted]) > 0:
                warnings.warn(f"Pos encoding not loaded: {keys}")
        print(f"Loaded checkpoint {last_checkpoint}")
    else:
        raise ValueError(f"Checkpoint path {args.checkpoint_path} does not exist")

    suffix = f'{last_checkpoint.stem}_16' if args.avg_char_width_16 else f'{last_checkpoint.stem}'
    fakes_path = args.checkpoint_path / 'saved_images' / suffix
    fakes_path.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for idx, batch in enumerate(tqdm(loader, desc='Generating images')):
            try:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                fakes = teddy.generate(batch['gen_text'], batch['style_text'], batch['style_img'])
                executor.submit(process_image, fakes.cpu(), batch['gen_text'], batch['dst_path'], fakes_path)
            except KeyboardInterrupt as e:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    parser.add_argument('--avg_char_width_16', action='store_true', help='Average the character width to 16')
    args = parser.parse_args()

    args.datasets_path = [Path(args.root_path, path) for path in args.datasets_path]
    args.checkpoint_path = Path(args.checkpoint_path, args.run_id)

    set_seed(args.seed)

    assert torch.cuda.is_available(), "You need a GPU to train Teddy"
    if args.device == 'auto':
        args.device = f'cuda:{np.argmax(free_mem_percent())}'

    generate_images(args.device, args)
