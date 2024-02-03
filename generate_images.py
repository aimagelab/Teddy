import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from files.ocr_checkpoints.train import add_arguments, set_seed, free_mem_percent
from datasets import dataset_factory
from model.teddy import Teddy
from torchvision.utils import save_image
from tqdm import tqdm
from util.functional import ChunkLoader

@torch.no_grad()
def generate_images(rank, args):
    device = torch.device(rank)

    args.datasets = ['iam_lines_eval']
    dataset = dataset_factory('test', **args.__dict__)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                         collate_fn=dataset.collate_fn, pin_memory=True, drop_last=False)

    ocr_checkpoint_a = torch.load(args.ocr_checkpoint_a, map_location=device)
    teddy = Teddy(ocr_checkpoint_a['charset'], **args.__dict__).to(device)
    teddy.eval()

    if args.checkpoint_path.exists() and len(list(args.checkpoint_path.glob('*_epochs.pth'))) > 0:
        last_checkpoint = sorted(args.checkpoint_path.glob('*_epochs.pth'))[-1]
        if args.start_epochs > 0 and Path(args.checkpoint_path, f'{args.start_epochs:06d}_epochs.pth').exists():
            last_checkpoint = Path(args.checkpoint_path, f'{args.start_epochs:06d}_epochs.pth')
        checkpoint = torch.load(last_checkpoint)
        teddy.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint {last_checkpoint}")
    else:
        raise ValueError(f"Checkpoint path {args.checkpoint_path} does not exist")

    fakes_path = args.checkpoint_path / 'saved_images' / f'{args.start_epochs:06d}'
    fakes_path.mkdir(parents=True, exist_ok=True)

    reals_path = args.checkpoint_path / 'saved_images' / f'real'
    reals_path.mkdir(parents=True, exist_ok=True)

    for idx, batch in enumerate(tqdm(loader, desc='Generating images')):
        try:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch['gen_texts'] = batch['style_texts']

            fakes = teddy.generate(batch['gen_texts'], batch['other_author_texts'], batch['other_author_imgs'])

            zip_args = [fakes, batch['style_imgs'], batch['style_imgs_len'], batch['gen_texts'], batch['style_authors']]
            for d, (fake, real, width, txt, author) in enumerate(zip(*zip_args)):
                fake = fake.squeeze(0)
                fake = fake[:, :, :16 * len(txt)]
                real = real[:, :, :width]
                fake_dst = fakes_path / author / f'{idx * len(fakes) + d:06d}.png'
                fake_dst.parent.mkdir(parents=True, exist_ok=True)
                save_image(fake, fake_dst)
                real_dst = reals_path / author / f'{idx * len(fakes) + d:06d}.png'
                real_dst.parent.mkdir(parents=True, exist_ok=True)
                if not real_dst.exists():
                    save_image(real, real_dst)
        except KeyboardInterrupt as e:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    args.datasets_path = [Path(args.root_path, path) for path in args.datasets_path]
    args.checkpoint_path = Path(args.checkpoint_path, args.run_id)

    for i in range(10, 500, 10):
        args.start_epochs = i
        set_seed(args.seed)

        assert torch.cuda.is_available(), "You need a GPU to train Teddy"
        if args.device == 'auto':
            args.device = f'cuda:{np.argmax(free_mem_percent())}'

        generate_images(args.device, args)
