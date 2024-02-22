from train import add_arguments, free_mem_percent
import argparse
from pathlib import Path
import torch
import numpy as np
from datasets import dataset_factory
from datasets import transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt

def compute_avg_char_width(args):
    dataset = dataset_factory('train', post_transform=T.ToTensor(), **args.__dict__)
    dataset.batch_keys('style')
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                         collate_fn=dataset.collate_fn, pin_memory=True, drop_last=False)

    ratios = []
    for batch in tqdm(loader, total=len(loader)):
        for l, t in zip(batch['style_img_len'], batch['style_text']):
            if l < 64:
                continue
            ratios.append(l / len(t))
    
    plt.hist(ratios, bins=50)
    plt.savefig(f'char_width_{"__".join(args.datasets)}.png')

    print(f'Average character width: {np.mean(ratios):.4f}')
    print(f'Std character width: {np.std(ratios):.4f}')
    print(f'Min character width: {np.min(ratios):.4f}')
    print(f'Max character width: {np.max(ratios):.4f}')

    print(f'25th percentile: {np.percentile(ratios, 25):.4f}')
    print(f'75th percentile: {np.percentile(ratios, 75):.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    args.checkpoint_path = Path(args.checkpoint_path, args.run_id)

    assert torch.cuda.is_available(), "You need a GPU to train Teddy"
    if args.device == 'auto':
        args.device = f'cuda:{np.argmax(free_mem_percent())}'

    compute_avg_char_width(args)