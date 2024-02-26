import argparse
import torch
from pathlib import Path
from datasets import dataset_factory
from datasets import transforms as T
from tqdm import tqdm
from torchvision.utils import save_image


@torch.no_grad()
def setup_loader(rank, args):
    datasets_kwargs = args.__dict__.copy()
    datasets_kwargs['gen_max_width'] = None
    # datasets_kwargs['pre_transform'] = lambda x: x
    datasets_kwargs['post_transform'] = T.Compose([
            T.FixedCharWidth(16) if args.eval_avg_char_width_16 else lambda x: x,
            T.ToTensor(),
            # T.PadNextDivisible(16),
            T.Normalize((0.5,), (0.5,))
        ])
    dataset = dataset_factory('test', **datasets_kwargs)
    dataset.batch_keys('style')

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=True, num_workers=args.num_workers,
                                         collate_fn=dataset.collate_fn, pin_memory=True, drop_last=False)
    return loader


if __name__ == '__main__':
    from train import add_arguments, set_seed

    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    parser.add_argument('--dst', type=Path, default='files/iam')
    args = parser.parse_args()

    set_seed(args.seed)

    loader = setup_loader(0, args)

    encountered = set()
    discarded = 0
    for sample in tqdm(loader):
        for img, img_len, author, lbl in zip(sample['style_img'], sample['style_img_len'], sample['style_author'], sample['style_text']):
            # if (author, lbl, img_len.item()) in encountered:
            #     discarded += 1
            #     continue
            img = img[..., :img_len]
            author_dir = args.dst / author
            author_dir.mkdir(parents=True, exist_ok=True)
            img_path = author_dir / f'{len(encountered):06d}.png'
            img_path = img_path.resolve()
            save_image(img, img_path)
            encountered.add((author, lbl, img_len.item()))
    
    print(f'Discarded {discarded} samples')

    
