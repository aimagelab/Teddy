import argparse
import torch
import torch.nn as nn
import numpy as np
import warnings
from pathlib import Path
from datasets import dataset_factory
from model.teddy import Teddy
from torchvision.utils import save_image
from tqdm import tqdm
from util.functional import ChunkLoader
from datasets import transforms as T
import concurrent.futures
from hwd.metrics import HWDScore, FIDScore, KIDScore, ProcessedDataset
from hwd.datasets import FolderDataset


class Evaluator(nn.Module):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.hwd = HWDScore()
        self.fid = FIDScore()
        self.kid = KIDScore()
        self.hwd_real_dataset = None
        self.fid_real_dataset = None
        self.kid_real_dataset = None
    
    @staticmethod
    def get_or_create(score, pkl_path, db_path=None):
        if pkl_path.exists():
            return ProcessedDataset.load(pkl_path)
        assert db_path is not None, 'No db_path provided'
        print(f'Creating {pkl_path}')
        dataset = FolderDataset(db_path, extension='png')
        dataset = score.digest(dataset)
        dataset.save(pkl_path)
        return dataset
    
    def set_real_dataset(self, real_path):
        self.hwd_real_dataset = self.get_or_create(self.hwd, real_path / 'hwd_real.pkl', real_path)
        self.fid_real_dataset = self.get_or_create(self.fid, real_path / 'fid_real.pkl', real_path)
        self.kid_real_dataset = self.get_or_create(self.kid, real_path / 'kid_real.pkl', real_path)
    
    def compute_metrics(self, fake_path, real_path=None):
        if real_path is not None:
            self.set_real_dataset(real_path)
        assert self.hwd_real_dataset is not None, 'No real dataset provided'

        hwd_fake_dataset = self.get_or_create(self.hwd, fake_path / 'hwd_fake.pkl', fake_path)
        hwd_result = self.hwd.distance(self.hwd_real_dataset, hwd_fake_dataset)

        fid_fake_dataset = self.get_or_create(self.fid, fake_path / 'fid_fake.pkl', fake_path)
        fid_result = self.fid.distance(self.fid_real_dataset, fid_fake_dataset)

        kid_fake_dataset = self.get_or_create(self.kid, fake_path / 'kid_fake.pkl', fake_path)
        kid_result = self.kid.distance(self.kid_real_dataset, kid_fake_dataset)

        return hwd_result, fid_result, kid_result


def process_image(fakes, gen_texts, dsts, fakes_path):
    for fake, txt, dst in zip(fakes, gen_texts, dsts):
        fake = fake[:, :, :16 * len(txt)]
        fake_dst = fakes_path / Path(dst)
        fake_dst.parent.mkdir(parents=True, exist_ok=True)
        save_image(fake, fake_dst)


@torch.no_grad()
def setup_loader(rank, args, eval_dataset='iam_eval'):
    args.datasets = [eval_dataset, ]
    datasets_kwargs = args.__dict__.copy()
    datasets_kwargs['pre_transform'] = T.Compose([
            T.Convert(1),
            T.ResizeFixedHeight(32),
            T.FixedCharWidth(16) if args.eval_avg_char_width_16 else lambda x: x,
            T.ToTensor(),
            T.PadNextDivisible(16),
            T.Normalize((0.5,), (0.5,))
        ])
    datasets_kwargs['post_transform'] = lambda x: x
    dataset = dataset_factory('all', **datasets_kwargs)
    dataset.batch_keys('style')
    dataset.datasets[0].filter_eval_set('test')

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=True, num_workers=args.num_workers,
                                         collate_fn=dataset.collate_fn, pin_memory=True, drop_last=False)
    return loader


@torch.no_grad()
def setup_loader_no_iam(rank, args):
    datasets_kwargs = args.__dict__.copy()
    # datasets_kwargs['pre_transform'] = T.Compose([
    #         T.Convert(1),
    #         T.ResizeFixedHeight(32),
    #         T.FixedCharWidth(16) if args.eval_avg_char_width_16 else lambda x: x,
    #         T.ToTensor(),
    #         T.PadNextDivisible(16),
    #         T.Normalize((0.5,), (0.5,))
    #     ])
    datasets_kwargs['post_transform'] = T.Compose([
            T.ToTensor(),
            T.PadNextDivisible(16),
            T.Normalize((0.5,), (0.5,))
        ])
    dataset = dataset_factory('test', **datasets_kwargs)
    dataset.batch_keys('style', 'same')

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=True, num_workers=args.num_workers,
                                         collate_fn=dataset.collate_fn, pin_memory=True, drop_last=False)
    return loader


@torch.no_grad()
def setup_teddy(rank, args):
    device = torch.device(rank)

    ocr_checkpoint_a = torch.load(args.ocr_checkpoint_a, map_location=device)
    teddy = Teddy(ocr_checkpoint_a['charset'], **args.__dict__).to(device)
    teddy.eval()
    return teddy


@torch.no_grad()
def generate_images_no_iam(rank, args, teddy=None, loader=None):
    device = torch.device(rank)

    teddy = teddy if teddy is not None else setup_teddy(rank, args)
    loader = loader if loader is not None else setup_loader_no_iam(rank, args)

    if args.checkpoint_path.exists() and len(list(args.checkpoint_path.glob('*_epochs.pth'))) > 0:
        last_checkpoint = sorted(args.checkpoint_path.glob('*_epochs.pth'))[-1]
        if args.eval_epoch is not None:
            last_checkpoint = Path(args.checkpoint_path, f'{args.eval_epoch:06d}_epochs.pth')
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

    suffix = f'{args.datasets[0]}_{last_checkpoint.stem}_16' if args.eval_avg_char_width_16 else f'{args.datasets[0]}_{last_checkpoint.stem}'
    fakes_path = args.checkpoint_path / 'saved_images' / suffix
    fakes_path.mkdir(parents=True, exist_ok=True)
    counter = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for idx, batch in enumerate(tqdm(loader, desc='Generating images')):
            try:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch['style_text'] = [''.join([c for c in txt if c in teddy.text_converter.charset]) for txt in batch['style_text']]
                batch['same_text'] = [''.join([c for c in txt if c in teddy.text_converter.charset]) for txt in batch['same_text']]
                fakes = teddy.generate(batch['style_text'], batch['same_text'], batch['same_img'])
                dst_paths = [Path(author) / f'{img_id:06d}.png' for author, img_id in zip(batch['same_author'], range(counter, 10*10))]
                executor.submit(process_image, fakes.cpu(), batch['style_text'], dst_paths, fakes_path)
            except KeyboardInterrupt as e:
                break


@torch.no_grad()
def generate_images(rank, args, teddy=None, loader=None):
    device = torch.device(rank)

    teddy = teddy if teddy is not None else setup_teddy(rank, args)
    loader = loader if loader is not None else setup_loader(rank, args)

    if args.checkpoint_path.exists() and len(list(args.checkpoint_path.glob('*_epochs.pth'))) > 0:
        last_checkpoint = sorted(args.checkpoint_path.glob('*_epochs.pth'))[-1]
        if args.eval_epoch is not None:
            last_checkpoint = Path(args.checkpoint_path, f'{args.eval_epoch:06d}_epochs.pth')
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

    suffix = f'{last_checkpoint.stem}_16' if args.eval_avg_char_width_16 else f'{last_checkpoint.stem}'
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
    from train import add_arguments, set_seed, free_mem_percent

    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    parser.add_argument('--eval_all_epochs', action='store_true', help='Evaluate all epochs')
    args = parser.parse_args()
    args.checkpoint_path = Path(args.checkpoint_path, args.run_id)

    set_seed(args.seed)

    assert torch.cuda.is_available(), "You need a GPU to train Teddy"
    if args.device == 'auto':
        args.device = f'cuda:{np.argmax(free_mem_percent())}'

    teddy = setup_teddy(args.device, args)
    # loader = setup_loader(args.device, args, args.eval_dataset)
    loader = setup_loader_no_iam(args.device, args)
    if args.eval_all_epochs:
        for epoch in sorted(args.checkpoint_path.glob('*_epochs.pth')):
            args.eval_epoch = int(epoch.stem.split('_')[0])
            generate_images_no_iam(args.device, args, teddy, loader)
    else:
        generate_images_no_iam(args.device, args, teddy, loader)
