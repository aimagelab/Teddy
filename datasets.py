import itertools
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import msgpack
from tqdm import tqdm
from torchvision import transforms as T
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
from einops import rearrange


def get_alphabet(labels):
    coll = ''.join(labels)
    unq = sorted(list(set(coll)))
    unq = [''.join(i) for i in itertools.product(unq, repeat=1)]
    alph = dict(zip(unq, range(len(unq))))
    return alph


def pad_images(images, padding_value=1):
    images = [rearrange(img, 'c h w -> w c h') for img in images]
    return rearrange(pad_sequence(images, padding_value=padding_value), 'w b c h -> b c h w')


class ResizeFixedHeight(object):
    def __init__(self, height):
        self.height = height

    def __call__(self, img):
        w, h = img.size
        ratio = h / self.height
        new_w = int(w / ratio)
        img = img.resize((new_w, self.height), Image.BILINEAR)
        return img


class PadNextDivisible(object):
    def __init__(self, divisible, padding_value=1):
        self.divisible = divisible
        self.padding_value = padding_value

    def __call__(self, img):
        width = img.shape[-1]
        if width % self.divisible == 0:
            return img
        pad_width = self.divisible - width % self.divisible
        return pad(img, (0, pad_width), value=self.padding_value)


class Base_dataset(Dataset):
    def __init__(self, path, nameset='train', transform=T.ToTensor()):
        super().__init__()
        self.transform = transform
        self.nameset = nameset
        self.path = path
        self.imgs = []
        self.imgs_to_label = {}
        self.imgs_to_author = {}
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.author_to_imgs = {}
        self.imgs_set = set()
        self.preloaded = False

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        style_img_path = self.imgs[idx]
        style_text = self.imgs_to_label[style_img_path.stem]

        author = self.imgs_to_author[style_img_path.stem]
        same_author_imgs = self.author_to_imgs[author]
        other_author_imgs = self.imgs_set - same_author_imgs

        # same_author_img_path = random.choice(list(same_author_imgs))
        # same_author_img = Image.open(same_author_img_path).convert('RGB')

        multi_author = len(other_author_imgs) > 0
        # other_author_imgs = same_author_imgs if not multi_author else other_author_imgs
        # other_author_img_path = random.choice(list(other_author_imgs))
        # other_author_img = Image.open(other_author_img_path).convert('RGB')

        if not self.preloaded:
            style_img = Image.open(style_img_path).convert('RGB')
            style_img = self.transform(style_img) if self.transform else style_img
            # same_author_img = self.transform(same_author_img)
            # other_author_img = self.transform(other_author_img)
        else:
            style_img = self.imgs_preloaded[idx]

        style_img_len = style_img.shape[-1]
        # same_author_img_len = same_author_img.shape[-1]
        # other_author_img_len = other_author_img.shape[-1]

        sample = {
            'style_img': style_img,
            'style_img_len': style_img_len,
            'style_text': style_text,
            # 'same_author_img': same_author_img,
            # 'same_author_img_len': same_author_img_len,
            # 'other_author_img': other_author_img,
            # 'other_author_img_len': other_author_img_len,
            'multi_author': multi_author,
        }
        return sample

    def load_img_sizes(self, img_sizes_path):
        if img_sizes_path.exists():
            with open(img_sizes_path, 'rb') as f:
                img_sizes = msgpack.load(f, strict_map_key=False)
            return {Path(filename).stem: (width, height) for filename, width, height in img_sizes}
        else:
            data = []
            img_sizes = {}
            for img_path in tqdm(self.imgs, desc='Loading image sizes'):
                try:
                    img = Image.open(img_path)
                except:
                    print(f'Error opening {img_path}')
                    continue
                img_sizes[img_path.stem] = img.size
                data.append((img_path.name, *img.size))
            with open(img_sizes_path, 'wb') as f:
                msgpack.dump(data, f)
            return img_sizes

    def preload(self):
        self.imgs_preloaded = [Image.open(img_path).convert('RGB') for img_path in self.imgs]
        self.imgs_preloaded = [self.transform(img) if self.transform else img for img in self.imgs_preloaded]
        self.preloaded = True


class IAM_dataset(Base_dataset):
    def __init__(self, path, nameset=None, transform=T.ToTensor(), max_width=None, max_height=None, dataset_type='lines', preloaded=True):
        super().__init__(path, nameset, transform)
        self.dataset_type = dataset_type

        self.imgs = list(Path(path, self.dataset_type).rglob('*.png'))

        xml_files = [ET.parse(xml_file) for xml_file in Path(path, 'xmls').rglob('*.xml')]
        tag = 'line' if self.dataset_type == 'lines' else 'word'
        self.imgs_to_label = {el.attrib['id']: el.attrib['text'] for xml_file in xml_files for el in xml_file.iter() if el.tag == tag}
        self.imgs_to_author = {el.attrib['id']: xml_file.getroot().attrib['writer-id'] for xml_file in xml_files for el in xml_file.iter() if el.tag == tag}

        img_sizes_path = Path(path, f'img_sizes_{dataset_type}.msgpack')
        self.imgs_to_sizes = self.load_img_sizes(img_sizes_path)
        assert set(self.imgs_to_label.keys()) == set(self.imgs_to_sizes.keys())

        htg_train_authors = Path('files/gan.iam.tr_va.gt.filter27.txt').read_text().splitlines()
        htg_train_authors = sorted({line.split(',')[0] for line in htg_train_authors})

        val_authors_count = round(len(htg_train_authors) * 0.1)
        val_authors = htg_train_authors[:val_authors_count]
        train_authors = htg_train_authors[val_authors_count:]
        assert len(set(train_authors) & set(val_authors)) == 0
        assert len(set(train_authors) | set(val_authors)) == len(htg_train_authors)

        if nameset == 'train':
            target_authors = train_authors
        elif nameset == 'val':
            target_authors = val_authors
        else:
            raise ValueError(f'Unknown nameset {nameset}')

        self.imgs = [img for img in self.imgs if self.imgs_to_author[img.stem] in target_authors]
        self.imgs_to_author = {k: v for k, v in self.imgs_to_author.items() if v in target_authors}
        self.imgs_to_label = {k: v for k, v in self.imgs_to_label.items() if k in self.imgs_to_author}

        assert all(path.stem in self.imgs_to_label for path in self.imgs), 'Images and labels do not match'
        assert len(self.imgs) > 0, f'No images found in {path}'
        self.char_to_idx = get_alphabet(self.imgs_to_label.values())
        self.idx_to_char = dict(zip(self.char_to_idx.values(), self.char_to_idx.keys()))

        if max_width and max_height:
            target_width = {filename: width * max_height / height for filename, (width, height) in self.imgs_to_sizes.items()}
            self.imgs = [img for img in self.imgs if target_width[img.stem] <= max_width]

        self.imgs_set = set(self.imgs)
        self.author_to_imgs = {author: {img for img in self.imgs if self.imgs_to_author[img.stem] == author} for author in target_authors}

        if preloaded:
            self.preload()


class Msgpack_dataset(Base_dataset):
    def __init__(self, path, nameset='train', transform=T.ToTensor(), max_width=None, max_height=None):
        super().__init__(path, nameset, transform)

        nameset_path = Path(path, f'{nameset}.msgpack')
        assert nameset_path.exists(), f'No msgpack file found in {path}'

        with open(nameset_path, 'rb') as f:
            data = msgpack.load(f)

        self.imgs_to_label = {Path(filename).stem: label for filename, label, *_ in data}
        self.imgs_to_author = {Path(filename).stem: '000' for filename, *_ in data}

        self.imgs = [Path(path, 'lines') / filename for filename, *_ in data]
        assert len(self.imgs) > 0, f'No images found in {path}'

        charset_path = Path(path, 'charset.msgpack')
        with open(charset_path, 'rb') as f:
            charset = msgpack.load(f, strict_map_key=False)
        self.char_to_idx = charset['char2idx']
        self.idx_to_char = charset['idx2char']

        img_sizes_path = Path(path, f'{nameset}_img_sizes.msgpack')
        self.imgs_to_sizes = self.load_img_sizes(img_sizes_path)
        assert set(self.imgs_to_label.keys()) == set(self.imgs_to_sizes.keys())

        if max_width and max_height:
            target_width = {filename: width * max_height / height for filename, (width, height) in self.imgs_to_sizes.items()}
            self.imgs = [img for img in self.imgs if target_width[img.stem] <= max_width]

        self.imgs_set = set(self.imgs)
        authors = set(self.imgs_to_author.values())
        self.author_to_imgs = {author: {img for img in self.imgs if self.imgs_to_author[img.stem] == author} for author in authors}


class Norhand_dataset(Msgpack_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Rimes_dataset(Msgpack_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ICFHR16_dataset(Msgpack_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ICFHR14_dataset(Msgpack_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LAM_dataset(Msgpack_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Rodrigo_dataset(Msgpack_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SaintGall_dataset(Msgpack_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Washington_dataset(Msgpack_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Leopardi_dataset(Msgpack_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MergedDataset(Dataset):
    def __init__(self, datasets, idx_to_char=None):
        super().__init__()
        self.datasets = datasets
        if idx_to_char:
            self.idx_to_char = idx_to_char
            self.char_to_idx = dict(zip(self.idx_to_char.values(), self.idx_to_char.keys()))
        else:
            self.char_to_idx = get_alphabet([''.join(list(d.idx_to_char.values())) for d in datasets])
            self.idx_to_char = dict(zip(self.char_to_idx.values(), self.char_to_idx.keys()))
        for dataset in self.datasets:
            dataset.char_to_idx = self.char_to_idx
            dataset.idx_to_char = self.idx_to_char

    @property
    def labels(self):
        return [label for dataset in self.datasets for label in dataset.imgs_to_label.values()]

    @property
    def vocab_size(self):
        return len(self.char_to_idx)

    @property
    def alphabet(self):
        return ''.join(sorted(self.char_to_idx.keys()))

    @property
    def imgs(self):
        return [img for dataset in self.datasets for img in dataset.imgs]

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            else:
                idx -= len(dataset)
        raise IndexError('Index out of range')

    def collate_fn(self, batch):
        collate_batch = {}
        collate_batch['style_imgs'] = pad_images([sample['style_img'] for sample in batch])
        collate_batch['style_imgs_len'] = torch.IntTensor([sample['style_img_len'] for sample in batch])
        collate_batch['style_texts'] = [sample['style_text'] for sample in batch]
        # collate_batch['same_author_imgs'] = pad_images([sample['same_author_img'] for sample in batch])
        # collate_batch['same_author_imgs_len'] = torch.IntTensor([sample['same_author_img_len'] for sample in batch])
        # collate_batch['other_author_imgs'] = pad_images([sample['other_author_img'] for sample in batch])
        # collate_batch['other_author_imgs_len'] = torch.IntTensor([sample['other_author_img_len'] for sample in batch])
        collate_batch['multi_authors'] = torch.BoolTensor([sample['multi_author'] for sample in batch])
        return collate_batch


def dataset_factory(datasets, datasets_path, nameset, idx_to_char=None, resize_height=32, divisible=16, max_width=None, channels=3):
    assert nameset in {'train', 'val'}, f'Unknown nameset {nameset}'
    transform = T.Compose([
        ResizeFixedHeight(resize_height),
        T.Grayscale() if channels == 1 else T.Lambda(lambda x: x),
        T.ToTensor(),
        PadNextDivisible(divisible),
        T.Normalize((0.5,), (0.5,))
    ])

    datasets_list = []
    kwargs = {'max_width': max_width, 'max_height': resize_height, 'transform': transform, 'nameset': nameset}
    for name, path in tqdm(zip(datasets, datasets_path), total=len(datasets), desc=f'Loading datasets {nameset}'):
        if name.lower() == 'iam_words':
            datasets_list.append(IAM_dataset(path, dataset_type='words', **kwargs))
        elif name.lower() == 'iam_lines':
            datasets_list.append(IAM_dataset(path, dataset_type='lines', **kwargs))
        elif name.lower() == 'rimes':
            datasets_list.append(Rimes_dataset(path, **kwargs))
        elif name.lower() == 'icfhr16':
            datasets_list.append(ICFHR16_dataset(path, **kwargs))
        elif name.lower() == 'icfhr14':
            datasets_list.append(ICFHR14_dataset(path, **kwargs))
        elif name.lower() == 'lam':
            datasets_list.append(LAM_dataset(path, **kwargs))
        elif name.lower() == 'rodrigo':
            datasets_list.append(Rodrigo_dataset(path, **kwargs))
        elif name.lower() == 'saintgall':
            datasets_list.append(SaintGall_dataset(path, **kwargs))
        elif name.lower() == 'washington':
            datasets_list.append(Washington_dataset(path, **kwargs))
        elif name.lower() == 'leopardi':
            datasets_list.append(Leopardi_dataset(path, **kwargs))
        elif name.lower() == 'norhand':
            datasets_list.append(Norhand_dataset(path, **kwargs))
        else:
            raise ValueError(f'Unknown dataset {name}')
    return MergedDataset(datasets_list, idx_to_char)


if __name__ == '__main__':
    for nameset in ('train', 'val'):
        print(len(IAM_dataset('/mnt/ssd/datasets/IAM', nameset=nameset)[0]))
        print(len(SaintGall_dataset('/mnt/ssd/datasets/SaintGall', nameset=nameset)[0]))
        print(len(Norhand_dataset('/mnt/ssd/datasets/Norhand', nameset=nameset)[0]))
        print(len(Rimes_dataset('/mnt/ssd/datasets/Rimes', nameset=nameset)[0]))
        print(len(ICFHR16_dataset('/mnt/ssd/datasets/ICFHR16', nameset=nameset)[0]))
        print(len(ICFHR14_dataset('/mnt/ssd/datasets/ICFHR14', nameset=nameset)[0]))
        print(len(LAM_dataset('/mnt/ssd/datasets/LAM_msgpack', nameset=nameset)[0]))
        print(len(Rodrigo_dataset('/mnt/ssd/datasets/Rodrigo', nameset=nameset)[0]))
        print(len(Washington_dataset('/mnt/ssd/datasets/Washington', nameset=nameset)[0]))
        print(len(Leopardi_dataset('/mnt/ssd/datasets/LEOPARDI/leopardi', nameset=nameset)[0]))
