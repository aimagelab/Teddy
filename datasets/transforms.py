from PIL import Image
import random
import torch
from torchvision.transforms import functional as F
from torchvision import transforms as T
from torchvision.transforms import Compose


class ResizeFixedHeight(object):
    def __init__(self, height):
        self.height = height

    def __call__(self, sample):
        img, lbl = sample
        w, h = img.size
        ratio = h / self.height
        new_w = int(w / ratio)
        img = img.resize((new_w, self.height), Image.BILINEAR)
        return img, lbl


class RandomShrink(object):
    def __init__(self, min_ratio, max_ratio, min_width=0, max_width=10 ** 9, snap_to=1):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_width = min_width
        self.max_width = max_width
        self.sanp_to = snap_to

    def __call__(self, sample):
        img, lbl = sample
        w, h = img.size
        min_width = max(int(w * self.min_ratio), self.min_width)
        max_width = min(int(w * self.max_ratio), self.max_width)
        new_w = random.randint(min_width, max_width)
        new_w = round(new_w / self.sanp_to) * self.sanp_to
        img = img.resize((new_w, h), Image.BILINEAR)
        return img, lbl


class PadNextDivisible(object):
    def __init__(self, divisible, padding_value=1):
        self.divisible = divisible
        self.padding_value = padding_value

    def __call__(self, sample):
        img, lbl = sample
        width = img.shape[-1]
        if width % self.divisible == 0:
            return img
        pad_width = self.divisible - width % self.divisible
        raise NotImplementedError
        return F.pad(img, (0, pad_width), value=self.padding_value), lbl


class Convert(object):
    def __init__(self, channels):
        if channels == 1:
            self.mode = 'L'
        elif channels == 3:
            self.mode = 'RGB'
        else:
            raise NotImplementedError

    def __call__(self, sample):
        img, lbl = sample
        return img.convert(self.mode), lbl


class ToTensor(T.ToTensor):
    def __call__(self, sample):
        img, lbl = sample
        return super().__call__(img), lbl


class ToPILImage(T.ToPILImage):
    def __call__(self, sample):
        img, lbl = sample
        return super().__call__(img), lbl


class Normalize(T.Normalize):
    def __call__(self, sample):
        img, lbl = sample
        return super().__call__(img), lbl


class FixedCharWidth(object):
    def __init__(self, width):
        self.width = width

    def __call__(self, sample):
        img, lbl = sample
        w, h = img.size
        new_w = self.width * len(lbl)
        img = img.resize((new_w, h), Image.BILINEAR)
        return img, lbl


class PadMinWidth(object):
    def __init__(self, min_width, padding_value=1):
        self.min_width = min_width
        self.padding_value = padding_value

    def __call__(self, sample):
        img, lbl = sample
        c, h, w = img.shape
        if w >= self.min_width:
            return img, lbl
        pad_width = self.min_width - w
        return F.pad(img, (0, 0, pad_width, 0), fill=self.padding_value), lbl
