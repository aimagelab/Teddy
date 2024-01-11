import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import pickle
import random

class UnifontModule(nn.Module):
    def __init__(self, charset, dim, transforms=None):
        super(UnifontModule, self).__init__()
        self.transforms = transforms
        self.charset = sorted(set(charset))
        self.symbols = self.get_symbols()
        self.symbols_size = self.symbols[0].numel()
        self.model = torch.nn.Linear(self.symbols_size, dim)

    def get_symbols(self):
        with open(f"files/unifont.pickle", "rb") as f:
            symbols = pickle.load(f)

        symbols = {sym['idx'][0]: sym['mat'].astype(np.float32) for sym in symbols}
        symbols = [symbols[ord(char)] for char in self.charset]
        symbols.insert(0, np.zeros_like(symbols[0]))
        symbols = np.stack(symbols)
        return torch.from_numpy(symbols).float().unsqueeze(1)

    def _apply(self, fn):
        super(UnifontModule, self)._apply(fn)
        self.symbols = fn(self.symbols)
        return self

    def forward(self, QR):
        mat = self.symbols[QR]
        if self.transforms is not None:
            mat = self.transforms(mat)
        mat = rearrange(mat, 'b l 1 h w -> b l (h w)')
        return self.model(mat)

    def __len__(self):
        return len(self.symbols)
    
class ConvUnifontModule(UnifontModule):
    def __init__(self, charset, dim, dimensions=(1, 64, 128, 256), transforms=None):
        super().__init__(charset, dim, transforms)
        blocks = []
        for i in range(len(dimensions) - 1):
            blocks.append(nn.Conv2d(dimensions[i], dimensions[i + 1], 3, padding=1))
            blocks.append(nn.MaxPool2d(2, 2))
            blocks.append(nn.ReLU())
        blocks.extend([
            nn.Flatten(),
            nn.Linear(dimensions[-1] * (2 ** (4 - len(dimensions) + 2)), dim),
        ])
        self.model = nn.Sequential(*blocks)

    def forward(self, QR):
        b, *_ = QR.shape
        mat = self.symbols[QR]
        if self.transforms is not None:
            mat = self.transforms(mat)
        mat = rearrange(mat, 'b l c h w -> (b l) c h w')
        mat = self.model(mat)
        mat = rearrange(mat, '(b l) e -> b l e', b=b)
        return mat

class OnehotModule(nn.Module):
    def __init__(self, charset, dim, transforms=None):
        super(OnehotModule, self).__init__()
        self.charset = sorted(set(charset))
        self.symbols = nn.Embedding(len(self.charset) + 1, dim)
    
    def forward(self, QR):
        return self.symbols.weight[QR]
    

class UnifontShiftTransform(nn.Module):
    def __init__(self, shift=0):
        super().__init__()
        if isinstance(shift, int):
            shift = (shift, shift)
        self.shift_x, self.shift_y = shift

    def forward(self, mat):
        b, l, c, h, w = mat.shape
        mat = nn.functional.pad(mat, (self.shift_x, self.shift_x, self.shift_y, self.shift_y))
        shift_x = random.randint(0, self.shift_x * 2)
        shift_y = random.randint(0, self.shift_y * 2)
        return mat[:, :, :, shift_y:shift_y + h, shift_x:shift_x + w]