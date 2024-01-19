import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
from .cnn_decoder import Conv2dBlock, ResBlocks
from einops import rearrange

class VariationalEncoder(nn.Module):
    def __init__(self, ups=3, n_res=2, dim=512, in_dim=1, res_norm='in', activ='relu', pad_type='reflect'):
        super(VariationalEncoder, self).__init__()

        self.model = []
        # self.model.append(ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type))

        dims = [dim // (2 ** i) for i in range(1, ups + 1)][::-1]

        self.model.append(Conv2dBlock(in_dim, dims[0], 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type))
        for d in dims:
            self.model.append(nn.MaxPool2d(2))
            self.model.append(Conv2dBlock(d, d * 2, 5, 1, 2, norm='in', activation=activ, pad_type=pad_type))
        self.model = nn.Sequential(*self.model)

        self.mu_linear = nn.Linear(dim * 4, dim)
        self.sigma_linear = nn.Linear(dim * 4, dim)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc # hack to get sampling on the GPU
        self.N.scale = self.N.scale
        self.kl = 0

    def _apply(self, fn, recurse=True):
        self.N.loc = fn(self.N.loc)
        self.N.scale = fn(self.N.scale)
        return super()._apply(fn, recurse)

    def forward(self, x):
        x = self.model(x)
        x = rearrange(x, 'b c h w -> b w (c h)')

        mu = self.mu_linear(x)
        sigma = torch.exp(self.sigma_linear(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1/2).sum()
        return z
    

class VariationalDecoder(nn.Module):
    def __init__(self, ups=3, n_res=2, dim=512, out_dim=1, res_norm='in', activ='relu', pad_type='reflect'):
        super(VariationalDecoder, self).__init__()

        self.fc = nn.Linear(dim, dim * 4)

        self.model = []
        self.model.append(ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type))
        for _ in range(ups):
            self.model.append(nn.Upsample(scale_factor=2))
            self.model.append(Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='in', activation=activ, pad_type=pad_type))
            dim = dim // 2
        self.model.append(Conv2dBlock(dim, out_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x = self.fc(x)
        x = rearrange(x, 'b w (c h) -> b c h w', h=4)
        return self.model(x)
    

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, channels=1):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(dim=latent_dims, in_dim=channels)
        self.decoder = VariationalDecoder(dim=latent_dims, out_dim=channels)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    

if __name__ == '__main__':
    vae = VariationalAutoencoder(512)

    img = torch.randn(16, 1, 32, 608)
    out = vae(img)
    print(out.shape)