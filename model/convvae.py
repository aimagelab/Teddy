from typing import Tuple
import torch.nn as nn
import torch
from torch import Tensor
from einops.layers.torch import Rearrange


class ChangeRange(nn.Module):
    def forward(self, x):
        return x * 2 - 1

class ConvVAE(nn.Module):
    def __init__(self, latent_dim, img_channels=1):
        super(ConvVAE, self).__init__()

        self.latent_dim = latent_dim
        self.name = 'convVAE'

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            Rearrange('b c h w -> b w (c h)'),
        )

        # flatten: (1,128,7,7) -> (1,128*7*7) = (1,6272)
        # hidden => mu
        self.fc1 = nn.Linear(1024, self.latent_dim)

        # hidden => logvar
        self.fc2 = nn.Linear(1024, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(),
            Rearrange('b w (c h) -> b c h w', c=256),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1),
            # nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def gen_from_noise(self, z):
        return self.decode(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

