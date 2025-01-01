import torch

from torch import nn
from typing import Optional, Tuple


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, d_model: int, d_hidden: Optional[int] = None, norm_eps: float = 1e-6):
        super(ConvNeXtBlock, self).__init__()
        d_hidden = 4 * d_model if d_hidden is None else d_hidden

        self.module = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model),
            LayerNorm2d(d_model, eps=norm_eps),
            nn.Conv2d(d_model, d_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_hidden, d_model, kernel_size=1)
        )

    def forward(self, x):
        return x + self.module(x)


class Encoder(nn.Module):
    def __init__(
            self, image_channels: int,
            dims: Tuple[int, ...] = (96, 192, 384, 768), depths: Tuple[int, ...] = (3, 3, 9, 4)
    ):
        super(Encoder, self).__init__()
        layers = [
            nn.Conv2d(image_channels, dims[0], kernel_size=5, padding=2)
        ]

        for i in range(len(dims) - 1):
            layers += [
                ConvNeXtBlock(dims[i]) for _ in range(depths[i])
            ]
            layers += [
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=4, stride=2, padding=1)
            ]

        layers += [
            ConvNeXtBlock(dims[-1]) for _ in range(depths[-1])
        ]

        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


class Decoder(nn.Module):
    def __init__(
            self, image_channels: int,
            dims: Tuple[int, ...] = (96, 192, 384, 768), depths: Tuple[int, ...] = (3, 3, 9, 4)
    ):
        super(Decoder, self).__init__()
        layers = [
            nn.Conv2d(image_channels, dims[-1], kernel_size=5, padding=2)
        ]
        layers += [
            ConvNeXtBlock(dims[-1]) for _ in range(depths[-1])
        ]

        for i in range(len(dims) - 2, -1, -1):
            layers += [
                nn.ConvTranspose2d(dims[i + 1], dims[i], kernel_size=4, stride=2, padding=1)
            ]
            layers += [
                ConvNeXtBlock(dims[i]) for _ in range(depths[i])
            ]

        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)
