from __future__ import annotations

from typing import Tuple

from einops import rearrange
from torch import nn, Tensor

from ..config import Config

from .resnet import ResNet
from .abstract import AbstractModel


class DiscriminatorConfig(Config):
    in_channels: int = 3
    patch_size: int | Tuple[int, int]

    dims: Tuple[int, ...]
    depths: Tuple[int, ...]


class Discriminator(ResNet, AbstractModel):
    def __init__(
            self, in_channels: int, patch_size: int | Tuple[int, int], dims: Tuple[int, ...], depths: Tuple[int, ...]
    ):
        ResNet.__init__(self, in_channels, dims, depths, sample="down")
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Linear(dims[-1], 1)

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape

        x = self.pool(
            ResNet.forward(
                self, rearrange(
                    x, "b c (p1 h) (p2 w) -> (b h w) c p1 p2",
                    p1=self.patch_size[0],
                    p2=self.patch_size[1]
                )
            )
        ).squeeze((2, 3))

        return rearrange(
            self.head(x).squeeze(-1),
            "(b h w) -> b h w", h=H // self.patch_size[0], w=W // self.patch_size[1]
        )

    @staticmethod
    def from_config(config: DiscriminatorConfig) -> Discriminator:
        return Discriminator(
            in_channels=config.in_channels,
            patch_size=config.patch_size,
            dims=config.dims,
            depths=config.depths
        )
