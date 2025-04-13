from __future__ import annotations

from typing import Tuple
from dataclasses import dataclass

import torch
from torch import nn, Tensor
from pydantic import BaseModel

from ..config import Config

from .abstract import AbstractModel
from .resnet import ResNet


@dataclass
class DiagonalGaussianDistribution:
    mean: Tensor
    log_var: Tensor

    def sample(self) -> Tensor:
        return torch.randn_like(self.log_var) * self.log_var.exp() + self.mean

    def kl(self) -> Tensor:
        return 0.5 * (
            self.mean.pow(2) + self.log_var.exp() - 1.0 - self.log_var
        ).sum((1, 2, 3))


class VAEEncoderConfig(Config):
    in_channels: int = 3
    latent_channels: int = 4

    dims: Tuple[int, ...]
    depths: Tuple[int, ...]


class VAEEncoder(ResNet, AbstractModel):
    def __init__(self, in_channels: int, latent_channels: int, dims: Tuple[int, ...], depths: Tuple[int, ...]):
        ResNet.__init__(self, in_channels, dims, depths, sample="down")
        self.latent_channels = latent_channels

        self.head = nn.Conv2d(dims[-1], 2 * latent_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> DiagonalGaussianDistribution:
        x = self.head(ResNet.forward(self, x))

        return DiagonalGaussianDistribution(
            mean=x[:, :self.latent_channels],
            log_var=x[:, self.latent_channels:]
        )

    @staticmethod
    def from_config(config: VAEEncoderConfig) -> VAEEncoder:
        return VAEEncoder(
            in_channels=config.in_channels,
            latent_channels=config.latent_channels,
            dims=config.dims,
            depths=config.depths
        )


class VAEDecoderConfig(Config):
    latent_channels: int = 4
    out_channels: int = 3

    dims: Tuple[int, ...]
    depths: Tuple[int, ...]


class VAEDecoder(ResNet, AbstractModel):
    def __init__(self, latent_channels: int, out_channels: int, dims: Tuple[int, ...], depths: Tuple[int, ...]):
        ResNet.__init__(self, latent_channels, dims, depths, sample="up")
        self.head = nn.Conv2d(dims[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(ResNet.forward(self, x))

    @staticmethod
    def from_config(config: VAEDecoderConfig) -> VAEDecoder:
        return VAEDecoder(
            latent_channels=config.latent_channels,
            out_channels=config.out_channels,
            dims=config.dims,
            depths=config.depths
        )
