from __future__ import annotations

import os
from typing import Optional, Tuple
from math import sqrt

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, Tensor
from pydantic import BaseModel

from ..config import Config

from .abstract import AbstractModel
from ..train.vae import VAETrainConfig


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super(Attention, self).__init__()
        assert d_model % n_heads == 0

        self.scale = sqrt(d_model / n_heads)
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def apply_rotary_emb(x: Tensor, freqs: Tensor) -> Tensor:
        return torch.view_as_real(
            torch.view_as_complex(x.unflatten(-1, (-1, 2)))
            * freqs
        ).flatten(-2)

    def forward(self, x: Tensor, freqs: Tensor) -> Tensor:
        q = rearrange(self.W_q(x), "b l (n d) -> b n l d", n=self.n_heads)
        k = rearrange(self.W_k(x), "b l (n d) -> b n l d", n=self.n_heads)
        v = rearrange(self.W_v(x), "b l (n d) -> b n l d", n=self.n_heads)

        attn = (
            self.apply_rotary_emb(q, freqs) @
            self.apply_rotary_emb(k, freqs).permute(-2, -1)
        ) / self.scale

        return self.W_o(
            rearrange(
                F.softmax(attn, dim=-1) @ v, "b n l d -> b l (n d)"
            )
        )


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: Optional[int] = None):
        super(SwiGLU, self).__init__()
        d_hidden = 4 * d_model if d_hidden is None else d_hidden

        self.gate = nn.Linear(d_model, d_hidden, bias=False)
        self.hidden = nn.Linear(d_model, d_hidden, bias=False)
        self.out = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(
            F.silu(self.gate(x)) * self.hidden(x)
        )


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class DiTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_hidden: Optional[int] = None, norm_eps: float = 1e-6):
        super(DiTBlock, self).__init__()
        d_hidden = 4 * d_model if d_hidden is None else d_hidden

        self.cond_emb = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_model)
        )

        self.attn = Attention(d_model, n_heads)
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn_alpha = nn.Linear(d_model, d_model, bias=False)
        self.attn_beta = nn.Linear(d_model, d_model, bias=False)
        self.attn_gamma = nn.Linear(d_model, d_model, bias=False)

        self.ffn = SwiGLU(d_model, d_hidden)
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn_alpha = nn.Linear(d_model, d_model, bias=False)
        self.ffn_beta = nn.Linear(d_model, d_model, bias=False)
        self.ffn_gamma = nn.Linear(d_model, d_model, bias=False)

        with torch.no_grad():
            self.attn_alpha.weight.zero_()
            self.ffn_alpha.weight.zero_()

    def forward(self, x: Tensor, c: Tensor, freqs: Tensor) -> Tensor:
        c = self.cond_emb(c)

        x = x + self.attn(
            self.attn_norm(x) * self.attn_gamma(c)[:, None] + self.attn_beta(c)[:, None],
            freqs
        ) * self.attn_alpha(c)[:, None]

        return x + self.ffn(
            self.ffn_norm(x) * self.ffn_gamma(c)[:, None] + self.ffn_beta(c)[:, None]
        ) * self.ffn_gamma(c)[:, None]


class SinusoidalEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super(SinusoidalEmbedding, self).__init__()
        assert d_model % 2 == 0

        self.register_buffer(
            "theta",
            1.0 / (1e4 ** (2 * torch.arange(d_model // 2) / d_model)),
            persistent=False
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x[:, None] * self.theta

        return torch.stack((
            x.cos(),
            x.sin()
        ), dim=-1).flatten(-2)


class DiTConfig(Config):
    class VAE(BaseModel):
        experiment: str
        checkpoint: int

    vae: VAE

    num_classes: int | Tuple[int, ...]

    patch_size: int

    d_model: int
    n_heads: int
    n_layers: int


class DiT(AbstractModel):
    def __init__(
            self, in_channels: int, in_size: int, num_classes: int | Tuple[int, ...],
            patch_size: int,
            d_model: int, n_heads: int, n_layers: int
    ):
        super(DiT, self).__init__()
        self.in_channels = in_channels
        self.in_size = in_size

        assert d_model % n_heads == 0

        d_attn = d_model // n_heads
        assert d_attn % 2 == 0

        self.patch_size = patch_size
        num_patches = in_size // patch_size

        theta = 1.0 / (1e4 ** (2 * torch.arange(d_attn // 2) / d_attn))
        freqs = torch.outer(torch.arange(num_patches * num_patches), theta)
        self.register_buffer(
            "freqs",
            torch.polar(torch.ones_like(freqs), freqs),
            persistent=False
        )

        self.patch_emb = nn.Linear(in_channels * patch_size * patch_size, d_model)

        self.layers = nn.ModuleList([
            DiTBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])

        num_classes = num_classes if isinstance(num_classes, tuple) else (num_classes,)
        self.class_embs = nn.ModuleList([
            nn.Embedding(n, d_model) for n in num_classes
        ])
        self.t_emb = SinusoidalEmbedding(d_model)

        self.head = nn.Linear(d_model, in_channels * patch_size * patch_size)

    def forward(self, x_t: Tensor, t: Tensor, cls: Tensor) -> Tensor:
        c = self.t_emb(t) + torch.stack([
            self.class_embs[i](cls[:, i])
            for i in range(len(self.class_embs))
        ], dim=-1).sum(-1)

        x = self.patch_emb(rearrange(
            x_t, "b c (p1 h) (p2 h) -> b (h w) (c p1 p2)",
            p1=self.patch_size, p2=self.patch_size
        ))

        for layer in self.layers:
            x = layer(x, c, self.freqs)

        return rearrange(
            self.head(x), "b (h w) (c p1 p2) -> b c (p1 h) (p2 w)",
            p1=self.patch_size, p2=self.patch_size, h=self.in_size, w=self.in_size
        )

    @staticmethod
    def from_config(config: DiTConfig) -> DiT:
        vae_config = VAETrainConfig.from_yaml(
            os.path.join(
                "experiments", config.vae.experiment
            )
        )

        return DiT(
            in_channels=vae_config.encoder.latent_channels,
            in_size=vae_config.data.image_size // (2 ** (len(vae_config.encoder.dims) - 1)),
            num_classes=config.num_classes,
            patch_size=config.patch_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers
        )
