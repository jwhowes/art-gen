# TODO:
#   1. FiLM                             x
#   2. FiLMUNet                         x
#   3. Diffusion Noise Scheduler        x
#   4. DDPM Loss (with pretrained VAE)
#   5. Diffusion Sampling
import torch
import os

from torch import nn
from typing import Tuple

from .util import ConvNeXtFiLMBlock, SinusoidalPosEmb, NoiseScheduler

from ..vae import VAEConfig, VAEEncoder, VAEDecoder
from ...config import Config


class DiffusionModel(nn.Module):
    def __init__(
            self, image_channels: int, image_size: int, vae_dir: str, vae_epoch: int, sample: bool = False,
            d_t: int = 768, dims: Tuple[int, ...] = (192, 384, 768, 1536), depths: Tuple[int, ...] = (3, 3, 5, 4),
            num_timesteps: int = 1000
    ):
        super(DiffusionModel, self).__init__()
        self._sample = sample

        vae_config = Config(VAEConfig, os.path.join(vae_dir, "config.yaml")).model.vae
        ckpt = torch.load(
            os.path.join(vae_dir, "model", f"checkpoint_{vae_epoch:02}.pt"),
            weights_only=True, map_location="cpu"
        )
        if self._sample:
            self.decoder = VAEDecoder(
                image_channels=image_channels,
                d_latent=vae_config.d_latent,
                dims=vae_config.dims,
                depths=vae_config.depths
            )
            self.decoder.load_state_dict(ckpt["decoder"])
            self.decoder.eval()
            self.decoder.requires_grad_(False)
        else:
            self.encoder = VAEEncoder(
                image_channels=image_channels,
                d_latent=vae_config.d_latent,
                dims=vae_config.dims,
                depths=vae_config.depths
            )
            self.encoder.load_state_dict(ckpt["encoder"])
            self.encoder.eval()
            self.encoder.requires_grad_(False)

        self.latent_size = image_size // (2 ** (len(vae_config.dims) - 1))
        self.d_latent = vae_config.d_latent

        self.t_model = nn.Sequential(
            SinusoidalPosEmb(d_t),
            nn.Linear(d_t, 4 * d_t),
            nn.GELU(),
            nn.Linear(4 * d_t, d_t)
        )

        self.stem = nn.Conv2d(self.d_latent, dims[0], kernel_size=5, padding=2)

        self.down_path = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.down_path.append(nn.ModuleList([
                ConvNeXtFiLMBlock(dims[i], d_t) for _ in range(depths[i])
            ]))
            self.down_samples.append(nn.Conv2d(
                dims[i], dims[i + 1], kernel_size=4, stride=2, padding=1
            ))

        self.mid_blocks = nn.ModuleList([
            ConvNeXtFiLMBlock(dims[-1], d_t) for _ in range(depths[-1])
        ])

        self.up_path = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_combines = nn.ModuleList()
        for i in range(len(dims) - 2, -1, -1):
            self.up_samples.append(nn.ConvTranspose2d(
                dims[i + 1], dims[i], kernel_size=4, stride=2, padding=1
            ))
            self.up_combines.append(nn.Conv2d(
                2 * dims[i], dims[i], kernel_size=5, padding=2
            ))
            self.up_path.append(nn.ModuleList([
                ConvNeXtFiLMBlock(dims[i], d_t) for _ in range(depths[-1])
            ]))

        self.head = nn.Conv2d(dims[0], self.d_latent, kernel_size=5, padding=2)

        self.scheduler = NoiseScheduler(num_timesteps)

    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(requires_grad)

        if self._sample:
            self.decoder.requires_grad_(False)
        else:
            self.encoder.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)

        if self._sample:
            self.decoder.eval()
        else:
            self.encoder.eval()

    def pred_noise(self, z_t, t):
        t_emb = self.t_model(t)

        x = self.stem(z_t)

        acts = []
        for down_blocks, down_sample in zip(self.down_path, self.down_samples):
            for block in down_blocks:
                x = block(x, t_emb)

            acts.append(x)
            x = down_sample(x)

        for block in self.mid_blocks:
            x = block(x, t_emb)

        for up_blocks, up_sample, up_combine, act in zip(self.up_path, self.up_samples, self.up_combines, acts[::-1]):
            x = up_combine(torch.concatenate((
                up_sample(x),
                act
            ), dim=1))

            for block in up_blocks:
                x = block(x, t)

        return self.head(x)

    def forward(self, image):
        assert not self._sample, "Cannot train if model was initiated in sample mode"

        z_0 = self.encoder(image).sample()

        z_t, eps, t = self.scheduler.add_noise(z_0)
