import torch
import torch.nn.functional as F

from torch import nn
from dataclasses import dataclass
from typing import Tuple, Mapping, Any, Optional

from ..config import SubConfig

from .util import Encoder, Decoder


class ModelConfig(SubConfig):
    def __init__(self, config: Optional[dict] = None):
        self.dims: Tuple[int, ...] = (96, 192, 384, 768)
        self.depths: Tuple[int, ...] = (3, 3, 9, 4)

        super().__init__(config)


class VAEConfig(SubConfig):
    def __init__(self, config: Optional[dict] = None):
        self.kl_weight: float = 1e-4
        self.adv_weight: float = 0.5
        self.adv_start: float = 0.5
        self.log_var_init: float = 0.0

        super().__init__(config)

        self.vae = ModelConfig(
            config.get("vae") if config is not None else None
        )

        if config is not None and "vae" in config:
            self.vae.d_latent = config["vae"].get("d_latent", 4)
        else:
            self.vae.d_latent = 4

        self.discriminator = ModelConfig(
            config.get("discriminator") if config is not None else None
        )


@dataclass
class DiagonalGaussian:
    mean: torch.FloatTensor
    log_var: torch.FloatTensor

    @property
    def kl(self):
        return 0.5 * (
            self.mean.pow(2) + self.log_var.exp() - 1.0 - self.log_var
        ).flatten(1).mean(1)

    def sample(self):
        return torch.randn_like(self.log_var) * (0.5 * self.log_var).exp() + self.mean

    @property
    def shape(self):
        return self.mean.shape

    @property
    def device(self):
        return self.mean.device

    def to(self, device: torch.device):
        self.mean.to(device)
        self.log_var.to(device)


class VAEEncoder(Encoder):
    def __init__(
            self, image_channels: int, d_latent: int,
            dims: Tuple[int, ...] = (96, 192, 384, 768), depths: Tuple[int, ...] = (3, 3, 9, 4)
    ):
        super(VAEEncoder, self).__init__(image_channels, dims, depths)
        self.head = nn.Conv2d(dims[-1], 2 * d_latent, kernel_size=5, padding=2)

    def forward(self, x) -> DiagonalGaussian:
        mean, log_var = self.head(super().forward(x)).chunk(2, 1)

        return DiagonalGaussian(mean, log_var)


class VAEDecoder(Decoder):
    def __init__(
            self, image_channels: int, d_latent: int,
            dims: Tuple[int, ...] = (96, 192, 384, 768), depths: Tuple[int, ...] = (3, 3, 9, 4)
    ):
        super(VAEDecoder, self).__init__(d_latent, dims, depths)
        self.head = nn.Conv2d(dims[0], image_channels, kernel_size=5, padding=2)

    def forward(self, z):
        return self.head(super().forward(z))


class Discriminator(Encoder):
    def __init__(
            self, image_channels: int,
            dims: Tuple[int, ...] = (96, 192, 384, 768), depths: Tuple[int, ...] = (3, 3, 9, 4)
    ):
        super(Discriminator, self).__init__(image_channels, dims, depths)
        self.head = nn.Conv2d(dims[-1], 1, kernel_size=1)

    def forward(self, x):
        return self.head(super().forward(x)).squeeze(1)


class VAE(nn.Module):
    def __init__(
            self, image_channels: int, d_latent: int,
            vae_dims: Tuple[int, ...] = (96, 192, 384, 768), vae_depths: Tuple[int, ...] = (3, 3, 9, 4),
            disc_dims: Tuple[int, ...] = (96, 192, 384, 768), disc_depths: Tuple[int, ...] = (3, 3, 9, 4),
            log_var_init: float = 0.0, kl_weight: float = 1e-4, adv_weight: float = 0.5
    ):
        super(VAE, self).__init__()
        self.kl_weight = kl_weight
        self.adv_weight = adv_weight

        self.encoder = VAEEncoder(image_channels, d_latent, vae_dims, vae_depths)
        self.decoder = VAEDecoder(image_channels, d_latent, vae_dims, vae_depths)
        self.discriminator = Discriminator(image_channels, disc_dims, disc_depths)
        self.log_var = nn.Parameter(torch.ones(size=()) * log_var_init)

    def state_dict(self, *args, **kwargs):
        return {
            "encoder": self.encoder.state_dict(*args, **kwargs),
            "decoder": self.decoder.state_dict(*args, **kwargs),
            "discriminator": self.discriminator.state_dict(*args, **kwargs),
            "log_var": self.log_var.data
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, *args, **kwargs):
        if strict or "encoder" in state_dict:
            self.encoder.load_state_dict(state_dict["encoder"], strict, *args, **kwargs)

        if strict or "decoder" in state_dict:
            self.decoder.load_state_dict(state_dict["decoder"], strict, *args, **kwargs)

        if strict or "discriminator" in state_dict:
            self.discriminator.load_state_dict(state_dict["discriminator"], strict, *args, **kwargs)

        if strict or "log_var" in state_dict:
            self.log_var.data = state_dict["log_var"]

    def forward(self, image, sample=True, return_dist=True):
        dist = self.encoder(image)

        if sample:
            z = dist.sample()
        else:
            z = dist.mean

        pred = self.decoder(z)

        if return_dist:
            return pred, dist

        return pred

    def vae_loss(self, image, do_adv=True):
        pred, dist = self(image)

        logits_fake = self.discriminator(pred)

        rec_loss = ((image - pred).abs() / self.log_var.exp() + self.log_var).mean()
        kl_loss = dist.kl.mean()
        adv_loss = -logits_fake.mean()

        if do_adv:
            # Calculate adaptive adversarial weight
            rec_grads = torch.autograd.grad(rec_loss, self.decoder.head.weight, retain_graph=True)[0]
            adv_grads = torch.autograd.grad(adv_loss, self.discriminator.head.weight, retain_graph=True)[0]

            with torch.no_grad():
                adaptive_adv_weight = (torch.norm(rec_grads) / (torch.norm(adv_grads) + 1e-4)).clamp(min=0.0, max=1e4)
        else:
            adaptive_adv_weight = 0.0
            adv_loss = torch.tensor(0.0, device=image.device)

        loss = rec_loss + self.kl_weight * kl_loss + self.adv_weight * adaptive_adv_weight * adv_loss

        return {
            "loss": loss,
            "metrics": {
                "rec loss": rec_loss.item(),
                "kl loss": kl_loss.item(),
                "adv loss": adv_loss.item()
            }
        }

    def disc_loss(self, image):
        with torch.no_grad():
            pred = self(image, return_dist=False)

        logits_real = self.discriminator(image)
        logits_fake = self.discriminator(pred)

        loss = 0.5 * (
            F.relu(1. - logits_real).mean() +
            F.relu(1. + logits_fake).mean()
        )

        return loss

    def vae_parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters()) + [self.log_var]

    def disc_parameters(self):
        return self.discriminator.parameters()
