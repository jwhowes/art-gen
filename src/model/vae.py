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


class VAEModelConfig(ModelConfig):
    def __init__(self, config: Optional[dict] = None):
        self.d_latent = 4
        super().__init__(config)


class DiscriminatorConfig(ModelConfig):
    def __init__(self, config: Optional[dict] = None):
        self.patch_size = 4
        super().__init__(config)


class VAEConfig(SubConfig):
    def __init__(self, config: Optional[dict] = None):
        self.kl_weight: float = 1e-4
        self.adv_weight: float = 0.5
        self.adv_start: float = 0.5

        super().__init__(config)

        self.vae: VAEModelConfig = VAEModelConfig(
            config.get("vae") if config is not None else None
        )

        self.discriminator: DiscriminatorConfig = DiscriminatorConfig(
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
            self, image_channels: int, patch_size: int = 4,
            dims: Tuple[int, ...] = (96, 192, 384, 768), depths: Tuple[int, ...] = (3, 3, 9, 4)
    ):
        super(Discriminator, self).__init__(image_channels=None, dims=dims, depths=depths)
        self.stem = nn.Conv2d(image_channels, dims[0], kernel_size=patch_size, stride=patch_size)
        self.head = nn.Conv2d(dims[-1], 1, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = super().forward(x)

        return self.head(x).squeeze(1)


class VAE(nn.Module):
    def __init__(
            self, image_channels: int, d_latent: int,
            vae_dims: Tuple[int, ...] = (96, 192, 384, 768), vae_depths: Tuple[int, ...] = (3, 3, 9, 4),
            disc_dims: Tuple[int, ...] = (96, 192, 384, 768), disc_depths: Tuple[int, ...] = (3, 3, 9, 4),
            disc_patch_size: int = 4,
            kl_weight: float = 1e-4, adv_weight: float = 0.5
    ):
        super(VAE, self).__init__()
        self.kl_weight = kl_weight
        self.adv_weight = adv_weight

        self.encoder = VAEEncoder(image_channels, d_latent, vae_dims, vae_depths)
        self.decoder = VAEDecoder(image_channels, d_latent, vae_dims, vae_depths)
        self.discriminator = Discriminator(image_channels, disc_patch_size, disc_dims, disc_depths)

    def state_dict(self, *args, **kwargs):
        return {
            "encoder": self.encoder.state_dict(*args, **kwargs),
            "decoder": self.decoder.state_dict(*args, **kwargs),
            "discriminator": self.discriminator.state_dict(*args, **kwargs)
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, *args, **kwargs):
        if strict or "encoder" in state_dict:
            self.encoder.load_state_dict(state_dict["encoder"], strict, *args, **kwargs)

        if strict or "decoder" in state_dict:
            self.decoder.load_state_dict(state_dict["decoder"], strict, *args, **kwargs)

        if strict or "discriminator" in state_dict:
            self.discriminator.load_state_dict(state_dict["discriminator"], strict, *args, **kwargs)

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

        rec_loss = (pred - image).abs().mean()
        kl_loss = dist.kl.mean()

        if do_adv:
            logits_fake = self.discriminator(pred)
            adv_loss = F.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))
        else:
            adv_loss = torch.tensor(0.0, device=image.device)

        loss = rec_loss + self.kl_weight * kl_loss + self.adv_weight * adv_loss

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
            pred = self(image, return_dist=False).detach()

        logits_real = self.discriminator(image)
        logits_fake = self.discriminator(pred)

        loss = (
            F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real)) +
            F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
        )

        return loss

    def vae_parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def disc_parameters(self):
        return self.discriminator.parameters()
