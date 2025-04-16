import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch.utils.data import DataLoader

from ..config import Config
from ..model.vae import VAEEncoder, VAEDecoder, VAEEncoderConfig, VAEDecoderConfig
from ..model.gan import Discriminator, DiscriminatorConfig
from ..data import UnconditionalDataset, DatasetConfig
from ..accelerator import accelerator

from .abstract import AbstractTrainer


class VAETrainConfig(Config):
    class VAETrain(BaseModel):
        num_epochs: int = 5
        log_interval: int = 100

        vae_lr: float = 1e-4
        disc_lr: float = 1e-4
        weight_decay: float = 0.0
        clip_grad: float = 1.0

        kl_weight: float = 1e-4
        disc_weight: float = 0.1

    encoder: VAEEncoderConfig
    decoder: VAEDecoderConfig
    discriminator: DiscriminatorConfig

    train: VAETrain = VAETrain()
    data: DatasetConfig


class VAETrainer(AbstractTrainer):
    metrics = ["kl", "disc", "recon"]
    config_cls = VAETrainConfig

    def train(self):
        encoder = VAEEncoder.from_config(self.config.encoder)
        decoder = VAEDecoder.from_config(self.config.decoder)
        discriminator = Discriminator.from_config(self.config.discriminator)

        dataset = UnconditionalDataset(self.config.data.image_size)
        dataloader: DataLoader = DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            pin_memory=True
        )

        vae_opt = torch.optim.AdamW(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=self.config.train.vae_lr,
            weight_decay=self.config.train.weight_decay
        )
        disc_opt = torch.optim.AdamW(
            discriminator.parameters(),
            lr=self.config.train.disc_lr,
            weight_decay=self.config.train.weight_decay
        )

        latent_size = (
            self.config.decoder.latent_channels,
            self.config.data.image_size // (2 ** (len(self.config.encoder.dims) - 1)),
            self.config.data.image_size // (2 ** (len(self.config.encoder.dims) - 1))
        )

        encoder, decoder, discriminator, dataloader, vae_opt, disc_opt = accelerator.prepare(
            encoder, decoder, discriminator, dataloader, vae_opt, disc_opt
        )

        encoder.train()
        decoder.train()
        discriminator.train()
        for epoch in range(self.config.train.num_epochs):
            if accelerator.is_main_process:
                print(f"EPOCH {epoch + 1} / {self.config.train.num_epochs}")

            total_kl = 0
            total_recon = 0
            total_disc = 0
            for i, image in enumerate(dataloader):
                vae_opt.zero_grad()

                with accelerator.autocast():
                    dist = encoder(image)
                    kl = dist.kl().mean()

                    z = dist.sample()
                    pred = decoder(z)

                    recon = F.mse_loss(pred, image)

                    pred_disc = discriminator(pred)

                    disc = F.binary_cross_entropy_with_logits(pred_disc, torch.ones_like(pred_disc))

                    loss = recon + self.config.train.kl_weight * kl + self.config.train.disc_weight * disc

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(encoder.parameters(), self.config.train.clip_grad)
                accelerator.clip_grad_norm_(decoder.parameters(), self.config.train.clip_grad)

                vae_opt.step()

                total_kl += kl.item()
                total_recon += recon.item()
                total_disc += disc.item()

                if accelerator.is_main_process and i % self.config.train.log_interval == 0:
                    print(f"\t{i} / {len(dataloader)} iters.\tKL: {kl.item():.4f}\tDisc: {disc.item():.4f}\tRecon: {recon.item():.4f}")

                disc_opt.zero_grad()

                with accelerator.autocast():
                    pred_real = discriminator(image)
                    pred_fake = discriminator(pred.detach())

                    loss = (
                        F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real)) +
                        F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
                    )

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(discriminator.parameters(), self.config.train.clip_grad)

                disc_opt.step()

            self.save_checkpoint(
                {
                    "encoder": accelerator.get_state_dict(encoder),
                    "decoder": accelerator.get_state_dict(decoder),
                    "discriminator": accelerator.get_state_dict(discriminator)
                },
                total_kl / (len(dataloader) // 2),
                total_disc / (len(dataloader) // 2),
                total_recon / (len(dataloader) // 2)
            )
