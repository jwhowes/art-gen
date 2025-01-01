import torch

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src import accelerator
from src.config import Config
from src.model.vae import VAE, VAEConfig
from src.data import get_dataloader, ArtDataset


def train(
        model: VAE,
        dataloader: DataLoader[ArtDataset],
        config: Config[VAEConfig]
):
    vae_opt = torch.optim.Adam(accelerator.unwrap_model(model).vae_parameters(), lr=config.lr)
    vae_lr_scheduler = get_cosine_schedule_with_warmup(
        vae_opt,
        num_warmup_steps=int(config.warmup * len(dataloader)),
        num_training_steps=config.num_epochs * len(dataloader)
    )

    disc_opt = torch.optim.Adam(accelerator.unwrap_model(model).disc_parameters(), lr=config.lr)
    disc_lr_scheduler = get_cosine_schedule_with_warmup(
        disc_opt,
        num_warmup_steps=int(config.warmup * len(dataloader)),
        num_training_steps=config.num_epochs * len(dataloader)
    )

    model, dataloader, vae_opt, vae_lr_scheduler, disc_opt, disc_lr_scheduler = accelerator.prepare(
        model, dataloader, vae_opt, vae_lr_scheduler, disc_opt, disc_lr_scheduler
    )

    model.train()
    for epoch in range(config.num_epochs):
        if accelerator.is_main_process:
            print(f"EPOCH {epoch + 1} / {config.num_epochs}")

        metrics = {
            "rec loss": 0,
            "kl loss": 0,
            "adv loss": 0
        }
        for i, image in enumerate(dataloader):
            do_adv = epoch + (i / len(dataloader)) >= config.model.adv_start

            vae_opt.zero_grad()

            with accelerator.autocast():
                vae_loss = accelerator.unwrap_model(model).vae_loss(image, do_adv=do_adv)

            for m in metrics:
                metrics[m] += vae_loss["metrics"][m]

            accelerator.backward(vae_loss["loss"])

            if config.clip_grad is not None:
                accelerator.clip_grad_norm_(
                    accelerator.unwrap_model(model).vae_parameters(),
                    config.clip_grad
                )

            vae_opt.step()
            vae_lr_scheduler.step()

            if do_adv:
                disc_opt.zero_grad()

                with accelerator.autocast():
                    disc_loss = accelerator.unwrap_model(model).disc_loss(image)

                accelerator.backward(disc_loss)

                if config.clip_grad is not None:
                    accelerator.clip_grad_norm_(
                        accelerator.unwrap_model(model).disc_parameters(),
                        config.clip_grad
                    )

                disc_opt.step()

            disc_lr_scheduler.step()

            if accelerator.is_main_process and i % config.log_interval == 0:
                print(
                    f"{i} / {len(dataloader)} iters.\t"
                    f"Rec Loss: {vae_loss['metrics']['rec loss']:.4f}\t"
                    f"KL Loss: {vae_loss['metrics']['kl loss']:.4f}\t"
                    f"Adv Loss: {vae_loss['metrics']['adv loss']:.4f}"
                )

        config.log(model, [m / len(dataloader) for m in metrics.values()])


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("config", type=str)
    args = parser.parse_args()

    config: Config[VAEConfig] = Config(
        VAEConfig, args.config, save=True, metrics=("rec loss", "kl loss", "adv loss")
    )

    dataloader: DataLoader[ArtDataset] = get_dataloader(config.dataset, shuffle=True)

    model = VAE(
        image_channels=dataloader.dataset.image_channels,

        d_latent=config.model.vae.d_latent,
        vae_dims=config.model.vae.dims,
        vae_depths=config.model.vae.depths,

        disc_dims=config.model.discriminator.dims,
        disc_depths=config.model.discriminator.depths,

        log_var_init=config.model.log_var_init,
        kl_weight=config.model.kl_weight,
        adv_weight=config.model.adv_weight
    )

    train(model, dataloader, config)
