from typing import Type

import click

from src.train.abstract import AbstractTrainer
from src.train.vae import VAETrainer


@click.group()
def cli():
    pass


def trainer(cls: Type[AbstractTrainer], command_name: str):
    @cli.command(command_name)
    @click.argument("config-path", type=click.Path(exists=True, dir_okay=False))
    def command(config_path: str):
        cls.from_config(config_path).train()

    return command


trainer(VAETrainer, "vae")


if __name__ == "__main__":
    cli()
