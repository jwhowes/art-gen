import yaml
import os
import torch

from typing import Optional, Tuple, TypeVar, Generic, Type
from datetime import datetime

from . import accelerator


class SubConfig:
    def __init__(self, config: Optional[dict] = None):
        if config is not None:
            for k, v in config.items():
                if hasattr(self, k):
                    if isinstance(getattr(self, k), float):
                        setattr(self, k, float(v))
                    else:
                        setattr(self, k, v)


class DatasetConfig(SubConfig):
    def __init__(self, config: Optional[dict] = None):
        self.batch_size: int = 32
        self.image_size: int = 256

        super().__init__(config)

class Config[T](SubConfig):
    def unknown_tag(self, loader, suffix, node):
        if isinstance(node, yaml.ScalarNode):
            constructor = loader.__class__.construct_scalar
        elif isinstance(node, yaml.SequenceNode):
            constructor = loader.__class__.construct_sequence
        elif isinstance(node, yaml.MappingNode):
            constructor = loader.__class__.construct_mapping
        else:
            raise NotImplementedError

        data = constructor(loader, node)

        return data

    def __init__(
            self, model_cls: Type[SubConfig], config_path: str,
            save: bool = False, metrics: Tuple[str, ...] = ("loss",)
    ):
        yaml.add_multi_constructor('!', self.unknown_tag)
        yaml.add_multi_constructor('tag:', self.unknown_tag)

        self.lr: float = 5e-5
        self.log_interval: int = 100
        self.num_epochs: int = 32
        self.warmup: float = 0.2
        self.clip_grad: Optional[float] = None

        with open(config_path) as f:
            config: Optional[dict] = yaml.load(f, Loader=yaml.FullLoader)

        super().__init__(config)

        exp_name: str = os.path.splitext(os.path.basename(config_path))[0]
        exp_dir: str = os.path.join("experiments", exp_name)
        model_dir = os.path.join(exp_dir, "model")

        self.dataset = DatasetConfig(
            config.get("dataset") if config is not None else None
        )
        self.model: T = model_cls(
            config.get("model") if config is not None else None
        )

        if save and accelerator.is_main_process:
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

            with open(os.path.join(exp_dir, "config.yaml"), "w+") as f:
                yaml.dump(self, f)

            with open(os.path.join(exp_dir, "log.csv"), "w+") as f:
                f.write("epoch,")
                f.write(",".join(metrics))
                f.write(",timestamp\n")

        self.metrics = metrics
        self._epoch = 0

        self.model_dir = model_dir
        self.exp_dir = exp_dir

    def log(self, model: torch.nn.Module, *metrics: Tuple[float, ...]):
        self._epoch += 1

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            torch.save(
                accelerator.get_state_dict(model),
                os.path.join(self.model_dir, f"checkpoint_{self._epoch:02}.pt")
            )

            with open(os.path.join(self.exp_dir, "log.csv"), "a") as f:
                f.write(str(self._epoch))
                f.write(",".join([
                    f"{float(m):.4f}" for m in metrics
                ]))
                f.write(f",{datetime.now()}\n")
