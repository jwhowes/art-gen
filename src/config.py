import os
from abc import ABC
from typing import Self

import yaml
from pydantic import BaseModel


class Config(ABC, BaseModel):
    @classmethod
    def from_yaml(cls, yaml_path: str) -> Self:
        if not os.path.isfile(yaml_path):
            return cls()

        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        if config is None:
            return cls()

        return cls.model_validate(config)

    def to_yaml(self, yaml_path: str):
        with open(yaml_path, "w+") as f:
            yaml.safe_dump(self.model_dump(), f)
