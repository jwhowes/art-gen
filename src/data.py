import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset

from . import accelerator
from .config import DatasetConfig


class ArtDataset(Dataset):
    @accelerator.main_process_first()
    def __init__(self, image_size: int = 256):
        self.ds = load_dataset("Artificio/WikiArt", split="train")

        self.image_size = image_size
        self.image_channels = 3

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            ),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            )
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.transform(self.ds[idx]["image"])


def get_dataloader(config: DatasetConfig, shuffle: bool = True) -> DataLoader[ArtDataset]:
    dataset = ArtDataset(
        image_size=config.image_size
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        pin_memory=True
    )
