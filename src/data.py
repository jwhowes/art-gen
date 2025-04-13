from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset

from .config import Config


class DatasetConfig(Config):
    batch_size: int
    image_size: int


class UnconditionalDataset(Dataset):
    def __init__(self, image_size: int):
        self.data = load_dataset("huggan/wikiart", split="train")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
            ),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx]["image"])
