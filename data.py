import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T


class FaceEmotionRecDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.has_labels = "emotion" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.array(row["pixels"].split(), dtype=np.float32).reshape(48, 48)
        pixels = pixels / 255.0
        img = torch.tensor(pixels, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            img = self.transform(img)

        if self.has_labels:
            label = int(row["emotion"])
            return img, label
        return img


def get_dataloaders(csv_path, batch_size=64, val_split=0.1, seed=42, augment=False):
    base_dataset = FER2013Dataset(csv_path)

    val_size = int(len(base_dataset) * val_split)
    train_size = len(base_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(
        range(len(base_dataset)), [train_size, val_size], generator=generator
    )

    if augment:
        train_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
        ])
    else:
        train_transform = None

    train_ds = FER2013Dataset(csv_path, transform=train_transform)
    train_ds = torch.utils.data.Subset(train_ds, train_idx.indices)

    val_ds = FER2013Dataset(csv_path)  # no augmentation on val
    val_ds = torch.utils.data.Subset(val_ds, val_idx.indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader