import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class SyllableDataset(Dataset):
    def __init__(self, folder_path, label, transform=None):
        self.folder_path = folder_path
        self.label = torch.tensor([label], dtype=torch.float32)
        self.transform = transform
        self.images = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.images[idx])
        image = Image.open(img_path).convert('RGB')  # Convert to RGB
        image = image.resize((600, 200))  # Resize to fixed dimensions
        image_tensor = torch.tensor(np.array(image) / 255.0, dtype=torch.float32).permute(2, 0, 1)  # Normalize and change to [C, H, W]
        return image_tensor, self.label
