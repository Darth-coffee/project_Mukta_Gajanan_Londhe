# dataset.py

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import config

class SyllableDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with 'syllable' and 'non-syllable' subdirectories.
        """
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []

        syllable_dir = os.path.join(root_dir, "syllable")
        nonsyllable_dir = os.path.join(root_dir, "non-syllable")

        for img_name in os.listdir(syllable_dir):
            self.image_paths.append(os.path.join(syllable_dir, img_name))
            self.labels.append(1.0)

        for img_name in os.listdir(nonsyllable_dir):
            self.image_paths.append(os.path.join(nonsyllable_dir, img_name))
            self.labels.append(0.0)

        # Sort consistently
        combined = list(zip(self.image_paths, self.labels))
        combined.sort()
        self.image_paths, self.labels = zip(*combined)

        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
            transforms.ToTensor(),  # Converts to [0,1] and moves channel to first dimension
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)
