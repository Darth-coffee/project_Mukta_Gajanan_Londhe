# config.py

import os
import torch

# === Directory Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")  # Training+Validation Data
DATA_DIR = os.path.join(BASE_DIR, "data")        # Testing Data
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "final_weights.pth")

# === Training Parameters ===
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATIENCE = 4  # For early stopping

# === Image Parameters ===
IMG_HEIGHT = 200  # height of spectrogram images
IMG_WIDTH = 600   # width of spectrogram images
INPUT_CHANNELS = 3  # RGB images
