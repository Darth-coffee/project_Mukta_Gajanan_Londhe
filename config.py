import os
import torch

# === Directory Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "final_weights.pth")

# === Training Parameters ===
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Image Parameters (for SimpleCNN) ===
IMG_HEIGHT = 200  # Corrected: height = 200
IMG_WIDTH = 600   # Corrected: width = 600
INPUT_CHANNELS = 3  # RGB images

# === Extra Settings (Optional) ===
# You can add config for data augmentation, logs, saving intervals, etc.
