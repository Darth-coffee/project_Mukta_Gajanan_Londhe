import os
import torch

# === Directory Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_POSITIVE_DIR = os.path.join(BASE_DIR, "data", "syllable")
TEST_NEGATIVE_DIR = os.path.join(BASE_DIR, "data", "non_syllable")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "final_weights.pth")

# === Training Parameters ===
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Image Parameters ===
IMG_HEIGHT = 200
IMG_WIDTH = 600
INPUT_CHANNELS = 3

# === Make sure checkpoints dir exists ===
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
