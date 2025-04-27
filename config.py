import os
import torch

# === Directory Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Training and testing directories
TRAIN_DIR = r"E:/new_data/cnn_ready_images/syllable_padded"
TEST_POSITIVE_DIR = r"E:/cnn_project/syllable"
TEST_NEGATIVE_DIR = r"E:/cnn_project/non_syllable"

# Checkpoints directory
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "final_weights.pth")

# === Training Parameters ===
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Image Parameters ===
IMG_HEIGHT = 600  # Corrected: height = 200
IMG_WIDTH = 200   # Corrected: width = 600
INPUT_CHANNELS = 3  # RGB images

# === Extra Settings (Optional) ===
# Future extension for data augmentation, logs, etc.
