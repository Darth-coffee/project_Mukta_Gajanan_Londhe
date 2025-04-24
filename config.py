# config.py
import numpy as np
import os
import torch  # Import torch to check for CUDA availability

# Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "final_weights.pth")

# Training Parameters 
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Corrected to use is_available()

# Image Resize and Input Information
IMG_HEIGHT = 600  # Fixed image height
IMG_WIDTH = 200   # Fixed image width
INPUT_CHANNELS = 3  # Number of color channels (3 for RGB, 1 for grayscale)

# You can add more specific configurations related to the model or dataset here
