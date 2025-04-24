# interface.py

import argparse
import os
from PIL import Image
import numpy as np
import torch

from model import SimpleCNN
from config import CHECKPOINT_PATH, DEVICE

# Manual image transformation
def image_to_tensor(img):
    img = img.resize((600, 200))  # Resize to (width, height)
    img = np.array(img).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC → CHW
    return torch.tensor(img).unsqueeze(0)  # Add batch dimension

def main():
    parser = argparse.ArgumentParser(description="Predict bird syllables from spectrogram images")
    parser.add_argument('--img', type=str, help='Path to a spectrogram image')
    args = parser.parse_args()

    if not args.img or not os.path.exists(args.img):
        print("Please provide a valid image path using --img")
        return

    # Load model
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # Load and process image
    image = Image.open(args.img).convert("RGB")
    tensor = image_to_tensor(image).to(DEVICE)

    # Predict
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()
        prediction = "syllable" if prob > 0.5 else "noise"

    print(f"{args.img}: {prediction} ({prob:.2f})")

if __name__ == "__main__":
    main()
