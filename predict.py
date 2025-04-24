import os
import torch
from PIL import Image
import numpy as np
from model import ResNet18
from config import CHECKPOINT_PATH, DATA_DIR

# Device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18(num_classes=1).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# image conversion to tensor
def image_to_tensor(img):
    img = np.array(img).astype(np.float32) / 255.0  # Normalize
    img = img.transpose(2, 0, 1)                    # HWC → CHW
    return torch.tensor(img).unsqueeze(0)           # Add batch dim


for fname in os.listdir(DATA_DIR):
    if fname.endswith(".jpg") or fname.endswith(".png"):
        path = os.path.join(DATA_DIR, fname)
        image = Image.open(path).convert("RGB")
        tensor = image_to_tensor(image).to(device)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.sigmoid(output).item()
            prediction = "syllable" if prob > 0.5 else "noise"

        print(f"{fname}: {prediction} ({prob:.2f})")
