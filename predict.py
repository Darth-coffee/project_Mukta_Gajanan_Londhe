import os
import torch
from PIL import Image
import numpy as np
from model import SimpleCNN  # ✅ Use your custom CNN model here
from config import CHECKPOINT_PATH, DATA_DIR

# === Device and Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)  # ✅ Instantiate your model
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# === Image Preprocessing ===
def image_to_tensor(img):
    img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0,1]
    img = img.transpose(2, 0, 1)                    # HWC → CHW
    return torch.tensor(img).unsqueeze(0)           # Add batch dimension

# === Prediction Loop ===
for fname in os.listdir(DATA_DIR):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(DATA_DIR, fname)
        try:
            image = Image.open(path).convert("RGB")
            tensor = image_to_tensor(image).to(device)

            with torch.no_grad():
                output = model(tensor)
                prob = torch.sigmoid(output).item()
                prediction = "syllable" if prob > 0.5 else "noise"

            print(f"{fname}: {prediction} ({prob:.2f})")

        except Exception as e:
            print(f"Error processing {fname}: {e}")
