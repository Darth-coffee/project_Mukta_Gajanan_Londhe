import os
import torch
from PIL import Image
import numpy as np
from model import DeeperCNN  # ✅ Your custom model
from config import CHECKPOINT_PATH, DATA_DIR

# Device and model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeeperCNN().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# Image preprocessing
def image_to_tensor(img):
    img = np.array(img).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return torch.tensor(img).unsqueeze(0)

# Prediction function
def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    tensor = image_to_tensor(image).to(device)
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()
        label = "syllable" if prob > 0.5 else "noise"
    return label, prob

# Main loop for batch predictions in DATA_DIR
def main():
    for fname in os.listdir(DATA_DIR):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(DATA_DIR, fname)
            try:
                label, prob = predict_image(path)
                print(f"{fname}: {label} ({prob:.2f})")
            except Exception as e:
                print(f"Error processing {fname}: {e}")

if __name__ == "__main__":
    main()
