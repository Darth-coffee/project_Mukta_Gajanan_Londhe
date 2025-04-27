import os
import torch
from PIL import Image
import numpy as np
from model import DeeperCNN  # Your custom model
from config import CHECKPOINT_PATH, TEST_POSITIVE_DIR, TEST_NEGATIVE_DIR  # Import paths from config

# Device and model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeeperCNN().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# Image preprocessing
def image_to_tensor(img):
    img = np.array(img).astype(np.float32) / 255.0  # Normalize image to [0, 1]
    img = img.transpose(2, 0, 1)  # Convert to channel-first format
    return torch.tensor(img).unsqueeze(0)  # Add batch dimension

# Prediction function
def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")  # Open image and ensure it's RGB
    tensor = image_to_tensor(image).to(device)  # Convert image to tensor and move to device
    with torch.no_grad():
        output = model(tensor)  # Get model output
        prob = torch.sigmoid(output).item()  # Apply sigmoid to get probability
        label = "syllable" if prob > 0.5 else "noise"  # Threshold probability for class
    return label, prob

# Main loop for batch predictions in TEST_POSITIVE_DIR and TEST_NEGATIVE_DIR
def main():
    for dir_path in [TEST_POSITIVE_DIR, TEST_NEGATIVE_DIR]:
        print(f"Processing images in {dir_path}")
        for fname in os.listdir(dir_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):  # Check image extensions
                path = os.path.join(dir_path, fname)  # Get full path of the image
                try:
                    label, prob = predict_image(path)  # Get prediction
                    print(f"{fname}: {label} ({prob:.2f})")  # Print prediction
                except Exception as e:
                    print(f"Error processing {fname}: {e}")  # Handle errors gracefully

if __name__ == "__main__":
    main()
