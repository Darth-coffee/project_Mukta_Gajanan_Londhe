# interface.py

import argparse
from predict import *

def main():
    parser = argparse.ArgumentParser(description="Predict bird syllables from spectrogram images")
    parser.add_argument('--img', type=str, help='Path to a spectrogram image')
    args = parser.parse_args()

    if not args.img or not os.path.exists(args.img):
        print("Please provide a valid image path using --img")
        return

    # Load model
    model = ResNet18(num_classes=1).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    # Load and transform image
    image = Image.open(args.img).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()
        prediction = "syllable" if prob > 0.5 else "noise"

    print(f"{args.img}: {prediction} ({prob:.2f})")

if __name__ == "__main__":
    main()
