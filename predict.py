# predict.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import os
from tqdm import tqdm

from model import DeeperCNN
from dataset import SyllableDataset
from config import *

def predict_and_save():
    # Load model
    model = DeeperCNN().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # Load test dataset
    test_dataset = SyllableDataset(root_dir=DATA_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    image_paths = []
    predictions = []

    with torch.no_grad():
        for inputs, paths in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().round().squeeze()

            image_paths.extend(paths)
            predictions.extend(preds.numpy().tolist())

    # Save predictions to CSV
    df = pd.DataFrame({
        "image_path": image_paths,
        "prediction": predictions
    })
    csv_path = os.path.join(CHECKPOINT_DIR, "predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nPredictions saved to {csv_path}")

def return_preds_as_list():
    model = DeeperCNN().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    test_dataset = SyllableDataset(root_dir=DATA_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    preds_list = []

    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Predicting (list only)"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().round().squeeze()

            preds_list.extend(preds.numpy().tolist())

    return preds_list

if __name__ == "__main__":
    predict_and_save()
