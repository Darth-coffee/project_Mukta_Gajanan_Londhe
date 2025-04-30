# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os

from LV_dataset import SyllableDataset
from model import DeeperCNN
from config import *

def train_model():
    # === Dataset setup ===
    full_dataset = SyllableDataset(root_dir=DATASET_DIR)

    # Create train-val split
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Balance classes in training using WeightedRandomSampler
    labels = [label for _, label in train_dataset]
    class_counts = np.bincount(np.array(labels, dtype=int))
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[int(label)] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # === Model setup ===
    model = DeeperCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # === Training helpers ===
    best_val_acc = 0.0
    patience_counter = 0

    train_acc_list, val_acc_list = [], []
    train_f1_list, val_f1_list = [], []

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # === Training Loop ===
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # Training
        model.train()
        running_preds, running_labels = [], []
        running_loss = 0.0

        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.sigmoid(outputs).detach().cpu().round()
            running_preds.extend(preds.numpy())
            running_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(running_labels, running_preds)
        train_f1 = f1_score(running_labels, running_preds)
        train_acc_list.append(train_acc)
        train_f1_list.append(train_f1)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0

        val_pbar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()

                preds = torch.sigmoid(outputs).cpu().round()
                val_preds.extend(preds.numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        val_acc_list.append(val_acc)
        val_f1_list.append(val_f1)

        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        # === Early Stopping ===
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(">> Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= 4:
                print("\nEarly stopping triggered.")
                break

    # === Plot training curves ===
    epochs_range = range(1, len(train_acc_list) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy plot
    axs[0].plot(epochs_range, train_acc_list, label="Train Acc")
    axs[0].plot(epochs_range, val_acc_list, label="Val Acc")
    axs[0].set_title("Accuracy over Epochs")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()
    axs[0].grid(True)

    # F1-score plot
    axs[1].plot(epochs_range, train_f1_list, label="Train F1")
    axs[1].plot(epochs_range, val_f1_list, label="Val F1")
    axs[1].set_title("F1 Score over Epochs")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("F1 Score")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "training_curves.png"))
    plt.show()

    # === Confusion Matrix ===
    final_cm = confusion_matrix(val_labels, val_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=final_cm, display_labels=["Non-syllable", "Syllable"])
    disp.plot(cmap='Blues')
    plt.title("Validation Confusion Matrix")
    plt.savefig(os.path.join(CHECKPOINT_DIR, "val_confusion_matrix.png"))
    plt.show()

if __name__ == "__main__":
    train_model()
