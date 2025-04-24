import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from dataset import SyllableDataset
from model import DeeperCNN
import config  # ← importing config variables

def train_model():
    # === Load config ===
    batch_size = config.batch_size
    epochs = config.epochs
    lr = config.lr
    device = config.device

    train_dir = config.train_dir
    test_pos_dir = config.test_pos_dir
    test_neg_dir = config.test_neg_dir
    model_save_path = config.model_save_path

    # === Datasets and Loaders ===
    train_full = SyllableDataset(train_dir, label=1.0)
    train_size = int(0.8 * len(train_full))
    val_size = len(train_full) - train_size
    train_dataset, val_dataset = random_split(
        train_full, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    test_pos = SyllableDataset(test_pos_dir, label=1.0)
    test_neg = SyllableDataset(test_neg_dir, label=0.0)
    test_dataset = ConcatDataset([test_pos, test_neg])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # === Model Setup ===
    model = DeeperCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, test_losses = [], [], []
    best_val_loss = float('inf')
    best_model_weights = None

    def calculate_f1(model, loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                preds = torch.sigmoid(model(x)).round()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        return f1_score(all_labels, all_preds)

    # === Training Loop ===
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.sigmoid(outputs).round()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train: {avg_train_loss:.4f}, "
              f"Val: {avg_val_loss:.4f}, "
              f"Test: {avg_test_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = model.state_dict()

    torch.save(best_model_weights, model_save_path)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
