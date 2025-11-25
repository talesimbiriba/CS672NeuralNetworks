"""
Comparison of small CNN, RNN, and Transformer on the UCI HAR Dataset
--------------------------------------------------------------------

- Uses raw Inertial Signals (real sensor 1D time series).
- Models:
    * CNN1DClassifier
    * RNNClassifier (LSTM)
    * TransformerClassifier
- Includes:
    * Device selection (MPS -> CUDA -> CPU)
    * Training loop with metrics
    * Visualization of sample sequences
    * Training curves
    * Confusion matrix plots

Requirements:
    pip install torch matplotlib numpy scikit-learn
"""

import os
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


# ---------------------------------------------------------------------
# Device selection (MPS for Mac, then CUDA, then CPU)
# ---------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU via MPS")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


device = get_device()


# ---------------------------------------------------------------------
# Download and load UCI HAR Dataset (Inertial Signals)
# ---------------------------------------------------------------------
def download_har(data_dir="data"):
    """
    Download and extract the UCI HAR Dataset if not already present.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "UCI_HAR_Dataset.zip"
    extract_path = data_dir / "UCI HAR Dataset"

    if extract_path.exists():
        print("UCI HAR Dataset already downloaded.")
        return extract_path

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    print(f"Downloading UCI HAR Dataset from {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete. Extracting...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    print("Extraction complete.")
    return extract_path


def load_har_inertial_signals(root_dir):
    """
    Load inertial signals from UCI HAR Dataset.
    We will use 9 channels:
        body_acc_(x,y,z), body_gyro_(x,y,z), total_acc_(x,y,z)
    Shapes:
        X_*: (num_samples, seq_len, num_channels)
        y_*: (num_samples,)  with labels 0..5
    """
    root_dir = Path(root_dir)
    base = root_dir

    signal_types = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]

    def load_split(split):
        signals = []
        inertial_path = base / split / "Inertial Signals"
        for sig in signal_types:
            fname = inertial_path / f"{sig}_{split}.txt" 
            # Each file: (num_samples, seq_len)
            data = np.loadtxt(fname)
            signals.append(data[:, :, None])  # add channel dim, generate a list of (num_samples, seq_len, 1) arrays.

        # Stack along channel dimension: (num_samples, seq_len, num_channels)
        X = np.concatenate(signals, axis=2).astype(np.float32) # final shape = (num_samples, seq_len, 9), with 9 channels.

        labels_path = base / split / f"y_{split}.txt"
        y = np.loadtxt(labels_path).astype(np.int64) - 1  # labels 1..6 -> 0..5

        return X, y

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    print("Train shape:", X_train.shape, "Labels:", np.bincount(y_train))
    print("Test  shape:", X_test.shape, "Labels:", np.bincount(y_test))

    return X_train, y_train, X_test, y_test


class HARSeqDataset(Dataset):
    """
    PyTorch Dataset for HAR inertial time series.
    Inputs: (seq_len, num_channels)
    """

    def __init__(self, X, y):
        self.X = torch.from_numpy(X)  # (N, T, C)
        self.y = torch.from_numpy(y)  # (N,)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class CNN1DClassifier(nn.Module):
    """
    Simple 1D CNN for sequence classification.
    Input: (B, T, C)  -> we internally transpose to (B, C, T) for Conv1d.
    """

    def __init__(self, num_channels, num_classes, hidden_channels=64):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, hidden_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pooling
        self.fc = nn.Linear(hidden_channels * 2, num_classes)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # (B, hidden*2)
        logits = self.fc(x)
        return logits


class RNNClassifier(nn.Module):
    """
    LSTM-based sequence classifier.
    Input: (B, T, C) with batch_first=True.
    """

    def __init__(self, num_channels, num_classes, hidden_size=64, num_layers=1, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * num_directions, num_classes)

    def forward(self, x):
        # x: (B, T, C)
        out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state from all directions
        if self.lstm.bidirectional:
            # h_n: (num_layers*2, B, H) -> concatenate final forward/backward
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, 2H)
        else:
            h_last = h_n[-1]  # (B, H)

        logits = self.fc(h_last)
        return logits


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.
    """

    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)) 

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, T, D)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        T = x.size(1)
        return x + self.pe[:, :T]


class TransformerClassifier(nn.Module):
    """
    Simple Transformer encoder for sequence classification.
    Input: (B, T, C)  -> project to d_model -> TransformerEncoder -> pool -> FC.
    """

    def __init__(self, num_channels, num_classes, d_model=64, num_heads=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(num_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, T, D)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, T, C)
        x = self.input_proj(x)  # (B, T, D)
        x = self.pos_encoder(x)
        x = self.encoder(x)  # (B, T, D)

        # Simple pooling: mean over time
        x = x.mean(dim=1)  # (B, D)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


# ---------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------
def accuracy_from_logits(logits, targets):
    #Compute accuracy given model logits (outputs before softmax) and true targets.
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3, model_name="model"):
    """
    Generic training loop for a classifier model.
    Returns history dict with train/val loss and accuracy.
    """
    model = model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_acc = accuracy_from_logits(logits, y_batch)
            running_loss += loss.item()
            running_acc += batch_acc
            n_batches += 1

        train_loss = running_loss / n_batches
        train_acc = running_acc / n_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                logits = model(X_val)
                loss = criterion(logits, y_val)
                batch_acc = accuracy_from_logits(logits, y_val)

                val_loss += loss.item()
                val_acc += batch_acc
                n_val_batches += 1

        val_loss /= n_val_batches
        val_acc /= n_val_batches

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"[{model_name}] Epoch {epoch}/{epochs} "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.3f}"
        )

    return history


# ---------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------
ACTIVITY_LABELS = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
}


def plot_sample_sequences(X, y, num_samples_per_class=1, savepath=None):
    """
    Plot a few example sequences (e.g., one per activity) with 3 selected channels.
    """
    num_classes = len(ACTIVITY_LABELS)
    num_plots = num_classes * num_samples_per_class

    # Use 3 channels for visualization: body_acc_x, body_gyro_x, total_acc_x
    channel_indices = [0, 3, 6]  # x-axis of the three sensor types
    channel_names = ["body_acc_x", "body_gyro_x", "total_acc_x"]

    plt.figure(figsize=(12, 2 * num_plots))

    plot_idx = 1
    for cls in range(num_classes):
        cls_indices = np.where(y == cls)[0][:num_samples_per_class]
        for idx in cls_indices:
            seq = X[idx]  # (T, C)
            t = np.arange(seq.shape[0])

            ax = plt.subplot(num_plots, 1, plot_idx)
            for ch_i, ch_name in zip(channel_indices, channel_names):
                ax.plot(t, seq[:, ch_i], label=ch_name, alpha=0.8)

            ax.set_title(f"Class {cls}: {ACTIVITY_LABELS[cls]} (sample {idx})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Sensor value")
            if plot_idx == 1:
                ax.legend(loc="upper right")
            plot_idx += 1

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()


def plot_training_curves(histories, model_names, savepath=None):
    """
    Plot training and validation loss/accuracy curves for several models.
    histories: list of history dicts
    model_names: list of strings
    """
    # different markers for different models
    markers = ["o", "s", "D", "^", "v", "<", ">"]

    plt.figure(figsize=(12, 5))
    # Loss
    plt.subplot(1, 2, 1)
    for hist, name in zip(histories, model_names):
        # plt.plot(hist["train_loss"], "--", label=f"{name} train")
        # plt.plot(hist["val_loss"], "-", label=f"{name} val")
        plt.plot(hist["train_loss"], linestyle="--", marker=markers[model_names.index(name) % len(markers)], markevery=2, label=f"{name} train")
        plt.plot(hist["val_loss"], linestyle="-", marker=markers[model_names.index(name) % len(markers)], markevery=2, label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    for hist, name in zip(histories, model_names):
        # plt.plot(hist["train_acc"], "--", label=f"{name} train")
        # plt.plot(hist["val_acc"], "-", label=f"{name} val")
        plt.plot(hist["train_acc"], linestyle="--", marker=markers[model_names.index(name) % len(markers)], markevery=2, label=f"{name} train")
        plt.plot(hist["val_acc"], linestyle="-", marker=markers[model_names.index(name) % len(markers)], markevery=2, label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()


def plot_confusion_matrix_for_model(model, data_loader, device, title="Confusion Matrix", savepath=None):
    """
    Compute and plot confusion matrix on a given dataset.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    cm = confusion_matrix(all_targets, all_preds)
    print(title)
    print(classification_report(all_targets, all_preds, target_names=list(ACTIVITY_LABELS.values())))

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(ACTIVITY_LABELS))
    plt.xticks(tick_marks, ACTIVITY_LABELS.values(), rotation=45, ha="right")
    plt.yticks(tick_marks, ACTIVITY_LABELS.values())

    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Download & load data
    data_root = download_har("data")
    X_train, y_train, X_test, y_test = load_har_inertial_signals(data_root)

    # Visualize a few sequences
    plot_sample_sequences(X_train, y_train, num_samples_per_class=1)

    # 2) Build datasets & loaders (with train/val split)
    full_train_dataset = HARSeqDataset(X_train, y_train)
    test_dataset = HARSeqDataset(X_test, y_test)

    # Train/val split (e.g., 80/20 on training set)
    val_ratio = 0.2
    n_train = int(len(full_train_dataset) * (1 - val_ratio))
    n_val = len(full_train_dataset) - n_train
    train_dataset, val_dataset = random_split(full_train_dataset, [n_train, n_val])

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    num_channels = X_train.shape[2]
    num_classes = len(ACTIVITY_LABELS)

    # 3) Initialize models
    cnn_model = CNN1DClassifier(num_channels=num_channels, num_classes=num_classes)
    rnn_model = RNNClassifier(num_channels=num_channels, num_classes=num_classes)
    transformer_model = TransformerClassifier(num_channels=num_channels, num_classes=num_classes)

    # 4) Train models (you can reduce epochs for a quick run)
    epochs = 20
    histories = []
    names = []

    print("\nTraining CNN...")
    hist_cnn = train_model(cnn_model, train_loader, val_loader, device, epochs=epochs, lr=1e-3, model_name="CNN")
    histories.append(hist_cnn)
    names.append("CNN")

    print("\nTraining RNN (LSTM)...")
    hist_rnn = train_model(rnn_model, train_loader, val_loader, device, epochs=epochs, lr=1e-3, model_name="RNN")
    histories.append(hist_rnn)
    names.append("RNN")

    print("\nTraining Transformer...")
    hist_trf = train_model(transformer_model, train_loader, val_loader, device, epochs=epochs, lr=1e-3, model_name="Transformer")
    histories.append(hist_trf)
    names.append("Transformer")

    # 5) Plot training curves
    plot_training_curves(histories, names)

    # 6) Confusion matrices on test set
    print("\nCNN on test set:")
    plot_confusion_matrix_for_model(cnn_model.to(device), test_loader, device, title="CNN Confusion Matrix")

    print("\nRNN on test set:")
    plot_confusion_matrix_for_model(rnn_model.to(device), test_loader, device, title="RNN Confusion Matrix")

    print("\nTransformer on test set:")
    plot_confusion_matrix_for_model(transformer_model.to(device), test_loader, device, title="Transformer Confusion Matrix")
