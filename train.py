"""
train.py
--------
Train the OT Autoencoder on normal traffic only.

Usage:
    python train.py [--epochs 50] [--batch_size 64] [--lr 1e-3]

Outputs:
    results/scaler.pkl      — fitted StandardScaler (needed at inference)
    results/model.pt        — trained model weights
    results/train_loss.png  — training loss curve
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from model import build_model

# ── Optional matplotlib ───────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

FEATURES = [
    "inter_arrival_ms",
    "payload_bytes",
    "entropy",
    "dst_port",
    "packet_rate_pps",
    "tcp_flags_syn",
    "tcp_flags_rst",
    "unique_dst_ports",
]


# ─────────────────────────────────────────────────────────────────────────────
def load_train_data(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    # Autoencoder trains ONLY on normal samples
    normal = df[df["label"] == 0][FEATURES]
    print(f"[✓] Loaded {len(normal)} normal samples for training")
    return normal.values.astype(np.float32)


def make_loader(X: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    tensor = torch.from_numpy(X)
    dataset = TensorDataset(tensor, tensor)   # input == target for autoencoder
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[✓] Device: {device}")

    # 1. Data
    X_raw = load_train_data(args.data)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    with open("results/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("[✓] Scaler saved → results/scaler.pkl")

    loader = make_loader(X.astype(np.float32), batch_size=args.batch_size)

    # 2. Model
    model = build_model(input_dim=len(FEATURES), latent_dim=args.latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 3. Training loop
    history = []
    print(f"\n{'Epoch':>6}  {'Loss':>12}")
    print("─" * 22)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            recon = model(x_batch)
            loss = criterion(recon, x_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(x_batch)

        epoch_loss /= len(loader.dataset)
        history.append(epoch_loss)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {epoch_loss:>12.6f}")

    # 4. Save model
    torch.save(model.state_dict(), "results/model.pt")
    print("\n[✓] Model saved → results/model.pt")

    # 5. Loss curve
    if HAS_MPL:
        plt.figure(figsize=(8, 4))
        plt.plot(history, color="#2563EB", linewidth=2)
        plt.title("Training Loss (MSE) — OT Autoencoder", fontsize=13)
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/train_loss.png", dpi=150)
        print("[✓] Loss curve → results/train_loss.png")


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train OT Autoencoder")
    p.add_argument("--data",        default="data/train.csv", help="Path to training CSV")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--latent_dim",  type=int,   default=4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
