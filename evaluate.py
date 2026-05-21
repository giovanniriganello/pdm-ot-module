"""
evaluate.py
-----------
Evaluate the trained autoencoder on the test set.

Usage:
    python evaluate.py [--threshold auto|<float>]

Outputs:
    results/roc_curve.png
    results/reconstruction_error_dist.png
    results/confusion_matrix.png
    Console: AUROC, F1, precision, recall, best threshold
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from model import build_model

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[!] matplotlib not found — skipping plots")

FEATURES = [
    "inter_arrival_ms", "payload_bytes", "entropy", "dst_port",
    "packet_rate_pps", "tcp_flags_syn", "tcp_flags_rst", "unique_dst_ports",
]


# ─────────────────────────────────────────────────────────────────────────────
def load_artifacts(model_path: str, scaler_path: str):
    with open(scaler_path, "rb") as f:
        scaler: StandardScaler = pickle.load(f)

    model = build_model(input_dim=len(FEATURES), latent_dim=4)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, scaler


def compute_errors(model, X_scaled: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(X_scaled.astype(np.float32))
    with torch.no_grad():
        errors = model.reconstruction_error(tensor).numpy()
    return errors


# ─────────────────────────────────────────────────────────────────────────────
def evaluate(args: argparse.Namespace) -> None:
    os.makedirs("results", exist_ok=True)

    # Load
    model, scaler = load_artifacts(args.model, args.scaler)
    df = pd.read_csv(args.data)
    X = scaler.transform(df[FEATURES].values)
    y_true = df["label"].values

    # Reconstruction error = anomaly score
    errors = compute_errors(model, X)

    # AUROC
    auroc = roc_auc_score(y_true, errors)
    fpr, tpr, thresholds = roc_curve(y_true, errors)

    # Best threshold (maximise F1)
    if args.threshold == "auto":
        f1_scores = [
            f1_score(y_true, (errors >= t).astype(int), zero_division=0)
            for t in thresholds
        ]
        best_idx = int(np.argmax(f1_scores))
        threshold = thresholds[best_idx]
    else:
        threshold = float(args.threshold)

    y_pred = (errors >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred)

    # ── Console report ────────────────────────────────────────────────────────
    print("\n" + "═" * 40)
    print("  OT Anomaly Detector — Evaluation")
    print("═" * 40)
    print(f"  AUROC      : {auroc:.4f}")
    print(f"  Threshold  : {threshold:.6f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print("═" * 40 + "\n")

    if not HAS_MPL:
        return

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    # 1. ROC Curve
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(fpr, tpr, color="#2563EB", lw=2, label=f"AUROC = {auroc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Reconstruction Error Distribution
    ax2 = fig.add_subplot(gs[1])
    err_normal  = errors[y_true == 0]
    err_anomaly = errors[y_true == 1]
    bins = np.linspace(0, np.percentile(errors, 99), 60)
    ax2.hist(err_normal,  bins=bins, alpha=0.6, color="#22C55E", label="Normal")
    ax2.hist(err_anomaly, bins=bins, alpha=0.6, color="#EF4444", label="Anomaly")
    ax2.axvline(threshold, color="black", lw=2, linestyle="--", label=f"Threshold={threshold:.4f}")
    ax2.set_xlabel("Reconstruction Error (MSE)")
    ax2.set_ylabel("Count")
    ax2.set_title("Reconstruction Error Distribution")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Confusion Matrix
    ax3 = fig.add_subplot(gs[2])
    im = ax3.imshow(cm, interpolation="nearest", cmap="Blues")
    ax3.set_title("Confusion Matrix")
    ax3.set_xlabel("Predicted label")
    ax3.set_ylabel("True label")
    ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Normal", "Anomaly"])
    ax3.set_yticklabels(["Normal", "Anomaly"])
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.colorbar(im, ax=ax3)

    plt.suptitle("OT Anomaly Detector — Evaluation Results", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/evaluation.png", dpi=150)
    print("[✓] Evaluation plots → results/evaluation.png")


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate OT Autoencoder")
    p.add_argument("--data",      default="data/test.csv")
    p.add_argument("--model",     default="results/model.pt")
    p.add_argument("--scaler",    default="results/scaler.pkl")
    p.add_argument("--threshold", default="auto",
                   help="'auto' finds best F1 threshold, or pass a float")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
