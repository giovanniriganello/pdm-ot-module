"""
detect.py
---------
Run the trained model on new network traffic samples.

Usage examples:
    # Classify a single sample (normal-looking)
    python detect.py --sample "1000,8,0.8,502,1.0,0,0,1"

    # Classify a sample (port-scan-like anomaly)
    python detect.py --sample "5,60,4.2,31337,120,1,1,300"

    # Run against a full CSV file
    python detect.py --file data/test.csv --output results/predictions.csv
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

from model import build_model

FEATURES = [
    "inter_arrival_ms", "payload_bytes", "entropy", "dst_port",
    "packet_rate_pps", "tcp_flags_syn", "tcp_flags_rst", "unique_dst_ports",
]

FEATURE_DESCRIPTIONS = [
    "inter_arrival_ms  : Time between packets (ms)",
    "payload_bytes     : Payload size (bytes)",
    "entropy           : Payload byte entropy (0-8)",
    "dst_port          : Destination port",
    "packet_rate_pps   : Packets per second",
    "tcp_flags_syn     : SYN flag (0/1)",
    "tcp_flags_rst     : RST flag (0/1)",
    "unique_dst_ports  : Distinct dst ports in flow",
]

DEFAULT_THRESHOLD = 0.05   # override with --threshold


# ─────────────────────────────────────────────────────────────────────────────
def load_artifacts(model_path: str, scaler_path: str):
    if not os.path.exists(model_path):
        sys.exit(f"[✗] Model not found: {model_path}\n    Run `python train.py` first.")
    if not os.path.exists(scaler_path):
        sys.exit(f"[✗] Scaler not found: {scaler_path}\n    Run `python train.py` first.")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model = build_model(input_dim=len(FEATURES), latent_dim=4)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, scaler


def predict_scores(model, scaler, X_raw: np.ndarray) -> np.ndarray:
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    tensor = torch.from_numpy(X_scaled)
    with torch.no_grad():
        return model.reconstruction_error(tensor).numpy()


# ─────────────────────────────────────────────────────────────────────────────
def run_single(args: argparse.Namespace) -> None:
    model, scaler = load_artifacts(args.model, args.scaler)

    try:
        values = [float(v.strip()) for v in args.sample.split(",")]
    except ValueError:
        sys.exit("[✗] --sample must be 8 comma-separated floats")

    if len(values) != len(FEATURES):
        sys.exit(f"[✗] Expected {len(FEATURES)} values, got {len(values)}")

    X = np.array([values])
    score = predict_scores(model, scaler, X)[0]
    label = "⚠ ANOMALY" if score >= args.threshold else "✓ NORMAL"

    print("\n" + "─" * 42)
    print(f"  {'Feature':<26}  {'Value':>10}")
    print("─" * 42)
    for name, val in zip(FEATURE_DESCRIPTIONS, values):
        tag = name.split(":")[0].strip()
        print(f"  {tag:<26}  {val:>10.2f}")
    print("─" * 42)
    print(f"  Reconstruction error  : {score:.6f}")
    print(f"  Threshold             : {args.threshold:.6f}")
    print(f"  Decision              : {label}")
    print("─" * 42 + "\n")


def run_file(args: argparse.Namespace) -> None:
    model, scaler = load_artifacts(args.model, args.scaler)
    df = pd.read_csv(args.file)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        sys.exit(f"[✗] Missing columns in CSV: {missing}")

    scores = predict_scores(model, scaler, df[FEATURES].values)
    df["anomaly_score"] = scores
    df["prediction"]    = (scores >= args.threshold).astype(int)

    n_anomalies = df["prediction"].sum()
    print(f"\n[✓] Processed {len(df)} samples → {n_anomalies} anomalies detected "
          f"({n_anomalies/len(df)*100:.1f}%)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[✓] Predictions saved → {args.output}\n")


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OT Anomaly Detector — Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal Modbus polling
  python detect.py --sample "1000,8,0.8,502,1.0,0,0,1"

  # Port scan anomaly
  python detect.py --sample "5,60,4.2,31337,120,1,1,300"

  # Full CSV
  python detect.py --file data/test.csv --output results/predictions.csv
        """
    )
    p.add_argument("--sample",    type=str, default=None,
                   help="Comma-separated feature values (8 fields)")
    p.add_argument("--file",      type=str, default=None,
                   help="Path to CSV file with traffic features")
    p.add_argument("--output",    type=str, default="results/predictions.csv")
    p.add_argument("--model",     type=str, default="results/model.pt")
    p.add_argument("--scaler",    type=str, default="results/scaler.pkl")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help=f"Anomaly threshold (default: {DEFAULT_THRESHOLD})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.sample:
        run_single(args)
    elif args.file:
        run_file(args)
    else:
        print("[!] Provide --sample or --file. Use --help for usage.")
