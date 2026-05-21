"""
generate_dataset.py
-------------------
Generates a synthetic OT/IoT network traffic dataset inspired by
industrial protocols (Modbus, DNP3, IEC 60870).

Normal traffic: periodic polling, small payloads, low entropy.
Anomalous traffic: port scans, replay attacks, unusual payload sizes.
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def generate_normal_traffic(n_samples: int = 5000) -> pd.DataFrame:
    """
    Simulate normal OT polling traffic:
    - Regular intervals (low jitter)
    - Small, fixed-size payloads (Modbus: 6-12 bytes)
    - Low entropy (structured data)
    - Destination ports: 502 (Modbus), 20000 (DNP3)
    - Low packet rate
    """
    records = []
    for _ in range(n_samples):
        record = {
            "inter_arrival_ms": np.random.normal(loc=1000, scale=30),       # ~1s polling
            "payload_bytes":    np.random.randint(6, 13),                    # small Modbus frame
            "entropy":          np.random.uniform(0.1, 1.5),                # structured, low entropy
            "dst_port":         np.random.choice([502, 20000], p=[0.7, 0.3]),
            "packet_rate_pps":  np.random.normal(loc=1.0, scale=0.1),       # 1 packet/sec
            "tcp_flags_syn":    np.random.choice([0, 1], p=[0.97, 0.03]),
            "tcp_flags_rst":    np.random.choice([0, 1], p=[0.99, 0.01]),
            "unique_dst_ports": np.random.randint(1, 3),
            "label": 0,  # normal
        }
        records.append(record)
    return pd.DataFrame(records)


def generate_anomalous_traffic(n_samples: int = 500) -> pd.DataFrame:
    """
    Simulate three attack/anomaly patterns:
    1. Port scan         – many unique dst ports, high SYN rate
    2. Replay attack     – repeated identical payloads, abnormal timing
    3. Payload injection – oversized payload, high entropy
    """
    records = []
    attack_types = ["port_scan", "replay", "injection"]

    for _ in range(n_samples):
        attack = np.random.choice(attack_types)

        if attack == "port_scan":
            record = {
                "inter_arrival_ms": np.random.normal(loc=10, scale=5),      # very fast
                "payload_bytes":    np.random.randint(40, 80),
                "entropy":          np.random.uniform(3.0, 5.0),
                "dst_port":         np.random.randint(1, 65535),
                "packet_rate_pps":  np.random.normal(loc=100, scale=20),
                "tcp_flags_syn":    1,
                "tcp_flags_rst":    np.random.choice([0, 1], p=[0.5, 0.5]),
                "unique_dst_ports": np.random.randint(50, 500),
                "label": 1,
            }
        elif attack == "replay":
            record = {
                "inter_arrival_ms": np.random.normal(loc=1000, scale=5),    # same timing
                "payload_bytes":    np.random.randint(6, 13),               # same size → suspicious
                "entropy":          np.random.uniform(0.0, 0.3),            # very low (copy-paste)
                "dst_port":         502,
                "packet_rate_pps":  np.random.normal(loc=1.0, scale=0.05),
                "tcp_flags_syn":    0,
                "tcp_flags_rst":    0,
                "unique_dst_ports": 1,
                "label": 1,
            }
        else:  # injection
            record = {
                "inter_arrival_ms": np.random.normal(loc=1000, scale=200),
                "payload_bytes":    np.random.randint(200, 1500),           # huge payload
                "entropy":          np.random.uniform(5.0, 8.0),            # encrypted/random
                "dst_port":         502,
                "packet_rate_pps":  np.random.normal(loc=2.0, scale=0.5),
                "tcp_flags_syn":    0,
                "tcp_flags_rst":    0,
                "unique_dst_ports": np.random.randint(1, 4),
                "label": 1,
            }
        records.append(record)

    return pd.DataFrame(records)


def build_and_save(output_dir: str = "data") -> None:
    os.makedirs(output_dir, exist_ok=True)

    normal    = generate_normal_traffic(5000)
    anomalous = generate_anomalous_traffic(500)

    df = pd.concat([normal, anomalous], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)
    df = df.clip(lower=0)   # no negative values

    # Split: 80% train (normal only for autoencoder), 20% test (mixed)
    normal_df = df[df["label"] == 0]
    anomaly_df = df[df["label"] == 1]

    train_df = normal_df.sample(frac=0.8, random_state=RANDOM_SEED)
    test_normal = normal_df.drop(train_df.index)
    test_df = pd.concat([test_normal, anomaly_df]).sample(frac=1, random_state=RANDOM_SEED)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"),  index=False)

    print(f"[✓] Train samples : {len(train_df)}  (all normal)")
    print(f"[✓] Test  samples : {len(test_df)}   ({len(anomaly_df)} anomalies)")
    print(f"[✓] Saved to '{output_dir}/'")


if __name__ == "__main__":
    build_and_save()
