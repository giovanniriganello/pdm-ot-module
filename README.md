# 🔒 OT Anomaly Detector

Unsupervised anomaly detection for **OT/IoT network traffic** using a PyTorch Autoencoder.

Designed to detect network-level threats in industrial environments:  
**port scans**, **replay attacks**, and **payload injection** on protocols like Modbus/TCP and DNP3.

---

## 📌 Why an Autoencoder?

In OT/ICS environments, labelled attack data is scarce. An autoencoder is trained **only on normal traffic** — it learns to reconstruct healthy patterns. When anomalous traffic arrives, the reconstruction error spikes, triggering an alert.

```
Normal traffic  →  low reconstruction error  →  ✓ NORMAL
Anomalous traffic →  high reconstruction error → ⚠ ANOMALY
```

---

## 🏗️ Architecture

```
Input (8 features)
      │
 ┌────▼────┐
 │ Linear  │  8 → 16
 │ BN+ReLU │
 │ Dropout │
 │ Linear  │  16 → 4   ← latent bottleneck
 └────┬────┘
      │  (encode)
 ┌────▼────┐
 │ Linear  │  4 → 16
 │ BN+ReLU │
 │ Dropout │
 │ Linear  │  16 → 8   ← reconstruction
 └────┬────┘
      │
Output (8 features)
```

Loss function: **MSE** between input and reconstruction.

---

## 📊 Features Used

| Feature | Description |
|---|---|
| `inter_arrival_ms` | Time between packets (ms) |
| `payload_bytes` | Payload size in bytes |
| `entropy` | Shannon entropy of payload bytes |
| `dst_port` | Destination port |
| `packet_rate_pps` | Packets per second in the flow |
| `tcp_flags_syn` | SYN flag present (0/1) |
| `tcp_flags_rst` | RST flag present (0/1) |
| `unique_dst_ports` | Distinct destination ports in flow |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/<your-username>/ot-anomaly-detector.git
cd ot-anomaly-detector
pip install -r requirements.txt
```

### 2. Generate the synthetic dataset

```bash
python data/generate_dataset.py
# → data/train.csv  (5 000 normal samples)
# → data/test.csv   (1 000 normal + 500 anomalies)
```

### 3. Train the model

```bash
python train.py --epochs 50 --batch_size 64 --lr 1e-3
# → results/model.pt
# → results/scaler.pkl
# → results/train_loss.png
```

### 4. Evaluate

```bash
python evaluate.py
# → AUROC, F1, Precision, Recall
# → results/evaluation.png
```

### 5. Run inference

```bash
# Single sample — normal Modbus polling
python detect.py --sample "1000,8,0.8,502,1.0,0,0,1"

# Single sample — port scan anomaly
python detect.py --sample "5,60,4.2,31337,120,1,1,300"

# Entire CSV
python detect.py --file data/test.csv --output results/predictions.csv
```

---

## 📁 Project Structure

```
ot-anomaly-detector/
│
├── data/
│   └── generate_dataset.py   # Synthetic traffic generator
│
├── model/
│   ├── __init__.py
│   └── autoencoder.py        # OTAutoencoder (PyTorch)
│
├── train.py                  # Training loop + scaler fitting
├── evaluate.py               # Metrics, ROC, confusion matrix
├── detect.py                 # CLI inference tool
├── requirements.txt
└── README.md
```

---

## 📈 Expected Results

With default hyperparameters on the synthetic dataset:

| Metric | Value |
|---|---|
| AUROC | ~0.97 |
| F1 Score | ~0.93 |
| Precision | ~0.91 |
| Recall | ~0.95 |

> Results will vary slightly due to random seed in dataset generation.

---

## 🧪 Anomaly Types Simulated

| Attack | Signature in features |
|---|---|
| **Port Scan** | Very low `inter_arrival_ms`, high `packet_rate_pps`, high `unique_dst_ports`, `tcp_flags_syn=1` |
| **Replay Attack** | Normal timing/size BUT near-zero `entropy` (copied packets) |
| **Payload Injection** | Large `payload_bytes` (200–1500), very high `entropy` |

---

## 🔧 Customisation

**Use your own traffic data:**  
Replace `data/train.csv` / `data/test.csv` with real captures. Ensure your CSV has the 8 feature columns listed above plus a `label` column (0=normal, 1=anomaly). Then re-run `train.py`.

**Adjust the threshold:**  
`evaluate.py` auto-selects the threshold that maximises F1. Pass it manually to `detect.py`:
```bash
python detect.py --file data/test.csv --threshold 0.08
```

**Deeper architecture:**  
Edit `model/autoencoder.py` — change `latent_dim` (default 4) or add more layers.

---

## 📚 Background Reading

- [Anomaly Detection in Industrial Control Systems](https://www.cisa.gov/ics)
- [Autoencoder-based anomaly detection survey](https://arxiv.org/abs/2101.03938)
- [Modbus Protocol Reference](https://modbus.org/specs.php)

---

## 📝 License

MIT — free to use, modify, and redistribute.

---

## 🤝 Contributing

Pull requests welcome! Ideas for improvement:
- Add LSTM autoencoder for temporal sequences
- Integrate with real PCAP capture via `scapy`
- Add a FastAPI inference endpoint
- Evaluate on public ICS datasets (e.g. [BATADAL](https://www.batadal.net/), [SWaT](https://itrust.sutd.edu.sg/testbeds/secure-water-treatment-swat/))
