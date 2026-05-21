"""
model/autoencoder.py
--------------------
Autoencoder for unsupervised anomaly detection on OT/IoT traffic.

Key idea:
  - Train ONLY on normal traffic.
  - The model learns to compress and reconstruct normal patterns.
  - At inference, anomalous traffic has a HIGH reconstruction error
    because the model never learned those patterns.

Architecture:
  Input(8) → Encoder → Latent(4) → Decoder → Output(8)
  (intentionally shallow and transparent for educational purposes)
"""

import torch
import torch.nn as nn


class OTAutoencoder(nn.Module):
    """
    Shallow symmetric autoencoder.

    Args:
        input_dim  : number of input features (default 8)
        latent_dim : size of the bottleneck (default 4)
        dropout    : dropout probability in encoder/decoder (default 0.1)
    """

    def __init__(self, input_dim: int = 8, latent_dim: int = 4, dropout: float = 0.1):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, latent_dim),
            nn.ReLU(),
        )

        # ── Decoder ──────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation."""
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-sample Mean Squared Error between input and reconstruction.
        Higher → more anomalous.
        """
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=1)


def build_model(input_dim: int = 8, latent_dim: int = 4) -> OTAutoencoder:
    """Factory function — keeps train.py clean."""
    return OTAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
