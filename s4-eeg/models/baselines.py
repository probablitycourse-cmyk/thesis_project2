"""
Baseline sequence models used to benchmark the S4 variants on the same task.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """Stacked LSTM, classified from the final hidden state of the top layer."""

    def __init__(
        self,
        input_dim: int,
        H: int = 64,
        num_layers: int = 2,
        out_dim: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            H,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.out_proj = nn.Linear(H * (2 if bidirectional else 1), out_dim)
        self.bidirectional = bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            last = h_n[-1]
        return self.out_proj(last)

    def param_groups(self, lr: float, **_):
        return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": lr}]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerModel(nn.Module):
    """Transformer encoder with learned positional embeddings and mean pooling."""

    def __init__(
        self,
        input_dim: int,
        H: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        out_dim: int = 2,
        dropout: float = 0.1,
        max_len: int = 2048,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(input_dim, H)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, H) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=H,
            nhead=num_heads,
            dim_feedforward=H * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(H, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        h = self.in_proj(x) + self.pos_embed[:, :L, :]
        h = self.encoder(h)
        return self.out_proj(h.mean(dim=1))

    def param_groups(self, lr: float, **_):
        return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": lr}]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPModel(nn.Module):
    """
    Flatten-then-MLP baseline.

    The whole window is flattened into a single vector (L * n_channels) and
    fed to a plain feed-forward network. This model has no notion of time
    order at all, so it cannot exploit temporal structure -- which makes it a
    useful leakage probe. If a shuffled-window split lets an MLP reach an
    accuracy far above chance, the split itself is putting near-duplicate
    windows on both sides rather than the model learning anything temporal.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        H: int = 256,
        num_layers: int = 2,
        out_dim: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        flat_dim = seq_len * input_dim

        layers: list[nn.Module] = []
        prev = flat_dim
        for _ in range(num_layers):
            layers += [nn.Linear(prev, H), nn.GELU(), nn.Dropout(dropout)]
            prev = H
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, L, input_dim) -> logits (batch, out_dim)."""
        return self.net(x.reshape(x.size(0), -1))

    def param_groups(self, lr: float, **_):
        return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": lr}]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
