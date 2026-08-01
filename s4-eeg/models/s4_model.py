"""
S4 block (S4 layer + channel mixing + residual) and the full sequence model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .s4_layer import S4Layer


class S4Block(nn.Module):
    """S4 layer -> GELU -> pointwise channel mixing -> dropout -> residual + LayerNorm."""

    def __init__(self, H: int, N: int, theta_mode: str = "none", dropout: float = 0.1) -> None:
        super().__init__()
        self.s4 = S4Layer(H, N, theta_mode=theta_mode)
        self.mix = nn.Linear(H, H)
        self.norm = nn.LayerNorm(H)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, L, H) -> (batch, L, H)."""
        y = self.s4(x.transpose(1, 2))       # S4Layer wants (batch, H, L)
        y = self.act(y).transpose(1, 2)      # back to (batch, L, H)
        y = self.drop(self.mix(y))
        return self.norm(x + y)


class S4Model(nn.Module):
    """Input projection -> stack of S4 blocks -> mean pool -> linear readout."""

    def __init__(
        self,
        input_dim: int,
        H: int = 64,
        N: int = 64,
        num_layers: int = 4,
        out_dim: int = 2,
        theta_mode: str = "none",
        dropout: float = 0.1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if pooling not in ("mean", "last", "max"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.pooling = pooling

        self.in_proj = nn.Linear(input_dim, H)
        self.blocks = nn.ModuleList(
            [S4Block(H, N, theta_mode=theta_mode, dropout=dropout) for _ in range(num_layers)]
        )
        self.out_proj = nn.Linear(H, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, L, input_dim) -> logits (batch, out_dim)."""
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)

        if self.pooling == "mean":
            pooled = h.mean(dim=1)
        elif self.pooling == "last":
            pooled = h[:, -1, :]
        else:
            pooled = h.max(dim=1).values

        return self.out_proj(pooled)

    def param_groups(self, lr: float, ssm_mult: float = 0.1, theta_mult: float = 0.1):
        """
        Split parameters into groups so the SSM timescale (log_dt) and the
        Theta generator (G) can use smaller learning rates than the rest.
        """
        theta_p, ssm_p, other_p = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith(".G"):
                theta_p.append(p)
            elif "log_dt" in name:
                ssm_p.append(p)
            else:
                other_p.append(p)

        groups = [
            {"params": other_p, "lr": lr},
            {"params": ssm_p, "lr": lr * ssm_mult},
        ]
        if theta_p:
            groups.append({"params": theta_p, "lr": lr * theta_mult})
        return groups

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
