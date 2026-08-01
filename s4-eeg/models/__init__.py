from .hippo import hippo_legs_matrices, nplr_decompose, verify_nplr
from .s4_layer import S4Layer, THETA_MODES
from .s4_model import S4Block, S4Model
from .baselines import LSTMModel, TransformerModel

__all__ = [
    "hippo_legs_matrices",
    "nplr_decompose",
    "verify_nplr",
    "S4Layer",
    "THETA_MODES",
    "S4Block",
    "S4Model",
    "LSTMModel",
    "TransformerModel",
    "build_model",
]


def build_model(cfg, input_dim: int, out_dim: int):
    """Factory that maps cfg.MODEL to a concrete model instance."""
    name = cfg.MODEL.lower()

    if name == "s4":
        return S4Model(
            input_dim=input_dim,
            H=cfg.H,
            N=cfg.N,
            num_layers=cfg.NUM_LAYERS,
            out_dim=out_dim,
            theta_mode=cfg.THETA_MODE,
            dropout=cfg.DROPOUT,
            pooling=cfg.POOLING,
        )
    if name == "lstm":
        return LSTMModel(
            input_dim=input_dim,
            H=cfg.H,
            num_layers=cfg.NUM_LAYERS,
            out_dim=out_dim,
            dropout=cfg.DROPOUT,
        )
    if name == "transformer":
        return TransformerModel(
            input_dim=input_dim,
            H=cfg.H,
            num_layers=cfg.NUM_LAYERS,
            num_heads=cfg.NUM_HEADS,
            out_dim=out_dim,
            dropout=cfg.DROPOUT,
        )
    raise ValueError(f"unknown MODEL {cfg.MODEL!r} (expected s4 | lstm | transformer)")
