"""
Entry point. Run a single experiment defined by config.CFG.

Command line:
    python main.py --model s4 --theta order --epochs 80 --target valence

From Colab:
    from config import CFG
    from main import run
    CFG.THETA_MODE = "order"
    hist, best = run(CFG)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from config import CFG
from data import make_loaders
from models import build_model
from training import train, plot_history


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(cfg=CFG, show_plot: bool = True):
    """Build data + model, train, save history, optionally plot."""
    cfg.show()
    set_seed(cfg.SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        print("WARNING: running on CPU will be very slow -- enable a GPU runtime")

    train_loader, test_loader, info = make_loaders(cfg)

    model = build_model(cfg, input_dim=info["n_channels"], out_dim=info["n_classes"])
    hist, best = train(model, train_loader, test_loader, cfg, device,
                       majority=info["majority_baseline"])

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.run_name()}.json")
    hist.save(json_path, extra={
        "run_name": cfg.run_name(),
        "config": {k: str(v) for k, v in cfg.as_dict().items()},
        "best_test_acc": best,
        "majority": info["majority_baseline"],
        "data_info": info,
    })
    print(f"history saved: {json_path}")

    if show_plot:
        plot_history(
            hist,
            majority=info["majority_baseline"],
            title=f"{cfg.run_name()}  (params={model.num_params():,})",
            save_path=os.path.join(cfg.OUTPUT_DIR, f"{cfg.run_name()}.png"),
        )

    return hist, best


def parse_args():
    p = argparse.ArgumentParser(description="S4 / baseline experiments on DREAMER EEG")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--model", type=str, default=None, choices=["s4", "lstm", "transformer"])
    p.add_argument("--theta", type=str, default=None, choices=["none", "const", "order"])
    p.add_argument("--target", type=str, default=None, choices=["valence", "arousal", "dominance"])
    p.add_argument("--split", type=str, default=None, choices=["random", "subject"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--H", type=int, default=None)
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--window-sec", type=float, default=None)
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def apply_args(cfg, args):
    mapping = {
        "data_dir": "DATA_DIR", "model": "MODEL", "theta": "THETA_MODE",
        "target": "TARGET", "split": "SPLIT_MODE", "epochs": "NUM_EPOCHS",
        "batch_size": "BATCH_SIZE", "lr": "LR", "H": "H", "N": "N",
        "layers": "NUM_LAYERS", "dropout": "DROPOUT",
        "window_sec": "WINDOW_SEC", "seed": "SEED",
    }
    for arg_name, cfg_name in mapping.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(cfg, cfg_name, val)
    if args.no_baseline:
        cfg.USE_BASELINE = False
    if getattr(args, "window_sec", None) is not None:
        cfg.STRIDE_SEC = cfg.WINDOW_SEC
    return cfg


if __name__ == "__main__":
    args = parse_args()
    apply_args(CFG, args)
    run(CFG, show_plot=not args.no_plot)
