"""ISRO BAS HAR — Standalone Temporal HAR Model Evaluation Script.

Evaluates a saved PyTorch checkpoint on an NPZ sliding-window dataset.

Usage
-----
    python -m scripts.evaluate_har --checkpoint models/har/conv1d_bigru_best.pt --dataset data/temporal_har/features/SES_001_NOMINAL_windows.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ─── Resolve project root ─────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.activity_engine.har_model import Conv1DBiGRUHAR, HAR_ACTION_CLASSES, NUM_HAR_CLASSES
from src.activity_engine.feature_extractor import FEATURE_DIM
from scripts.train_har import compute_metrics, benchmark_latency


def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_path: Path,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict:
    """Evaluate a trained checkpoint against an NPZ dataset."""
    checkpoint_path = Path(checkpoint_path)
    dataset_path = Path(dataset_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dev = torch.device(device)

    # 1. Load dataset
    data = np.load(dataset_path)
    windows = data["windows"].astype(np.float32)
    labels = data["labels"].astype(np.int64)

    # 2. Load model
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    config = ckpt.get("config", {})

    model = Conv1DBiGRUHAR(
        input_dim=config.get("input_dim", FEATURE_DIM),
        seq_len=config.get("seq_len", 60),
        conv_channels=config.get("conv_channels", 128),
        gru_hidden_dim=config.get("gru_hidden_dim", 64),
        num_classes=config.get("num_classes", NUM_HAR_CLASSES),
    ).to(dev)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = TensorDataset(torch.from_numpy(windows), torch.from_numpy(labels))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds_list = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for bx, _ in loader:
            bx = bx.to(dev)
            logits = model(bx)
            preds_list.extend(logits.argmax(dim=-1).cpu().tolist())
    eval_time = time.perf_counter() - t0

    metrics = compute_metrics(labels, np.array(preds_list))
    latency = benchmark_latency(model, device=device, iterations=100)

    print("=" * 68)
    print("EVALUATION RESULTS")
    print("=" * 68)
    print(f"  Checkpoint       : {checkpoint_path.name}")
    print(f"  Dataset Samples  : {len(windows)} windows")
    print(f"  Accuracy         : {metrics['accuracy']*100:.2f}%")
    print(f"  Macro-F1         : {metrics['macro_f1']:.4f}")
    print(f"  Latency (mean)   : {latency['mean_latency_ms']:.2f} ms")
    print("=" * 68)

    return {
        "metrics": metrics,
        "latency": latency,
        "total_eval_time_sec": round(eval_time, 3),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HAR Checkpoint")
    parser.add_argument("--checkpoint", default="models/har/conv1d_bigru_best.pt")
    parser.add_argument("--dataset", default="data/temporal_har/features/SES_001_NOMINAL_windows.npz")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        dataset_path=Path(args.dataset),
        device=args.device,
    )
