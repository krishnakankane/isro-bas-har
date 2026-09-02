"""ISRO BAS HAR — Temporal HAR Model Training Pipeline.

Trains the Conv1D-BiGRU-Attention temporal classifier on pre-extracted feature tensors (N x 60 x 154).
Preserves 100% offline reproducibility and tracks all performance metrics.

Usage
-----
    python -m scripts.train_har --dataset data/temporal_har/features/SES_001_NOMINAL_windows.npz --epochs 60
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─── Resolve project root ─────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.activity_engine.har_model import (
    Conv1DBiGRUHAR,
    HAR_ACTION_CLASSES,
    NUM_HAR_CLASSES,
)
from src.activity_engine.feature_extractor import FEATURE_DIM


def stratified_train_val_split(
    labels: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform deterministic stratified train/val split across classes."""
    rng = np.random.RandomState(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    unique_classes = np.unique(labels)
    for c in unique_classes:
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        n_val = max(1, int(round(len(c_idx) * val_ratio)))
        val_indices.extend(c_idx[:n_val])
        train_indices.extend(c_idx[n_val:])

    train_arr = np.array(train_indices, dtype=np.int64)
    val_arr = np.array(val_indices, dtype=np.int64)
    rng.shuffle(train_arr)
    rng.shuffle(val_arr)
    return train_arr, val_arr


def compute_class_weights(labels: np.ndarray, num_classes: int = NUM_HAR_CLASSES) -> torch.Tensor:
    """Compute smoothed inverse class frequencies for weighted CrossEntropyLoss."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total_samples = float(len(labels))
    # Smoothed inverse frequency: sqrt(total / (num_classes * count + eps))
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = math.sqrt(total_samples / (num_classes * counts[c]))
        else:
            weights[c] = 1.0
    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    return torch.from_numpy(weights).float()


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_HAR_CLASSES,
) -> Dict[str, object]:
    """Compute accuracy, per-class precision/recall/F1, and confusion matrix."""
    accuracy = float(np.mean(y_true == y_pred))

    # Confusion matrix: rows = true, cols = pred
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class_metrics: Dict[int, Dict[str, float]] = {}
    f1_list: List[float] = []

    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(np.sum(cm[:, c]) - tp)
        fn = int(np.sum(cm[c, :]) - tp)
        support = int(np.sum(cm[c, :]))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        if support > 0:
            f1_list.append(f1)

        per_class_metrics[c] = {
            "class_name": HAR_ACTION_CLASSES.get(c, f"ACTION_{c}"),
            "support": support,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        }

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_metrics,
    }


def benchmark_latency(
    model: nn.Module,
    input_shape: Tuple[int, int, int] = (1, 60, FEATURE_DIM),
    device: str = "cpu",
    iterations: int = 100,
) -> Dict[str, float]:
    """Benchmark inference latency (ms per window)."""
    model.eval()
    dummy = torch.randn(*input_shape, dtype=torch.float32, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    times: List[float] = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    return {
        "mean_latency_ms": round(float(np.mean(times)), 3),
        "std_latency_ms": round(float(np.std(times)), 3),
        "p95_latency_ms": round(float(np.percentile(times, 95)), 3),
        "min_latency_ms": round(float(np.min(times)), 3),
        "max_latency_ms": round(float(np.max(times)), 3),
    }


def train_har_model(
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    val_ratio: float = 0.2,
    seed: int = 42,
    patience: int = 15,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Execute end-to-end training of the Conv1D-BiGRU HAR model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev_str)

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Feature dataset not found at: {dataset_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "conv1d_bigru_best.pt"
    metrics_path = output_dir / "training_metrics.json"

    print("=" * 68)
    print("ISRO BAS HAR — TEMPORAL HAR MODEL TRAINING")
    print("=" * 68)
    print(f"  Dataset Path    : {dataset_path}")
    print(f"  Target Device   : {dev_str.upper()}")
    print(f"  Max Epochs      : {epochs} (Early Stopping patience={patience})")
    print(f"  Batch Size      : {batch_size}")
    print(f"  Initial LR      : {lr}")
    print(f"  Val Ratio       : {val_ratio * 100:.0f}%")
    print("=" * 68)

    # 1. Load Feature Tensors
    npz_data = np.load(dataset_path)
    windows = npz_data["windows"].astype(np.float32)  # (N, 60, 154)
    labels = npz_data["labels"].astype(np.int64)      # (N,)

    N, T, D = windows.shape
    assert T == 60, f"Expected 60 frames per window, got {T}"
    assert D == FEATURE_DIM, f"Expected {FEATURE_DIM} features per frame, got {D}"
    assert len(labels) == N, "Window count and label count mismatch!"

    print(f"[DataLoader] Loaded {N} sliding windows with shape ({N}, {T}, {D})")

    # 2. Stratified Split
    train_idx, val_idx = stratified_train_val_split(labels, val_ratio=val_ratio, seed=seed)
    X_train, y_train = windows[train_idx], labels[train_idx]
    X_val, y_val = windows[val_idx], labels[val_idx]

    print(f"[DataLoader] Split: {len(X_train)} Train windows | {len(X_val)} Validation windows")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Model & Loss Initialization
    model = Conv1DBiGRUHAR(
        input_dim=FEATURE_DIM,
        seq_len=60,
        conv_channels=128,
        gru_hidden_dim=64,
        gru_num_layers=2,
        num_classes=NUM_HAR_CLASSES,
        dropout_conv=0.2,
        dropout_gru=0.2,
        dropout_fc=0.3,
    ).to(dev)

    param_count = model.count_parameters()
    print(f"[Model] Conv1D-BiGRU initialized with {param_count:,} trainable parameters.")

    class_weights = compute_class_weights(y_train, num_classes=NUM_HAR_CLASSES).to(dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 4. Training Loop
    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    t_start_train = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train Step
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for bx, by in train_loader:
            bx, by = bx.to(dev), by.to(dev)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(by)
            preds = logits.argmax(dim=-1)
            train_correct += int((preds == by).sum().item())
            train_total += len(by)

        scheduler.step()
        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # ── Validation Step
        model.eval()
        val_loss_sum = 0.0
        val_preds_list: List[int] = []
        val_true_list: List[int] = []

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(dev), by.to(dev)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss_sum += loss.item() * len(by)
                preds = logits.argmax(dim=-1)
                val_preds_list.extend(preds.cpu().tolist())
                val_true_list.extend(by.cpu().tolist())

        val_loss = val_loss_sum / len(y_val)
        val_eval = compute_metrics(np.array(val_true_list), np.array(val_preds_list))
        val_acc = val_eval["accuracy"]
        val_f1 = val_eval["macro_f1"]

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "lr": round(float(scheduler.get_last_lr()[0]), 6),
        }
        history.append(epoch_stats)

        # Log every 5 epochs or on improvement
        improved = (val_loss < best_val_loss) or (val_loss == best_val_loss and val_f1 > best_val_f1)
        if improved:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_f1,
                    "param_count": param_count,
                    "classes": HAR_ACTION_CLASSES,
                    "config": {
                        "input_dim": FEATURE_DIM,
                        "seq_len": 60,
                        "num_classes": NUM_HAR_CLASSES,
                        "conv_channels": 128,
                        "gru_hidden_dim": 64,
                    },
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or improved:
            flag = " [BEST SAVED]" if improved else ""
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.1f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.1f}% F1: {val_f1:.4f}{flag}"
            )

        if patience_counter >= patience:
            print(f"[EarlyStopping] Triggered at epoch {epoch} (No improvement for {patience} epochs).")
            break

    total_train_time = time.time() - t_start_train

    # 5. Final Best Model Evaluation & Latency Benchmark
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_preds_final: List[int] = []
    with torch.no_grad():
        for bx, _ in val_loader:
            bx = bx.to(dev)
            logits = model(bx)
            val_preds_final.extend(logits.argmax(dim=-1).cpu().tolist())

    final_metrics = compute_metrics(y_val, np.array(val_preds_final))
    cpu_latency = benchmark_latency(model.to("cpu"), device="cpu", iterations=100)
    cuda_latency = benchmark_latency(model.to("cuda"), device="cuda", iterations=100) if torch.cuda.is_available() else None

    # Print Final Summary & Confusion Matrix
    print("\n" + "=" * 68)
    print("TRAINING COMPLETE — BASELINE PERFORMANCE SUMMARY")
    print("=" * 68)
    print(f"  Best Epoch        : {best_epoch}")
    print(f"  Total Train Time  : {total_train_time:.2f}s")
    print(f"  Validation Loss   : {best_val_loss:.4f}")
    print(f"  Validation Acc    : {final_metrics['accuracy']*100:.2f}%")
    print(f"  Validation Macro-F1: {final_metrics['macro_f1']:.4f}")
    print(f"  Model Parameters  : {param_count:,}")
    print(f"  CPU Latency (mean): {cpu_latency['mean_latency_ms']:.2f} ms / window")
    if cuda_latency:
        print(f"  GPU Latency (mean): {cuda_latency['mean_latency_ms']:.2f} ms / window")
    print(f"  Saved Checkpoint  : {checkpoint_path}")

    print("\nPer-Class Breakdown on Validation Split:")
    print(f"  {'ID':<3} {'Action Name':<25} {'Support':<8} {'Precision':<10} {'Recall':<8} {'F1-Score':<8}")
    print("  " + "-" * 64)
    for c, d in final_metrics["per_class"].items():
        print(
            f"  {c:<3} {d['class_name']:<25} {d['support']:<8} "
            f"{d['precision']*100:6.1f}%    {d['recall']*100:6.1f}%  {d['f1_score']:6.4f}"
        )

    print("\nValidation Confusion Matrix (Rows: True, Cols: Predicted):")
    cm_arr = np.array(final_metrics["confusion_matrix"])
    header = "      " + " ".join([f"[{c}]" for c in range(NUM_HAR_CLASSES)])
    print(header)
    for r in range(NUM_HAR_CLASSES):
        row_str = f"[{r:02d}] " + "  ".join([f"{cm_arr[r, c]:2d}" for c in range(NUM_HAR_CLASSES)])
        print(row_str)
    print("=" * 68)

    full_report = {
        "best_epoch": best_epoch,
        "total_train_time_sec": round(total_train_time, 2),
        "param_count": param_count,
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "final_validation_metrics": final_metrics,
        "cpu_latency": cpu_latency,
        "gpu_latency": cuda_latency,
        "history": history,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    return full_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ISRO BAS HAR — Train Temporal Classifier")
    parser.add_argument(
        "--dataset",
        default="data/temporal_har/features/SES_001_NOMINAL_windows.npz",
        help="Path to .npz feature tensor file",
    )
    parser.add_argument(
        "--output_dir",
        default="models/har",
        help="Directory to save model checkpoints and logs",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Max training epochs (default: 60)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_har_model(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        seed=args.seed,
        patience=args.patience,
    )
