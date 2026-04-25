"""
xView3 SAR Vessel Detection — Multi-class Classifier

Trains a CNN classifier on Sentinel-1 SAR vessel chips from the xView3
dataset, predicting one of:

    non_vessel | small_vessel | medium_vessel | large_vessel
    fishing_vessel | dark_vessel | dark_fishing_vessel

The full xView3 task is detection (bbox + class), not classification.
This trainer builds a starter classifier on cropped detection chips
(default 64x64) — useful as a pre-screen ranker before forwarding to a
full detector, and as a stand-alone vessel-type classifier when the
operator already has bounding boxes from another source (AIS/radar/visual).

Civilian use: USCG SAR demand prediction, illegal fishing, MDA awareness.
Federal use: Maritime Domain Awareness, dark-vessel surveillance.

Output:
    packages/c2_intel/models/xview3_vessel_classifier.pt
    packages/c2_intel/models/xview3_vessel_classifier_meta.json

Prerequisites:
    Download xView3 from https://iuu.xview.us/dataset and unpack to
    packages/training/data/xview3/ (manifest CSVs + chips/<scene_id>/...).

Usage:
    python train_xview3_vessel_detector.py [--epochs 20 --crop 64]

If chips are not present, falls back to a sanity-check on synthetic
SAR-like noise so the training pipeline is testable without the data.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "c2_intel" / "models"
MODEL_PATH = MODELS_DIR / "xview3_vessel_classifier.pt"
META_PATH  = MODELS_DIR / "xview3_vessel_classifier_meta.json"

DEFAULT_CROP    = 64
N_CHANNELS      = 2     # VV + VH dB
CLASSES = [
    "non_vessel", "small_vessel", "medium_vessel", "large_vessel",
    "fishing_vessel", "dark_vessel", "dark_fishing_vessel",
]


def _crop_chip(scene_dir: Path, det_row: int, det_col: int,
               crop: int) -> Optional[np.ndarray]:
    """Read VV+VH chip rasters and crop a (crop x crop x 2) array around det.
    Falls back to None if rasterio isn't installed or files missing."""
    try:
        import rasterio  # type: ignore
    except ImportError:
        logger.debug("[xview3-train] rasterio not installed — cannot crop chips")
        return None

    vv_path = scene_dir / "VV_dB.tif"
    vh_path = scene_dir / "VH_dB.tif"
    if not (vv_path.exists() and vh_path.exists()):
        return None

    half = crop // 2
    chip = np.zeros((crop, crop, 2), dtype=np.float32)

    for ch_idx, p in enumerate((vv_path, vh_path)):
        try:
            with rasterio.open(p) as src:
                # Window the read so we don't load the whole scene
                from rasterio.windows import Window
                row_off = max(0, det_row - half)
                col_off = max(0, det_col - half)
                window = Window(col_off, row_off, crop, crop)
                arr = src.read(1, window=window, boundless=True, fill_value=-30.0)
                # Pad/crop in case window was clipped at edge
                if arr.shape != (crop, crop):
                    fixed = np.full((crop, crop), -30.0, dtype=np.float32)
                    fixed[:arr.shape[0], :arr.shape[1]] = arr
                    arr = fixed
                chip[..., ch_idx] = arr
        except Exception as e:
            logger.debug("[xview3-train] read fail %s: %s", p, e)
            return None

    # Normalize from dB scale (~ -40..0) to [0, 1]
    chip = (chip - (-40.0)) / 40.0
    return np.clip(chip, 0.0, 1.0)


def _build_real_dataset(crop: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (X[N, crop, crop, 2], y[N]) cropped from xView3 chips."""
    from datasets.xview3 import iter_chips_with_labels  # type: ignore

    X_list, y_list = [], []
    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}
    for chip_dir, dets in iter_chips_with_labels(split="train"):
        for det in dets:
            chip = _crop_chip(chip_dir, det.row, det.col, crop)
            if chip is None:
                continue
            cls = det.class_label
            if cls not in cls_to_idx:
                continue
            X_list.append(chip)
            y_list.append(cls_to_idx[cls])
    if not X_list:
        return np.zeros((0, crop, crop, 2), dtype=np.float32), np.zeros(0, dtype=np.int64)
    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    logger.info("[xview3-train] real chips loaded: %d (class dist: %s)",
                len(X), {CLASSES[c]: int(n) for c, n in Counter(y.tolist()).items()})
    return X, y


def _build_synthetic_fallback(crop: int, n: int = 4000
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Generate SAR-like synthetic chips so the pipeline is testable without
    real xView3 data. Each class has a distinct radiometric signature
    (vessel = bright spot in dark sea; dark vessel = lower SNR; large
    vessel = wider extent). Not for production — pipeline test only."""
    rng = np.random.default_rng(7)
    X = rng.normal(0.2, 0.05, (n, crop, crop, 2)).astype(np.float32)  # dim sea
    y = rng.integers(0, len(CLASSES), n).astype(np.int64)
    half = crop // 2
    for i in range(n):
        cls = CLASSES[y[i]]
        if cls == "non_vessel":
            continue
        # Inject a bright spot per class
        if "large" in cls:
            radius = 6
            intensity = 0.85
        elif "medium" in cls:
            radius = 4
            intensity = 0.75
        elif "fishing" in cls:
            radius = 3
            intensity = 0.70
        elif "dark" in cls:
            radius = 3
            intensity = 0.45    # lower contrast — dark vessel
        else:
            radius = 2
            intensity = 0.60
        for ch in range(2):
            for r in range(half - radius, half + radius):
                for c in range(half - radius, half + radius):
                    if 0 <= r < crop and 0 <= c < crop:
                        X[i, r, c, ch] += intensity
    X = np.clip(X, 0.0, 1.0)
    logger.info("[xview3-train] synthetic fallback: %d samples", n)
    return X, y


def train(epochs: int = 20, batch_size: int = 64, crop: int = DEFAULT_CROP,
          device_str: str = "auto") -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if device_str == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("[xview3-train] Device: %s", device)

    # Try real xView3 chips first
    X, y = _build_real_dataset(crop=crop)
    used_real = len(X) > 0
    if not used_real:
        logger.warning(
            "[xview3-train] No real xView3 chips found — falling back to "
            "synthetic SAR-like noise. Download xView3 from "
            "https://iuu.xview.us/dataset for real training.")
        X, y = _build_synthetic_fallback(crop=crop)

    # NCHW for torch
    X = np.transpose(X, (0, 3, 1, 2)).astype(np.float32)

    rng = np.random.default_rng(7)
    order = rng.permutation(len(X))
    X, y = X[order], y[order]
    split = int(len(X) * 0.8)
    X_train, X_val = torch.tensor(X[:split]), torch.tensor(X[split:])
    y_train, y_val = torch.tensor(y[:split]), torch.tensor(y[split:])

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=batch_size, shuffle=False, num_workers=0)

    class _CNN(nn.Module):
        def __init__(self, n_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(N_CHANNELS, 32, 3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(128 * 4 * 4, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            return self.head(self.features(x))

    model = _CNN(n_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                preds = model(X_b.to(device)).argmax(1).cpu()
                correct += int((preds == y_b).sum())
                total += len(y_b)
        acc = correct / max(total, 1)
        logger.info("[xview3-train] Epoch %d/%d  val_acc=%.4f", epoch + 1, epochs, acc)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "n_classes": len(CLASSES),
        "n_channels": N_CHANNELS,
        "crop": crop,
        "classes": CLASSES,
    }, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "CNN (3-conv block + MLP head)",
        "task": "xView3 SAR vessel multi-class (Sentinel-1 chip classifier)",
        "classes": CLASSES,
        "n_classes": len(CLASSES),
        "n_channels": N_CHANNELS,
        "crop": crop,
        "n_samples": int(len(X)),
        "data_source": "xview3_real" if used_real else "synthetic_fallback",
        "metrics": {"val_acc_best": round(best_acc, 4)},
        "note": ("Real xView3 chips required for production performance; "
                 "synthetic fallback exists only to test the pipeline. "
                 "Download data from https://iuu.xview.us/dataset."),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[xview3-train] Saved -> %s (val_acc=%.4f, real=%s)",
                MODEL_PATH, best_acc, used_real)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--crop", type=int, default=DEFAULT_CROP)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, crop=args.crop,
          device_str=args.device)
