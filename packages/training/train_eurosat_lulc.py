"""
EuroSAT LULC Classifier — Sentinel-2 Land-Use / Land-Cover

Real Sentinel-2 RGB imagery, 27,000 labeled patches × 10 land-use classes.
Direct HTTPS download (no auth, no portal). CNN classifier suitable as a
pre-trained backbone for downstream EO models (burn-scar, smoke plume,
flood inundation, vessel detection).

Output:
    packages/c2_intel/models/eurosat_lulc_classifier.pt
    packages/c2_intel/models/eurosat_lulc_classifier_meta.json

Usage:
    python train_eurosat_lulc.py [--epochs 15]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "c2_intel" / "models"
MODEL_PATH = MODELS_DIR / "eurosat_lulc_classifier.pt"
META_PATH  = MODELS_DIR / "eurosat_lulc_classifier_meta.json"


def train(epochs: int = 15, batch_size: int = 64, device_str: str = "auto") -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if device_str == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("[eurosat-train] Device: %s", device)

    from datasets.eurosat import (   # type: ignore
        ensure_downloaded, load_eurosat_rgb,
    )
    ensure_downloaded("rgb")
    X, y, classes = load_eurosat_rgb()
    if len(X) < 100:
        raise RuntimeError(f"Only {len(X)} EuroSAT samples — download failed?")

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
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),                          # 32
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),                          # 16
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),                          # 8
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(256 * 4 * 4, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            return self.head(self.features(x))

    model = _CNN(n_classes=len(classes)).to(device)
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
        logger.info("[eurosat-train] Epoch %d/%d  val_acc=%.4f", epoch + 1, epochs, acc)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "n_classes": len(classes),
                "classes": classes}, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "CNN (4-conv block + MLP head)",
        "task": "EuroSAT 10-class land-use / land-cover (Sentinel-2 RGB)",
        "classes": classes,
        "n_classes": len(classes),
        "n_samples": int(len(X)),
        "data_source": "eurosat_real_dfki",
        "metrics": {"val_acc_best": round(best_acc, 4)},
        "transfer_learning": (
            "Pretrained CNN backbone for downstream EO models — burn-scar "
            "U-Net, smoke plume CNN, flood inundation, vessel detection."),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[eurosat-train] Saved -> %s (val_acc=%.4f)", MODEL_PATH, best_acc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, device_str=args.device)
