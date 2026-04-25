"""
RadioML RF Modulation Classifier (1D CNN)

Trains a 1D-CNN on DeepSig RadioML I/Q samples to classify the
modulation type (24 classes for 2018, 11 for 2016) of a 1024-sample
(or 128-sample) baseband signal.

Used for: counter-UAS RF signature awareness, drone-radio detection,
EW spectrum monitoring, RF anomaly detection.

Civilian: airport perimeter UAS surveillance, prisons, critical infra.
Federal: counter-UAS, EW awareness, signal exploitation.

Output:
    packages/c2_intel/models/radioml_modulation_classifier.pt
    packages/c2_intel/models/radioml_modulation_classifier_meta.json

Prerequisites (one of):
    1. RML2018.01a HDF5 at packages/training/data/radioml/GOLD_XYZ_OSC.0001_1024.hdf5
       (preferred — 24 classes, 2.6M samples)
    2. RML2016.10a pickle at packages/training/data/radioml/RML2016.10a_dict.pkl
       (smaller — 11 classes, 220k samples)

If neither is present, falls back to synthetic I/Q with per-class
signatures so the pipeline is testable. Synthetic accuracy is not
representative of real performance.

Usage:
    python train_radioml_classifier.py [--edition 2018 --epochs 15 --min-snr 0]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "c2_intel" / "models"
MODEL_PATH = MODELS_DIR / "radioml_modulation_classifier.pt"
META_PATH  = MODELS_DIR / "radioml_modulation_classifier_meta.json"


def train(edition: str = "2018", epochs: int = 15, batch_size: int = 256,
          min_snr: int = 0, max_samples: int = 200000,
          device_str: str = "auto") -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if device_str == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("[radioml-train] Device: %s", device)

    from datasets.radioml import (   # type: ignore
        load_radioml_2016, load_radioml_2018, synthetic_radioml,
        MODULATION_CLASSES_2016, MODULATION_CLASSES_2018,
    )

    used_real = True
    if edition == "2018":
        X, y, y_snr = load_radioml_2018(max_samples=max_samples, min_snr=min_snr)
        classes = MODULATION_CLASSES_2018
    elif edition == "2016":
        X, y, y_snr = load_radioml_2016(max_samples=max_samples)
        classes = MODULATION_CLASSES_2016
        if min_snr is not None and len(X) > 0:
            mask = y_snr >= int(min_snr)
            X, y, y_snr = X[mask], y[mask], y_snr[mask]
    else:
        raise ValueError(f"edition must be 2016 or 2018, got {edition!r}")

    if len(X) == 0:
        logger.warning(
            "[radioml-train] No real RadioML data — falling back to "
            "synthetic. Register at deepsig.ai/datasets and download "
            "RML2018.01a (or RML2016.10a) for real training.")
        X, y, y_snr = synthetic_radioml(n_per_class=300,
                                        seq_len=1024 if edition == "2018" else 128,
                                        classes=classes)
        used_real = False

    seq_len = X.shape[-1]

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

    class _Conv1D(nn.Module):
        def __init__(self, n_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(2, 64, 7, padding=3),
                nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 64, 5, padding=2),
                nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 3, padding=1),
                nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(8),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(128 * 8, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            return self.head(self.features(x))

    model = _Conv1D(n_classes=len(classes)).to(device)
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
        logger.info("[radioml-train] Epoch %d/%d  val_acc=%.4f",
                    epoch + 1, epochs, acc)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "classes": classes,
        "edition": edition,
        "seq_len": seq_len,
    }, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "1D CNN (3-conv block + MLP head)",
        "task": f"RadioML modulation classification ({len(classes)} classes)",
        "edition": edition,
        "classes": classes,
        "n_classes": len(classes),
        "n_samples": int(len(X)),
        "seq_len": int(seq_len),
        "min_snr_db": int(min_snr) if min_snr is not None else None,
        "data_source": ("deepsig_radioml_real" if used_real else
                        "synthetic_fallback (register at deepsig.ai/datasets for real)"),
        "metrics": {"val_acc_best": round(best_acc, 4)},
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[radioml-train] Saved -> %s (val_acc=%.4f, real=%s)",
                MODEL_PATH, best_acc, used_real)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--edition", choices=["2016", "2018"], default="2018")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--min-snr", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=200000)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    train(edition=args.edition, epochs=args.epochs, batch_size=args.batch_size,
          min_snr=args.min_snr, max_samples=args.max_samples,
          device_str=args.device)
