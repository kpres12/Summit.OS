"""
Building Damage Vision Classifier (CNN)

Fine-tunes EfficientNet-B0 (pretrained on ImageNet) to classify building damage
from satellite image patches: no-damage / minor / major / destroyed.

Uses real xBD imagery if available; generates synthetic patches as fallback.
Synthetic patches use texture variance as a proxy for damage severity — real
imagery will dramatically improve accuracy.

Output: packages/c2_intel/models/damage_vision_classifier.pt
        packages/c2_intel/models/damage_vision_meta.json

Usage:
    python train_damage_vision.py [--epochs 10] [--batch-size 32] [--device auto]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "c2_intel" / "models"
MODEL_PATH = MODELS_DIR / "damage_vision_classifier.pt"
META_PATH  = MODELS_DIR / "damage_vision_meta.json"

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
IMG_SIZE = 224


def _synthetic_dataset(n_per_class: int = 200):
    """
    Generate synthetic image patches with texture correlating to damage class.
    Intensity variance and edge density increase with damage severity.
    Not a substitute for real imagery — demonstrates the pipeline only.
    """
    import torch
    from torch.utils.data import TensorDataset

    images, labels = [], []
    rng = np.random.default_rng(42)

    for cls_idx in range(4):
        for _ in range(n_per_class):
            # Base texture: more chaotic = more damage
            noise_scale = 0.05 + cls_idx * 0.20
            base = rng.uniform(0.3, 0.7, (IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
            noise = rng.normal(0, noise_scale, (IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
            # Structural edges: fewer as damage increases
            if cls_idx < 2:
                for _ in range(20 - cls_idx * 8):
                    r = rng.integers(10, IMG_SIZE - 10)
                    c = rng.integers(10, IMG_SIZE - 10)
                    base[r, c:c+rng.integers(5, 30), :] = 0.2 + cls_idx * 0.1
            img = np.clip(base + noise, 0.0, 1.0)
            images.append(torch.from_numpy(img.transpose(2, 0, 1)))
            labels.append(cls_idx)

    X = torch.stack(images)
    y = torch.tensor(labels, dtype=torch.long)
    # Normalize to ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    X = (X - mean) / std
    return TensorDataset(X, y)


def _real_dataset(data_dir: Path):
    """Load xBD patch images if real imagery was downloaded."""
    try:
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
        xbd_patches = data_dir / "patches"
        if not xbd_patches.exists():
            return None
        tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return ImageFolder(str(xbd_patches), transform=tf)
    except Exception:
        return None


def train(epochs: int = 10, batch_size: int = 32, device_str: str = "auto") -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    from sklearn.metrics import f1_score

    if device_str == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("[DamageVision] Device: %s", device)

    # Dataset
    data_dir = Path(__file__).parent / "data" / "xbd"
    dataset = _real_dataset(data_dir)
    source = "real"
    if dataset is None:
        logger.warning("[DamageVision] No real xBD patches found — using synthetic imagery")
        dataset = _synthetic_dataset(n_per_class=200)
        source = "synthetic"

    n_val = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Model: EfficientNet-B0, replace classifier head
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, len(DAMAGE_CLASSES)),
    )
    model = model.to(device)

    # Freeze backbone for first half of training
    for p in model.features.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1, best_state = 0.0, None

    for epoch in range(epochs):
        # Unfreeze backbone halfway through
        if epoch == epochs // 2:
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch)

        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X, y in val_loader:
                preds = model(X.to(device)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.numpy())

        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        logger.info("[DamageVision] Epoch %d/%d — loss %.3f — val F1-macro %.3f",
                    epoch + 1, epochs, total_loss / len(train_loader), f1)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "classes": DAMAGE_CLASSES,
                "img_size": IMG_SIZE}, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "EfficientNet-B0",
        "backbone": "imagenet1k_v1",
        "img_size": IMG_SIZE,
        "classes": DAMAGE_CLASSES,
        "n_epochs": epochs,
        "n_samples": len(dataset),
        "data_source": source,
        "metrics": {"f1_macro_val": round(best_f1, 4)},
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[DamageVision] Model saved → %s (val F1: %.3f)", MODEL_PATH, best_f1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device",     default="auto")
    args = p.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, device_str=args.device)
