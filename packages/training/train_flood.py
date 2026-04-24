"""
Flood Extent Classifier Trainer

Gradient-boosted binary classifier: is this area flooded?
Features from SAR (Sentinel-1 VV/VH backscatter) + optical (NDWI, NDVI) + terrain.

Output: packages/c2_intel/models/flood_classifier.joblib
        packages/c2_intel/models/flood_classifier_meta.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "c2_intel" / "models"
MODEL_PATH = MODELS_DIR / "flood_classifier.joblib"
META_PATH  = MODELS_DIR / "flood_classifier_meta.json"

FEATURES = ["sar_vv_mean", "sar_vh_mean", "sar_vv_std", "ndwi", "ndvi",
            "blue_mean", "slope_deg", "dem_m"]


def _build_features(s: dict) -> list[float]:
    return [
        float(s.get("sar_vv_mean", -15)) / -30.0,       # normalise dB
        float(s.get("sar_vh_mean", -20)) / -30.0,
        float(s.get("sar_vv_std",  2.5)) / 5.0,
        (float(s.get("ndwi",  0.0)) + 1) / 2.0,         # -1..1 → 0..1
        (float(s.get("ndvi",  0.3)) + 1) / 2.0,
        float(s.get("blue_mean",  0.1)),
        min(float(s.get("slope_deg", 5)) / 30.0, 1.0),
        min(float(s.get("dem_m", 50))   / 500.0, 1.0),
    ]


def train(n_estimators: int = 200) -> None:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report
    import joblib

    sys.path.insert(0, str(Path(__file__).parent))
    from datasets.floodnet import download, load_as_training_samples

    logger.info("[Flood] Loading training data...")
    data_dir = download()
    samples  = load_as_training_samples(data_dir)

    X = np.array([_build_features(s) for s in samples], dtype=np.float32)
    y = np.array([int(s.get("is_flooded", s.get("is_deforested", 0))) for s in samples],
                 dtype=np.int32)

    logger.info("[Flood] %d samples, %d features, class balance: %.1f%% flooded",
                len(X), X.shape[1], y.mean() * 100)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=4,
        learning_rate=0.1, subsample=0.8, random_state=42,
    )

    scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    logger.info("[Flood] CV F1: %.3f ± %.3f", scores.mean(), scores.std())

    model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "GradientBoostingClassifier",
        "n_estimators": n_estimators,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "features": FEATURES,
        "classes": ["not_flooded", "flooded"],
        "metrics": {
            "f1_cv": round(float(scores.mean()), 4),
            "f1_cv_std": round(float(scores.std()), 4),
        },
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[Flood] Model saved → %s", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).parent))
    train()
