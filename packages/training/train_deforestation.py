"""
Deforestation Cause Classifier Trainer

Multi-class classifier: given NDVI change + SAR texture + context features,
predict what caused the deforestation (logging / agriculture / fire / infrastructure / mining).

Output: packages/c2_intel/models/deforestation_classifier.joblib
        packages/c2_intel/models/deforestation_classifier_meta.json
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
MODEL_PATH = MODELS_DIR / "deforestation_classifier.joblib"
META_PATH  = MODELS_DIR / "deforestation_classifier_meta.json"

CAUSES = ["logging", "agriculture", "fire", "infrastructure", "mining"]


def _build_features(s: dict) -> list[float]:
    return [
        (float(s.get("ndvi_delta", -0.3)) + 1) / 2.0,
        min(float(s.get("ndvi_t0", 0.7)), 1.0),
        min(float(s.get("ndvi_t1", 0.4)), 1.0),
        min(float(s.get("sar_texture_delta", 0.3)), 1.0),
        min(float(s.get("patch_area_ha", 10)) / 500.0, 1.0),
        float(s.get("edge_regularity", 0.5)),
        min(float(s.get("distance_to_road_km", 5)) / 50.0, 1.0),
        float(s.get("protected_area", 0)),
    ]


def train(n_estimators: int = 200) -> None:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    import joblib

    sys.path.insert(0, str(Path(__file__).parent))
    from datasets.deforestation import download, load_as_training_samples

    logger.info("[Deforestation] Loading training data...")
    data_dir = download()
    samples  = load_as_training_samples(data_dir)
    samples  = [s for s in samples if s.get("cause") in CAUSES]

    X = np.array([_build_features(s) for s in samples], dtype=np.float32)
    y = np.array([CAUSES.index(s["cause"]) for s in samples], dtype=np.int32)

    logger.info("[Deforestation] %d samples, %d features, %d classes",
                len(X), X.shape[1], len(CAUSES))

    model = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=4,
        learning_rate=0.1, subsample=0.8, random_state=42,
    )

    scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro")
    logger.info("[Deforestation] CV F1-macro: %.3f ± %.3f", scores.mean(), scores.std())

    model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "GradientBoostingClassifier",
        "n_estimators": n_estimators,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "classes": CAUSES,
        "metrics": {
            "f1_macro_cv": round(float(scores.mean()), 4),
            "f1_macro_cv_std": round(float(scores.std()), 4),
        },
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[Deforestation] Model saved → %s", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).parent))
    train()
