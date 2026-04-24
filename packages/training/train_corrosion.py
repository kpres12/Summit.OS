"""
Structural Corrosion / Defect Classifier Trainer

Multi-label classifier predicting which defect types are present in an
infrastructure inspection image, and an overall severity score (0-1).

Output: packages/c2_intel/models/corrosion_classifier.joblib
        packages/c2_intel/models/corrosion_classifier_meta.json

Input features (14):
  - thermal_delta_c: temperature above ambient (from thermal adapter)
  - surface_roughness: normalized (from LiDAR or structured light)
  - rgb_rust_ratio: fraction of rust-colored pixels (HSV analysis)
  - rgb_crack_score: edge density in Canny filter output (0-1)
  - context_type (one-hot 5): bridge, pipeline, tank, steel_beam, concrete_deck
  - age_years: asset age in years (normalized)
  - last_inspection_months: months since last inspection
  - environment_marine: bool (saltwater proximity)

Target: multi-label binary vector [crack, spalling, efflorescence,
         exposed_rebar, corrosion, delamination]
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
MODEL_PATH = MODELS_DIR / "corrosion_classifier.joblib"
META_PATH = MODELS_DIR / "corrosion_classifier_meta.json"

DEFECT_CLASSES = ["crack", "spalling", "efflorescence", "exposed_rebar", "corrosion", "delamination"]
CONTEXT_TYPES = ["bridge", "pipeline", "tank", "steel_beam", "concrete_deck"]


def _build_features(sample: dict) -> list[float]:
    thermal = min(float(sample.get("thermal_delta_c", 0)) / 50.0, 1.0)
    roughness = min(float(sample.get("surface_roughness", 0.2)), 1.0)
    rust_ratio = min(float(sample.get("rgb_rust_ratio", 0.1)), 1.0)
    crack_score = min(float(sample.get("rgb_crack_score", 0.1)), 1.0)
    ctx = sample.get("context", sample.get("context_type", "concrete_deck")).lower()
    ctx_oh = [1.0 if c in ctx else 0.0 for c in CONTEXT_TYPES]
    age = min(float(sample.get("age_years", 10)) / 100.0, 1.0)
    last_insp = min(float(sample.get("last_inspection_months", 12)) / 120.0, 1.0)
    marine = float(sample.get("environment_marine", 0))
    return [thermal, roughness, rust_ratio, crack_score] + ctx_oh + [age, last_insp, marine]


def _build_labels(sample: dict) -> list[int]:
    defects = sample.get("defect_classes", [])
    return [1 if cls in defects else 0 for cls in DEFECT_CLASSES]


def train(n_estimators: int = 150) -> None:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.model_selection import cross_val_score
    import joblib

    sys.path.insert(0, str(Path(__file__).parent))
    from datasets.corrosion import download, load_as_training_samples

    logger.info("[Corrosion] Loading training data...")
    data_dir = download()
    samples = load_as_training_samples(data_dir)

    X = np.array([_build_features(s) for s in samples], dtype=np.float32)
    y = np.array([_build_labels(s) for s in samples], dtype=np.int32)

    logger.info("[Corrosion] %d samples, %d features, %d label dimensions", len(X), X.shape[1], y.shape[1])

    base = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=3, random_state=42)
    model = MultiOutputClassifier(base, n_jobs=-1)

    # Score on first defect class (corrosion) as proxy
    corrosion_idx = DEFECT_CLASSES.index("corrosion")
    scores = cross_val_score(
        GradientBoostingClassifier(n_estimators=n_estimators, max_depth=3, random_state=42),
        X, y[:, corrosion_idx], cv=5, scoring="f1",
    )
    logger.info("[Corrosion] Corrosion F1 CV: %.3f ± %.3f", scores.mean(), scores.std())

    model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "MultiOutputClassifier(GradientBoostingClassifier)",
        "n_estimators": n_estimators,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "defect_classes": DEFECT_CLASSES,
        "context_types": CONTEXT_TYPES,
        "metrics": {
            "corrosion_f1_cv": round(float(scores.mean()), 4),
        },
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[Corrosion] Model saved → %s", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).parent))
    train()
