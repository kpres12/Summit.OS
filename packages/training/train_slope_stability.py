"""
Slope Stability Classifier Trainer

Gradient-boosted multiclass classifier for mining/geohazard slope stability assessment.
Synthetic dataset — physics-grounded distributions based on geotechnical monitoring.

Output: packages/c2_intel/models/slope_stability_classifier.joblib
        packages/c2_intel/models/slope_stability_classifier_meta.json

Input features (10):
  - displacement_mm_day   : surface displacement rate (mm/day) from prism/radar
  - displacement_accel    : acceleration of displacement (mm/day²)
  - crack_width_mm        : maximum observed tension crack width (mm)
  - pore_pressure_kpa     : piezometer pore pressure (kPa)
  - rainfall_mm_24h       : rainfall in last 24 hours (mm)
  - slope_angle_deg       : slope face angle (degrees)
  - material_type         : 0=rock, 1=soil, 2=waste_dump, 3=tailings
  - saturation_pct        : degree of saturation (%)
  - vibration_ppv         : peak particle velocity from blasting (mm/s)
  - historical_movement_mm: cumulative historical displacement (mm)

Classes:
  0 = stable
  1 = watch       (monitor closely — elevated risk)
  2 = critical    (evacuate/halt operations)
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
MODEL_PATH = MODELS_DIR / "slope_stability_classifier.joblib"
META_PATH  = MODELS_DIR / "slope_stability_classifier_meta.json"

FEATURES = [
    "displacement_mm_day", "displacement_accel", "crack_width_mm",
    "pore_pressure_kpa", "rainfall_mm_24h", "slope_angle_deg",
    "material_type", "saturation_pct", "vibration_ppv", "historical_movement_mm",
]

CLASSES = ["stable", "watch", "critical"]


def _synthetic_samples(n: int = 900) -> tuple[np.ndarray, np.ndarray]:
    import random
    rng = random.Random(42)

    X, y = [], []
    per_class = n // 3

    # Class 0: stable — low displacement, dry conditions
    for _ in range(per_class):
        X.append([
            rng.uniform(0.0, 1.0),       # displacement_mm_day: negligible
            rng.uniform(-0.05, 0.05),    # displacement_accel: near zero
            rng.uniform(0.0, 5.0),       # crack_width_mm: hairline / absent
            rng.uniform(0, 50),          # pore_pressure_kpa: low
            rng.uniform(0, 20),          # rainfall_mm_24h: dry
            rng.uniform(20, 45),         # slope_angle_deg: moderate
            rng.choices([0, 1], weights=[0.6, 0.4])[0],  # material_type: rock/soil
            rng.uniform(10, 40),         # saturation_pct: dry
            rng.uniform(0, 50),          # vibration_ppv: low blasting
            rng.uniform(0, 50),          # historical_movement_mm: minimal
        ])
        y.append(0)

    # Class 1: watch — moderate displacement + rainfall, elevated pore pressure
    for _ in range(per_class):
        X.append([
            rng.uniform(1.0, 10.0),      # displacement_mm_day: measurable
            rng.uniform(0.05, 0.5),      # displacement_accel: slight increase
            rng.uniform(5.0, 30.0),      # crack_width_mm: developing cracks
            rng.uniform(50, 150),        # pore_pressure_kpa: elevated
            rng.uniform(20, 60),         # rainfall_mm_24h: moderate rain
            rng.uniform(35, 60),         # slope_angle_deg: steeper
            rng.choices([1, 2, 3], weights=[0.4, 0.35, 0.25])[0],  # softer materials
            rng.uniform(40, 70),         # saturation_pct: getting wet
            rng.uniform(30, 120),        # vibration_ppv: moderate blasting
            rng.uniform(30, 200),        # historical_movement_mm: some history
        ])
        y.append(1)

    # Class 2: critical — accelerating displacement, high pore pressure, saturated
    for _ in range(per_class):
        X.append([
            rng.uniform(10.0, 100.0),    # displacement_mm_day: rapid
            rng.uniform(0.5, 5.0),       # displacement_accel: accelerating (Fukuzono)
            rng.uniform(30.0, 200.0),    # crack_width_mm: wide open cracks
            rng.uniform(150, 400),       # pore_pressure_kpa: very high
            rng.uniform(50, 150),        # rainfall_mm_24h: heavy rain event
            rng.uniform(45, 75),         # slope_angle_deg: steep
            rng.choices([2, 3], weights=[0.5, 0.5])[0],  # waste dump / tailings
            rng.uniform(70, 100),        # saturation_pct: saturated
            rng.uniform(80, 250),        # vibration_ppv: heavy blasting nearby
            rng.uniform(100, 1000),      # historical_movement_mm: prior failures
        ])
        y.append(2)

    # Shuffle
    combined = list(zip(X, y))
    rng.shuffle(combined)
    X_arr, y_arr = zip(*combined)
    return np.array(X_arr, dtype=np.float32), np.array(y_arr, dtype=np.int32)


def train(n_estimators: int = 200) -> None:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    import joblib

    logger.info("[SlopeModel] Generating synthetic training data...")
    X, y = _synthetic_samples(n=900)
    logger.info("[SlopeModel] %d samples, %d features, %d classes",
                len(X), X.shape[1], len(set(y.tolist())))

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )

    scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro")
    logger.info("[SlopeModel] CV F1-macro: %.3f ± %.3f", scores.mean(), scores.std())

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
        "classes": CLASSES,
        "metrics": {
            "f1_macro_cv": round(float(scores.mean()), 4),
            "f1_macro_cv_std": round(float(scores.std()), 4),
        },
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[SlopeModel] Model saved → %s", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).parent))
    train()
