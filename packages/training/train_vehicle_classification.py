"""
Vehicle Classification Model Trainer

Gradient-boosted classifier to identify ground vehicle class from aerial/sensor data.
Purely synthetic dataset — physics-grounded feature distributions per class.

Output: packages/c2_intel/models/vehicle_classifier.joblib
        packages/c2_intel/models/vehicle_classifier_meta.json

Input features (12):
  - speed_mps             : current speed in metres per second
  - heading_change_rate   : degrees per second (manoeuvrability indicator)
  - formation_spacing_m   : distance to nearest peer vehicle (m)
  - time_of_day_h         : hour 0-23
  - area_type             : 0=urban, 1=rural, 2=off-road
  - convoy_member         : bool 0/1
  - stop_frequency        : stops per hour
  - route_deviation       : metres off expected route
  - size_class            : 0=small, 1=medium, 2=large, 3=heavy
  - thermal_signature     : normalised 0-1
  - acoustic_level        : normalised 0-1
  - payload_indicator     : bool 0/1

Classes:
  0 = civilian_passenger
  1 = civilian_commercial
  2 = emergency_services
  3 = military_logistics
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
MODEL_PATH = MODELS_DIR / "vehicle_classifier.joblib"
META_PATH  = MODELS_DIR / "vehicle_classifier_meta.json"

FEATURES = [
    "speed_mps", "heading_change_rate", "formation_spacing_m", "time_of_day_h",
    "area_type", "convoy_member", "stop_frequency", "route_deviation",
    "size_class", "thermal_signature", "acoustic_level", "payload_indicator",
]

CLASSES = ["civilian_passenger", "civilian_commercial", "emergency_services", "military_logistics"]


def _synthetic_samples(n: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    import random
    rng = random.Random(42)

    X, y = [], []
    per_class = n // 4

    # Class 0: civilian_passenger — typical urban/rural commuter
    for _ in range(per_class):
        X.append([
            rng.uniform(5, 30),          # speed_mps
            rng.uniform(0.0, 3.0),       # heading_change_rate (normal turns)
            rng.uniform(50, 500),        # formation_spacing_m (no formation)
            rng.uniform(6, 22),          # time_of_day_h (commute hours)
            rng.choices([0, 1], weights=[0.7, 0.3])[0],  # area_type: urban/rural
            0,                           # convoy_member: no
            rng.uniform(0.5, 4.0),       # stop_frequency
            rng.uniform(0, 50),          # route_deviation: follows roads
            rng.choices([0, 1], weights=[0.7, 0.3])[0],  # size_class: small/medium
            rng.uniform(0.05, 0.25),     # thermal_signature: low
            rng.uniform(0.1, 0.35),      # acoustic_level: low
            0,                           # payload_indicator: no
        ])
        y.append(0)

    # Class 1: civilian_commercial — delivery/freight vehicles
    for _ in range(per_class):
        X.append([
            rng.uniform(8, 25),          # speed_mps (slower, loaded)
            rng.uniform(0.0, 2.0),       # heading_change_rate (route-following)
            rng.uniform(30, 300),        # formation_spacing_m
            rng.uniform(5, 20),          # time_of_day_h (business hours)
            rng.choices([0, 1], weights=[0.6, 0.4])[0],  # area_type
            0,                           # convoy_member: rarely
            rng.uniform(2.0, 8.0),       # stop_frequency: deliveries
            rng.uniform(0, 30),          # route_deviation: follows delivery routes
            rng.choices([1, 2], weights=[0.5, 0.5])[0],  # size_class: medium/large
            rng.uniform(0.1, 0.3),       # thermal_signature
            rng.uniform(0.2, 0.5),       # acoustic_level: engine noise
            1,                           # payload_indicator: cargo
        ])
        y.append(1)

    # Class 2: emergency_services — police/fire/ambulance
    for _ in range(per_class):
        X.append([
            rng.uniform(15, 45),         # speed_mps: fast response
            rng.uniform(3.0, 12.0),      # heading_change_rate: erratic routing
            rng.uniform(20, 400),        # formation_spacing_m
            rng.uniform(0, 24),          # time_of_day_h: 24/7
            rng.choices([0, 1], weights=[0.8, 0.2])[0],  # area_type: mostly urban
            0,                           # convoy_member: rare (except motorcades)
            rng.uniform(0.0, 2.0),       # stop_frequency: incident response
            rng.uniform(20, 200),        # route_deviation: non-standard routes
            rng.choices([0, 1], weights=[0.6, 0.4])[0],  # size_class: small/medium
            rng.uniform(0.2, 0.5),       # thermal_signature: engine heat
            rng.uniform(0.5, 1.0),       # acoustic_level: sirens
            rng.choices([0, 1], weights=[0.5, 0.5])[0],  # payload_indicator
        ])
        y.append(2)

    # Class 3: military_logistics — convoy vehicles
    for _ in range(per_class):
        X.append([
            rng.uniform(10, 30),         # speed_mps: convoy speed
            rng.uniform(0.0, 1.5),       # heading_change_rate: disciplined
            rng.uniform(10, 80),         # formation_spacing_m: tight convoy spacing
            rng.uniform(4, 23),          # time_of_day_h
            rng.choices([1, 2], weights=[0.4, 0.6])[0],  # area_type: rural/off-road
            1,                           # convoy_member: yes
            rng.uniform(0.0, 1.0),       # stop_frequency: planned halts only
            rng.uniform(0, 100),         # route_deviation: deviates for security
            rng.choices([2, 3], weights=[0.5, 0.5])[0],  # size_class: large/heavy
            rng.uniform(0.35, 0.8),      # thermal_signature: diesel heat
            rng.uniform(0.4, 0.8),       # acoustic_level: heavy engines
            1,                           # payload_indicator: always loaded
        ])
        y.append(3)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train(n_estimators: int = 200) -> None:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    import joblib

    logger.info("[VehicleModel] Generating synthetic training data...")
    X, y = _synthetic_samples(n=1200)
    logger.info("[VehicleModel] %d samples, %d features, %d classes",
                len(X), X.shape[1], len(set(y.tolist())))

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )

    scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro")
    logger.info("[VehicleModel] CV F1-macro: %.3f ± %.3f", scores.mean(), scores.std())

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
    logger.info("[VehicleModel] Model saved → %s", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).parent))
    train()
