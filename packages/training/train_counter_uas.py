"""
Counter-UAS Classifier Trainer

Gradient-boosted binary classifier: is this UAS authorized or rogue/threat?
Synthetic dataset — physics-grounded feature distributions.

Output: packages/c2_intel/models/counter_uas_classifier.joblib
        packages/c2_intel/models/counter_uas_classifier_meta.json

Input features (10):
  - altitude_m            : flight altitude in metres
  - speed_mps             : speed in metres per second
  - rcs_dbsm              : radar cross section (dBsm)
  - rf_power_dbm          : RF emissions power (dBm)
  - flight_pattern        : 0=direct, 1=orbit, 2=erratic, 3=hover
  - distance_to_asset_m   : distance to nearest protected asset (m)
  - time_active_min       : how long the UAS has been active
  - operator_id_confirmed : bool 0/1
  - geofence_compliant    : bool 0/1
  - payload_detected      : bool 0/1

Classes:
  0 = authorized
  1 = rogue/threat
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
MODEL_PATH = MODELS_DIR / "counter_uas_classifier.joblib"
META_PATH  = MODELS_DIR / "counter_uas_classifier_meta.json"

FEATURES = [
    "altitude_m", "speed_mps", "rcs_dbsm", "rf_power_dbm",
    "flight_pattern", "distance_to_asset_m", "time_active_min",
    "operator_id_confirmed", "geofence_compliant", "payload_detected",
]

CLASSES = ["authorized", "rogue_threat"]


def _synthetic_samples(n: int = 800) -> tuple[np.ndarray, np.ndarray]:
    import random
    rng = random.Random(42)

    X, y = [], []
    n_auth = n // 2
    n_rogue = n - n_auth

    # Class 0: authorized UAS — compliant, predictable operations
    for _ in range(n_auth):
        X.append([
            rng.uniform(50, 400),        # altitude_m: proper AGL bands
            rng.uniform(2, 20),          # speed_mps: normal survey/delivery speed
            rng.uniform(-30, -15),       # rcs_dbsm: small drone signature
            rng.uniform(-90, -60),       # rf_power_dbm: normal control link
            rng.choices([0, 1], weights=[0.6, 0.4])[0],  # flight_pattern: direct/orbit
            rng.uniform(200, 5000),      # distance_to_asset_m: away from assets
            rng.uniform(5, 120),         # time_active_min: normal mission duration
            1,                           # operator_id_confirmed: yes
            1,                           # geofence_compliant: yes
            rng.choices([0, 1], weights=[0.7, 0.3])[0],  # payload_detected: delivery possible
        ])
        y.append(0)

    # Class 1: rogue/threat UAS — low, sneaky, non-compliant
    for _ in range(n_rogue):
        X.append([
            rng.uniform(10, 120),        # altitude_m: low NOE flight
            rng.uniform(0, 18),          # speed_mps: varies (hover to fast ingress)
            rng.uniform(-25, -5),        # rcs_dbsm: possibly modified/larger
            rng.uniform(-110, -50),      # rf_power_dbm: may be low (pre-programmed)
            rng.choices([1, 2, 3], weights=[0.25, 0.45, 0.3])[0],  # orbit/erratic/hover
            rng.uniform(0, 800),         # distance_to_asset_m: close to assets
            rng.uniform(2, 60),          # time_active_min: often shorter / loitering
            0,                           # operator_id_confirmed: no
            0,                           # geofence_compliant: no
            rng.choices([0, 1], weights=[0.4, 0.6])[0],  # payload_detected: often yes
        ])
        y.append(1)

    # Shuffle
    combined = list(zip(X, y))
    rng.shuffle(combined)
    X_arr, y_arr = zip(*combined)
    return np.array(X_arr, dtype=np.float32), np.array(y_arr, dtype=np.int32)


def train(n_estimators: int = 200) -> None:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    import joblib

    logger.info("[CounterUAS] Generating synthetic training data...")
    X, y = _synthetic_samples(n=800)
    logger.info("[CounterUAS] %d samples, %d features, class balance: %.1f%% rogue",
                len(X), X.shape[1], y.mean() * 100)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )

    scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    logger.info("[CounterUAS] CV F1: %.3f ± %.3f", scores.mean(), scores.std())

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
            "f1_cv": round(float(scores.mean()), 4),
            "f1_cv_std": round(float(scores.std()), 4),
        },
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[CounterUAS] Model saved → %s", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).parent))
    train()
