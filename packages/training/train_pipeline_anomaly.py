"""
Pipeline Anomaly Classifier Trainer

Gradient-boosted multiclass classifier: normal vs. operational anomaly vs. leak suspected.
Synthetic dataset — physics-grounded distributions based on pipeline integrity monitoring.

Output: packages/c2_intel/models/pipeline_anomaly_classifier.joblib
        packages/c2_intel/models/pipeline_anomaly_classifier_meta.json

Input features (10):
  - pressure_bar          : operating pressure (bar)
  - pressure_delta_pct    : % change in pressure vs. rolling baseline
  - flow_rate_m3h         : volumetric flow rate (m³/h)
  - flow_delta_pct        : % change in flow vs. rolling baseline
  - temp_delta_c          : temperature deviation from baseline (°C)
  - acoustic_db           : acoustic emission level (dB) — AE sensing
  - cp_mv                 : cathodic protection potential (mV) — corrosion indicator
  - wall_loss_pct         : pipe wall thickness loss percentage
  - time_since_pig_days   : days since last inline inspection run
  - segment_age_years     : age of this pipeline segment in years

Classes:
  0 = normal
  1 = anomaly_operational  (transient event, no leak — e.g. pressure surge)
  2 = leak_suspected
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
MODEL_PATH = MODELS_DIR / "pipeline_anomaly_classifier.joblib"
META_PATH  = MODELS_DIR / "pipeline_anomaly_classifier_meta.json"

FEATURES = [
    "pressure_bar", "pressure_delta_pct", "flow_rate_m3h", "flow_delta_pct",
    "temp_delta_c", "acoustic_db", "cp_mv", "wall_loss_pct",
    "time_since_pig_days", "segment_age_years",
]

CLASSES = ["normal", "anomaly_operational", "leak_suspected"]


def _synthetic_samples(n: int = 900) -> tuple[np.ndarray, np.ndarray]:
    import random
    rng = random.Random(42)

    X, y = [], []
    per_class = n // 3

    # Class 0: normal — stable operations
    for _ in range(per_class):
        X.append([
            rng.uniform(40, 80),         # pressure_bar: normal operating
            rng.uniform(-2.0, 2.0),      # pressure_delta_pct: stable
            rng.uniform(100, 500),       # flow_rate_m3h: steady
            rng.uniform(-3.0, 3.0),      # flow_delta_pct: stable
            rng.uniform(-1.0, 1.0),      # temp_delta_c: minimal deviation
            rng.uniform(40, 65),         # acoustic_db: low background noise
            rng.uniform(-900, -800),     # cp_mv: healthy cathodic protection
            rng.uniform(0.0, 5.0),       # wall_loss_pct: minimal
            rng.uniform(0, 180),         # time_since_pig_days: recent inspection
            rng.uniform(1, 20),          # segment_age_years
        ])
        y.append(0)

    # Class 1: anomaly_operational — pressure transient, no leak (valve ops, slug flow)
    for _ in range(per_class):
        X.append([
            rng.uniform(50, 100),        # pressure_bar: elevated
            rng.uniform(5.0, 20.0),      # pressure_delta_pct: spike
            rng.uniform(90, 480),        # flow_rate_m3h: relatively stable
            rng.uniform(-5.0, 5.0),      # flow_delta_pct: not significantly changed
            rng.uniform(-2.0, 4.0),      # temp_delta_c: slight change
            rng.uniform(55, 80),         # acoustic_db: elevated (valve noise)
            rng.uniform(-950, -750),     # cp_mv: still reasonable
            rng.uniform(2.0, 10.0),      # wall_loss_pct: some wear
            rng.uniform(30, 365),        # time_since_pig_days
            rng.uniform(5, 30),          # segment_age_years
        ])
        y.append(1)

    # Class 2: leak_suspected — pressure drop + flow anomaly + acoustic spike
    for _ in range(per_class):
        X.append([
            rng.uniform(20, 60),         # pressure_bar: dropping
            rng.uniform(-20.0, -5.0),    # pressure_delta_pct: significant drop
            rng.uniform(120, 600),       # flow_rate_m3h: may increase (loss from system)
            rng.uniform(5.0, 30.0),      # flow_delta_pct: imbalance
            rng.uniform(1.0, 8.0),       # temp_delta_c: Joule-Thomson cooling
            rng.uniform(70, 110),        # acoustic_db: high AE from escaping fluid
            rng.uniform(-1100, -600),    # cp_mv: corrosion / stray current issues
            rng.uniform(8.0, 25.0),      # wall_loss_pct: significant
            rng.uniform(90, 730),        # time_since_pig_days: long since inspection
            rng.uniform(10, 50),         # segment_age_years: older segments
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

    logger.info("[PipelineModel] Generating synthetic training data...")
    X, y = _synthetic_samples(n=900)
    logger.info("[PipelineModel] %d samples, %d features, %d classes",
                len(X), X.shape[1], len(set(y.tolist())))

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )

    scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro")
    logger.info("[PipelineModel] CV F1-macro: %.3f ± %.3f", scores.mean(), scores.std())

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
    logger.info("[PipelineModel] Model saved → %s", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).parent))
    train()
