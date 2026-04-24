"""
Crowd Density Estimator Trainer

Trains a regression model to estimate crowd count from aerial image features.
Lightweight tabular model (not a full CNN) — designed to run on features
already extracted by the YOLO detection pipeline.

Output: packages/c2_intel/models/crowd_estimator.joblib
        packages/c2_intel/models/crowd_estimator_meta.json

Input features (10):
  - person_detections: raw YOLO person count in frame
  - frame_coverage: fraction of frame with detections (0-1)
  - detection_density: detections per 100px²
  - mean_bbox_area: average bounding box area (px²), normalized
  - altitude_m: sensor altitude (higher altitude = occlusion → undercount)
  - fov_deg: camera field of view
  - overlap_ratio: estimated occlusion ratio from bbox overlap
  - time_of_day: 0-1 (normalized hour)
  - thermal_count: person count from co-registered thermal frame (if available)
  - scenario_type (0-4): disaster/event/street/evacuation/sar

Target: true_count (integer) — corrected crowd count
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
MODEL_PATH = MODELS_DIR / "crowd_estimator.joblib"
META_PATH = MODELS_DIR / "crowd_estimator_meta.json"

SCENARIO_TYPES = ["disaster", "event", "street", "evacuation", "sar"]


def _build_features(sample: dict) -> list[float]:
    detections = float(sample.get("person_detections", sample.get("count", 0)))
    coverage = min(float(sample.get("frame_coverage", 0.3)), 1.0)
    density = min(float(sample.get("detection_density", 0.01)), 1.0)
    bbox_area = min(float(sample.get("mean_bbox_area", 2000)) / 10000.0, 1.0)
    alt = min(float(sample.get("altitude_m", 100)) / 500.0, 1.0)
    fov = min(float(sample.get("fov_deg", 45)) / 120.0, 1.0)
    overlap = min(float(sample.get("overlap_ratio", 0.2)), 1.0)
    tod = min(float(sample.get("time_of_day", 0.5)), 1.0)
    thermal = min(float(sample.get("thermal_count", detections)) / max(detections + 1, 1), 2.0)
    scenario = sample.get("scenario", sample.get("scenario_type", "disaster"))
    scen_idx = SCENARIO_TYPES.index(scenario) / 4.0 if scenario in SCENARIO_TYPES else 0.0
    return [detections / 1000.0, coverage, density, bbox_area, alt, fov, overlap, tod, thermal, scen_idx]


def train(n_estimators: int = 200) -> None:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    import joblib

    sys.path.insert(0, str(Path(__file__).parent))
    from datasets.crowd import download, load_as_training_samples

    logger.info("[Crowd] Loading training data...")
    data_dir = download()
    samples = load_as_training_samples(data_dir)

    # Synthesize YOLO-feature proxies from count labels
    augmented = []
    import random
    random.seed(42)
    for s in samples:
        true_count = int(s.get("count", 10))
        # Model: YOLO detects ~70-90% of true count due to occlusion
        detection_rate = random.uniform(0.55, 0.92)
        detected = max(1, int(true_count * detection_rate))
        augmented.append({
            **s,
            "person_detections": detected,
            "frame_coverage": min(detected / max(true_count, 1) * 0.8, 1.0),
            "detection_density": detected / 640000.0,  # 800x800 frame
            "mean_bbox_area": max(200, 5000 / max(detected, 1)),
            "altitude_m": random.uniform(40, 300),
            "fov_deg": random.uniform(30, 90),
            "overlap_ratio": min((1 - detection_rate) * 1.5, 0.9),
            "time_of_day": random.random(),
            "thermal_count": detected + random.randint(-2, 5),
        })

    X = np.array([_build_features(s) for s in augmented], dtype=np.float32)
    y = np.array([float(s.get("count", 0)) for s in augmented], dtype=np.float32)

    logger.info("[Crowd] %d samples, %d features", len(X), X.shape[1])

    model = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=4,
        learning_rate=0.1, subsample=0.8, random_state=42,
    )

    from sklearn.metrics import mean_absolute_error
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    mae_cv = -scores.mean()
    logger.info("[Crowd] CV MAE: %.1f persons", mae_cv)

    model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "GradientBoostingRegressor",
        "n_estimators": n_estimators,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "scenario_types": SCENARIO_TYPES,
        "metrics": {"mae_cv_persons": round(float(mae_cv), 1)},
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[Crowd] Model saved → %s", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).parent))
    train()
