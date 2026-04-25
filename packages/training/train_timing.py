"""
C2 Action Timing Predictor Training
=====================================
Trains three QuantileRegressor models (p25, median, p75) on OpenSky-derived
and doctrine-prior timing data. Output plugs directly into c2_intel/timing.py.

Feature vector (must match C2TimingPredictor._build_features — 39 features):
  [one-hot event_type (30)] + [one-hot context (6)] + [score, n_obs, urgency_tier]

Targets: minutes_to_action (p25 / median / p75 quantiles)

Usage:
    python train_timing.py \\
        --data   /tmp/heli-training-data/opensky \\
        --output ../../packages/c2_intel/models
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Must match timing.py exactly
C2_EVENT_TYPES = [
    "COMMS_DEGRADED", "COMMS_RESTORED", "THREAT_IDENTIFIED", "THREAT_NEUTRALIZED",
    "ASSET_OFFLINE", "ASSET_ONLINE", "AUTHORITY_DELEGATED", "AUTHORITY_REVOKED",
    "MISSION_STARTED", "MISSION_COMPLETED", "MISSION_ABORTED",
    "SENSOR_LOSS", "SENSOR_RESTORED", "GEOFENCE_BREACH", "GEOFENCE_CLEARED",
    "ENGAGEMENT_AUTHORIZED", "ENGAGEMENT_DENIED", "ENGAGEMENT_COMPLETE",
    "BATTERY_CRITICAL", "BATTERY_LOW", "HANDOFF_INITIATED", "HANDOFF_COMPLETE",
    "NODE_DEGRADED", "NODE_FAILED", "NODE_RECOVERED",
    "WEATHER_ALERT", "AIRSPACE_CONFLICT", "LINK_DEGRADED", "LINK_LOST",
    "PEER_OBSERVATION",
]

C2_CONTEXTS = [
    "urban_sar",
    "wildfire",
    "disaster_response",
    "military_ace",
    "border_patrol",
    "other",
]


def _build_feature_matrix(df) -> np.ndarray:
    """Build feature matrix matching C2TimingPredictor._build_features."""
    rows = []
    for _, row in df.iterrows():
        evt = str(row["event_type"])
        ctx = str(row["context"])
        score = float(row.get("score", 50))
        n_obs = float(row.get("n_obs", 1))
        urgency_tier = float(row.get("urgency_tier", 1))

        type_vec = [1.0 if t == evt else 0.0 for t in C2_EVENT_TYPES]
        ctx_vec  = [1.0 if c == ctx else 0.0 for c in C2_CONTEXTS]
        rows.append(type_vec + ctx_vec + [score, n_obs, urgency_tier])
    return np.array(rows, dtype=np.float32)


def train(data_dir: str, output_dir: str) -> None:
    try:
        import joblib
        import pandas as pd
        from sklearn.linear_model import QuantileRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import mean_absolute_error
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
    except ImportError:
        raise RuntimeError("scikit-learn, joblib, pandas required")

    data = Path(data_dir)
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    parquet_path = data / "timing_training.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Timing training data not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d timing records", len(df))

    # Filter valid rows
    df = df[df["minutes_to_action"] > 0]
    df = df[df["event_type"].isin(C2_EVENT_TYPES)]
    df = df.dropna()
    logger.info("After filtering: %d records", len(df))

    X = _build_feature_matrix(df)
    y = df["minutes_to_action"].values.astype(np.float32)

    logger.info("Feature matrix: %s, target range: [%.1f, %.1f] min",
                X.shape, y.min(), y.max())

    quantile_map = {
        "median": (0.50, "c2_timing_predictor.joblib"),
        "p25":    (0.25, "c2_timing_predictor_p25.joblib"),
        "p75":    (0.75, "c2_timing_predictor_p75.joblib"),
    }

    cv_maes = []
    for name, (q, filename) in quantile_map.items():
        # QuantileRegressor works best without scaling (it's already linear)
        model = QuantileRegressor(
            quantile=q,
            alpha=0.01,          # mild L1 regularization
            solver="highs",
        )
        model.fit(X, y)

        y_pred = model.predict(X)
        mae    = mean_absolute_error(y, y_pred)
        cv_maes.append(mae)
        logger.info("%-10s quantile=%.2f  MAE=%.2f min", name, q, mae)

        model_path = out / filename
        joblib.dump(model, model_path)
        logger.info("Saved → %s", model_path.name)

    # Per-event-type summary stats for logging
    event_summary = {}
    for evt in C2_EVENT_TYPES:
        mask = df["event_type"] == evt
        if mask.sum() > 0:
            event_summary[evt] = {
                "n": int(mask.sum()),
                "median_min": round(float(df.loc[mask, "minutes_to_action"].median()), 1),
            }

    meta = {
        "trained_at":    datetime.now(timezone.utc).isoformat(),
        "n_samples":     len(df),
        "n_features":    X.shape[1],
        "feature_dims":  {
            "event_types": len(C2_EVENT_TYPES),
            "contexts":    len(C2_CONTEXTS),
            "scalar":      3,
        },
        "cv_mae_mean":   round(float(np.mean(cv_maes)), 2),
        "event_summary": event_summary,
        "source": "OpenSky Network state vectors + operational doctrine priors",
    }
    meta_path = out / "c2_timing_predictor_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Timing predictor training complete. Models at %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Train C2 timing predictor on OpenSky data")
    p.add_argument("--data",   required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    train(args.data, args.output)
