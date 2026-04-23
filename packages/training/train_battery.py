"""
Battery Degradation Predictor Training
========================================
Trains a gradient-boosted regression model on NASA Li-ion battery discharge
data to predict minutes remaining until critical (15% SOC) and empty (0%).

Output models plug into packages/c2_intel/battery.py.

Features:
  soc_pct            — current state of charge (0-100)
  discharge_rate_c   — C-rate (1.0 = 1-hour discharge, 2.0 = 30-min, etc.)
  temp_celsius       — ambient temperature
  capacity_ratio     — relative capacity vs nominal (1.0 = fresh, <1 = degraded)

Targets:
  minutes_to_critical  — minutes until SOC ≤ 15%
  minutes_to_empty     — minutes until SOC ≤ 0%

Usage:
    python train_battery.py \\
        --data   /tmp/heli-training-data/battery \\
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

_FEATURES = ["soc_pct", "discharge_rate_c", "temp_celsius", "capacity_ratio"]
_TARGETS  = ["minutes_to_critical", "minutes_to_empty"]


def train(data_dir: str, output_dir: str) -> None:
    try:
        import joblib
        import pandas as pd
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import mean_absolute_error
    except ImportError:
        raise RuntimeError("scikit-learn, joblib, pandas required")

    data = Path(data_dir)
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load training data (parquet from nasa_battery.py downloader)
    parquet_path = data / "battery_training.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Battery training data not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d discharge records", len(df))

    # Validate columns
    missing = [c for c in _FEATURES + _TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in training data: {missing}")

    # Drop outliers and invalid rows
    df = df[df["minutes_to_critical"] >= 0]
    df = df[df["soc_pct"].between(0, 100)]
    df = df[df["discharge_rate_c"].between(0.1, 5.0)]
    df = df.dropna(subset=_FEATURES + _TARGETS)

    X = df[_FEATURES].values.astype(np.float32)
    metrics = {}

    for target in _TARGETS:
        y = df[target].values.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("gbr",    GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=42,
            )),
        ])

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        mae    = mean_absolute_error(y_test, y_pred)
        logger.info("%-25s MAE: %.1f minutes", target, mae)
        metrics[f"{target}_mae_min"] = round(mae, 2)

        model_key = "critical" if "critical" in target else "empty"
        model_path = out / f"battery_predictor_{model_key}.joblib"
        joblib.dump(pipe, model_path)
        logger.info("Saved → %s", model_path.name)

    # Feature importance from the GBR on 'minutes_to_critical'
    pipe_crit = joblib.load(out / "battery_predictor_critical.joblib")
    importances = dict(zip(_FEATURES, pipe_crit.named_steps["gbr"].feature_importances_))

    meta = {
        "trained_at":       datetime.now(timezone.utc).isoformat(),
        "n_samples":        len(df),
        "features":         _FEATURES,
        "targets":          _TARGETS,
        "metrics":          metrics,
        "feature_importances": {k: round(v, 4) for k, v in importances.items()},
        "critical_threshold_pct": 15,
        "source": "NASA Li-ion battery aging dataset (synthetic augmented)",
    }
    meta_path = out / "battery_predictor_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Battery predictor training complete. Models at %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Train battery degradation predictor")
    p.add_argument("--data",   required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    train(args.data, args.output)
