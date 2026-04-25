"""
Fire Danger Classifier + FRP Intensity Regressor

Uses live NASA FIRMS VIIRS detections (375 m, 5-day global) fused with
OpenMeteo historical weather per 0.5° cluster to train two models:

  1. fire_danger_classifier  — 5-class (low/moderate/high/very_high/extreme)
     Input: weather + fire cluster features
     Output: fire danger level for a given location/date

  2. fire_intensity_regressor — predicts log1p(frp_max) MW
     Input: same features
     Output: expected fire radiative power at detection site

Set FIRMS_MAP_KEY env var to use live 375m VIIRS data.
Without key → falls back to 1km MODIS public CSV.

Output:
  packages/c2_intel/models/fire_danger_classifier.joblib
  packages/c2_intel/models/fire_danger_classifier_meta.json
  packages/c2_intel/models/fire_intensity_regressor.joblib
  packages/c2_intel/models/fire_intensity_regressor_meta.json
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

DANGER_CLASSES = ["low", "moderate", "high", "very_high", "extreme"]

# Features used for both models
FEATURE_NAMES = [
    "temp_max",        # °C — hot = fire-prone
    "rh_max",          # % — dry = fire-prone
    "wind_max",        # km/h
    "wind_gust",       # km/h
    "vpd_max",         # kPa — vapour pressure deficit (atmospheric dryness)
    "et0",             # mm — evapotranspiration (fuel moisture proxy)
    "precip",          # mm — recent rain (suppression)
    "hotspot_count",   # number of pixels in cluster
    "confidence_score", # VIIRS detection confidence (0.5–1.5)
    "daynight_day_frac", # fraction of day detections (crown fires more daytime)
    "fire_weather_index",  # computed FWI proxy (0–100)
    "lat_abs",         # abs(latitude) — distance from equator
]


def _build_features(s: dict) -> list[float]:
    def safe(k, default=0.0):
        v = s.get(k)
        return default if v is None or (isinstance(v, float) and v != v) else float(v)

    return [
        safe("temp_max", 25.0),
        safe("rh_max",   50.0),
        safe("wind_max", 15.0),
        safe("wind_gust", 20.0),
        min(safe("vpd_max", 1.5), 8.0),
        min(safe("et0", 5.0), 20.0),
        min(safe("precip", 0.0), 50.0),
        min(safe("hotspot_count", 1), 500.0),
        safe("confidence_score", 1.0),
        safe("daynight_day_frac", 0.5),
        safe("fire_weather_index", 30.0),
        abs(safe("lat", 30.0)),
    ]


def _danger_label_int(s: dict) -> int:
    return DANGER_CLASSES.index(s.get("fire_danger_class", "low"))


def train() -> None:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import LabelEncoder
    import joblib

    sys.path.insert(0, str(Path(__file__).parent))
    from datasets.firms_weather import load_as_training_samples

    logger.info("[FireDanger] Fetching FIRMS + OpenMeteo samples...")
    samples = load_as_training_samples(max_clusters=400, fetch_weather=True)
    logger.info("[FireDanger] Got %d cluster samples", len(samples))

    if len(samples) < 20:
        logger.error("[FireDanger] Not enough samples to train — check FIRMS_MAP_KEY")
        return

    # ── Build matrices ────────────────────────────────────────────────────────
    X_raw = [_build_features(s) for s in samples]
    X = np.array(X_raw, dtype=np.float32)
    y_cls = np.array([_danger_label_int(s) for s in samples], dtype=np.int32)
    y_frp = np.log1p([max(0.0, s.get("frp_max", 0.0)) for s in samples]).astype(np.float32)

    # Drop NaN rows
    valid = ~np.isnan(X).any(axis=1)
    X, y_cls, y_frp = X[valid], y_cls[valid], y_frp[valid]

    logger.info("[FireDanger] %d samples, %d features | class dist: %s",
                len(X), X.shape[1],
                {c: int((y_cls == i).sum()) for i, c in enumerate(DANGER_CLASSES)})

    # ── Train danger classifier ───────────────────────────────────────────────
    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=2,
        random_state=42,
    )

    # StratifiedKFold only valid if all classes have >= n_splits samples
    min_class_count = int(np.bincount(y_cls).min())
    n_splits = min(5, max(2, min_class_count))
    cv_clf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf_scores = cross_val_score(clf, X, y_cls, cv=cv_clf, scoring="f1_weighted")
    logger.info("[FireDanger] Classifier CV F1 (weighted): %.3f ± %.3f",
                clf_scores.mean(), clf_scores.std())
    clf.fit(X, y_cls)

    clf_importances = sorted(
        zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1]
    )
    logger.info("[FireDanger] Top classifier features: %s",
                ", ".join(f"{n}={v:.3f}" for n, v in clf_importances[:5]))

    # ── Train intensity regressor ─────────────────────────────────────────────
    reg = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=2,
        random_state=42,
    )

    cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)
    reg_scores = cross_val_score(reg, X, y_frp, cv=cv_reg, scoring="neg_mean_absolute_error")
    reg_mae = -reg_scores.mean()
    logger.info("[FireDanger] Regressor CV MAE (log1p FRP MW): %.4f", reg_mae)
    reg.fit(X, y_frp)

    reg_importances = sorted(
        zip(FEATURE_NAMES, reg.feature_importances_), key=lambda x: -x[1]
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, MODELS_DIR / "fire_danger_classifier.joblib")
    joblib.dump(reg, MODELS_DIR / "fire_intensity_regressor.joblib")

    data_sources = list({s.get("source", "unknown") for s in samples})
    real_count = sum(1 for s in samples if "viirs" in s.get("source", "") or "real" in s.get("source", ""))

    clf_meta = {
        "trained_at":   datetime.now(timezone.utc).isoformat(),
        "model":        "GradientBoostingClassifier",
        "task":         "fire_danger_class",
        "classes":      DANGER_CLASSES,
        "n_samples":    int(len(X)),
        "n_real":       real_count,
        "n_features":   len(FEATURE_NAMES),
        "features":     FEATURE_NAMES,
        "data_sources": data_sources,
        "metrics": {
            "f1_cv_weighted":     round(float(clf_scores.mean()), 4),
            "f1_cv_weighted_std": round(float(clf_scores.std()), 4),
        },
        "feature_importances": {n: round(float(v), 4) for n, v in clf_importances},
    }

    reg_meta = {
        "trained_at":   datetime.now(timezone.utc).isoformat(),
        "model":        "GradientBoostingRegressor",
        "task":         "log1p_frp_max_mw",
        "n_samples":    int(len(X)),
        "n_real":       real_count,
        "n_features":   len(FEATURE_NAMES),
        "features":     FEATURE_NAMES,
        "data_sources": data_sources,
        "metrics": {
            "mae_cv_log1p_frp": round(float(reg_mae), 4),
        },
        "feature_importances": {n: round(float(v), 4) for n, v in reg_importances},
        "note": "Predict with np.expm1(model.predict(X)) to recover FRP in MW",
    }

    (MODELS_DIR / "fire_danger_classifier_meta.json").write_text(json.dumps(clf_meta, indent=2))
    (MODELS_DIR / "fire_intensity_regressor_meta.json").write_text(json.dumps(reg_meta, indent=2))

    logger.info("[FireDanger] Classifier saved → fire_danger_classifier.joblib (F1=%.3f)", clf_scores.mean())
    logger.info("[FireDanger] Regressor saved  → fire_intensity_regressor.joblib (MAE=%.4f)", reg_mae)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).parent))
    train()
