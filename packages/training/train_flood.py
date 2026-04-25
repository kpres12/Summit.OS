"""
Flood Extent Classifier Trainer

Gradient-boosted binary classifier: is this area flooded?

Features:
  Tier 1 (always): SAR VV/VH mean, SAR VV std, NDWI, NDVI, blue_mean, slope, DEM
  Tier 2 (real S1): SAR VV p10/p90, SAR VH mean, polarisation ratio
  Tier 3 (real S2): red_mean, nir_mean, nbr, ndre, ndmi

Data preference (best → fallback):
  1. Real Sentinel-1 chips from Element84 STAC (requires rasterio)
  2. Hurricane Harvey real segmentation masks (Kaggle)
  3. Physics-informed synthetic SAR proxies

Output: packages/c2_intel/models/flood_classifier.joblib
        packages/c2_intel/models/flood_classifier_meta.json
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
MODEL_PATH = MODELS_DIR / "flood_classifier.joblib"
META_PATH  = MODELS_DIR / "flood_classifier_meta.json"

# Tier-1 features always present
FEATURES_BASE = [
    "sar_vv_mean", "sar_vh_mean", "sar_vv_std",
    "ndwi", "ndvi", "blue_mean", "slope_deg", "dem_m",
]

# Tier-2 features from real Sentinel-1
FEATURES_S1 = ["sar_vv_p10", "sar_vv_p90", "pol_ratio"]

# Tier-3 features from real Sentinel-2
FEATURES_S2 = ["red_mean", "nir_mean", "nbr", "ndre"]


def _build_features(s: dict, use_s1: bool, use_s2: bool) -> list[float]:
    def safe(key: str, default: float = 0.0) -> float:
        v = s.get(key)
        if v is None or (isinstance(v, float) and (v != v)):  # nan check
            return default
        return float(v)

    feats = [
        safe("sar_vv_mean", -15) / -30.0,
        safe("sar_vh_mean", -20) / -30.0,
        safe("sar_vv_std",  2.5) / 5.0,
        (safe("ndwi",  0.0) + 1) / 2.0,
        (safe("ndvi",  0.3) + 1) / 2.0,
        safe("blue_mean", 0.1),
        min(safe("slope_deg", 5) / 30.0, 1.0),
        min(safe("dem_m", 50) / 500.0, 1.0),
    ]

    if use_s1:
        vv_mean = safe("sar_vv_mean", -15)
        vh_mean = safe("sar_vh_mean", -20)
        feats += [
            safe("sar_vv_p10", vv_mean - 2) / -30.0,
            safe("sar_vv_p90", vv_mean + 2) / -30.0,
            # Polarisation ratio VH/VV — sensitive to surface roughness
            min(abs(vh_mean - vv_mean) / 10.0, 1.0),
        ]

    if use_s2:
        feats += [
            safe("red_mean", 0.08),
            safe("nir_mean", 0.35),
            (safe("nbr",  0.2) + 1) / 2.0,
            (safe("ndre", 0.1) + 1) / 2.0,
        ]

    return feats


def train(n_estimators: int = 300, prefer_real: bool = True) -> None:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    import joblib

    sys.path.insert(0, str(Path(__file__).parent))
    from datasets.floodnet import download as floodnet_download, load_as_training_samples as load_floodnet
    from datasets.sentinel1_stac import load_as_training_samples as load_s1

    # ── Gather samples ────────────────────────────────────────────────────────
    samples: list[dict] = []
    use_s1_features = False

    if prefer_real:
        logger.info("[Flood] Attempting real Sentinel-1 data from STAC...")
        s1_samples = load_s1(use_real=True)
        real_s1 = [s for s in s1_samples if s.get("source") == "sentinel1_stac_real"]
        if real_s1:
            logger.info("[Flood] Got %d real Sentinel-1 samples", len(real_s1))
            samples.extend(s1_samples)
            use_s1_features = True
        else:
            logger.info("[Flood] No real S1 data — using physics-informed synthetic S1 + Harvey")
            samples.extend(s1_samples)
            use_s1_features = True  # physics-synthetic still has p10/p90

    if len(samples) < 100:
        floodnet_download()
        harvey_samples = load_floodnet(Path(__file__).parent / "data" / "flood")
        logger.info("[Flood] Loaded %d Harvey/FloodNet samples", len(harvey_samples))
        samples.extend(harvey_samples)

    # ── Build feature matrix ──────────────────────────────────────────────────
    X = np.array([_build_features(s, use_s1_features, False) for s in samples],
                 dtype=np.float32)
    y = np.array([int(s.get("is_flooded", 0)) for s in samples], dtype=np.int32)

    # Remove any rows with NaN (edge cases from proxy computation)
    valid_mask = ~np.isnan(X).any(axis=1)
    X, y = X[valid_mask], y[valid_mask]

    real_count = sum(1 for s in samples if "real" in s.get("source", ""))
    logger.info("[Flood] %d samples (%d real), %d features, %.1f%% flooded",
                len(X), real_count, X.shape[1], y.mean() * 100)

    # ── Train ─────────────────────────────────────────────────────────────────
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    logger.info("[Flood] CV F1: %.3f ± %.3f", scores.mean(), scores.std())

    model.fit(X, y)

    # Feature importances
    feat_names = FEATURES_BASE[:]
    if use_s1_features:
        feat_names += FEATURES_S1
    importances = sorted(zip(feat_names, model.feature_importances_), key=lambda x: -x[1])
    logger.info("[Flood] Top features: %s",
                ", ".join(f"{n}={v:.3f}" for n, v in importances[:5]))

    # ── Save ──────────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "trained_at":    datetime.now(timezone.utc).isoformat(),
        "model":         "GradientBoostingClassifier",
        "n_estimators":  n_estimators,
        "n_samples":     int(len(X)),
        "n_real_samples": real_count,
        "n_features":    int(X.shape[1]),
        "features":      feat_names,
        "use_s1_features": use_s1_features,
        "classes":       ["not_flooded", "flooded"],
        "data_sources": list({s.get("source", "unknown") for s in samples}),
        "metrics": {
            "f1_cv":     round(float(scores.mean()), 4),
            "f1_cv_std": round(float(scores.std()), 4),
        },
        "feature_importances": {n: round(float(v), 4) for n, v in importances},
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[Flood] Model saved → %s  (F1=%.3f)", MODEL_PATH, scores.mean())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).parent))
    train()
