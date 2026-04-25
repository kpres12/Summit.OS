"""
Compound Hazard Risk Scorer

Trains per-hazard regressors (fire, flood, seismic) and a compound
0-100 risk scorer from three real data sources:

  Fire    — NASA FIRMS VIIRS 375m + OpenMeteo (live via FIRMS_MAP_KEY)
  Seismic — USGS ComCat M2.5+ past 30 days (no key needed)
  Flood   — physics-informed synthetic (precip + RH + terrain features)
             augmented with fire sample weather (inverted: wet = flood risk)

Features (unified across all four models):
  weather: temp_max, rh_max, wind_max, vpd_max, precip, et0
  fire:    fire_weather_index, log1p_frp, hotspot_count_norm
  seismic: eq_magnitude, eq_depth_km, eq_sig_norm
  spatial: lat_abs, lon_sin, lon_cos
  time:    day_sin, day_cos

Output:
  packages/c2_intel/models/fire_risk_regressor.joblib
  packages/c2_intel/models/flood_risk_regressor.joblib
  packages/c2_intel/models/seismic_risk_regressor.joblib
  packages/c2_intel/models/compound_hazard_scorer.joblib
  packages/c2_intel/models/compound_hazard_scorer_meta.json
"""

from __future__ import annotations

import json
import logging
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "c2_intel" / "models"

FEATURE_NAMES = [
    "temp_max",            # °C
    "rh_max",              # %
    "wind_max",            # km/h
    "vpd_max",             # kPa
    "precip",              # mm (recent)
    "et0",                 # mm (evapotranspiration — fuel moisture proxy)
    "fire_weather_index",  # 0-100 FWI proxy
    "log1p_frp",           # log1p(FRP MW) — 0 if no active fire
    "hotspot_count_norm",  # hotspot count / 100, clamped 0-1
    "eq_magnitude",        # nearest recent M2.5+ magnitude (0 = none nearby)
    "eq_depth_km",         # hypocentre depth km (0 = none)
    "eq_sig_norm",         # USGS significance / 1000, clamped 0-1
    "lat_abs",             # abs(latitude) — 0 equator, 90 pole
    "lon_sin",             # sin(lon_rad) — circular encoding
    "lon_cos",             # cos(lon_rad)
    "day_sin",             # sin(2π × doy / 365) — seasonal
    "day_cos",             # cos(2π × doy / 365)
]
N_FEAT = len(FEATURE_NAMES)

DOY = datetime.now(timezone.utc).timetuple().tm_yday


# ── Feature builder ────────────────────────────────────────────────────────────

def _featurise(s: dict) -> list[float]:
    def sf(k, default=0.0):
        v = s.get(k)
        return default if v is None or (isinstance(v, float) and v != v) else float(v)

    lat  = sf("lat", 0.0)
    lon  = sf("lon", 0.0)
    doy  = int(sf("doy", DOY))
    lon_rad = math.radians(lon)

    return [
        sf("temp_max",   25.0),
        sf("rh_max",     50.0),
        min(sf("wind_max", 15.0), 100.0),
        min(sf("vpd_max",  1.5),   8.0),
        min(sf("precip",   0.0),  200.0),
        min(sf("et0",      5.0),  20.0),
        sf("fire_weather_index", 20.0),
        min(math.log1p(max(0.0, sf("frp_total", 0.0))), 10.0),  # log1p_frp
        min(sf("hotspot_count", 0) / 100.0, 1.0),
        min(sf("eq_magnitude", 0.0), 9.5),
        min(sf("eq_depth_km", 0.0), 300.0),
        min(sf("eq_sig", 0) / 1000.0, 1.0),
        abs(lat),
        math.sin(lon_rad),
        math.cos(lon_rad),
        math.sin(2 * math.pi * doy / 365),
        math.cos(2 * math.pi * doy / 365),
    ]


# ── Score functions (physics-informed labels) ─────────────────────────────────

def _fire_score(s: dict) -> float:
    fwi = float(s.get("fire_weather_index", 20.0))
    frp = float(s.get("frp_total", 0.0))
    base = fwi / 100.0
    frp_boost = min(math.log1p(frp) / 8.0, 0.25)
    return float(np.clip(base + frp_boost, 0.0, 1.0))


def _flood_score(s: dict) -> float:
    precip = float(s.get("precip", 0.0))      # mm
    rh     = float(s.get("rh_max", 50.0))     # %
    fwi    = float(s.get("fire_weather_index", 20.0))
    # High precip + high humidity + low fire-weather → flood risk
    p_score  = 1 / (1 + math.exp(-0.15 * (precip - 10)))  # sigmoid, inflects at 10mm
    rh_score = max(0.0, rh - 40) / 60.0                    # 0 below 40%, 1 at 100%
    dry_inv  = max(0.0, 1.0 - fwi / 80.0)                 # fires are anti-flood
    return float(np.clip(p_score * rh_score * dry_inv * 1.5, 0.0, 1.0))


def _seismic_score(s: dict) -> float:
    mag   = float(s.get("eq_magnitude", 0.0))
    depth = float(s.get("eq_depth_km",  50.0))
    sig   = float(s.get("eq_sig",        0.0))
    if mag < 2.5:
        return 0.0
    mag_score   = (mag - 2.5) / 6.5              # 0 at M2.5, 1 at M9
    depth_mult  = max(0.2, 1.0 - depth / 200.0)  # shallow = more dangerous
    sig_boost   = min(sig / 800.0, 0.2)
    return float(np.clip(mag_score * depth_mult + sig_boost, 0.0, 1.0))


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_fire_samples() -> list[dict]:
    from datasets.firms_weather import load_as_training_samples
    samples = load_as_training_samples(max_clusters=400, fetch_weather=True)
    for s in samples:
        s["fire_score"]    = _fire_score(s)
        s["flood_score"]   = _flood_score(s)
        s["seismic_score"] = 0.0
        s["doy"]           = DOY
    logger.info("[Compound] Loaded %d fire samples (FIRMS+OpenMeteo)", len(samples))
    return samples


def _load_seismic_samples(days: int = 30, limit: int = 500) -> list[dict]:
    from datetime import timedelta
    end   = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    url   = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query"
        f"?format=geojson&minmagnitude=2.5&orderby=magnitude"
        f"&starttime={start}&endtime={end}&limit={limit}"
    )
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "HeliOS/1.0"})
        resp.raise_for_status()
        features = resp.json().get("features", [])
    except Exception as e:
        logger.warning("[Compound] USGS fetch failed: %s", e)
        return []

    rng = random.Random(42)
    samples = []
    for feat in features:
        props  = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [0, 0, 10])
        lon, lat, depth = float(coords[0]), float(coords[1]), float(coords[2] if len(coords) > 2 else 10)

        mag = float(props.get("mag") or 0.0)
        sig = int(props.get("sig") or 0)
        ts  = props.get("time", 0)
        doy = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).timetuple().tm_yday if ts else DOY

        # Rough weather proxies for the quake location (no OpenMeteo call — seismic dominates)
        temp  = 15 + abs(lat) * -0.2 + rng.gauss(5, 5)  # crude lat→temp proxy
        s = {
            "lat":                 lat,
            "lon":                 lon,
            "doy":                 doy,
            "temp_max":            round(temp, 1),
            "rh_max":              round(rng.gauss(55, 20), 1),
            "wind_max":            round(abs(rng.gauss(15, 8)), 1),
            "vpd_max":             round(max(0.1, rng.gauss(1.2, 0.6)), 2),
            "precip":              round(max(0.0, rng.gauss(2, 4)), 2),
            "et0":                 round(max(0.1, rng.gauss(4, 2)), 2),
            "fire_weather_index":  round(max(0, rng.gauss(25, 15)), 1),
            "frp_total":           0.0,
            "hotspot_count":       0,
            "eq_magnitude":        mag,
            "eq_depth_km":         depth,
            "eq_sig":              sig,
        }
        s["fire_score"]    = _fire_score(s)
        s["flood_score"]   = _flood_score(s)
        s["seismic_score"] = _seismic_score(s)
        samples.append(s)

    logger.info("[Compound] Loaded %d seismic samples (USGS M2.5+ past %d days)", len(samples), days)
    return samples


def _synthetic_flood_samples(n: int = 300) -> list[dict]:
    """Heavy precipitation events as flood training samples."""
    rng = random.Random(99)
    flood_regions = [
        (23.0, 90.0),   # Bangladesh
        (10.0, 6.0),    # Nigeria
        (-15.0, -60.0), # Amazon basin
        (30.0, 105.0),  # Southeast China
        (15.0, 75.0),   # India monsoon belt
        (52.0, 10.0),   # Central Europe
        (-34.0, 150.0), # Eastern Australia
    ]
    samples = []
    for _ in range(n):
        lat_base, lon_base = rng.choice(flood_regions)
        lat = lat_base + rng.uniform(-8, 8)
        lon = lon_base + rng.uniform(-8, 8)
        doy = rng.randint(1, 365)
        precip = max(5, rng.gauss(50, 30))   # heavy rain
        rh     = min(100, max(60, rng.gauss(85, 8)))
        temp   = rng.gauss(22, 8)
        fwi    = max(0, rng.gauss(10, 8))    # wet → low FWI
        s = {
            "lat":                 round(lat, 2),
            "lon":                 round(lon, 2),
            "doy":                 doy,
            "temp_max":            round(temp, 1),
            "rh_max":              round(rh, 1),
            "wind_max":            round(abs(rng.gauss(20, 10)), 1),
            "vpd_max":             round(max(0.1, rng.gauss(0.5, 0.3)), 2),
            "precip":              round(precip, 1),
            "et0":                 round(max(0.1, rng.gauss(3, 1.5)), 2),
            "fire_weather_index":  round(fwi, 1),
            "frp_total":           0.0,
            "hotspot_count":       0,
            "eq_magnitude":        0.0,
            "eq_depth_km":         0.0,
            "eq_sig":              0,
        }
        s["fire_score"]    = _fire_score(s)
        s["flood_score"]   = _flood_score(s)
        s["seismic_score"] = 0.0
        samples.append(s)
    logger.info("[Compound] Generated %d synthetic flood samples", len(samples))
    return samples


def _synthetic_baseline(n: int = 400) -> list[dict]:
    """Low-risk background locations — stable weather, no active hazards."""
    rng = random.Random(7)
    samples = []
    for _ in range(n):
        lat = rng.uniform(-60, 70)
        lon = rng.uniform(-180, 180)
        doy = rng.randint(1, 365)
        s = {
            "lat": round(lat, 2),
            "lon": round(lon, 2),
            "doy": doy,
            "temp_max":            round(rng.gauss(18, 10), 1),
            "rh_max":              round(min(100, max(20, rng.gauss(55, 20))), 1),
            "wind_max":            round(abs(rng.gauss(12, 6)), 1),
            "vpd_max":             round(max(0.1, rng.gauss(1.0, 0.5)), 2),
            "precip":              round(max(0.0, rng.gauss(3, 4)), 2),
            "et0":                 round(max(0.1, rng.gauss(4, 2)), 2),
            "fire_weather_index":  round(max(0, rng.gauss(20, 12)), 1),
            "frp_total":           0.0,
            "hotspot_count":       0,
            "eq_magnitude":        0.0,
            "eq_depth_km":         0.0,
            "eq_sig":              0,
        }
        s["fire_score"]    = _fire_score(s)
        s["flood_score"]   = _flood_score(s)
        s["seismic_score"] = 0.0
        samples.append(s)
    logger.info("[Compound] Generated %d synthetic baseline samples", len(samples))
    return samples


# ── Training ──────────────────────────────────────────────────────────────────

def train() -> None:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_score
    import joblib

    sys.path.insert(0, str(Path(__file__).parent))

    fire_samples     = _load_fire_samples()
    seismic_samples  = _load_seismic_samples(days=30, limit=500)
    flood_samples    = _synthetic_flood_samples(n=300)
    baseline_samples = _synthetic_baseline(n=400)

    all_samples = fire_samples + seismic_samples + flood_samples + baseline_samples
    logger.info("[Compound] Total: %d samples (%d fire, %d seismic, %d flood, %d baseline)",
                len(all_samples), len(fire_samples), len(seismic_samples),
                len(flood_samples), len(baseline_samples))

    X   = np.array([_featurise(s) for s in all_samples], dtype=np.float32)
    y_f = np.array([s["fire_score"]    for s in all_samples], dtype=np.float32)
    y_l = np.array([s["flood_score"]   for s in all_samples], dtype=np.float32)
    y_s = np.array([s["seismic_score"] for s in all_samples], dtype=np.float32)
    y_c = np.clip(np.maximum(np.maximum(y_f, y_l), y_s) * 100, 0, 100).astype(np.float32)

    valid = ~np.isnan(X).any(axis=1)
    X, y_f, y_l, y_s, y_c = X[valid], y_f[valid], y_l[valid], y_s[valid], y_c[valid]
    logger.info("[Compound] Feature matrix: %s | compound score mean=%.1f std=%.1f",
                X.shape, y_c.mean(), y_c.std())

    def _gbt(seed=42):
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.85, min_samples_leaf=3, random_state=seed,
        )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    models  = {}

    for name, y_target in [
        ("fire_risk",     y_f),
        ("flood_risk",    y_l),
        ("seismic_risk",  y_s),
        ("compound",      y_c / 100.0),  # train on 0-1, scale output ×100
    ]:
        model = _gbt()
        scores = cross_val_score(model, X, y_target, cv=cv,
                                 scoring="neg_mean_absolute_error")
        mae = -scores.mean()
        model.fit(X, y_target)
        results[name] = {"mae_cv": round(float(mae), 4)}
        models[name]  = model
        importances = sorted(zip(FEATURE_NAMES, model.feature_importances_),
                             key=lambda x: -x[1])
        results[name]["top5"] = {n: round(float(v), 4) for n, v in importances[:5]}
        logger.info("[Compound] %-16s CV MAE=%.4f | top: %s",
                    name, mae,
                    ", ".join(f"{n}={v:.3f}" for n, v in importances[:3]))

    # ── Save ──────────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["fire_risk"],    MODELS_DIR / "fire_risk_regressor.joblib")
    joblib.dump(models["flood_risk"],   MODELS_DIR / "flood_risk_regressor.joblib")
    joblib.dump(models["seismic_risk"], MODELS_DIR / "seismic_risk_regressor.joblib")
    joblib.dump(models["compound"],     MODELS_DIR / "compound_hazard_scorer.joblib")

    data_sources = list({s.get("source", "synthetic") for s in all_samples})
    meta = {
        "trained_at":   datetime.now(timezone.utc).isoformat(),
        "model":        "GradientBoostingRegressor × 4",
        "n_samples":    int(len(X)),
        "n_real":       len(fire_samples) + len(seismic_samples),
        "n_features":   N_FEAT,
        "features":     FEATURE_NAMES,
        "data_sources": data_sources,
        "sample_counts": {
            "fire_viirs":   len(fire_samples),
            "seismic_usgs": len(seismic_samples),
            "flood_synth":  len(flood_samples),
            "baseline":     len(baseline_samples),
        },
        "models": {
            "fire_risk_regressor":    "fire_risk_regressor.joblib",
            "flood_risk_regressor":   "flood_risk_regressor.joblib",
            "seismic_risk_regressor": "seismic_risk_regressor.joblib",
            "compound_hazard_scorer": "compound_hazard_scorer.joblib (output ×100 = 0–100 score)",
        },
        "metrics": results,
        "compound_formula": "max(fire_score, flood_score, seismic_score) × 100",
        "usage": (
            "import joblib, numpy as np; "
            "m = joblib.load('compound_hazard_scorer.joblib'); "
            "score = float(np.clip(m.predict([features])[0] * 100, 0, 100))"
        ),
    }
    (MODELS_DIR / "compound_hazard_scorer_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("[Compound] All 4 models saved to %s", MODELS_DIR)
    logger.info("[Compound] Compound scorer MAE=%.2f points (0-100 scale)",
                results["compound"]["mae_cv"] * 100)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.path.insert(0, str(Path(__file__).parent))
    train()
