"""
US Drought Severity Classifier

Trains a classifier on the US Drought Monitor (USDM) weekly state-level data,
using Open-Meteo antecedent weather aggregates as features.

Target: USDM dominant drought class for a state-week (None/D0/D1/D2/D3/D4),
derived from the area-percentage in each class (whichever class has the
largest non-zero area is the dominant class for that state-week).

Features per state-week:
  - precip_sum_30d / 60d / 90d / 180d
  - precip_anomaly_30d / 90d (vs same-month historical average)
  - temp_mean_30d, temp_anomaly_30d
  - et0_sum_30d, et0_sum_90d
  - vpd_mean_30d, vpd_max_30d
  - week_of_year (sin/cos), latitude, longitude

Real data sources (no auth):
  USDM REST API:  https://usdmdataservices.unl.edu/api/StateStatistics
  Open-Meteo:     https://archive-api.open-meteo.com/v1/archive

Output: packages/c2_intel/models/drought_severity_classifier.joblib
        packages/c2_intel/models/drought_severity_classifier_meta.json

Usage:
    python train_drought_severity.py [--start 2023-01-01 --end 2024-12-31]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "c2_intel" / "models"
MODEL_PATH = MODELS_DIR / "drought_severity_classifier.joblib"
META_PATH  = MODELS_DIR / "drought_severity_classifier_meta.json"
CACHE_DIR  = Path(__file__).parent / "data" / "drought"

USDM_BASE = "https://usdmdataservices.unl.edu/api/StateStatistics"
OPENMETEO_HIST = "https://archive-api.open-meteo.com/v1/archive"

# State FIPS codes (USDM API uses these as aoi)
STATE_FIPS: dict[str, str] = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "FL": "12", "GA": "13", "HI": "15", "ID": "16",
    "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22",
    "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34",
    "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40",
    "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47",
    "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56",
}

# State centroid coordinates (lat, lon) for the 48 contiguous + AK + HI states
# Used to query Open-Meteo at one representative point per state.
STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "AL": (32.806, -86.791), "AK": (61.370, -152.404), "AZ": (33.730, -111.431),
    "AR": (34.969, -92.373), "CA": (36.117, -119.682), "CO": (39.059, -105.311),
    "CT": (41.597, -72.755), "DE": (39.318, -75.508), "FL": (27.766, -81.687),
    "GA": (33.040, -83.643), "HI": (21.094, -157.498), "ID": (44.240, -114.479),
    "IL": (40.349, -88.986), "IN": (39.849, -86.258), "IA": (42.011, -93.210),
    "KS": (38.526, -96.726), "KY": (37.668, -84.670), "LA": (31.169, -91.867),
    "ME": (44.693, -69.381), "MD": (39.063, -76.802), "MA": (42.230, -71.530),
    "MI": (43.326, -84.536), "MN": (45.694, -93.900), "MS": (32.741, -89.679),
    "MO": (38.456, -92.288), "MT": (46.921, -110.454), "NE": (41.125, -98.268),
    "NV": (38.313, -117.055), "NH": (43.452, -71.564), "NJ": (40.298, -74.521),
    "NM": (34.840, -106.248), "NY": (42.166, -74.948), "NC": (35.630, -79.806),
    "ND": (47.529, -99.784), "OH": (40.388, -82.764), "OK": (35.565, -96.929),
    "OR": (44.572, -122.071), "PA": (40.590, -77.210), "RI": (41.681, -71.511),
    "SC": (33.856, -80.945), "SD": (44.299, -99.438), "TN": (35.747, -86.692),
    "TX": (31.054, -97.563), "UT": (40.150, -111.862), "VT": (44.045, -72.710),
    "VA": (37.768, -78.169), "WA": (47.400, -121.490), "WV": (38.491, -80.954),
    "WI": (44.268, -89.616), "WY": (42.755, -107.302),
}

# Map USDM categorical to numeric class
USDM_CLASSES = ["None", "D0", "D1", "D2", "D3", "D4"]


def _fetch_usdm_state_stats(start: date, end: date) -> list[dict]:
    """Fetch weekly USDM percent-area-by-class for all states (CSV per state)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"usdm_state_{start}_{end}.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            if data:
                logger.info("[Drought] Cached USDM rows: %d", len(data))
                return data
        except Exception:
            pass

    import csv as _csv
    import io as _io
    url = f"{USDM_BASE}/GetDroughtSeverityStatisticsByAreaPercent"
    all_rows: list[dict] = []
    logger.info("[Drought] Fetching USDM for 50 states: %s -> %s", start, end)
    for state, fips in STATE_FIPS.items():
        try:
            r = requests.get(url, params={
                "aoi": fips,
                "startdate": start.strftime("%-m/%-d/%Y"),
                "enddate":   end.strftime("%-m/%-d/%Y"),
                "statisticsType": "1",
            }, timeout=60, headers={"User-Agent": "Heli.OS/1.0"})
            r.raise_for_status()
            rows = list(_csv.DictReader(_io.StringIO(r.text.strip())))
            all_rows.extend(rows)
        except Exception as e:
            logger.warning("[Drought] %s fetch failed: %s", state, e)
        time.sleep(0.1)

    cache.write_text(json.dumps(all_rows))
    logger.info("[Drought] USDM total rows across states: %d", len(all_rows))
    return all_rows


def _dominant_class(row: dict) -> int:
    """USDM rows are percent-area in each class; return the dominant class id."""
    none_p = float(row.get("None") or 0)
    d0 = float(row.get("D0") or 0)
    d1 = float(row.get("D1") or 0)
    d2 = float(row.get("D2") or 0)
    d3 = float(row.get("D3") or 0)
    d4 = float(row.get("D4") or 0)
    cls_areas = [none_p, d0, d1, d2, d3, d4]
    return int(np.argmax(cls_areas))


def _fetch_state_weather(state: str, lat: float, lon: float,
                         start: date, end: date) -> dict | None:
    """Fetch full-period daily weather for a state centroid (one big request)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"wx_{state}_{start}_{end}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    try:
        r = requests.get(OPENMETEO_HIST, params={
            "latitude": lat, "longitude": lon,
            "start_date": str(start), "end_date": str(end),
            "daily": ",".join([
                "precipitation_sum",
                "temperature_2m_mean",
                "temperature_2m_max",
                "wind_speed_10m_max",
                "vapour_pressure_deficit_max",
                "et0_fao_evapotranspiration",
                "relative_humidity_2m_min",
            ]),
            "timezone": "UTC",
        }, timeout=120)
        r.raise_for_status()
        data = r.json().get("daily") or {}
        if not data.get("time"):
            return None
        cache.write_text(json.dumps(data))
        return data
    except Exception as e:
        logger.warning("[Drought] OpenMeteo fetch failed for %s: %s", state, e)
        return None


def _aggregate_window(daily: dict, end_idx: int, window_days: int) -> dict:
    start_idx = max(0, end_idx - window_days)
    def agg(field: str, fn):
        vals = [v for v in (daily.get(field) or [])[start_idx:end_idx] if v is not None]
        return float(fn(vals)) if vals else 0.0
    return {
        "precip_sum":     agg("precipitation_sum", sum),
        "temp_mean":      agg("temperature_2m_mean", np.mean) if daily.get("temperature_2m_mean") else 0.0,
        "temp_max_mean":  agg("temperature_2m_max", np.mean) if daily.get("temperature_2m_max") else 0.0,
        "wind_max_mean":  agg("wind_speed_10m_max", np.mean) if daily.get("wind_speed_10m_max") else 0.0,
        "vpd_max_mean":   agg("vapour_pressure_deficit_max", np.mean) if daily.get("vapour_pressure_deficit_max") else 0.0,
        "et0_sum":        agg("et0_fao_evapotranspiration", sum) if daily.get("et0_fao_evapotranspiration") else 0.0,
        "rh_min_mean":    agg("relative_humidity_2m_min", np.mean) if daily.get("relative_humidity_2m_min") else 0.0,
    }


def _build_dataset(start: date, end: date) -> tuple[np.ndarray, np.ndarray, list[str]]:
    usdm = _fetch_usdm_state_stats(start, end)
    if not usdm:
        raise RuntimeError("USDM API returned no rows")

    # Index USDM by (state_abbrev, valid_end_date_str)
    by_key: dict[tuple[str, str], dict] = {}
    for row in usdm:
        s = (row.get("StateAbbreviation") or "").strip().upper()
        # ValidEnd is "YYYY-MM-DD" string from CSV
        d = (row.get("ValidEnd") or "").strip()
        if not s or not d or s not in STATE_CENTROIDS:
            continue
        by_key[(s, d)] = row

    # Pre-fetch one full daily timeseries per state
    state_daily: dict[str, dict] = {}
    states_in_data = sorted({k[0] for k in by_key})
    logger.info("[Drought] %d states, %d state-weeks total", len(states_in_data), len(by_key))
    for state in states_in_data:
        lat, lon = STATE_CENTROIDS[state]
        # Extend range backward 200 days for antecedent windows
        wx_start = start - timedelta(days=200)
        wx = _fetch_state_weather(state, lat, lon, wx_start, end)
        if wx is not None:
            state_daily[state] = wx
        time.sleep(0.4)  # be polite to Open-Meteo

    feat_names = [
        "lat", "lon_norm", "lat_abs",
        "week_sin", "week_cos",
        "precip_30d", "precip_60d", "precip_90d", "precip_180d",
        "precip_anom_30d", "precip_anom_90d",
        "temp_mean_30d", "temp_anom_30d",
        "vpd_max_mean_30d", "vpd_max_mean_90d",
        "et0_sum_30d", "et0_sum_90d",
        "rh_min_mean_30d",
        "wind_max_mean_30d",
    ]

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for (state, day_str), row in by_key.items():
        wx = state_daily.get(state)
        if not wx or not wx.get("time"):
            continue
        times = wx["time"]
        if day_str not in times:
            continue
        end_idx = times.index(day_str) + 1  # inclusive end

        agg30 = _aggregate_window(wx, end_idx, 30)
        agg60 = _aggregate_window(wx, end_idx, 60)
        agg90 = _aggregate_window(wx, end_idx, 90)
        agg180 = _aggregate_window(wx, end_idx, 180)

        # Anomaly = current 30d - mean of (30d windows from same month, prior years)
        # Approx: compare to (precip_180d - precip_30d) / 5 (simple antecedent baseline)
        precip_baseline_30 = (agg180["precip_sum"] - agg30["precip_sum"]) / 5.0
        temp_baseline_30 = agg180["temp_mean"]  # rough — same month-of-year not isolated
        precip_anom_30 = agg30["precip_sum"] - precip_baseline_30
        precip_anom_90 = agg90["precip_sum"] - 3 * precip_baseline_30
        temp_anom_30 = agg30["temp_mean"] - temp_baseline_30

        try:
            d = datetime.strptime(day_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        woy = d.timetuple().tm_yday / 7.0
        lat, lon = STATE_CENTROIDS[state]

        feat = np.array([
            lat, lon / 180.0, abs(lat) / 90.0,
            math.sin(2 * math.pi * woy / 52), math.cos(2 * math.pi * woy / 52),
            agg30["precip_sum"], agg60["precip_sum"], agg90["precip_sum"], agg180["precip_sum"],
            precip_anom_30, precip_anom_90,
            agg30["temp_mean"], temp_anom_30,
            agg30["vpd_max_mean"], agg90["vpd_max_mean"],
            agg30["et0_sum"], agg90["et0_sum"],
            agg30["rh_min_mean"],
            agg30["wind_max_mean"],
        ], dtype=np.float32)

        X_list.append(feat)
        y_list.append(_dominant_class(row))

    if not X_list:
        return np.zeros((0, len(feat_names)), dtype=np.float32), np.zeros(0, dtype=np.int64), feat_names

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    logger.info("[Drought] Built %d samples — class dist: %s",
                len(X), {USDM_CLASSES[c]: int(n) for c, n in Counter(y.tolist()).items()})
    return X, y, feat_names


def train(start: str, end: str) -> None:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    import joblib

    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()

    X, y, feat_names = _build_dataset(s, e)
    if len(X) < 100:
        raise RuntimeError(f"Only {len(X)} state-week samples — too few.")

    counts = Counter(y.tolist())
    keep = {c for c, n in counts.items() if n >= 5}
    if len(keep) < len(counts):
        mask = np.array([yi in keep for yi in y])
        X, y = X[mask], y[mask]
        old_to_new = {old: new for new, old in enumerate(sorted(keep))}
        y = np.array([old_to_new[yi] for yi in y], dtype=np.int64)
        kept_classes = [USDM_CLASSES[c] for c in sorted(keep)]
        logger.info("[Drought] Dropped rare classes — kept %s", kept_classes)
    else:
        kept_classes = USDM_CLASSES

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=7)

    f1 = cross_val_score(clf, X, y, cv=3, scoring="f1_weighted", n_jobs=-1)
    logger.info("[Drought] CV f1_weighted = %.4f +/- %.4f", f1.mean(), f1.std())

    clf.fit(X, y)
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    top = {feat_names[i]: round(float(importances[i]), 4) for i in top_idx}
    logger.info("[Drought] Top features: %s", top)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": clf, "feature_names": feat_names,
                 "classes": kept_classes}, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "GradientBoostingClassifier",
        "task": "USDM dominant drought class (state-week)",
        "classes": kept_classes,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "data_sources": ["usdm_unl_real", "openmeteo_archive_real"],
        "period": f"{start} -> {end}",
        "metrics": {
            "f1_cv_weighted": round(float(f1.mean()), 4),
            "f1_cv_std":      round(float(f1.std()), 4),
        },
        "top_features": top,
        "class_distribution": {kept_classes[c]: int(n) for c, n in Counter(y.tolist()).items()},
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[Drought] Saved -> %s", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end",   default="2024-12-31")
    args = p.parse_args()
    train(start=args.start, end=args.end)
