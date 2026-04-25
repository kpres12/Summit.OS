"""
NOAA Storm Severity Model

Predicts severity class (none / minor / moderate / severe / catastrophic) for
NOAA Storm Events using rich tabular features:
  event_type, state, magnitude, magnitude_type, duration_hours,
  injuries_direct, injuries_indirect, deaths_direct, deaths_indirect,
  damage_property_usd, damage_crops_usd, lat, lon, month, hour_of_day,
  begin_range_mi, tor_f_scale (categorical), flood_cause (categorical)

Target: severity class derived from casualties + property damage:
  catastrophic: deaths>=10 OR property>=$10M
  severe       : deaths>=1 OR property>=$1M OR injuries>=10
  moderate    : property>=$100k OR injuries>=1
  minor       : property>0 OR magnitude > 0
  none        : recorded but no impact

Data: NOAA Storm Events Database, multi-year CSV from NCEI.
License: US Government work, public domain.

Output: packages/c2_intel/models/storm_severity_classifier.joblib
        packages/c2_intel/models/storm_severity_classifier_meta.json

Usage:
    python train_storm_severity.py [--years 2022 2023 2024]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import logging
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "c2_intel" / "models"
MODEL_PATH = MODELS_DIR / "storm_severity_classifier.joblib"
META_PATH  = MODELS_DIR / "storm_severity_classifier_meta.json"
CACHE_DIR  = Path(__file__).parent / "data" / "noaa_storms"

NCEI_BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

SEVERITY_CLASSES = ["none", "minor", "moderate", "severe", "catastrophic"]


def _parse_money(s: str) -> float:
    if not s:
        return 0.0
    s = s.strip().upper().replace("$", "").replace(",", "")
    if not s:
        return 0.0
    mult = 1.0
    if s.endswith("K"):
        mult, s = 1e3, s[:-1]
    elif s.endswith("M"):
        mult, s = 1e6, s[:-1]
    elif s.endswith("B"):
        mult, s = 1e9, s[:-1]
    elif s.endswith("H"):
        mult, s = 1e2, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return 0.0


def _severity_from_impact(deaths: int, injuries: int, prop_usd: float) -> int:
    """0..4 severity class."""
    if deaths >= 10 or prop_usd >= 10_000_000:
        return 4
    if deaths >= 1 or prop_usd >= 1_000_000 or injuries >= 10:
        return 3
    if prop_usd >= 100_000 or injuries >= 1:
        return 2
    if prop_usd > 0:
        return 1
    return 0


def _parse_datetime(date_str: str, time_str: str) -> Optional[datetime]:
    """Parse NOAA BEGIN_DATE_TIME, BEGIN_YEARMONTH+BEGIN_DAY+BEGIN_TIME."""
    if not date_str:
        return None
    fmts = [
        "%d-%b-%y %H:%M:%S",   # 12-MAY-23 14:30:00
        "%d-%b-%y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for f in fmts:
        try:
            return datetime.strptime(date_str.strip(), f)
        except ValueError:
            pass
    return None


def _list_year_files(year: int) -> list[str]:
    """Find StormEvents_details*_d{year}_c*.csv.gz on the NCEI index."""
    try:
        idx = requests.get(NCEI_BASE, timeout=30,
                           headers={"User-Agent": "Heli.OS/1.0"}).text
    except Exception as e:
        logger.warning("[Storm] Index fetch failed: %s", e)
        return []
    matches = re.findall(
        rf"StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz", idx)
    return sorted(set(matches))


def _download_year(year: int) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"storms_{year}.csv"
    if cache.exists():
        with cache.open("r", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
        logger.info("[Storm] Cached %d rows for %d", len(rows), year)
        return rows

    files = _list_year_files(year)
    if not files:
        logger.warning("[Storm] No NCEI file for %d", year)
        return []
    url = NCEI_BASE + files[-1]
    logger.info("[Storm] Downloading %s", url)
    r = requests.get(url, timeout=120, headers={"User-Agent": "Heli.OS/1.0"})
    r.raise_for_status()
    text = gzip.decompress(r.content).decode("utf-8", errors="replace")
    cache.write_text(text)
    rows = list(csv.DictReader(io.StringIO(text)))
    logger.info("[Storm] Parsed %d rows for %d", len(rows), year)
    return rows


def _featurize(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Convert raw NOAA records to (X, y, feature_names, event_type_vocab)."""
    # First pass: vocabularies
    event_type_vocab: list[str] = []
    seen: set[str] = set()
    for rec in rows:
        et = (rec.get("EVENT_TYPE") or "").strip().lower()
        if et and et not in seen:
            seen.add(et)
            event_type_vocab.append(et)
    event_type_vocab = sorted(event_type_vocab)
    et_index = {e: i for i, e in enumerate(event_type_vocab)}

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for rec in rows:
        try:
            lat = float(rec.get("BEGIN_LAT") or 0)
            lon = float(rec.get("BEGIN_LON") or 0)
        except ValueError:
            continue
        if lat == 0 and lon == 0:
            continue

        deaths = int(rec.get("DEATHS_DIRECT") or 0) + int(rec.get("DEATHS_INDIRECT") or 0)
        injuries = int(rec.get("INJURIES_DIRECT") or 0) + int(rec.get("INJURIES_INDIRECT") or 0)
        prop = _parse_money(rec.get("DAMAGE_PROPERTY", "")) + _parse_money(rec.get("DAMAGE_CROPS", ""))

        # Magnitude
        mag = 0.0
        try:
            mag = float(rec.get("MAGNITUDE") or 0)
        except ValueError:
            pass

        # Tornado scale
        tor = (rec.get("TOR_F_SCALE") or "").strip().upper()
        tor_num = 0
        if tor.startswith("EF"):
            try: tor_num = int(tor[2:])
            except ValueError: pass
        elif tor.startswith("F"):
            try: tor_num = int(tor[1:])
            except ValueError: pass

        # Duration
        bdt = _parse_datetime(rec.get("BEGIN_DATE_TIME") or "", "")
        edt = _parse_datetime(rec.get("END_DATE_TIME") or "", "")
        duration_h = 0.0
        if bdt and edt and edt > bdt:
            duration_h = (edt - bdt).total_seconds() / 3600.0

        # Time-of-year, hour-of-day cyclic encoding
        if bdt:
            doy = bdt.timetuple().tm_yday
            day_sin = math.sin(2 * math.pi * doy / 365.25)
            day_cos = math.cos(2 * math.pi * doy / 365.25)
            hr_sin = math.sin(2 * math.pi * bdt.hour / 24)
            hr_cos = math.cos(2 * math.pi * bdt.hour / 24)
        else:
            day_sin = day_cos = hr_sin = hr_cos = 0.0

        et = (rec.get("EVENT_TYPE") or "").strip().lower()
        et_idx = et_index.get(et, -1)
        et_one_hot = np.zeros(len(event_type_vocab), dtype=np.float32)
        if et_idx >= 0:
            et_one_hot[et_idx] = 1.0

        # Numeric features (note: deaths/injuries/prop are NOT included as features
        # since severity target is derived from them — that would be label leakage)
        numeric = np.array([
            mag,
            tor_num,
            min(duration_h, 240.0),     # cap
            lat,
            lon / 180.0,
            day_sin, day_cos, hr_sin, hr_cos,
        ], dtype=np.float32)

        X_list.append(np.concatenate([numeric, et_one_hot]))
        y_list.append(_severity_from_impact(deaths, injuries, prop))

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)

    feat_names = [
        "magnitude", "tor_scale", "duration_h", "lat", "lon_norm",
        "day_sin", "day_cos", "hour_sin", "hour_cos",
    ] + [f"event_type::{e}" for e in event_type_vocab]
    return X, y, feat_names, event_type_vocab


def train(years: list[int], device_str: str = "auto") -> None:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib

    all_rows: list[dict] = []
    for y in years:
        all_rows.extend(_download_year(y))
    if not all_rows:
        raise RuntimeError("No NOAA storm rows downloaded.")
    logger.info("[Storm] Total rows across %d years: %d", len(years), len(all_rows))

    X, y, feat_names, et_vocab = _featurize(all_rows)
    logger.info("[Storm] Features: %d  | classes: %s", X.shape[1],
                dict(Counter(y.tolist())))

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=7)),
    ])

    # Subsample for speed if huge
    if len(X) > 80_000:
        rng = np.random.default_rng(7)
        idx = rng.choice(len(X), 80_000, replace=False)
        X, y = X[idx], y[idx]
        logger.info("[Storm] Subsampled to %d rows for training", len(X))

    f1 = cross_val_score(pipe, X, y, cv=3, scoring="f1_weighted", n_jobs=-1)
    logger.info("[Storm] CV f1_weighted = %.4f +/- %.4f", f1.mean(), f1.std())

    pipe.fit(X, y)

    # Per-feature importance — need to drill into the classifier
    importances = pipe.named_steps["clf"].feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    top = {feat_names[i]: round(float(importances[i]), 4) for i in top_idx}
    logger.info("[Storm] Top 10 features: %s", top)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "feature_names": feat_names,
                 "event_type_vocab": et_vocab,
                 "classes": SEVERITY_CLASSES}, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "GradientBoostingClassifier",
        "task": "Storm severity (5-class)",
        "classes": SEVERITY_CLASSES,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_event_types": len(et_vocab),
        "years": years,
        "data_source": "noaa_storm_events_real",
        "metrics": {
            "f1_cv_weighted": round(float(f1.mean()), 4),
            "f1_cv_weighted_std": round(float(f1.std()), 4),
        },
        "top_features": top,
        "class_distribution": dict(Counter(y.tolist())),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[Storm] Saved -> %s", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024])
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    train(years=args.years, device_str=args.device)
