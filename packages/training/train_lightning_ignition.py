"""
Lightning-Caused Fire Ignition Risk Model

Trains a binary classifier: given a lightning strike location, predict the
probability that a wildfire hotspot appears within 25 km / 72 hours.

Real data sources (no auth):
  NOAA Storm Events Database — events with type LIGHTNING (date, lat, lon, casualties)
  NASA FIRMS MODIS 7-day CSV — recent hotspots with lat, lon, acq_date, FRP
  Open-Meteo Historical API — daily weather (T, RH, wind, VPD, FWI, precip)

Training shape:
  Each lightning event becomes a sample. Label = 1 if any FIRMS hotspot
  exists within 25 km and the next 72 h, else 0. Features:
    fuel/weather (post-strike): temp_max, rh_min, wind_max, vpd, fwi, precip
    geo:                        lat, lon (sin/cos), lat_abs
    time:                       day_sin, day_cos, hour_sin, hour_cos
    lightning meta:             casualties_indicator, strike_count_in_state

Output: packages/c2_intel/models/lightning_ignition_classifier.joblib
        packages/c2_intel/models/lightning_ignition_classifier_meta.json

Usage:
    python train_lightning_ignition.py [--years 2022 2023 2024]
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
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "c2_intel" / "models"
MODEL_PATH = MODELS_DIR / "lightning_ignition_classifier.joblib"
META_PATH  = MODELS_DIR / "lightning_ignition_classifier_meta.json"
CACHE_DIR  = Path(__file__).parent / "data" / "lightning_ignition"
NOAA_CACHE = Path(__file__).parent / "data" / "noaa_storms"  # reuse storm cache

NCEI_BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
FIRMS_MODIS_URL = (
    "https://firms.modaps.eosdis.nasa.gov/data/active_fire/"
    "modis-c6.1/csv/MODIS_C6_1_Global_7d.csv"
)
OPENMETEO_HIST = "https://archive-api.open-meteo.com/v1/archive"

EARTH_R_KM = 6371.0


def _list_year_files(year: int) -> list[str]:
    try:
        text = requests.get(NCEI_BASE, timeout=30,
                            headers={"User-Agent": "Heli.OS/1.0"}).text
    except Exception as e:
        logger.warning("[Lightning] NCEI index failed: %s", e)
        return []
    return sorted(set(re.findall(
        rf"StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz", text)))


def _download_year(year: int) -> list[dict]:
    NOAA_CACHE.mkdir(parents=True, exist_ok=True)
    cache = NOAA_CACHE / f"storms_{year}.csv"
    if cache.exists():
        with cache.open("r", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
        return rows
    files = _list_year_files(year)
    if not files:
        return []
    url = NCEI_BASE + files[-1]
    r = requests.get(url, timeout=120, headers={"User-Agent": "Heli.OS/1.0"})
    r.raise_for_status()
    text = gzip.decompress(r.content).decode("utf-8", errors="replace")
    cache.write_text(text)
    return list(csv.DictReader(io.StringIO(text)))


def _parse_dt(date_str: str) -> datetime | None:
    if not date_str:
        return None
    for f in ("%d-%b-%y %H:%M:%S", "%d-%b-%y %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str.strip(), f).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


FIRMS_API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
# CONUS bbox roughly (lon_min, lat_min, lon_max, lat_max)
CONUS_BBOX    = "-125,24,-66,49"


def _firms_key() -> str:
    """Read FIRMS_MAP_KEY from .env or env var. Never log the value."""
    import os
    key = os.environ.get("FIRMS_MAP_KEY") or os.environ.get("FIRMS_API_KEY") or ""
    if not key:
        # Try project .env
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("FIRMS_MAP_KEY=") or line.startswith("FIRMS_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        break
    return key


def _download_firms_archive(start: str, days_per_chunk: int = 10,
                            n_chunks: int = 1, source: str = "VIIRS_SNPP_SP",
                            bbox: str = CONUS_BBOX) -> list[dict]:
    """Download historical FIRMS hotspots over CONUS via the keyed API.

    The API returns up to 10 days per call; we chain calls forward in time.
    Result is cached per (source, start, days_per_chunk, n_chunks, bbox).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"firms_{source}_{start}_d{days_per_chunk}x{n_chunks}.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            if data:
                logger.info("[Lightning] Cached FIRMS-archive: %d hotspots", len(data))
                return data
        except Exception:
            pass

    key = _firms_key()
    if not key:
        logger.error("[Lightning] FIRMS_MAP_KEY not set — cannot fetch archive")
        return []

    rows: list[dict] = []
    cursor = datetime.strptime(start, "%Y-%m-%d").date()
    for chunk in range(n_chunks):
        url = f"{FIRMS_API_BASE}/{key}/{source}/{bbox}/{days_per_chunk}/{cursor}"
        try:
            r = requests.get(url, timeout=120, headers={"User-Agent": "Heli.OS/1.0"})
            r.raise_for_status()
            text = r.text
            reader = csv.DictReader(io.StringIO(text))
            n_added = 0
            for row in reader:
                try:
                    acq_time = row.get("acq_time", "0")
                    # acq_time is HHMM like "1247" — convert to hour
                    hh = int(acq_time[:2]) if len(acq_time) >= 2 else 0
                    rows.append({
                        "lat": float(row["latitude"]),
                        "lon": float(row["longitude"]),
                        "acq_date": row.get("acq_date", ""),
                        "acq_hour": hh,
                        "frp": float(row.get("frp", 0) or 0),
                        "confidence": str(row.get("confidence", "")),
                    })
                    n_added += 1
                except (ValueError, KeyError):
                    continue
            logger.info("[Lightning] FIRMS %s %s+%dd  -> %d hotspots (total %d)",
                        source, cursor, days_per_chunk, n_added, len(rows))
        except Exception as e:
            logger.warning("[Lightning] FIRMS chunk %s failed: %s", cursor, e)
        cursor = cursor + timedelta(days=days_per_chunk)

    cache.write_text(json.dumps(rows))
    return rows


def _fetch_weather(lat: float, lon: float, date: str) -> dict | None:
    """Open-Meteo historical archive — 1 day. Cached on disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"wx_{round(lat,2)}_{round(lon,2)}_{date}.json"
    cache = CACHE_DIR / key
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    try:
        r = requests.get(OPENMETEO_HIST, params={
            "latitude": lat, "longitude": lon,
            "start_date": date, "end_date": date,
            "daily": ",".join([
                "temperature_2m_max", "relative_humidity_2m_min",
                "wind_speed_10m_max", "vapour_pressure_deficit_max",
                "precipitation_sum", "et0_fao_evapotranspiration",
            ]),
            "timezone": "UTC",
        }, timeout=20)
        r.raise_for_status()
        data = r.json().get("daily") or {}
        if not data.get("time"):
            return None
        out = {
            "temp_max": (data.get("temperature_2m_max") or [None])[0],
            "rh_min":   (data.get("relative_humidity_2m_min") or [None])[0],
            "wind_max": (data.get("wind_speed_10m_max") or [None])[0],
            "vpd_max":  (data.get("vapour_pressure_deficit_max") or [None])[0],
            "precip":   (data.get("precipitation_sum") or [None])[0],
            "et0":      (data.get("et0_fao_evapotranspiration") or [None])[0],
        }
        cache.write_text(json.dumps(out))
        return out
    except Exception as e:
        logger.debug("[Lightning] weather fetch failed for %.3f,%.3f %s: %s",
                     lat, lon, date, e)
        return None


def _fwi_proxy(temp: float, rh: float, wind: float, precip: float) -> float:
    """Simple FWI-like fire weather index (0-100)."""
    if temp is None or rh is None or wind is None:
        return 0.0
    dryness = max(0.0, 100.0 - rh) / 100.0
    heat = max(0.0, temp - 5) / 40.0
    wind_factor = min(wind / 30.0, 1.0)
    rain_pen = math.exp(-(precip or 0.0) / 5.0)
    return 100.0 * dryness * (0.5 + 0.5 * heat) * (0.6 + 0.4 * wind_factor) * rain_pen


def _build_dataset(years: list[int],
                   match_radius_km: float = 25.0,
                   match_window_h: int = 72,
                   max_lightning: int = 3000,
                   skip_weather: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """For each lightning strike row, label by FIRMS proximity in next 72 h."""
    storm_rows: list[dict] = []
    for y in years:
        storm_rows.extend(_download_year(y))
    logger.info("[Lightning] Loaded %d storm rows total", len(storm_rows))

    lightning: list[dict] = []
    for r in storm_rows:
        if (r.get("EVENT_TYPE") or "").strip().lower() != "lightning":
            continue
        try:
            lat = float(r.get("BEGIN_LAT") or 0)
            lon = float(r.get("BEGIN_LON") or 0)
        except ValueError:
            continue
        if lat == 0 and lon == 0:
            continue
        dt = _parse_dt(r.get("BEGIN_DATE_TIME") or "")
        if dt is None:
            continue
        deaths = int(r.get("DEATHS_DIRECT") or 0) + int(r.get("DEATHS_INDIRECT") or 0)
        injuries = int(r.get("INJURIES_DIRECT") or 0) + int(r.get("INJURIES_INDIRECT") or 0)
        lightning.append({
            "lat": lat, "lon": lon, "dt": dt,
            "state": (r.get("STATE") or "").strip().upper(),
            "casualties": deaths + injuries,
        })
    logger.info("[Lightning] %d lightning events parsed", len(lightning))

    # Strike-count per state (rough activity prior)
    by_state = Counter(s["state"] for s in lightning)

    if max_lightning and len(lightning) > max_lightning:
        # Stratified sample — keep all lightning with casualties, sample the rest
        with_cas = [s for s in lightning if s["casualties"] > 0]
        without = [s for s in lightning if s["casualties"] == 0]
        rng = np.random.default_rng(7)
        n_take = max(0, max_lightning - len(with_cas))
        take_idx = rng.choice(len(without), size=min(n_take, len(without)), replace=False)
        lightning = with_cas + [without[i] for i in take_idx]
        rng.shuffle(lightning)
        logger.info("[Lightning] Subsampled to %d events (%d w/ casualties + %d random)",
                    len(lightning), len(with_cas), n_take)

    # Pull FIRMS archive covering the whole training year(s).
    # FIRMS area API caps day_range to [1..5]; use 5-day chunks × 73 chunks/year.
    firms: list[dict] = []
    for y in years:
        # VIIRS SNPP_SP: 375m, 2012-present. Best signal for ignition proximity.
        firms.extend(_download_firms_archive(
            start=f"{y}-01-01", days_per_chunk=5, n_chunks=73,
            source="VIIRS_SNPP_SP", bbox=CONUS_BBOX))
        # NOAA-20 SP doubles spatial coverage (different overpass)
        firms.extend(_download_firms_archive(
            start=f"{y}-01-01", days_per_chunk=5, n_chunks=73,
            source="VIIRS_NOAA20_SP", bbox=CONUS_BBOX))
    logger.info("[Lightning] Total FIRMS archive hotspots: %d", len(firms))

    # Index FIRMS by 1deg lat/lon cell for fast filtering
    firms_idx: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for h in firms:
        firms_idx[(int(math.floor(h["lat"])), int(math.floor(h["lon"])))].append(h)

    feat_names = [
        "lat", "lon_norm", "lat_abs",
        "lon_sin", "lon_cos",
        "day_sin", "day_cos", "hour_sin", "hour_cos",
        "temp_max", "rh_min", "wind_max", "vpd_max", "precip", "et0",
        "fwi_proxy",
        "casualties_indicator",
        "state_strike_density",
    ]

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    n_processed = 0
    n_pos = 0
    n_skipped_wx = 0
    for s in lightning:
        n_processed += 1
        date_str = s["dt"].strftime("%Y-%m-%d")
        if skip_weather:
            wx = {"temp_max": None, "rh_min": None, "wind_max": None,
                  "vpd_max": None, "precip": None, "et0": None}
        else:
            wx = _fetch_weather(s["lat"], s["lon"], date_str)
            if wx is None:
                n_skipped_wx += 1
                continue
            time.sleep(0.06)

        # Label: any FIRMS hotspot within 25 km AND within +72h of strike?
        ymin, ymax = int(math.floor(s["lat"] - 1)), int(math.floor(s["lat"] + 1))
        xmin, xmax = int(math.floor(s["lon"] - 1)), int(math.floor(s["lon"] + 1))
        label = 0
        for ya in range(ymin, ymax + 1):
            if label: break
            for xa in range(xmin, xmax + 1):
                for h in firms_idx.get((ya, xa), ()):
                    try:
                        h_dt = datetime.strptime(h["acq_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    except (ValueError, KeyError):
                        continue
                    dt_diff = (h_dt - s["dt"]).total_seconds() / 3600.0
                    if 0 <= dt_diff <= match_window_h:
                        if _haversine_km(s["lat"], s["lon"], h["lat"], h["lon"]) <= match_radius_km:
                            label = 1
                            break
                if label: break
        if label:
            n_pos += 1

        doy = s["dt"].timetuple().tm_yday
        feat = np.array([
            s["lat"],
            s["lon"] / 180.0,
            abs(s["lat"]) / 90.0,
            math.sin(math.radians(s["lon"])),
            math.cos(math.radians(s["lon"])),
            math.sin(2 * math.pi * doy / 365.25),
            math.cos(2 * math.pi * doy / 365.25),
            math.sin(2 * math.pi * s["dt"].hour / 24),
            math.cos(2 * math.pi * s["dt"].hour / 24),
            wx["temp_max"] or 25.0,
            wx["rh_min"] or 50.0,
            wx["wind_max"] or 5.0,
            wx["vpd_max"] or 1.5,
            wx["precip"] or 0.0,
            wx["et0"] or 4.0,
            _fwi_proxy(wx["temp_max"], wx["rh_min"], wx["wind_max"], wx["precip"]),
            float(s["casualties"] > 0),
            min(by_state.get(s["state"], 0) / 5000.0, 1.0),
        ], dtype=np.float32)

        X_list.append(feat)
        y_list.append(label)

        if n_processed % 100 == 0:
            logger.info("[Lightning] processed=%d  pos=%d  skipped_wx=%d",
                        n_processed, n_pos, n_skipped_wx)

    if not X_list:
        return np.zeros((0, len(feat_names)), dtype=np.float32), np.zeros(0, dtype=np.int64), feat_names

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    logger.info("[Lightning] Built %d samples, positive rate=%.3f", len(X), y.mean())
    return X, y, feat_names


def train(years: list[int], max_lightning: int = 3000, skip_weather: bool = False) -> None:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    import joblib

    X, y, feat_names = _build_dataset(years=years, max_lightning=max_lightning,
                                       skip_weather=skip_weather)
    if len(X) < 100:
        raise RuntimeError(f"Only {len(X)} samples — too few to train.")

    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        raise RuntimeError(f"Single-class dataset (pos={pos}, neg={neg}) — adjust radius/window.")

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.06,
        subsample=0.8, random_state=7)

    f1 = cross_val_score(clf, X, y, cv=3, scoring="f1_weighted", n_jobs=-1)
    auc = cross_val_score(clf, X, y, cv=3, scoring="roc_auc", n_jobs=-1)
    logger.info("[Lightning] CV f1=%.4f +/- %.4f  auc=%.4f +/- %.4f",
                f1.mean(), f1.std(), auc.mean(), auc.std())

    clf.fit(X, y)
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    top = {feat_names[i]: round(float(importances[i]), 4) for i in top_idx}
    logger.info("[Lightning] Top features: %s", top)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": clf, "feature_names": feat_names}, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "GradientBoostingClassifier",
        "task": "P(wildfire ignition within 25km / 72h of lightning strike)",
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "positive_rate": round(float(y.mean()), 4),
        "data_sources": ["noaa_storm_events_real",
                         "nasa_firms_viirs_archive_real (SNPP_SP + NOAA20_SP)",
                         "openmeteo_archive_real" if not skip_weather else "openmeteo_skipped"],
        "years": years,
        "metrics": {
            "f1_cv_weighted": round(float(f1.mean()), 4),
            "f1_cv_std": round(float(f1.std()), 4),
            "auc_cv": round(float(auc.mean()), 4),
            "auc_cv_std": round(float(auc.std()), 4),
        },
        "top_features": top,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[Lightning] Saved -> %s", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs="+", type=int, default=[2024])
    p.add_argument("--max-lightning", type=int, default=2000)
    p.add_argument("--skip-weather", action="store_true",
                   help="Skip Open-Meteo weather fetch (use when rate-limited)")
    args = p.parse_args()
    train(years=args.years, max_lightning=args.max_lightning,
          skip_weather=args.skip_weather)
