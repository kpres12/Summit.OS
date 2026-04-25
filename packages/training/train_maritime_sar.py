"""
Maritime SAR / Marine Hazard Risk Classifier

Trains a classifier to predict the probability that a reported marine
hazard event (NOAA Storm Events: marine thunderstorm wind, marine hail,
sneaker wave, rip current, storm surge, etc.) occurs within 100 km / 24 h
of a NOAA NDBC buoy reading.

This is a real-data civilian-SAR risk model — quantifies SAR/USCG demand
from observable maritime conditions.

Real data sources (no auth):
  NOAA NDBC realtime buoy data (~45 days, hundreds of stations):
    https://www.ndbc.noaa.gov/data/realtime2/{station}.txt
  NOAA Storm Events Database (already cached locally from prior run)

Training shape:
  Each (buoy_station x reading_time) becomes a sample. Features:
    wind_speed, wind_gust, wave_height, dom_period, avg_period,
    mean_wave_dir, atmos_pressure, air_temp, water_temp, dewpoint,
    visibility, lat, lon, hour_sin/cos, day_sin/cos
  Label = 1 if any NOAA marine event within 100 km / 24 h, else 0.

Output: packages/c2_intel/models/maritime_sar_classifier.joblib
        packages/c2_intel/models/maritime_sar_classifier_meta.json

Usage:
    python train_maritime_sar.py [--max-buoys 50]
"""

from __future__ import annotations

import argparse
import csv
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
MODEL_PATH = MODELS_DIR / "maritime_sar_classifier.joblib"
META_PATH  = MODELS_DIR / "maritime_sar_classifier_meta.json"
CACHE_DIR  = Path(__file__).parent / "data" / "maritime_sar"
NOAA_CACHE = Path(__file__).parent / "data" / "noaa_storms"

NDBC_BASE  = "https://www.ndbc.noaa.gov/data/realtime2"
NDBC_HIST  = "https://www.ndbc.noaa.gov/view_text_file.php"
NDBC_INDEX = "https://www.ndbc.noaa.gov/data/stations/station_table.txt"

EARTH_R_KM = 6371.0

# NOAA Storm Events that count as "marine SAR-relevant"
MARINE_EVENT_TYPES = {
    "marine thunderstorm wind",
    "marine high wind",
    "marine strong wind",
    "marine hail",
    "marine dense fog",
    "rip current",
    "sneakerwave",
    "sneaker wave",
    "storm surge/tide",
    "high surf",
    "marine tropical storm",
    "marine hurricane/typhoon",
    "marine tropical depression",
    "tsunami",
    "waterspout",
    "seiche",
}


def _haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = p2 - p1
    dl = math.radians(b_lon - a_lon)
    a = math.sin(dp/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


def _parse_dt(date_str: str) -> datetime | None:
    if not date_str:
        return None
    for f in ("%d-%b-%y %H:%M:%S", "%d-%b-%y %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str.strip(), f).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _list_ndbc_stations(limit: int) -> list[dict]:
    """Get a list of active NDBC buoy stations with lat/lon."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "stations.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            return data[:limit]
        except Exception:
            pass

    try:
        r = requests.get(NDBC_INDEX, timeout=60, headers={"User-Agent": "Heli.OS/1.0"})
        r.raise_for_status()
    except Exception as e:
        logger.error("[Maritime] Station index fetch failed: %s", e)
        return []

    stations = []
    for line in r.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 7:
            continue
        station = parts[0]
        if not station or len(station) > 6:
            continue
        # Lat/lon are in column 7 as something like "26.030 N 89.668 W"
        loc = parts[6]
        m = re.match(r"\s*(\d+\.\d+)\s*([NS])\s+(\d+\.\d+)\s*([EW])", loc)
        if not m:
            continue
        lat = float(m.group(1)) * (1 if m.group(2) == "N" else -1)
        lon = float(m.group(3)) * (1 if m.group(4) == "E" else -1)
        # Prefer ocean buoys (filter by station type field if present)
        stype = parts[2].lower() if len(parts) > 2 else ""
        if stype in ("buoy", "fixed", "fixed buoy", "tao", "iws", "drifting buoy"):
            stations.append({"station": station, "lat": lat, "lon": lon, "type": stype})

    cache.write_text(json.dumps(stations))
    logger.info("[Maritime] Indexed %d NDBC ocean stations", len(stations))
    return stations[:limit]


def _fetch_ndbc_historical(station: str, year: int) -> list[dict]:
    """Fetch a full year of standard meteorological data for a buoy.

    Format: gzipped text file at /data/historical/stdmet/{station}h{year}.txt.gz
    Same column layout as realtime data.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{station}_{year}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass

    try:
        r = requests.get(NDBC_HIST, params={
            "filename": f"{station}h{year}.txt.gz",
            "dir": "data/historical/stdmet/",
        }, timeout=60, headers={"User-Agent": "Heli.OS/1.0"})
        r.raise_for_status()
        text = r.text
        # NDBC view_text_file.php returns the decompressed text directly when
        # filename ends in .gz — but if station/year doesn't exist, returns HTML
        if "<html" in text.lower() or "<body" in text.lower():
            return []
    except Exception as e:
        logger.debug("[Maritime] %s/%d historical fetch failed: %s", station, year, e)
        return []

    lines = text.splitlines()
    if len(lines) < 3:
        return []
    # Header line 0 starts with '#YY  MM DD ...'  Line 1 is units. Data starts line 2.
    header = re.sub(r"\s+", " ", lines[0].lstrip("#")).strip().split()
    rows = []
    for raw in lines[2:]:
        parts = re.sub(r"\s+", " ", raw.strip()).split()
        if len(parts) < len(header):
            continue
        rec = dict(zip(header, parts))
        try:
            dt = datetime(
                int("20" + rec["YY"]) if len(rec["YY"]) == 2 else int(rec["YY"]),
                int(rec["MM"]), int(rec["DD"]), int(rec["hh"]), int(rec["mm"]),
                tzinfo=timezone.utc,
            )
        except (KeyError, ValueError):
            continue

        def _f(key: str) -> float:
            v = rec.get(key, "MM")
            try:
                fv = float(v)
                if fv >= 99.0 and key in ("WSPD", "GST", "WVHT", "DPD", "APD",
                                          "PRES", "ATMP", "WTMP", "DEWP", "VIS"):
                    return float("nan")
                return fv
            except (ValueError, TypeError):
                return float("nan")

        rows.append({
            "ts": int(dt.timestamp()),
            "wdir":  _f("WDIR"),
            "wspd":  _f("WSPD"),
            "gst":   _f("GST"),
            "wvht":  _f("WVHT"),
            "dpd":   _f("DPD"),
            "apd":   _f("APD"),
            "mwd":   _f("MWD"),
            "pres":  _f("PRES"),
            "atmp":  _f("ATMP"),
            "wtmp":  _f("WTMP"),
            "dewp":  _f("DEWP"),
            "vis":   _f("VIS"),
        })
    cache.write_text(json.dumps(rows))
    return rows


def _fetch_ndbc_realtime(station: str) -> list[dict]:
    """Fetch last ~45 days of buoy data — kept for online inference but not training."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{station}_rt.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    try:
        r = requests.get(f"{NDBC_BASE}/{station}.txt", timeout=30,
                         headers={"User-Agent": "Heli.OS/1.0"})
        r.raise_for_status()
        text = r.text
    except Exception:
        return []
    return _parse_ndbc_text(text, cache)


def _load_marine_events(years: list[int]) -> list[dict]:
    """Load marine subset of NOAA storm events from cached CSVs."""
    out: list[dict] = []
    for y in years:
        cache = NOAA_CACHE / f"storms_{y}.csv"
        if not cache.exists():
            logger.warning("[Maritime] No storm cache for %d", y)
            continue
        with cache.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for r in reader:
                et = (r.get("EVENT_TYPE") or "").strip().lower()
                if et not in MARINE_EVENT_TYPES:
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
                out.append({"ts": int(dt.timestamp()), "lat": lat, "lon": lon, "type": et})
    logger.info("[Maritime] Loaded %d marine events from cached storm data", len(out))
    return out


def _build_dataset(max_buoys: int, match_radius_km: float = 100.0,
                   match_window_h: int = 24, max_samples: int = 30000
                   ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    stations = _list_ndbc_stations(limit=max_buoys)
    if not stations:
        raise RuntimeError("No NDBC stations available.")

    # Marine events from already-cached NOAA storms
    events = _load_marine_events(years=[2024, 2025, 2026])
    # Index events by 1deg lat/lon cell for fast neighborhood lookup
    ev_idx: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for e in events:
        ev_idx[(int(math.floor(e["lat"])), int(math.floor(e["lon"])))].append(e)

    feat_names = [
        "wspd", "gst", "wvht", "dpd", "apd",
        "wdir_sin", "wdir_cos", "mwd_sin", "mwd_cos",
        "pres_anom", "atmp", "wtmp", "atmp_minus_wtmp",
        "dewp_dep", "vis_norm",
        "lat", "lon_norm", "lat_abs",
        "hour_sin", "hour_cos", "day_sin", "day_cos",
    ]

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    n_with_data = 0
    for st in stations:
        rows = _fetch_ndbc_realtime(st["station"])
        time.sleep(0.1)  # politeness
        if not rows:
            continue
        n_with_data += 1
        cell_neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cell_neighbors.extend(ev_idx.get(
                    (int(math.floor(st["lat"])) + dy, int(math.floor(st["lon"])) + dx),
                    [],
                ))

        # Subsample buoy readings per station (every Nth) to control dataset size
        rows = rows[::3]  # every 3rd hourly reading

        for r in rows:
            # Skip if too many fields missing
            if math.isnan(r.get("wspd", float("nan"))) and math.isnan(r.get("wvht", float("nan"))):
                continue

            # Label
            label = 0
            t_ev_lo = r["ts"]
            t_ev_hi = r["ts"] + match_window_h * 3600
            for e in cell_neighbors:
                if not (t_ev_lo - 3600 <= e["ts"] <= t_ev_hi):
                    continue
                if _haversine_km(st["lat"], st["lon"], e["lat"], e["lon"]) <= match_radius_km:
                    label = 1
                    break

            wdir = r.get("wdir", float("nan"))
            mwd = r.get("mwd", float("nan"))
            pres = r.get("pres", float("nan"))
            wspd = r.get("wspd", 0.0); wspd = 0.0 if math.isnan(wspd) else wspd
            gst = r.get("gst", 0.0); gst = 0.0 if math.isnan(gst) else gst
            wvht = r.get("wvht", 0.0); wvht = 0.0 if math.isnan(wvht) else wvht
            dpd = r.get("dpd", 0.0); dpd = 0.0 if math.isnan(dpd) else dpd
            apd = r.get("apd", 0.0); apd = 0.0 if math.isnan(apd) else apd
            atmp = r.get("atmp", 15.0); atmp = 15.0 if math.isnan(atmp) else atmp
            wtmp = r.get("wtmp", 15.0); wtmp = 15.0 if math.isnan(wtmp) else wtmp
            dewp = r.get("dewp", 10.0); dewp = 10.0 if math.isnan(dewp) else dewp
            vis = r.get("vis", 10.0); vis = 10.0 if math.isnan(vis) else vis

            wdir_sin = math.sin(math.radians(wdir)) if not math.isnan(wdir) else 0.0
            wdir_cos = math.cos(math.radians(wdir)) if not math.isnan(wdir) else 0.0
            mwd_sin = math.sin(math.radians(mwd)) if not math.isnan(mwd) else 0.0
            mwd_cos = math.cos(math.radians(mwd)) if not math.isnan(mwd) else 0.0
            pres_anom = (pres - 1013.25) if not math.isnan(pres) else 0.0

            dt = datetime.fromtimestamp(r["ts"], tz=timezone.utc)
            doy = dt.timetuple().tm_yday

            feat = np.array([
                wspd, gst, wvht, dpd, apd,
                wdir_sin, wdir_cos, mwd_sin, mwd_cos,
                pres_anom, atmp, wtmp, atmp - wtmp,
                atmp - dewp,  # dewpoint depression (humidity proxy)
                min(vis, 30.0) / 30.0,
                st["lat"], st["lon"] / 180.0, abs(st["lat"]) / 90.0,
                math.sin(2 * math.pi * dt.hour / 24),
                math.cos(2 * math.pi * dt.hour / 24),
                math.sin(2 * math.pi * doy / 365.25),
                math.cos(2 * math.pi * doy / 365.25),
            ], dtype=np.float32)

            X_list.append(feat)
            y_list.append(label)

            if len(X_list) >= max_samples:
                break
        if len(X_list) >= max_samples:
            break

    if not X_list:
        return np.zeros((0, len(feat_names)), dtype=np.float32), np.zeros(0, dtype=np.int64), feat_names

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    logger.info("[Maritime] Built %d samples from %d buoys, positive rate=%.3f",
                len(X), n_with_data, y.mean())
    return X, y, feat_names


def train(max_buoys: int = 80) -> None:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    import joblib

    X, y, feat_names = _build_dataset(max_buoys=max_buoys)
    if len(X) < 200:
        raise RuntimeError(f"Only {len(X)} samples — too few.")
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        raise RuntimeError(f"Single-class dataset (pos={pos}, neg={neg}).")

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.06,
        subsample=0.8, random_state=7)

    f1 = cross_val_score(clf, X, y, cv=3, scoring="f1_weighted", n_jobs=-1)
    auc = cross_val_score(clf, X, y, cv=3, scoring="roc_auc", n_jobs=-1)
    logger.info("[Maritime] CV f1=%.4f  auc=%.4f", f1.mean(), auc.mean())

    clf.fit(X, y)
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    top = {feat_names[i]: round(float(importances[i]), 4) for i in top_idx}
    logger.info("[Maritime] Top features: %s", top)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": clf, "feature_names": feat_names}, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "GradientBoostingClassifier",
        "task": "P(marine SAR-relevant event within 100km/24h of buoy reading)",
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "positive_rate": round(float(y.mean()), 4),
        "data_sources": ["noaa_ndbc_realtime_real", "noaa_storm_events_real_marine_subset"],
        "metrics": {
            "f1_cv_weighted": round(float(f1.mean()), 4),
            "auc_cv": round(float(auc.mean()), 4),
        },
        "top_features": top,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[Maritime] Saved -> %s", MODEL_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--max-buoys", type=int, default=80)
    args = p.parse_args()
    train(max_buoys=args.max_buoys)
