"""
MarineCadastre.gov AIS Historical Data Loader
=================================================
The US Bureau of Ocean Energy Management + USCG publishes complete US-
waters AIS broadcasts as monthly CSV bundles via marinecadastre.gov.
**Free, no auth, direct HTTPS download.**

Index URL pattern (monthly):
    https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{YEAR}/AIS_{YEAR}_{MM}_{DD}.zip

Each daily zip is ~50-200 MB and contains a single CSV (~1-5M AIS records).
Schema (representative columns):
    MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselName, IMO,
    CallSign, VesselType, Status, Length, Width, Draft, Cargo

Coverage: US Coast Guard receivers from 2017-present, US territorial
waters + EEZ. Roughly 100k-200k unique vessels per month.

Why this matters:
  - Vessel trajectory anomaly training (replaces what xView3 vessel
    behavior labels would give us)
  - Dark-vessel inference: for any AIS-broadcast track, gaps + sudden
    course changes = candidate dark-period flag for vessel CNN training
  - Civilian USCG SAR demand modeling
  - Federal MDA + IUU fishing pattern-of-life

Usage:
    from packages.training.datasets.marinecadastre import (
        list_available_days, download_day, iter_records, vessel_summary,
    )
    days = list_available_days(2024, 6)
    csv_path = download_day(2024, 6, 15)
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import requests

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent.parent / "data" / "marinecadastre"
BASE_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler"


# Vessel type codes per the ITU-R M.1371 standard
VESSEL_TYPE_NAMES = {
    0: "not_available",
    20: "wig", 21: "wig_a", 22: "wig_b", 23: "wig_c",
    30: "fishing", 31: "towing", 32: "towing_long",
    33: "dredging", 34: "diving", 35: "military",
    36: "sailing", 37: "pleasure",
    40: "high_speed", 41: "high_speed_a", 42: "high_speed_b",
    43: "high_speed_c", 44: "high_speed_d",
    50: "pilot", 51: "search_rescue", 52: "tug",
    53: "port_tender", 54: "anti_pollution", 55: "law_enforcement",
    58: "medical_transport",
    60: "passenger", 61: "passenger_a", 62: "passenger_b",
    63: "passenger_c", 64: "passenger_d",
    70: "cargo", 71: "cargo_a", 72: "cargo_b", 73: "cargo_c", 74: "cargo_d",
    80: "tanker", 81: "tanker_a", 82: "tanker_b", 83: "tanker_c", 84: "tanker_d",
    90: "other", 91: "other_a", 92: "other_b", 93: "other_c", 94: "other_d",
}


def _vessel_class(code: int) -> str:
    return VESSEL_TYPE_NAMES.get(code, "other")


def _build_url(d: date) -> str:
    return f"{BASE_URL}/{d.year}/AIS_{d.year}_{d.month:02d}_{d.day:02d}.zip"


def list_available_days(year: int, month: int) -> list[date]:
    """Best-effort enumeration — MarineCadastre exposes daily zips for
    every calendar day in the AIS coverage period."""
    if month < 1 or month > 12:
        raise ValueError(f"invalid month {month}")
    if month in (1, 3, 5, 7, 8, 10, 12):
        days_in_month = 31
    elif month == 2:
        # leap-year-aware
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            days_in_month = 29
        else:
            days_in_month = 28
    else:
        days_in_month = 30
    return [date(year, month, d) for d in range(1, days_in_month + 1)]


def download_day(year: int, month: int, day: int,
                 dest_dir: Path = OUT_DIR) -> Optional[Path]:
    """Download a single day's AIS zip and return the local CSV path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    d = date(year, month, day)
    csv_name = f"AIS_{d.year}_{d.month:02d}_{d.day:02d}.csv"
    csv_path = dest_dir / csv_name
    if csv_path.exists():
        return csv_path
    url = _build_url(d)
    logger.info("[marinecadastre] downloading %s", url)
    try:
        r = requests.get(url, timeout=600,
                         headers={"User-Agent": "Heli.OS/1.0"})
        r.raise_for_status()
    except Exception as e:
        logger.warning("[marinecadastre] %s download failed: %s", d, e)
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            inner = next((n for n in zf.namelist() if n.endswith(".csv")), None)
            if not inner:
                logger.warning("[marinecadastre] no CSV in %s", url)
                return None
            with zf.open(inner) as src, csv_path.open("wb") as dst:
                dst.write(src.read())
    except Exception as e:
        logger.warning("[marinecadastre] extract failed for %s: %s", d, e)
        return None
    logger.info("[marinecadastre] %s -> %s (%.1f MB)",
                d, csv_path, csv_path.stat().st_size / 1024 / 1024)
    return csv_path


def iter_records(csv_path: Path, max_rows: Optional[int] = None
                 ) -> Iterator[dict]:
    """Stream parsed AIS records from a daily CSV.

    Each yielded dict contains: mmsi, ts (datetime UTC), lat, lon, sog,
    cog, heading, vessel_name, imo, callsign, vessel_type_code,
    vessel_class, status, length, width, draft, cargo.
    """
    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            try:
                lat = float(row.get("LAT") or 0)
                lon = float(row.get("LON") or 0)
                if lat == 0 and lon == 0:
                    continue
                ts_str = row.get("BaseDateTime") or ""
                try:
                    ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
                except ValueError:
                    try:
                        ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
                vt = int(float(row.get("VesselType") or 0))
                yield {
                    "mmsi":            row.get("MMSI"),
                    "ts":              ts,
                    "lat":             lat,
                    "lon":             lon,
                    "sog":             float(row.get("SOG") or 0),
                    "cog":             float(row.get("COG") or 0),
                    "heading":         float(row.get("Heading") or 0),
                    "vessel_name":     (row.get("VesselName") or "").strip(),
                    "imo":             (row.get("IMO") or "").strip(),
                    "callsign":        (row.get("CallSign") or "").strip(),
                    "vessel_type_code": vt,
                    "vessel_class":     _vessel_class(vt),
                    "status":          row.get("Status"),
                    "length":          float(row.get("Length") or 0),
                    "width":           float(row.get("Width") or 0),
                    "draft":           float(row.get("Draft") or 0),
                    "cargo":           int(float(row.get("Cargo") or 0)),
                }
            except (ValueError, KeyError, TypeError):
                continue


def vessel_summary(csv_path: Path, max_rows: int = 500_000) -> dict:
    """One-pass summary stats for a daily CSV."""
    by_class: dict[str, int] = {}
    n = 0
    for rec in iter_records(csv_path, max_rows=max_rows):
        by_class[rec["vessel_class"]] = by_class.get(rec["vessel_class"], 0) + 1
        n += 1
    return {"total_records": n, "by_class": dict(sorted(by_class.items()))}
