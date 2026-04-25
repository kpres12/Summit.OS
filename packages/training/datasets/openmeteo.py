"""
Open-Meteo Weather API — Free, No API Key Required
https://open-meteo.com/

Fetches hourly historical weather for a (lat, lon, date) point.
Used to enrich fire event records from NASA FIRMS with real atmospheric
conditions for wildfire LSTM training sequences.

Variables fetched:
  - temperature_2m (°C)
  - relative_humidity_2m (%)
  - wind_speed_10m (m/s)
  - wind_direction_10m (°)
  - precipitation (mm)

Rate limit: ~10k req/day free tier — we batch by location.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_BASE = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"

_VARIABLES = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation"


def fetch_hourly(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    max_retries: int = 2,
) -> Optional[dict]:
    """
    Fetch hourly weather for a location over a date range.

    Args:
        lat, lon: coordinates
        start_date, end_date: ISO date strings (YYYY-MM-DD)

    Returns:
        Dict with 'time', 'temperature_2m', etc. arrays — or None on failure.
    """
    params = {
        "latitude": round(lat, 3),
        "longitude": round(lon, 3),
        "start_date": start_date,
        "end_date": end_date,
        "hourly": _VARIABLES,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(_BASE, params=params, timeout=15,
                                headers={"User-Agent": "Heli.OS/1.0"})
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data.get("hourly")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                logger.debug("[OpenMeteo] Failed for %.2f,%.2f: %s", lat, lon, e)
    return None


def fetch_recent(lat: float, lon: float, hours: int = 24) -> Optional[dict]:
    """Fetch recent weather (last N hours) using forecast endpoint."""
    params = {
        "latitude": round(lat, 3),
        "longitude": round(lon, 3),
        "hourly": _VARIABLES,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "past_days": min(hours // 24 + 1, 7),
        "forecast_days": 0,
    }
    try:
        resp = requests.get(_FORECAST_BASE, params=params, timeout=10,
                            headers={"User-Agent": "Heli.OS/1.0"})
        resp.raise_for_status()
        return resp.json().get("hourly")
    except Exception as e:
        logger.debug("[OpenMeteo] Recent fetch failed: %s", e)
        return None


def weather_at_hour(hourly: dict, target_hour: int) -> dict:
    """Extract weather values at a specific hour index from hourly response."""
    idx = min(target_hour, len(hourly.get("time", [1])) - 1)
    return {
        "temp_c":       _safe(hourly, "temperature_2m", idx, 25.0),
        "rh_pct":       _safe(hourly, "relative_humidity_2m", idx, 40.0),
        "wind_speed_ms":_safe(hourly, "wind_speed_10m", idx, 5.0),
        "wind_dir_deg": _safe(hourly, "wind_direction_10m", idx, 180.0),
        "precip_mm":    _safe(hourly, "precipitation", idx, 0.0),
    }


def _safe(d: dict, key: str, idx: int, default: float) -> float:
    try:
        v = d[key][idx]
        return float(v) if v is not None else default
    except (KeyError, IndexError, TypeError):
        return default
