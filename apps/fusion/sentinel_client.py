"""
Sentinel prediction client for Summit.OS Fusion.

Provides an async helper to invoke Sentinel's fire spread simulation API
whenever a smoke/ignition detection is confirmed.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp


SENTINEL_API_URL = os.getenv("SENTINEL_API_URL", os.getenv("API_BASE_URL", "http://localhost:8000")).rstrip("/")
SENTINEL_API_KEY = os.getenv("SENTINEL_API_KEY")  # optional
DEFAULT_FUEL_MODEL = int(os.getenv("SENTINEL_DEFAULT_FUEL_MODEL", "4"))  # Chaparral
DEFAULT_MC_RUNS = int(os.getenv("SENTINEL_DEFAULT_MC_RUNS", "50"))
DEFAULT_SIM_HOURS = int(os.getenv("SENTINEL_DEFAULT_SIM_HOURS", "12"))
DEFAULT_DT_MIN = float(os.getenv("SENTINEL_DEFAULT_TIME_STEP_MIN", "15"))


async def simulate_spread(
    lat: float,
    lon: float,
    conditions: Optional[Dict[str, Any]] = None,
    session: Optional[aiohttp.ClientSession] = None,
) -> Dict[str, Any]:
    """
    Call Sentinel's /api/v1/prediction/simulate endpoint with minimal required fields.

    Args:
        lat, lon: ignition coordinates
        conditions: optional weather/terrain dict; sensible defaults are applied if missing
        session: optional shared aiohttp session
    Returns:
        JSON dict of SpreadResult
    """
    now = datetime.now(timezone.utc).isoformat()
    c = conditions or {}

    env = {
        "timestamp": c.get("timestamp", now),
        "latitude": float(c.get("latitude", lat)),
        "longitude": float(c.get("longitude", lon)),
        "temperature_c": float(c.get("temperature_c", c.get("temperature", 28.0))),
        "relative_humidity": float(c.get("relative_humidity", c.get("humidity", 30.0))),
        "wind_speed_mps": float(c.get("wind_speed_mps", c.get("wind_speed", 5.0))),
        "wind_direction_deg": float(c.get("wind_direction_deg", c.get("wind_direction", 270.0))),
        "fuel_moisture": float(c.get("fuel_moisture", 0.15)),
        "soil_moisture": float(c.get("soil_moisture", 0.1)),
        "fuel_model": int(c.get("fuel_model", DEFAULT_FUEL_MODEL)),
        "slope_deg": float(c.get("slope_deg", c.get("slope", 0.0))),
        "aspect_deg": float(c.get("aspect_deg", c.get("aspect", 0.0))),
        "canopy_cover": float(c.get("canopy_cover", 0.0)),
        "elevation_m": float(c.get("elevation_m", c.get("elevation", 0.0))),
    }

    body = {
        "ignition_points": [
            {"latitude": lat, "longitude": lon, "altitude": float(c.get("altitude", 0.0))}
        ],
        "conditions": env,
        "simulation_hours": int(c.get("simulation_hours", DEFAULT_SIM_HOURS)),
        "time_step_minutes": float(c.get("time_step_minutes", DEFAULT_DT_MIN)),
        "monte_carlo_runs": int(c.get("monte_carlo_runs", DEFAULT_MC_RUNS)),
        "custom_parameters": {},
    }

    headers = {"Content-Type": "application/json"}
    if SENTINEL_API_KEY:
        headers["Authorization"] = f"Bearer {SENTINEL_API_KEY}"

    own_session = False
    if session is None:
        own_session = True
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))

    try:
        url = f"{SENTINEL_API_URL}/api/v1/prediction/simulate"
        async with session.post(url, json=body, headers=headers) as resp:
            resp.raise_for_status()
            return await resp.json()
    finally:
        if own_session and session:
            await session.close()