"""
Heli.OS — OpenSky Network ADS-B Adapter
==========================================
Pulls real-time aircraft positions from the OpenSky Network REST API and
ingests them as AIRCRAFT entities into Heli.OS.

Zero hardware required — uses the public free tier (400 requests/day,
no auth) or an OpenSky account for higher limits.

What you get:
  - Real aircraft positions updated every 10-30s
  - ICAO24 hex + callsign
  - Lat/lon/altitude (barometric)
  - Ground speed, heading, vertical rate
  - On-ground flag (entity_type: neutral when airborne, unknown when ground)

Config (AdapterConfig.extra fields):
  bbox_min_lat   — bounding box south edge   (default: 24.0  = southern US)
  bbox_max_lat   — bounding box north edge   (default: 49.0  = northern US)
  bbox_min_lon   — bounding box west edge    (default: -125.0)
  bbox_max_lon   — bounding box east edge    (default: -66.0)
  username       — OpenSky username          (optional, increases rate limit)
  password       — OpenSky password          (optional)
  stale_seconds  — drop aircraft not refreshed for N seconds (default: 120)

Environment:
  OPENSKY_USERNAME / OPENSKY_PASSWORD — credentials (optional)
  OPENSKY_BBOX     — "min_lat,min_lon,max_lat,max_lon" override

Register in adapters.json:
  {
    "adapter_type": "opensky",
    "name": "OpenSky ADS-B",
    "poll_interval_seconds": 30,
    "extra": {
      "bbox_min_lat": 33.5, "bbox_max_lat": 35.0,
      "bbox_min_lon": -119.0, "bbox_max_lon": -116.5
    }
  }
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import AsyncIterator, Optional

try:
    import httpx
except ImportError:
    raise ImportError("httpx is required: pip install httpx")

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("heli.adapters.opensky")

OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"

# OpenSky states/all response column indices
_COL = {
    "icao24": 0, "callsign": 1, "origin_country": 2,
    "time_position": 3, "last_contact": 4,
    "lon": 5, "lat": 6, "baro_altitude": 7,
    "on_ground": 8, "velocity": 9, "true_track": 10,
    "vertical_rate": 11, "sensors": 12, "geo_altitude": 13,
    "squawk": 14, "spi": 15, "position_source": 16,
}


class OpenSkyAdapter(BaseAdapter):
    adapter_type = "opensky"

    def __init__(self, config: AdapterConfig, mqtt_client=None):
        super().__init__(config, mqtt_client)
        ex = config.extra or {}
        env_bbox = os.getenv("OPENSKY_BBOX", "").split(",")

        if len(env_bbox) == 4:
            self._bbox = {
                "min_lat": float(env_bbox[0]), "min_lon": float(env_bbox[1]),
                "max_lat": float(env_bbox[2]), "max_lon": float(env_bbox[3]),
            }
        else:
            self._bbox = {
                "min_lat": float(ex.get("bbox_min_lat", 24.0)),
                "max_lat": float(ex.get("bbox_max_lat", 49.0)),
                "min_lon": float(ex.get("bbox_min_lon", -125.0)),
                "max_lon": float(ex.get("bbox_max_lon", -66.0)),
            }

        self._username = ex.get("username") or os.getenv("OPENSKY_USERNAME")
        self._password = ex.get("password") or os.getenv("OPENSKY_PASSWORD")
        self._stale_s  = int(ex.get("stale_seconds", 120))
        self._client: Optional[httpx.AsyncClient] = None

    async def connect(self) -> None:
        auth = (self._username, self._password) if self._username else None
        self._client = httpx.AsyncClient(auth=auth, timeout=15.0)
        logger.info("OpenSky adapter connected (bbox=%s)", self._bbox)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def stream_observations(self) -> AsyncIterator[dict]:
        assert self._client is not None
        while True:
            try:
                params = {
                    "lamin": self._bbox["min_lat"], "lomin": self._bbox["min_lon"],
                    "lamax": self._bbox["max_lat"], "lomax": self._bbox["max_lon"],
                }
                resp = await self._client.get(OPENSKY_STATES_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
                states = data.get("states") or []
                now = time.time()

                for state in states:
                    if len(state) < 17:
                        continue
                    lat = state[_COL["lat"]]
                    lon = state[_COL["lon"]]
                    if lat is None or lon is None:
                        continue
                    last_contact = state[_COL["last_contact"]] or 0
                    if now - last_contact > self._stale_s:
                        continue

                    icao24   = (state[_COL["icao24"]] or "").upper()
                    callsign = (state[_COL["callsign"]] or icao24).strip()
                    alt      = state[_COL["baro_altitude"]] or state[_COL["geo_altitude"]] or 0
                    speed    = state[_COL["velocity"]] or 0
                    heading  = state[_COL["true_track"]] or 0
                    on_ground = bool(state[_COL["on_ground"]])

                    yield {
                        "entity_id":   f"adsb-{icao24.lower()}",
                        "type":        "neutral",
                        "callsign":    callsign or icao24,
                        "position":    {"lat": lat, "lon": lon, "alt": alt},
                        "last_seen":   int(last_contact or now),
                        "properties": {
                            "asset_type":    "AIRCRAFT",
                            "icao24":        icao24,
                            "speed_ms":      round(speed, 1),
                            "heading":       round(heading, 1),
                            "on_ground":     on_ground,
                            "origin":        state[_COL["origin_country"]] or "",
                            "source":        "opensky",
                            "controllable":  False,
                        },
                    }

                logger.debug("OpenSky: ingested %d aircraft", len(states))

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning("OpenSky rate limit hit — backing off 60s")
                    await asyncio.sleep(60)
                else:
                    raise

            await asyncio.sleep(self.config.poll_interval_seconds)
