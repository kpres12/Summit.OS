"""
Summit.OS CelesTrak Satellite Adapter

Fetches Two-Line Element (TLE) sets from CelesTrak, propagates satellite
positions to the current epoch using SGP4, and publishes each satellite
as an Entity into the Summit.OS data fabric via MQTT.

Environment variables:
    CELESTRAK_ENABLED        - "true" to enable (default: "true")
    CELESTRAK_MAX_SATS       - max satellites to track (default: 200)
    CELESTRAK_TLE_REFRESH    - seconds between TLE re-fetch (default: 21600 = 6h)
    CELESTRAK_PROPAGATE_INTERVAL - seconds between position updates (default: 30)
    CELESTRAK_GROUP           - TLE group (default: "active")
    MQTT_HOST                - MQTT broker host (default: "localhost")
    MQTT_PORT                - MQTT broker port (default: 1883)
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

try:
    from sgp4.api import Satrec, jday
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False

logger = logging.getLogger("summit.adapter.celestrak")

# Earth radius in km for ECEF → lat/lon conversion
EARTH_RADIUS_KM = 6371.0
KM_TO_M = 1000.0


def _ecef_to_lla(x_km: float, y_km: float, z_km: float) -> Tuple[float, float, float]:
    """Convert ECEF (km) from SGP4 to lat, lon (degrees), alt (meters)."""
    r = math.sqrt(x_km**2 + y_km**2 + z_km**2)
    lat = math.degrees(math.asin(z_km / r)) if r > 0 else 0.0
    lon = math.degrees(math.atan2(y_km, x_km))
    alt_m = (r - EARTH_RADIUS_KM) * KM_TO_M
    return lat, lon, alt_m


def _velocity_magnitude(vx: float, vy: float, vz: float) -> float:
    """Compute velocity magnitude in m/s from SGP4 km/s components."""
    return math.sqrt(vx**2 + vy**2 + vz**2) * KM_TO_M


class CelesTrakAdapter:
    """Fetches TLEs from CelesTrak and publishes satellite positions via MQTT."""

    TLE_URL = "https://celestrak.org/NORAD/elements/gp.php"

    def __init__(
        self,
        mqtt_client: Any,
        max_sats: int = int(os.getenv("CELESTRAK_MAX_SATS", "200")),
        tle_refresh: float = float(os.getenv("CELESTRAK_TLE_REFRESH", "21600")),
        propagate_interval: float = float(os.getenv("CELESTRAK_PROPAGATE_INTERVAL", "30")),
        group: str = os.getenv("CELESTRAK_GROUP", "active"),
    ):
        self.mqtt = mqtt_client
        self.max_sats = max_sats
        self.tle_refresh = tle_refresh
        self.propagate_interval = max(propagate_interval, 5)
        self.group = group
        self._stop = asyncio.Event()
        self._satellites: List[Dict[str, Any]] = []  # [{name, norad_id, satrec}, ...]
        self._last_tle_fetch: float = 0
        self._stats = {"tle_fetches": 0, "propagations": 0, "satellites": 0, "errors": 0}

    @property
    def enabled(self) -> bool:
        return os.getenv("CELESTRAK_ENABLED", "true").lower() == "true" and SGP4_AVAILABLE

    async def start(self):
        if not self.enabled:
            if not SGP4_AVAILABLE:
                logger.warning("CelesTrak adapter disabled: sgp4 not installed (pip install sgp4)")
            else:
                logger.info("CelesTrak adapter disabled")
            return

        logger.info(
            f"CelesTrak adapter starting (max_sats={self.max_sats}, "
            f"group={self.group}, propagate_interval={self.propagate_interval}s)"
        )

        async with httpx.AsyncClient(timeout=60.0) as client:
            while not self._stop.is_set():
                try:
                    # Refresh TLEs if needed
                    if time.time() - self._last_tle_fetch > self.tle_refresh:
                        await self._fetch_tles(client)

                    # Propagate and publish
                    if self._satellites:
                        self._propagate_and_publish()

                except Exception as e:
                    self._stats["errors"] += 1
                    logger.error(f"CelesTrak error: {e}")

                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self.propagate_interval
                    )
                    break
                except asyncio.TimeoutError:
                    pass

        logger.info(f"CelesTrak adapter stopped (stats={self._stats})")

    async def stop(self):
        self._stop.set()

    async def _fetch_tles(self, client: httpx.AsyncClient):
        """Fetch TLE data from CelesTrak and parse into Satrec objects."""
        params = {"GROUP": self.group, "FORMAT": "tle"}
        resp = await client.get(self.TLE_URL, params=params)
        resp.raise_for_status()
        text = resp.text

        lines = [l.rstrip() for l in text.strip().split("\n") if l.strip()]
        satellites = []

        i = 0
        while i + 2 < len(lines) and len(satellites) < self.max_sats:
            name_line = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()

            if not line1.startswith("1 ") or not line2.startswith("2 "):
                i += 1
                continue

            try:
                satrec = Satrec.twoline2rv(line1, line2)
                norad_id = line1[2:7].strip()
                satellites.append({
                    "name": name_line,
                    "norad_id": norad_id,
                    "satrec": satrec,
                    "line1": line1,
                    "line2": line2,
                })
            except Exception as e:
                logger.debug(f"Failed to parse TLE for {name_line}: {e}")

            i += 3

        self._satellites = satellites
        self._last_tle_fetch = time.time()
        self._stats["tle_fetches"] += 1
        self._stats["satellites"] = len(satellites)
        logger.info(f"CelesTrak: loaded {len(satellites)} satellite TLEs")

    def _propagate_and_publish(self):
        """Propagate all satellites to current time and publish positions."""
        now = datetime.now(timezone.utc)
        jd, fr = jday(
            now.year, now.month, now.day,
            now.hour, now.minute, now.second + now.microsecond / 1e6,
        )

        now_iso = now.isoformat()
        published = 0

        for sat in self._satellites:
            try:
                satrec = sat["satrec"]
                e, r, v = satrec.sgp4(jd, fr)
                if e != 0:
                    continue

                lat, lon, alt_m = _ecef_to_lla(r[0], r[1], r[2])
                speed_mps = _velocity_magnitude(v[0], v[1], v[2])

                entity_id = f"celestrak-{sat['norad_id']}"
                payload = {
                    "entity_id": entity_id,
                    "id": entity_id,
                    "entity_type": "TRACK",
                    "domain": "AERIAL",
                    "state": "ACTIVE",
                    "name": sat["name"],
                    "class_label": "satellite",
                    "confidence": 1.0,
                    "kinematics": {
                        "position": {
                            "latitude": lat,
                            "longitude": lon,
                            "altitude_msl": alt_m,
                            "altitude_agl": 0.0,
                        },
                        "heading_deg": 0.0,
                        "speed_mps": speed_mps,
                        "climb_rate": 0.0,
                    },
                    "provenance": {
                        "source_id": "celestrak",
                        "source_type": "tle",
                        "org_id": "",
                        "created_at": time.time(),
                        "updated_at": time.time(),
                        "version": 1,
                    },
                    "metadata": {
                        "norad_id": sat["norad_id"],
                        "tle_line1": sat["line1"],
                        "tle_line2": sat["line2"],
                        "source": "celestrak",
                    },
                    "ttl_seconds": 120,
                    "ts": now_iso,
                }

                topic = f"entities/{entity_id}/update"
                self.mqtt.publish(topic, json.dumps(payload), qos=0)
                published += 1

            except Exception as e:
                logger.debug(f"Propagation error for {sat.get('name')}: {e}")

        self._stats["propagations"] += 1
        logger.info(f"CelesTrak: propagated {published}/{len(self._satellites)} satellites")
