"""
Summit.OS CelesTrak Satellite Adapter

Fetches Two-Line Element (TLE) sets from CelesTrak, propagates satellite
positions to the current epoch using SGP4, and publishes each satellite
as a Summit.OS TRACK entity into the data fabric via MQTT.

Environment variables:
    CELESTRAK_ENABLED            - "true" to enable (default: "true")
    CELESTRAK_MAX_SATS           - max satellites to track (default: 200)
    CELESTRAK_TLE_REFRESH        - seconds between TLE re-fetch (default: 21600)
    CELESTRAK_PROPAGATE_INTERVAL - seconds between position updates (default: 30)
    CELESTRAK_GROUP              - TLE group (default: "active")
    MQTT_HOST / MQTT_PORT        - broker connection
"""
from __future__ import annotations

import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from sdk import BaseAdapter, AdapterManifest, EntityBuilder, Protocol, Capability

try:
    from sgp4.api import Satrec, jday
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False

logger = logging.getLogger("summit.adapter.celestrak")

EARTH_RADIUS_KM = 6371.0


def _ecef_to_lla(x_km: float, y_km: float, z_km: float) -> Tuple[float, float, float]:
    r = math.sqrt(x_km**2 + y_km**2 + z_km**2)
    lat = math.degrees(math.asin(z_km / r)) if r > 0 else 0.0
    lon = math.degrees(math.atan2(y_km, x_km))
    alt_m = (r - EARTH_RADIUS_KM) * 1000.0
    return lat, lon, alt_m


def _velocity_mps(vx: float, vy: float, vz: float) -> float:
    return math.sqrt(vx**2 + vy**2 + vz**2) * 1000.0


class CelesTrakAdapter(BaseAdapter):
    """Fetches TLEs from CelesTrak and publishes satellite TRACK entities."""

    MANIFEST = AdapterManifest(
        name="celestrak",
        version="1.0.0",
        protocol=Protocol.TLE,
        capabilities=[Capability.READ],
        entity_types=["TRACK"],
        description="Satellite positions propagated from CelesTrak TLE data via SGP4",
        homepage="https://celestrak.org",
        optional_env=["CELESTRAK_GROUP", "CELESTRAK_MAX_SATS"],
    )

    def __init__(
        self,
        max_sats: int = int(os.getenv("CELESTRAK_MAX_SATS", "200")),
        tle_refresh: float = float(os.getenv("CELESTRAK_TLE_REFRESH", "21600")),
        propagate_interval: float = float(os.getenv("CELESTRAK_PROPAGATE_INTERVAL", "30")),
        group: str = os.getenv("CELESTRAK_GROUP", "active"),
        **kwargs,
    ):
        super().__init__(device_id="celestrak", **kwargs)
        self.max_sats = max_sats
        self.tle_refresh = tle_refresh
        self.propagate_interval = max(propagate_interval, 5.0)
        self.group = group
        self._satellites: List[Dict[str, Any]] = []
        self._last_tle_fetch: float = 0
        self._stats = {"tle_fetches": 0, "propagations": 0, "satellites": 0, "errors": 0}

    @property
    def enabled(self) -> bool:
        if not SGP4_AVAILABLE:
            logger.warning("CelesTrak disabled: sgp4 not installed (pip install sgp4)")
            return False
        return os.getenv("CELESTRAK_ENABLED", "true").lower() == "true"

    async def run(self):
        logger.info(
            f"CelesTrak adapter running (max_sats={self.max_sats}, "
            f"group={self.group}, interval={self.propagate_interval}s)"
        )
        async with httpx.AsyncClient(timeout=60.0) as client:
            while not self.stopped:
                try:
                    if time.time() - self._last_tle_fetch > self.tle_refresh:
                        await self._fetch_tles(client)
                    if self._satellites:
                        self._propagate_and_publish()
                except Exception as e:
                    self._stats["errors"] += 1
                    logger.error(f"CelesTrak error: {e}")
                await self.sleep(self.propagate_interval)

    async def _fetch_tles(self, client: httpx.AsyncClient):
        resp = await client.get(
            "https://celestrak.org/NORAD/elements/gp.php",
            params={"GROUP": self.group, "FORMAT": "tle"},
        )
        resp.raise_for_status()
        lines = [l.rstrip() for l in resp.text.strip().split("\n") if l.strip()]
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
                satellites.append({"name": name_line, "norad_id": norad_id,
                                   "satrec": satrec, "line1": line1, "line2": line2})
            except Exception as e:
                logger.debug(f"Failed to parse TLE for {name_line}: {e}")
            i += 3

        self._satellites = satellites
        self._last_tle_fetch = time.time()
        self._stats["tle_fetches"] += 1
        self._stats["satellites"] = len(satellites)
        logger.info(f"CelesTrak: loaded {len(satellites)} satellite TLEs")

    def _propagate_and_publish(self):
        now = datetime.now(timezone.utc)
        jd, fr = jday(now.year, now.month, now.day,
                      now.hour, now.minute, now.second + now.microsecond / 1e6)
        published = 0
        for sat in self._satellites:
            try:
                e, r, v = sat["satrec"].sgp4(jd, fr)
                if e != 0:
                    continue
                lat, lon, alt_m = _ecef_to_lla(r[0], r[1], r[2])
                speed = _velocity_mps(v[0], v[1], v[2])
                entity = (
                    EntityBuilder(f"celestrak-{sat['norad_id']}", sat["name"])
                    .track()
                    .aerial()
                    .at(lat, lon, alt_m)
                    .moving(0.0, speed)
                    .label("satellite")
                    .source("tle", "celestrak")
                    .org(self.org_id)
                    .ttl(120)
                    .meta_dict({
                        "norad_id": sat["norad_id"],
                        "tle_line1": sat["line1"],
                        "tle_line2": sat["line2"],
                    })
                    .build()
                )
                self.publish(entity, qos=0)
                published += 1
            except Exception as e:
                logger.debug(f"Propagation error for {sat.get('name')}: {e}")
        self._stats["propagations"] += 1
        logger.info(f"CelesTrak: propagated {published}/{len(self._satellites)} satellites")
