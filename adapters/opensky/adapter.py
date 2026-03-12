"""
Summit.OS OpenSky Network Adapter

Polls the OpenSky Network REST API for live ADS-B aircraft positions
and publishes each aircraft as a Summit.OS Entity into the data fabric.

Built on BaseAdapter — manifests its capabilities, uses EntityBuilder
for schema-consistent entity construction, and self-manages MQTT.

Environment variables:
    OPENSKY_ENABLED          - "true" to enable (default: "true")
    OPENSKY_POLL_INTERVAL    - seconds between polls (default: 10)
    OPENSKY_BBOX             - bounding box "lat_min,lon_min,lat_max,lon_max"
    OPENSKY_USERNAME         - optional credentials for higher rate limits
    OPENSKY_PASSWORD         - optional
    MQTT_HOST / MQTT_PORT    - broker connection
"""
from __future__ import annotations

import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Make SDK importable when running from adapters/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from sdk import BaseAdapter, AdapterManifest, EntityBuilder, Protocol, Capability

logger = logging.getLogger("summit.adapter.opensky")

# OpenSky state vector field indices
IDX_ICAO24, IDX_CALLSIGN, IDX_ORIGIN_COUNTRY = 0, 1, 2
IDX_LONGITUDE, IDX_LATITUDE = 5, 6
IDX_BARO_ALTITUDE, IDX_ON_GROUND = 7, 8
IDX_VELOCITY, IDX_TRUE_TRACK, IDX_VERTICAL_RATE = 9, 10, 11
IDX_GEO_ALTITUDE, IDX_SQUAWK = 13, 14


class OpenSkyAdapter(BaseAdapter):
    """Polls OpenSky Network and publishes aircraft TRACK entities."""

    MANIFEST = AdapterManifest(
        name="opensky",
        version="1.0.0",
        protocol=Protocol.ADSB,
        capabilities=[Capability.READ, Capability.SUBSCRIBE],
        entity_types=["TRACK"],
        description="Live ADS-B aircraft positions from the OpenSky Network",
        homepage="https://opensky-network.org",
        optional_env=["OPENSKY_USERNAME", "OPENSKY_PASSWORD", "OPENSKY_BBOX"],
    )

    def __init__(
        self,
        poll_interval: float = float(os.getenv("OPENSKY_POLL_INTERVAL", "10")),
        bbox: Optional[str] = os.getenv("OPENSKY_BBOX", ""),
        username: Optional[str] = os.getenv("OPENSKY_USERNAME"),
        password: Optional[str] = os.getenv("OPENSKY_PASSWORD"),
        **kwargs,
    ):
        super().__init__(device_id="opensky", **kwargs)
        self.poll_interval = max(poll_interval, 5.0)  # OpenSky rate-limits below 5s
        self.username = username
        self.password = password
        self._bbox: Optional[Tuple[float, float, float, float]] = None
        self._stats = {"polls": 0, "aircraft": 0, "errors": 0}

        if bbox and bbox.strip():
            parts = [float(x.strip()) for x in bbox.split(",")]
            if len(parts) == 4:
                self._bbox = (parts[0], parts[1], parts[2], parts[3])

    @property
    def enabled(self) -> bool:
        return os.getenv("OPENSKY_ENABLED", "true").lower() == "true"

    async def run(self):
        logger.info(
            f"OpenSky adapter running (interval={self.poll_interval}s, "
            f"bbox={self._bbox or 'worldwide'})"
        )
        async with httpx.AsyncClient(timeout=30.0) as client:
            while not self.stopped:
                try:
                    await self._poll(client)
                except Exception as e:
                    self._stats["errors"] += 1
                    logger.error(f"OpenSky poll error: {e}")
                await self.sleep(self.poll_interval)

    async def _poll(self, client: httpx.AsyncClient):
        params: Dict[str, Any] = {}
        if self._bbox:
            lat_min, lon_min, lat_max, lon_max = self._bbox
            params.update({"lamin": lat_min, "lomin": lon_min,
                           "lamax": lat_max, "lomax": lon_max})

        auth = httpx.BasicAuth(self.username, self.password) if (self.username and self.password) else None
        resp = await client.get("https://opensky-network.org/api/states/all", params=params, auth=auth)
        resp.raise_for_status()

        states: List[list] = resp.json().get("states") or []
        self._stats["polls"] += 1
        self._stats["aircraft"] = len(states)
        published = 0

        for sv in states:
            try:
                entity = self._state_vector_to_entity(sv)
                if entity:
                    self.publish(entity, qos=0)
                    published += 1
            except Exception as e:
                logger.debug(f"Skipping aircraft: {e}")

        logger.info(f"OpenSky: published {published}/{len(states)} aircraft")

    def _state_vector_to_entity(self, sv: list) -> Optional[Dict[str, Any]]:
        icao24 = sv[IDX_ICAO24]
        lat = sv[IDX_LATITUDE]
        lon = sv[IDX_LONGITUDE]
        if not icao24 or lat is None or lon is None:
            return None

        callsign = (sv[IDX_CALLSIGN] or "").strip()
        baro_alt = float(sv[IDX_BARO_ALTITUDE] or 0)
        geo_alt  = float(sv[IDX_GEO_ALTITUDE] or baro_alt)
        velocity = float(sv[IDX_VELOCITY] or 0)
        heading  = float(sv[IDX_TRUE_TRACK] or 0)
        vert_rate = float(sv[IDX_VERTICAL_RATE] or 0)
        on_ground = bool(sv[IDX_ON_GROUND])
        squawk   = sv[IDX_SQUAWK] or ""
        origin   = sv[IDX_ORIGIN_COUNTRY] or ""

        return (
            EntityBuilder(f"opensky-{icao24}", callsign or icao24.upper())
            .track()
            .aerial()
            .at(float(lat), float(lon), baro_alt)
            .moving(heading, velocity, vert_rate)
            .label("aircraft")
            .source("adsb", "opensky")
            .org(self.org_id)
            .ttl(60)
            .aerial_telemetry(
                flight_mode="GROUND" if on_ground else "AIRBORNE",
                airspeed_mps=velocity,
            )
            .meta_dict({
                "icao24": icao24,
                "callsign": callsign,
                "origin_country": origin,
                "squawk": squawk,
                "on_ground": str(on_ground).lower(),
            })
            .build()
        )
