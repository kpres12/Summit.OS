"""
Summit.OS OpenSky Network Adapter

Polls the OpenSky Network REST API for live ADS-B aircraft positions
and publishes each aircraft as an Entity into the Summit.OS data fabric
via MQTT.

OpenSky API docs: https://openskynetwork.github.io/opensky-api/rest.html

Environment variables:
    OPENSKY_ENABLED          - "true" to enable (default: "true")
    OPENSKY_POLL_INTERVAL    - seconds between polls (default: 10)
    OPENSKY_BBOX             - bounding box "lat_min,lon_min,lat_max,lon_max"
                               (default: empty = worldwide)
    OPENSKY_USERNAME         - optional OpenSky credentials for higher rate limits
    OPENSKY_PASSWORD         - optional
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

logger = logging.getLogger("summit.adapter.opensky")

# OpenSky state vector indices
# https://openskynetwork.github.io/opensky-api/rest.html#all-state-vectors
IDX_ICAO24 = 0
IDX_CALLSIGN = 1
IDX_ORIGIN_COUNTRY = 2
IDX_TIME_POSITION = 3
IDX_LAST_CONTACT = 4
IDX_LONGITUDE = 5
IDX_LATITUDE = 6
IDX_BARO_ALTITUDE = 7
IDX_ON_GROUND = 8
IDX_VELOCITY = 9
IDX_TRUE_TRACK = 10
IDX_VERTICAL_RATE = 11
IDX_SENSORS = 12
IDX_GEO_ALTITUDE = 13
IDX_SQUAWK = 14
IDX_SPI = 15
IDX_POSITION_SOURCE = 16


class OpenSkyAdapter:
    """Polls OpenSky Network and publishes aircraft entities to MQTT."""

    API_URL = "https://opensky-network.org/api/states/all"

    def __init__(
        self,
        mqtt_client: Any,
        poll_interval: float = float(os.getenv("OPENSKY_POLL_INTERVAL", "10")),
        bbox: Optional[str] = os.getenv("OPENSKY_BBOX", ""),
        username: Optional[str] = os.getenv("OPENSKY_USERNAME"),
        password: Optional[str] = os.getenv("OPENSKY_PASSWORD"),
    ):
        self.mqtt = mqtt_client
        self.poll_interval = max(poll_interval, 5)  # OpenSky rate-limits below 5s
        self.username = username
        self.password = password
        self._stop = asyncio.Event()
        self._bbox: Optional[Tuple[float, float, float, float]] = None
        self._stats = {"polls": 0, "aircraft": 0, "errors": 0}

        if bbox and bbox.strip():
            parts = [float(x.strip()) for x in bbox.split(",")]
            if len(parts) == 4:
                self._bbox = (parts[0], parts[1], parts[2], parts[3])

    @property
    def enabled(self) -> bool:
        return os.getenv("OPENSKY_ENABLED", "true").lower() == "true"

    async def start(self):
        """Run the polling loop until stop() is called."""
        if not self.enabled:
            logger.info("OpenSky adapter disabled")
            return

        logger.info(
            f"OpenSky adapter starting (interval={self.poll_interval}s, "
            f"bbox={self._bbox or 'worldwide'})"
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            while not self._stop.is_set():
                try:
                    await self._poll(client)
                except Exception as e:
                    self._stats["errors"] += 1
                    logger.error(f"OpenSky poll error: {e}")

                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval)
                    break  # stop was set
                except asyncio.TimeoutError:
                    pass  # normal — just means interval elapsed

        logger.info(f"OpenSky adapter stopped (stats={self._stats})")

    async def stop(self):
        self._stop.set()

    async def _poll(self, client: httpx.AsyncClient):
        """Fetch aircraft states from OpenSky and publish to MQTT."""
        params: Dict[str, Any] = {}
        if self._bbox:
            lat_min, lon_min, lat_max, lon_max = self._bbox
            params.update({
                "lamin": lat_min, "lomin": lon_min,
                "lamax": lat_max, "lomax": lon_max,
            })

        auth = None
        if self.username and self.password:
            auth = httpx.BasicAuth(self.username, self.password)

        resp = await client.get(self.API_URL, params=params, auth=auth)
        resp.raise_for_status()
        data = resp.json()

        states: List[list] = data.get("states") or []
        self._stats["polls"] += 1
        self._stats["aircraft"] = len(states)

        now_iso = datetime.now(timezone.utc).isoformat()
        published = 0

        for sv in states:
            try:
                entity_payload = self._state_vector_to_entity(sv, now_iso)
                if entity_payload:
                    topic = f"entities/{entity_payload['entity_id']}/update"
                    self.mqtt.publish(topic, json.dumps(entity_payload), qos=0)
                    published += 1
            except Exception as e:
                logger.debug(f"Skipping aircraft: {e}")

        logger.info(f"OpenSky: published {published}/{len(states)} aircraft")

    @staticmethod
    def _state_vector_to_entity(sv: list, now_iso: str) -> Optional[Dict[str, Any]]:
        """Convert an OpenSky state vector array to a Summit.OS entity dict."""
        icao24 = sv[IDX_ICAO24]
        if not icao24:
            return None

        lat = sv[IDX_LATITUDE]
        lon = sv[IDX_LONGITUDE]
        if lat is None or lon is None:
            return None

        baro_alt = sv[IDX_BARO_ALTITUDE] or 0
        geo_alt = sv[IDX_GEO_ALTITUDE] or baro_alt
        velocity = sv[IDX_VELOCITY] or 0
        heading = sv[IDX_TRUE_TRACK] or 0
        vert_rate = sv[IDX_VERTICAL_RATE] or 0
        callsign = (sv[IDX_CALLSIGN] or "").strip()
        on_ground = sv[IDX_ON_GROUND] or False
        squawk = sv[IDX_SQUAWK] or ""
        origin = sv[IDX_ORIGIN_COUNTRY] or ""

        entity_id = f"opensky-{icao24}"

        return {
            "entity_id": entity_id,
            "id": entity_id,
            "entity_type": "TRACK",
            "domain": "AERIAL",
            "state": "ACTIVE",
            "name": callsign or icao24.upper(),
            "class_label": "aircraft",
            "confidence": 1.0,
            "kinematics": {
                "position": {
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "altitude_msl": float(baro_alt),
                    "altitude_agl": 0.0,
                },
                "heading_deg": float(heading),
                "speed_mps": float(velocity),
                "climb_rate": float(vert_rate),
            },
            "aerial": {
                "altitude_agl": 0.0,
                "altitude_msl": float(geo_alt),
                "airspeed_mps": float(velocity),
                "flight_mode": "GROUND" if on_ground else "AIRBORNE",
                "battery_pct": 0.0,
                "link_quality": "",
            },
            "provenance": {
                "source_id": "opensky",
                "source_type": "adsb",
                "org_id": "",
                "created_at": time.time(),
                "updated_at": time.time(),
                "version": 1,
            },
            "metadata": {
                "icao24": icao24,
                "callsign": callsign,
                "origin_country": origin,
                "squawk": squawk,
                "on_ground": str(on_ground).lower(),
                "source": "opensky",
            },
            "ttl_seconds": 60,
            "ts": now_iso,
        }
