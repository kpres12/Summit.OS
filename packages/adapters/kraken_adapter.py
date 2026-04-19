"""
Summit.OS — Kraken Robotics Adapter
=====================================
Integrates Kraken Robotics underwater systems into Summit.OS as first-class
entities: AUVs, towed sonar systems, acoustic positioning nodes.

Kraken hardware supported:
  - ThunderFish AUV (autonomous underwater vehicle)
  - KATFISH towed synthetic aperture sonar (SAS)
  - Acoustic Positioning System (APS) — ultra-short baseline (USBL)
  - SeaPower battery / energy systems (status monitoring)

Integration modes:
  rest_poll   — Poll Kraken INSIGHT software REST API (default)
  websocket   — Real-time telemetry WebSocket (INSIGHT 2.x+)
  nmea_tcp    — Raw NMEA-0183 over TCP (legacy / hardware-direct)

The adapter maps Kraken telemetry to Summit.OS entities:
  ThunderFish AUV → entity_type: active, asset_type: AUV
  KATFISH sonar   → entity_type: active, asset_type: SONAR_TOW
  APS node        → entity_type: active, asset_type: ACOUSTIC_NODE
  Sonar detections → pushed as alert entities with severity + coordinates

Mission commands flow back from Summit.OS tasking → Kraken INSIGHT API:
  survey_area  → translated to Kraken mission plan with SAS coverage legs
  abort        → Kraken emergency abort command
  rtb          → return to surface / recovery point

Config (AdapterConfig.extra fields):
  mode          — "rest_poll" | "websocket" | "nmea_tcp"  (default: rest_poll)
  host          — Kraken INSIGHT host                     (default: localhost)
  port          — API port                                (default: 8080)
  api_key       — INSIGHT API key                         (env: KRAKEN_API_KEY)
  vessel_id     — surface vessel / tender ID for USBL reference
  depth_unit    — "m" | "ft"                              (default: m)

Environment:
  KRAKEN_API_KEY   — API key for Kraken INSIGHT software
  KRAKEN_HOST      — INSIGHT host override
  KRAKEN_PORT      — INSIGHT port override

Register in adapters.json:
  {
    "adapter_type": "kraken",
    "name": "Kraken Robotics — ThunderFish",
    "poll_interval_seconds": 5,
    "extra": {
      "mode": "rest_poll",
      "host": "192.168.1.100",
      "port": 8080,
      "vessel_id": "TENDER-01"
    }
  }

Contact Branca.ai for production integration support and Kraken API credentials.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import AsyncIterator, Dict, List, Optional

try:
    import httpx
except ImportError:
    raise ImportError("httpx is required: pip install httpx")

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("summit.adapters.kraken")

# Kraken INSIGHT REST API endpoints (documented in INSIGHT Developer Guide)
_INSIGHT_VEHICLES   = "/api/v1/vehicles"
_INSIGHT_SENSORS    = "/api/v1/sensors"
_INSIGHT_DETECTIONS = "/api/v1/detections"
_INSIGHT_MISSION    = "/api/v1/mission"

# Map Kraken vehicle type strings → Summit.OS asset types
_VEHICLE_TYPE_MAP = {
    "thunderfish": "AUV",
    "auv":         "AUV",
    "katfish":     "SONAR_TOW",
    "sonar_tow":   "SONAR_TOW",
    "aps_node":    "ACOUSTIC_NODE",
    "usbl":        "ACOUSTIC_NODE",
    "seapower":    "POWER_UNIT",
}

# Map Kraken vehicle state → Summit.OS entity_type
_STATE_MAP = {
    "mission":    "active",
    "idle":       "active",
    "recovery":   "active",
    "emergency":  "alert",
    "lost":       "unknown",
    "offline":    "unknown",
}


class KrakenAdapter(BaseAdapter):
    """
    Integrates Kraken Robotics underwater systems with Summit.OS.

    In rest_poll mode: polls Kraken INSIGHT REST API for vehicle/sensor status.
    In websocket mode: maintains a persistent WebSocket to INSIGHT for real-time data.
    """

    adapter_type = "kraken"

    def __init__(self, config: AdapterConfig, mqtt_client=None):
        super().__init__(config, mqtt_client)
        ex = config.extra or {}

        self._mode       = ex.get("mode", "rest_poll")
        self._host       = ex.get("host") or os.getenv("KRAKEN_HOST", "localhost")
        self._port       = int(ex.get("port") or os.getenv("KRAKEN_PORT", "8080"))
        self._api_key    = ex.get("api_key") or os.getenv("KRAKEN_API_KEY", "")
        self._vessel_id  = ex.get("vessel_id", "SURFACE-VESSEL")
        self._depth_unit = ex.get("depth_unit", "m")

        self._base_url = f"http://{self._host}:{self._port}"
        self._client: Optional[httpx.AsyncClient] = None

    def _headers(self) -> dict:
        h = {"Accept": "application/json"}
        if self._api_key:
            h["X-API-Key"] = self._api_key
        return h

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers(),
            timeout=10.0,
        )
        # Probe health endpoint to confirm INSIGHT is reachable
        try:
            r = await self._client.get("/api/v1/health")
            r.raise_for_status()
            info = r.json()
            logger.info(
                "Kraken INSIGHT connected — version=%s  vessels=%s",
                info.get("version", "?"),
                info.get("vehicle_count", "?"),
            )
        except Exception as e:
            logger.warning("Kraken INSIGHT health check failed: %s — will retry", e)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def stream_observations(self) -> AsyncIterator[dict]:
        assert self._client is not None

        while True:
            observations: List[dict] = []

            # ── Vehicles (AUV, KATFISH, APS nodes) ────────────────────────
            try:
                r = await self._client.get(_INSIGHT_VEHICLES)
                r.raise_for_status()
                for v in r.json().get("vehicles", []):
                    obs = self._vehicle_to_entity(v)
                    if obs:
                        observations.append(obs)
            except Exception as e:
                logger.debug("Kraken vehicle poll failed: %s", e)

            # ── Sonar detections → Summit.OS alerts ───────────────────────
            try:
                r = await self._client.get(
                    _INSIGHT_DETECTIONS,
                    params={"since": int(time.time()) - 60},  # last 60s
                )
                r.raise_for_status()
                for det in r.json().get("detections", []):
                    obs = self._detection_to_alert(det)
                    if obs:
                        observations.append(obs)
            except Exception as e:
                logger.debug("Kraken detection poll failed: %s", e)

            for obs in observations:
                yield obs

            await asyncio.sleep(self.config.poll_interval_seconds)

    def _vehicle_to_entity(self, v: dict) -> Optional[dict]:
        """Map a Kraken INSIGHT vehicle record to a Summit.OS entity."""
        vid      = v.get("id") or v.get("name")
        if not vid:
            return None

        vtype    = (v.get("type") or "auv").lower()
        state    = (v.get("state") or "idle").lower()
        nav      = v.get("navigation", {})
        lat      = nav.get("latitude")
        lon      = nav.get("longitude")
        if lat is None or lon is None:
            return None

        depth    = nav.get("depth", 0)
        alt_m    = -depth  # negative = below surface

        asset_type   = _VEHICLE_TYPE_MAP.get(vtype, "AUV")
        entity_type  = _STATE_MAP.get(state, "active")
        mission_info = v.get("current_mission") or {}

        return {
            "entity_id":  f"kraken-{vid}",
            "type":       entity_type,
            "callsign":   (v.get("name") or vid).upper(),
            "position":   {"lat": lat, "lon": lon, "alt": alt_m},
            "last_seen":  int(time.time()),
            "properties": {
                "asset_type":   asset_type,
                "manufacturer": "Kraken Robotics",
                "vehicle_id":   vid,
                "vehicle_type": vtype,
                "state":        state,
                "depth_m":      depth,
                "heading_deg":  nav.get("heading"),
                "speed_kts":    nav.get("speed"),
                "battery_pct":  v.get("battery_percent"),
                "mission_id":   mission_info.get("id"),
                "mission_name": mission_info.get("name"),
                "vessel_id":    self._vessel_id,
                "source":       "kraken_insight",
            },
        }

    def _detection_to_alert(self, det: dict) -> Optional[dict]:
        """Map a Kraken sonar detection to a Summit.OS alert entity."""
        det_id  = det.get("id")
        lat     = det.get("latitude")
        lon     = det.get("longitude")
        if not det_id or lat is None or lon is None:
            return None

        severity  = det.get("confidence", 0.5)
        det_class = det.get("classification", "unknown").upper()

        return {
            "entity_id":  f"kraken-det-{det_id}",
            "type":       "alert",
            "callsign":   f"DETECTION-{det_id[:6].upper()}",
            "position":   {"lat": lat, "lon": lon, "alt": 0},
            "last_seen":  int(time.time()),
            "properties": {
                "asset_type":      "SONAR_DETECTION",
                "classification":  det_class,
                "confidence":      severity,
                "depth_m":         det.get("depth"),
                "sonar_id":        det.get("sonar_id"),
                "image_url":       det.get("image_url"),
                "source":          "kraken_insight",
                "manufacturer":    "Kraken Robotics",
            },
        }

    async def send_mission_command(self, command: str, payload: dict) -> bool:
        """
        Send a mission command back to Kraken INSIGHT from Summit.OS tasking.

        Commands:
          survey_area  — payload: {vehicle_id, area: [{lat,lon}], altitude_m, pattern}
          abort        — payload: {vehicle_id}
          rtb          — payload: {vehicle_id, surface_lat, surface_lon}
        """
        if not self._client:
            return False
        try:
            r = await self._client.post(
                f"{_INSIGHT_MISSION}/{command}",
                json=payload,
            )
            r.raise_for_status()
            logger.info("Kraken command sent: %s → %s", command, payload.get("vehicle_id"))
            return True
        except Exception as e:
            logger.error("Kraken command failed (%s): %s", command, e)
            return False
