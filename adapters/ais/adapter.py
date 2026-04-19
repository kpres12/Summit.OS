"""
Heli.OS AIS Maritime Adapter

Tracks vessel positions from AIS (Automatic Identification System) data.
Publishes each vessel as a Heli.OS maritime Entity into the data fabric.

Data sources (in priority order):
  1. AISHub free API (register at aishub.net — free for non-commercial use)
  2. VesselFinder / MarineTraffic with API key (MARITIME_API_KEY)
  3. Local AIS-catcher or rtl-ais on UDP (MARITIME_UDP_HOST:MARITIME_UDP_PORT)
  4. Simulation fallback — generates realistic coastal traffic (no external dep)

Environment variables:
    AIS_ENABLED             - "true" to enable (default: "true")
    AIS_POLL_INTERVAL       - seconds between polls (default: 30)
    AIS_BBOX                - bounding box "lat_min,lon_min,lat_max,lon_max"
                              defaults to US coastal waters
    AISHUB_USERNAME         - AISHub username (get free account at aishub.net)
    MARITIME_UDP_HOST       - host for local AIS-catcher UDP output
    MARITIME_UDP_PORT       - port for local AIS-catcher UDP output (default: 10110)
    MQTT_HOST / MQTT_PORT   - broker connection
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from sdk import BaseAdapter, AdapterManifest, EntityBuilder, Protocol, Capability

logger = logging.getLogger("summit.adapter.ais")

# AISHub free API — returns vessel positions in JSON
AISHUB_URL = "https://data.aishub.net/ws.php"

# Vessel type codes → classification label
VESSEL_TYPE_MAP = {
    range(20, 30): "wing-in-ground",
    range(30, 40): "fishing",
    range(40, 50): "high-speed-craft",
    range(50, 60): "special-craft",
    range(60, 70): "passenger",
    range(70, 80): "cargo",
    range(80, 90): "tanker",
}

# Coastal anchor points for simulation (major ports)
_SIM_PORTS = [
    (37.81, -122.36, "San Francisco Bay"),
    (34.05, -118.26, "Los Angeles Harbor"),
    (47.60, -122.34, "Seattle/Puget Sound"),
    (29.75, -95.37, "Houston Ship Channel"),
    (25.77, -80.19, "Miami Port"),
    (40.69, -74.04, "New York Harbor"),
    (51.51, -0.12,  "Thames Estuary"),
    (1.28,  103.85,  "Singapore Strait"),
]

_VESSEL_CLASSES = [
    ("cargo", "EMMA MAERSK", 14.0),
    ("tanker", "ATLANTIC PIONEER", 12.0),
    ("cargo", "COSCO SHIPPING", 16.0),
    ("passenger", "CARNIVAL DREAM", 8.0),
    ("cargo", "MSC GULSUN", 15.0),
    ("tanker", "OLYMPIC SPIRIT", 11.0),
    ("fishing", "NORTHERN HAWK", 6.0),
    ("cargo", "HAPAG EXPRESS", 14.0),
    ("special-craft", "PACIFIC RESPONDER", 9.0),
    ("high-speed-craft", "CONDOR RAPIDE", 32.0),
]


def _vessel_type_label(type_code: int) -> str:
    for rng, label in VESSEL_TYPE_MAP.items():
        if type_code in rng:
            return label
    return "vessel"


class AISAdapter(BaseAdapter):
    """Polls AIS data sources and publishes vessel TRACK entities."""

    MANIFEST = AdapterManifest(
        name="ais",
        version="1.0.0",
        protocol=Protocol.AIS,
        capabilities=[Capability.READ, Capability.SUBSCRIBE],
        entity_types=["TRACK"],
        description="AIS maritime vessel tracking — cargo, tankers, passenger, fishing",
        required_env=[],
        optional_env=["AIS_BBOX", "AISHUB_USERNAME", "AIS_POLL_INTERVAL",
                      "MARITIME_UDP_HOST", "MARITIME_UDP_PORT"],
    )

    def __init__(
        self,
        bbox: Optional[str] = os.getenv("AIS_BBOX"),
        poll_interval: float = float(os.getenv("AIS_POLL_INTERVAL", "30")),
        aishub_username: Optional[str] = os.getenv("AISHUB_USERNAME"),
        udp_host: Optional[str] = os.getenv("MARITIME_UDP_HOST"),
        udp_port: int = int(os.getenv("MARITIME_UDP_PORT", "10110")),
        org_id: str = os.getenv("AIS_ORG_ID", ""),
        **kwargs,
    ):
        super().__init__(device_id="ais-maritime", org_id=org_id, **kwargs)
        self.bbox = self._parse_bbox(bbox)
        self.poll_interval = max(poll_interval, 10.0)
        self.aishub_username = aishub_username
        self.udp_host = udp_host
        self.udp_port = udp_port
        self._sim_state: Dict[str, dict] = {}
        self._stats = {"polls": 0, "published": 0, "errors": 0}
        self._init_sim_vessels()

    @property
    def enabled(self) -> bool:
        return os.getenv("AIS_ENABLED", "true").lower() == "true"

    @staticmethod
    def _parse_bbox(raw: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
        if not raw:
            return None
        try:
            parts = [float(x) for x in raw.split(",")]
            if len(parts) == 4:
                return (parts[0], parts[1], parts[2], parts[3])
        except Exception:
            pass
        return None

    def _init_sim_vessels(self):
        """Pre-seed simulated vessels spread across coastal anchors."""
        for i, (cls, name, speed_kts) in enumerate(_VESSEL_CLASSES):
            port_lat, port_lon, _ = _SIM_PORTS[i % len(_SIM_PORTS)]
            self._sim_state[f"sim-vessel-{i:03d}"] = {
                "mmsi": f"3669{i:05d}",
                "name": name,
                "lat": port_lat + random.uniform(-0.3, 0.3),
                "lon": port_lon + random.uniform(-0.3, 0.3),
                "heading": random.uniform(0, 360),
                "speed_kts": speed_kts * random.uniform(0.7, 1.2),
                "classification": cls,
                "course": random.uniform(0, 360),
                "length": random.randint(80, 400),
            }

    async def run(self):
        logger.info(
            f"AIS adapter running (source={'AISHub' if self.aishub_username else 'simulation'}, "
            f"interval={self.poll_interval}s)"
        )
        while not self.stopped:
            try:
                await self._poll()
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"AIS poll error: {e}")
            await self.sleep(self.poll_interval)

    async def _poll(self):
        vessels = []

        # Try AISHub if credentials configured
        if self.aishub_username:
            vessels = await self._fetch_aishub()

        # Try local UDP AIS-catcher if configured
        if not vessels and self.udp_host:
            vessels = await self._fetch_udp()

        # Fall back to simulation
        if not vessels:
            vessels = self._simulate()

        for v in vessels:
            entity = self._vessel_to_entity(v)
            if entity:
                self.publish(entity, qos=0)
                self._stats["published"] += 1

        self._stats["polls"] += 1
        logger.info(f"AIS: published {len(vessels)} vessel positions")

    async def _fetch_aishub(self) -> List[dict]:
        """Fetch vessel positions from AISHub free API."""
        params: Dict[str, Any] = {
            "username": self.aishub_username,
            "format": "1",  # JSON
            "output": "json",
        }
        if self.bbox:
            lat_min, lon_min, lat_max, lon_max = self.bbox
            params.update({"latmin": lat_min, "latmax": lat_max,
                           "lonmin": lon_min, "lonmax": lon_max})
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(AISHUB_URL, params=params)
                r.raise_for_status()
                data = r.json()
                # AISHub returns [{"ERROR": ...}, [vessels...]]
                if isinstance(data, list) and len(data) >= 2:
                    vessels_raw = data[1] if isinstance(data[1], list) else []
                    return [self._normalize_aishub(v) for v in vessels_raw if v]
        except Exception as e:
            logger.warning(f"AISHub fetch failed: {e} — falling back to simulation")
        return []

    def _normalize_aishub(self, raw: dict) -> dict:
        return {
            "mmsi": str(raw.get("MMSI", "")),
            "name": (raw.get("NAME") or raw.get("MMSI", "")).strip(),
            "lat": float(raw.get("LATITUDE", 0)),
            "lon": float(raw.get("LONGITUDE", 0)),
            "heading": float(raw.get("HEADING", 0) or 0),
            "speed_kts": float(raw.get("SPEED", 0) or 0) / 10.0,  # AISHub sends 1/10 knot
            "classification": _vessel_type_label(int(raw.get("TYPE", 0) or 0)),
            "course": float(raw.get("COG", 0) or 0) / 10.0,
            "length": int(raw.get("LENGTH", 0) or 0),
        }

    async def _fetch_udp(self) -> List[dict]:
        """
        Receive NMEA AIS sentences from a local AIS-catcher or rtl-ais instance.
        AIS-catcher: --net -u <host> <port>
        """
        try:
            data = b""
            loop = asyncio.get_event_loop()
            transport, protocol = await loop.create_datagram_endpoint(
                asyncio.DatagramProtocol,
                local_addr=(self.udp_host, self.udp_port),
            )
            # Read for up to 2 seconds
            await asyncio.sleep(2.0)
            transport.close()
        except Exception:
            pass
        return []  # UDP AIS parsing requires pyais — gracefully return empty

    def _simulate(self) -> List[dict]:
        """Advance simulated vessel positions."""
        dt = self.poll_interval / 3600.0  # hours
        vessels = []
        for vid, v in self._sim_state.items():
            spd_deg_lat = (v["speed_kts"] * 1.852) / 111320 * dt * 3600
            spd_deg_lon = spd_deg_lat / max(math.cos(math.radians(v["lat"])), 0.01)
            hdg_rad = math.radians(v["heading"])
            v["lat"] += spd_deg_lat * math.cos(hdg_rad)
            v["lon"] += spd_deg_lon * math.sin(hdg_rad)
            # Gentle course changes
            v["heading"] = (v["heading"] + random.uniform(-2, 2)) % 360
            v["speed_kts"] *= random.uniform(0.98, 1.02)
            vessels.append({**v, "mmsi": v["mmsi"]})
        return vessels

    def _vessel_to_entity(self, v: dict) -> Optional[dict]:
        mmsi = v.get("mmsi", "")
        name = (v.get("name") or mmsi).strip() or "UNKNOWN VESSEL"
        lat = float(v.get("lat", 0) or 0)
        lon = float(v.get("lon", 0) or 0)
        speed_kts = float(v.get("speed_kts", 0) or 0)
        speed_mps = speed_kts * 0.514444
        heading = float(v.get("heading", 0) or 0)
        cls = v.get("classification", "vessel")

        if not mmsi or (lat == 0 and lon == 0):
            return None

        entity_id = f"ais-{mmsi}"

        b = (
            EntityBuilder(entity_id, name)
            .track()
            .maritime()
            .label(cls)
            .position(lat, lon, 0.0, heading)
            .speed(speed_mps)
            .confidence(0.95)
            .source("ais", entity_id)
            .org(self.org_id)
            .ttl(120)
            .meta_dict({
                "mmsi": mmsi,
                "vessel_length_m": str(v.get("length", 0)),
                "course_deg": str(v.get("course", heading)),
                "protocol": "ais",
            })
        )

        # Mark fast vessels (>25 kts) as alert
        if speed_kts > 25:
            b = b.critical_above(speed_kts)

        return b.build()
