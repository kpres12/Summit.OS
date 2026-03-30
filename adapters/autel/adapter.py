"""
Summit.OS Autel Adapter

Connects to Autel EVO series drones (EVO II, EVO Nano, EVO Lite, EVO Max 4T)
via the Autel SDK / USB bridge or MAVLink-compatible telemetry stream, and
publishes real-time telemetry as Summit.OS TRACK entities.

Autel EVO drones support MAVLink output when connected to a companion
computer via USB or serial. The Autel SDK (Android/iOS) exposes a REST
interface when the controller is connected to the same LAN.

Connection modes
----------------
MAVLINK (default):
    Reads MAVLink telemetry from the drone autopilot. Requires pymavlink.
    Compatible with EVO II Pro (ArduPilot-based firmware).
    Set AUTEL_CONNECTION to a pymavlink connection string.

SDK:
    Polls the Autel Smart Controller web API (HTTP/JSON) at AUTEL_POLL_HZ.
    Requires the Autel Smart Controller on the same LAN.
    Set AUTEL_MODE=sdk and AUTEL_CONTROLLER_HOST.

Environment variables
---------------------
    AUTEL_ENABLED           "true" to enable (default: "false")
    AUTEL_MODE              "mavlink" | "sdk" (default: "mavlink")
    AUTEL_CONNECTION        MAVLink connection string (mavlink mode,
                            default: "udp:127.0.0.1:14550")
    AUTEL_VEHICLE_ID        Vehicle identifier (default: "autel-evo-01")
    AUTEL_VEHICLE_NAME      Human-readable label (default: AUTEL_VEHICLE_ID)
    AUTEL_CONTROLLER_HOST   Smart Controller LAN IP (sdk mode)
    AUTEL_CONTROLLER_PORT   Smart Controller API port (sdk mode, default: 8080)
    AUTEL_POLL_HZ           SDK poll rate in Hz (default: 2, max: 10)
    AUTEL_ORG_ID            org_id tag (default: "")
    MQTT_HOST / MQTT_PORT   Broker connection
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from sdk import BaseAdapter, AdapterManifest, EntityBuilder, Protocol, Capability

logger = logging.getLogger("summit.adapter.autel")

_MODE = os.getenv("AUTEL_MODE", "mavlink").lower()
_CONNECTION = os.getenv("AUTEL_CONNECTION", "udp:127.0.0.1:14550")
_VEHICLE_ID = os.getenv("AUTEL_VEHICLE_ID", "autel-evo-01")
_VEHICLE_NAME = os.getenv("AUTEL_VEHICLE_NAME", _VEHICLE_ID)
_CONTROLLER_HOST = os.getenv("AUTEL_CONTROLLER_HOST", "")
_CONTROLLER_PORT = int(os.getenv("AUTEL_CONTROLLER_PORT", "8080"))
_POLL_HZ = min(float(os.getenv("AUTEL_POLL_HZ", "2")), 10.0)
_ORG_ID = os.getenv("AUTEL_ORG_ID", "")


class AutelAdapter(BaseAdapter):
    """Publishes Autel EVO drone telemetry as Summit.OS TRACK entities.

    Supports EVO II, EVO II Pro, EVO Nano, EVO Lite, and EVO Max 4T.
    EVO II Pro supports ArduPilot-based MAVLink firmware.
    Other models connect via the Autel Smart Controller SDK API.
    Falls back to simulated telemetry if drivers are unavailable.
    """

    MANIFEST = AdapterManifest(
        name="autel",
        version="1.0.0",
        protocol=Protocol.MAVLINK if _MODE == "mavlink" else Protocol.HTTP,
        capabilities=[Capability.READ, Capability.STREAM],
        entity_types=["TRACK"],
        description="Autel EVO drone adapter — MAVLink or Smart Controller SDK",
        optional_env=[
            "AUTEL_CONNECTION",
            "AUTEL_CONTROLLER_HOST",
            "AUTEL_CONTROLLER_PORT",
            "AUTEL_POLL_HZ",
            "AUTEL_MODE",
            "AUTEL_VEHICLE_ID",
        ],
    )

    def __init__(self, org_id: str = _ORG_ID, **kwargs):
        super().__init__(device_id="autel", org_id=org_id, **kwargs)
        self._mode = _MODE
        self._connection = _CONNECTION
        self._vehicle_id = _VEHICLE_ID
        self._vehicle_name = _VEHICLE_NAME
        self._poll_interval = 1.0 / max(_POLL_HZ, 0.1)

    @property
    def enabled(self) -> bool:
        return os.getenv("AUTEL_ENABLED", "false").lower() == "true"

    async def run(self):
        logger.info(f"Autel adapter running (mode={self._mode}, vehicle={self._vehicle_id})")
        if self._mode == "sdk":
            await self._run_sdk()
        else:
            await self._run_mavlink()

    # ── MAVLink mode ─────────────────────────────────────────────────────────

    async def _run_mavlink(self):
        """Read MAVLink telemetry — same path as the MAVLink adapter."""
        try:
            packages_path = os.path.join(os.path.dirname(__file__), "..", "..", "packages")
            if packages_path not in sys.path:
                sys.path.insert(0, packages_path)
            from hal.drivers.mavlink import MAVLinkDriver

            driver = MAVLinkDriver(connection_string=self._connection, system_id=1)
            connected = await driver.connect()
            if not connected:
                logger.error(f"Autel MAVLink: could not connect at {self._connection}")
                await self._run_simulated()
                return

            await driver.start_telemetry()
            logger.info(f"Autel MAVLink telemetry streaming from {self._connection}")

            while not self.stopped:
                entity = self._mavlink_to_entity(driver.telemetry)
                self.publish(entity, qos=0)
                await self.sleep(self._poll_interval)

            await driver.disconnect()

        except ImportError:
            logger.warning("MAVLink driver unavailable — falling back to simulation")
            await self._run_simulated()

    def _mavlink_to_entity(self, telem: Any) -> Dict[str, Any]:
        battery_pct = float(telem.battery_remaining or 0)
        b = (
            EntityBuilder(f"autel-{self._vehicle_id}", self._vehicle_name)
            .track()
            .aerial()
            .at(float(telem.lat), float(telem.lon), float(telem.alt))
            .moving(float(telem.heading), float(telem.groundspeed), float(telem.climb_rate))
            .label("quadcopter")
            .source("autel", f"autel-{self._vehicle_id}")
            .org(self.org_id)
            .ttl(10)
            .aerial_telemetry(
                flight_mode=telem.flight_mode,
                battery_pct=battery_pct,
                airspeed_mps=float(telem.airspeed),
                link_quality="good" if telem.gps_fix_type >= 3 else "degraded",
            )
            .meta_dict({
                "vehicle_id": self._vehicle_id,
                "battery_pct": str(round(battery_pct, 1)),
                "flight_mode": telem.flight_mode,
                "protocol": "autel/mavlink",
            })
        )
        if battery_pct < 10.0:
            b = b.critical()
        elif battery_pct < 20.0:
            b = b.warning()
        elif telem.armed:
            b = b.active()
        else:
            b = b.standby()
        return b.build()

    # ── Smart Controller SDK mode ─────────────────────────────────────────────

    async def _run_sdk(self):
        """Poll Autel Smart Controller LAN API for vehicle state."""
        if not _CONTROLLER_HOST:
            logger.warning("AUTEL_CONTROLLER_HOST not set — falling back to simulation")
            await self._run_simulated()
            return

        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed — falling back to simulation")
            await self._run_simulated()
            return

        base_url = f"http://{_CONTROLLER_HOST}:{_CONTROLLER_PORT}"
        logger.info(f"Autel SDK polling {base_url}")

        async with httpx.AsyncClient(timeout=5.0) as client:
            while not self.stopped:
                try:
                    # Autel Smart Controller exposes a local REST API.
                    # Endpoint: GET /autel/dronestate  (proprietary — check SDK docs)
                    resp = await client.get(f"{base_url}/autel/dronestate")
                    resp.raise_for_status()
                    state = resp.json()
                    entity = self._sdk_to_entity(state)
                    self.publish(entity, qos=0)
                except Exception as e:
                    logger.error(f"Autel SDK poll error: {e}")

                await self.sleep(self._poll_interval)

    def _sdk_to_entity(self, state: Dict[str, Any]) -> Dict[str, Any]:
        lat = float(state.get("latitude", 0.0))
        lon = float(state.get("longitude", 0.0))
        alt = float(state.get("altitude", 0.0))
        heading = float(state.get("heading", 0.0))
        speed = float(state.get("horizontalSpeed", 0.0))
        battery_pct = float(state.get("batteryPercent", 0.0))
        flight_state = str(state.get("flightStatus", "UNKNOWN")).upper()

        b = (
            EntityBuilder(f"autel-{self._vehicle_id}", self._vehicle_name)
            .track()
            .aerial()
            .at(lat, lon, alt)
            .moving(heading, speed, float(state.get("verticalSpeed", 0.0)))
            .label("quadcopter")
            .source("autel", f"autel-{self._vehicle_id}")
            .org(self.org_id)
            .ttl(15)
            .aerial_telemetry(
                flight_mode=flight_state,
                battery_pct=battery_pct,
                airspeed_mps=speed,
                link_quality="good",
            )
            .meta_dict({
                "vehicle_id": self._vehicle_id,
                "battery_pct": str(round(battery_pct, 1)),
                "flight_state": flight_state,
                "protocol": "autel/sdk",
            })
        )
        if battery_pct < 10.0:
            b = b.critical()
        elif battery_pct < 20.0:
            b = b.warning()
        elif flight_state in ("FLYING", "HOVERING"):
            b = b.active()
        else:
            b = b.standby()
        return b.build()

    # ── Simulation fallback ──────────────────────────────────────────────────

    async def _run_simulated(self):
        logger.info("Autel adapter: running in simulation mode")
        t = 0.0
        while not self.stopped:
            t += self._poll_interval
            state = {
                "latitude": 33.4484 + 0.001 * math.sin(t * 0.05),
                "longitude": -112.0740 + 0.001 * math.cos(t * 0.05),
                "altitude": 60.0 + 10.0 * math.sin(t * 0.03),
                "heading": (t * 2.5) % 360.0,
                "horizontalSpeed": 6.0 + 1.5 * math.sin(t * 0.1),
                "verticalSpeed": 0.3 * math.sin(t * 0.12),
                "batteryPercent": max(0, 90 - int(t * 0.03)),
                "flightStatus": "FLYING",
            }
            entity = self._sdk_to_entity(state)
            entity["metadata"]["simulated"] = "true"
            self.publish(entity, qos=0)
            await self.sleep(self._poll_interval)
