"""
Heli.OS Parrot Adapter

Connects to Parrot drones (ANAFI Series: ANAFI 4K, ANAFI Thermal, ANAFI USA,
ANAFI Ai) via the Parrot Olympe Python SDK or the ARSDK REST API, and
publishes real-time telemetry as Heli.OS TRACK entities.

Connection modes
----------------
OLYMPE (default):
    Uses the Parrot Olympe SDK (Python) for direct MAVLink-style telemetry
    over the Parrot private Wi-Fi network.
    Requires: pip install parrot-olympe
    Set PARROT_DRONE_IP to the drone's default IP (192.168.42.1).

HTTP:
    Polls the ARSDK HTTP API exposed by the Skycontroller 4 or
    companion app on the same LAN.
    Set PARROT_MODE=http and PARROT_CONTROLLER_HOST.

Environment variables
---------------------
    PARROT_ENABLED          "true" to enable (default: "false")
    PARROT_MODE             "olympe" | "http" (default: "olympe")
    PARROT_DRONE_IP         Drone Wi-Fi IP (olympe mode, default: "192.168.42.1")
    PARROT_CONTROLLER_HOST  Controller LAN IP (http mode)
    PARROT_CONTROLLER_PORT  Controller API port (http mode, default: 14550)
    PARROT_VEHICLE_ID       Vehicle identifier (default: "parrot-anafi-01")
    PARROT_VEHICLE_NAME     Human-readable label (default: PARROT_VEHICLE_ID)
    PARROT_POLL_HZ          Poll rate in Hz (default: 2, max: 10)
    PARROT_ORG_ID           org_id tag (default: "")
    MQTT_HOST / MQTT_PORT   Broker connection
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from sdk import BaseAdapter, AdapterManifest, EntityBuilder, Protocol, Capability

logger = logging.getLogger("summit.adapter.parrot")

_MODE = os.getenv("PARROT_MODE", "olympe").lower()
_DRONE_IP = os.getenv("PARROT_DRONE_IP", "192.168.42.1")
_CONTROLLER_HOST = os.getenv("PARROT_CONTROLLER_HOST", "")
_CONTROLLER_PORT = int(os.getenv("PARROT_CONTROLLER_PORT", "14550"))
_VEHICLE_ID = os.getenv("PARROT_VEHICLE_ID", "parrot-anafi-01")
_VEHICLE_NAME = os.getenv("PARROT_VEHICLE_NAME", _VEHICLE_ID)
_POLL_HZ = min(float(os.getenv("PARROT_POLL_HZ", "2")), 10.0)
_ORG_ID = os.getenv("PARROT_ORG_ID", "")


class ParrotAdapter(BaseAdapter):
    """Publishes Parrot ANAFI drone telemetry as Heli.OS TRACK entities.

    Supports ANAFI 4K, ANAFI Thermal, ANAFI USA, and ANAFI Ai.
    Uses the Olympe SDK for direct Wi-Fi connection or the ARSDK HTTP API
    via a Skycontroller 4 on LAN. Falls back to simulated telemetry.
    """

    MANIFEST = AdapterManifest(
        name="parrot",
        version="1.0.0",
        protocol=Protocol.HTTP,
        capabilities=[Capability.READ, Capability.STREAM],
        entity_types=["TRACK"],
        description="Parrot ANAFI adapter — Olympe SDK or ARSDK HTTP",
        optional_env=[
            "PARROT_DRONE_IP",
            "PARROT_CONTROLLER_HOST",
            "PARROT_CONTROLLER_PORT",
            "PARROT_POLL_HZ",
            "PARROT_MODE",
            "PARROT_VEHICLE_ID",
        ],
    )

    def __init__(self, org_id: str = _ORG_ID, **kwargs):
        super().__init__(device_id="parrot", org_id=org_id, **kwargs)
        self._mode = _MODE
        self._drone_ip = _DRONE_IP
        self._vehicle_id = _VEHICLE_ID
        self._vehicle_name = _VEHICLE_NAME
        self._poll_interval = 1.0 / max(_POLL_HZ, 0.1)

    @property
    def enabled(self) -> bool:
        return os.getenv("PARROT_ENABLED", "false").lower() == "true"

    async def run(self):
        logger.info(f"Parrot adapter running (mode={self._mode}, vehicle={self._vehicle_id})")
        if self._mode == "http":
            await self._run_http()
        else:
            await self._run_olympe()

    # ── Olympe SDK mode ──────────────────────────────────────────────────────

    async def _run_olympe(self):
        """Connect via Parrot Olympe SDK over the drone's Wi-Fi network."""
        try:
            import olympe  # type: ignore
            from olympe.messages.ardrone3.Piloting import TakeOff, Landing  # noqa
            from olympe.messages.ardrone3.PilotingState import (  # type: ignore
                FlyingStateChanged,
                PositionChanged,
                SpeedChanged,
                AttitudeChanged,
            )
            from olympe.messages.common.CommonState import BatteryStateChanged  # type: ignore
        except ImportError:
            logger.warning("parrot-olympe not installed — falling back to simulation. "
                           "Install with: pip install parrot-olympe")
            await self._run_simulated()
            return

        drone = olympe.Drone(self._drone_ip)
        try:
            drone.connect()
            logger.info(f"Parrot Olympe connected to {self._drone_ip}")
        except Exception as e:
            logger.error(f"Parrot Olympe connect failed: {e}")
            await self._run_simulated()
            return

        try:
            while not self.stopped:
                try:
                    # Read state from Olympe's cached state dictionary
                    pos = drone.get_state(PositionChanged)
                    speed = drone.get_state(SpeedChanged)
                    battery = drone.get_state(BatteryStateChanged)
                    flying = drone.get_state(FlyingStateChanged)

                    state = {
                        "latitude": pos.get("latitude", 0.0),
                        "longitude": pos.get("longitude", 0.0),
                        "altitude": pos.get("altitude", 0.0),
                        "heading": 0.0,  # derived from attitude if needed
                        "speedX": speed.get("speedX", 0.0),
                        "speedY": speed.get("speedY", 0.0),
                        "battery_percent": float(battery.get("percent", 0)),
                        "flying_state": str(flying.get("state", "landed")),
                    }
                    entity = self._state_to_entity(state)
                    self.publish(entity, qos=0)
                except Exception as e:
                    logger.debug(f"Parrot state read error: {e}")

                await self.sleep(self._poll_interval)
        finally:
            drone.disconnect()

    # ── HTTP / ARSDK mode ─────────────────────────────────────────────────────

    async def _run_http(self):
        """Poll the ARSDK HTTP API from a Skycontroller 4 on LAN."""
        if not _CONTROLLER_HOST:
            logger.warning("PARROT_CONTROLLER_HOST not set — falling back to simulation")
            await self._run_simulated()
            return

        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed — falling back to simulation")
            await self._run_simulated()
            return

        base_url = f"http://{_CONTROLLER_HOST}:{_CONTROLLER_PORT}"
        logger.info(f"Parrot HTTP polling {base_url}")

        async with httpx.AsyncClient(timeout=5.0) as client:
            while not self.stopped:
                try:
                    # ARSDK REST: GET /api/v1/drone/state
                    resp = await client.get(f"{base_url}/api/v1/drone/state")
                    resp.raise_for_status()
                    entity = self._state_to_entity(resp.json())
                    self.publish(entity, qos=0)
                except Exception as e:
                    logger.error(f"Parrot HTTP poll error: {e}")

                await self.sleep(self._poll_interval)

    # ── Entity mapping ───────────────────────────────────────────────────────

    def _state_to_entity(self, state: Dict[str, Any]) -> Dict[str, Any]:
        lat = float(state.get("latitude", 0.0))
        lon = float(state.get("longitude", 0.0))
        alt = float(state.get("altitude", 0.0))
        heading = float(state.get("heading", 0.0))

        # Olympe uses speedX/speedY components; HTTP API uses groundspeed
        speed_x = float(state.get("speedX", state.get("groundspeed", 0.0)))
        speed_y = float(state.get("speedY", 0.0))
        speed = math.hypot(speed_x, speed_y)

        battery_pct = float(state.get("battery_percent", state.get("batteryPercent", 0.0)))
        flying_state = str(
            state.get("flying_state", state.get("flightStatus", "landed"))
        ).lower()

        b = (
            EntityBuilder(f"parrot-{self._vehicle_id}", self._vehicle_name)
            .track()
            .aerial()
            .at(lat, lon, alt)
            .moving(heading, speed, float(state.get("speedZ", 0.0)))
            .label("quadcopter")
            .source("parrot", f"parrot-{self._vehicle_id}")
            .org(self.org_id)
            .ttl(15)
            .aerial_telemetry(
                flight_mode=flying_state.upper(),
                battery_pct=battery_pct,
                airspeed_mps=speed,
                link_quality="good",
            )
            .meta_dict({
                "vehicle_id": self._vehicle_id,
                "battery_pct": str(round(battery_pct, 1)),
                "flying_state": flying_state,
                "protocol": f"parrot/{self._mode}",
            })
        )

        if battery_pct < 10.0:
            b = b.critical()
        elif battery_pct < 20.0:
            b = b.warning()
        elif flying_state in ("flying", "hovering"):
            b = b.active()
        else:
            b = b.standby()

        return b.build()

    # ── Simulation fallback ──────────────────────────────────────────────────

    async def _run_simulated(self):
        logger.info("Parrot adapter: running in simulation mode")
        t = 0.0
        while not self.stopped:
            t += self._poll_interval
            state = {
                "latitude": 48.8566 + 0.001 * math.sin(t * 0.04),
                "longitude": 2.3522 + 0.001 * math.cos(t * 0.04),
                "altitude": 50.0 + 8.0 * math.sin(t * 0.03),
                "heading": (t * 2.0) % 360.0,
                "speedX": 4.0 * math.cos(math.radians(t * 2.0)),
                "speedY": 4.0 * math.sin(math.radians(t * 2.0)),
                "speedZ": 0.2 * math.sin(t * 0.15),
                "battery_percent": max(0, 88 - int(t * 0.03)),
                "flying_state": "flying",
            }
            entity = self._state_to_entity(state)
            entity["metadata"]["simulated"] = "true"
            self.publish(entity, qos=0)
            await self.sleep(self._poll_interval)
