"""
Heli.OS Skydio Adapter

Connects to Skydio drones via the Skydio Cloud API (REST/WebSocket) or
the on-vehicle UDP telemetry stream, and publishes real-time telemetry as
Heli.OS TRACK entities into the data fabric.

Connection modes
----------------
CLOUD (default):
    Polls the Skydio Cloud API for vehicle state at SKYDIO_POLL_HZ.
    Requires SKYDIO_API_KEY and SKYDIO_VEHICLE_SERIAL.

UDP (direct):
    Listens for JSON telemetry packets broadcast by the Skydio app on
    UDP port 14590 (the default Skydio SDK telemetry port).
    Set SKYDIO_MODE=udp and SKYDIO_UDP_HOST/SKYDIO_UDP_PORT.

Environment variables
---------------------
    SKYDIO_ENABLED          "true" to enable (default: "false")
    SKYDIO_MODE             "cloud" | "udp" (default: "cloud")
    SKYDIO_API_KEY          Skydio Cloud API key (cloud mode)
    SKYDIO_VEHICLE_SERIAL   Vehicle serial number (cloud mode)
    SKYDIO_POLL_HZ          Cloud poll rate in Hz (default: 1, max: 5)
    SKYDIO_UDP_HOST         UDP bind host (udp mode, default: "0.0.0.0")
    SKYDIO_UDP_PORT         UDP bind port (udp mode, default: 14590)
    SKYDIO_ORG_ID           org_id tag (default: "")
    MQTT_HOST / MQTT_PORT   Broker connection
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from sdk import BaseAdapter, AdapterManifest, EntityBuilder, Protocol, Capability

logger = logging.getLogger("summit.adapter.skydio")

_MODE = os.getenv("SKYDIO_MODE", "cloud").lower()
_API_KEY = os.getenv("SKYDIO_API_KEY", "")
_VEHICLE_SERIAL = os.getenv("SKYDIO_VEHICLE_SERIAL", "")
_POLL_HZ = min(float(os.getenv("SKYDIO_POLL_HZ", "1")), 5.0)
_UDP_HOST = os.getenv("SKYDIO_UDP_HOST", "0.0.0.0")
_UDP_PORT = int(os.getenv("SKYDIO_UDP_PORT", "14590"))
_ORG_ID = os.getenv("SKYDIO_ORG_ID", "")

# Skydio Cloud API base URL
_CLOUD_API_BASE = "https://api.skydio.com/api/v0"


class SkydioAdapter(BaseAdapter):
    """Publishes Skydio drone telemetry as Heli.OS TRACK entities.

    Supports Skydio X2D, X10, and Dock-based autonomous flight.
    Falls back to simulated telemetry if the SDK / credentials are unavailable.
    """

    MANIFEST = AdapterManifest(
        name="skydio",
        version="1.0.0",
        protocol=Protocol.HTTP,
        capabilities=[Capability.READ, Capability.STREAM],
        entity_types=["TRACK"],
        description="Skydio drone adapter — cloud API or direct UDP telemetry",
        required_env=["SKYDIO_API_KEY"] if _MODE == "cloud" else [],
        optional_env=[
            "SKYDIO_VEHICLE_SERIAL",
            "SKYDIO_POLL_HZ",
            "SKYDIO_UDP_HOST",
            "SKYDIO_UDP_PORT",
            "SKYDIO_MODE",
        ],
    )

    def __init__(self, org_id: str = _ORG_ID, **kwargs):
        super().__init__(device_id="skydio", org_id=org_id, **kwargs)
        self._mode = _MODE
        self._api_key = _API_KEY
        self._vehicle_serial = _VEHICLE_SERIAL
        self._poll_interval = 1.0 / max(_POLL_HZ, 0.1)

    @property
    def enabled(self) -> bool:
        return os.getenv("SKYDIO_ENABLED", "false").lower() == "true"

    async def run(self):
        logger.info(f"Skydio adapter running (mode={self._mode})")
        if self._mode == "udp":
            await self._run_udp()
        else:
            await self._run_cloud()

    # ── Cloud mode ──────────────────────────────────────────────────────────

    async def _run_cloud(self):
        """Poll the Skydio Cloud API for vehicle state."""
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed — falling back to simulation")
            await self._run_simulated()
            return

        if not self._api_key:
            logger.warning("SKYDIO_API_KEY not set — falling back to simulation")
            await self._run_simulated()
            return

        headers = {"Authorization": f"Bearer {self._api_key}"}
        vehicle_id = self._vehicle_serial or "skydio-01"

        async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
            while not self.stopped:
                try:
                    # GET /vehicles/{serial}/state
                    # Returns: lat, lon, alt, heading, speed, battery_percent,
                    #          flight_state, gimbal_pitch, signal_quality, etc.
                    resp = await client.get(
                        f"{_CLOUD_API_BASE}/vehicles/{vehicle_id}/state"
                    )
                    resp.raise_for_status()
                    state = resp.json().get("data", {})
                    entity = self._state_to_entity(state, vehicle_id)
                    self.publish(entity, qos=0)
                except httpx.HTTPStatusError as e:
                    logger.error(f"Skydio API error: {e.response.status_code}")
                except Exception as e:
                    logger.error(f"Skydio cloud poll error: {e}")

                await self.sleep(self._poll_interval)

    # ── UDP mode ─────────────────────────────────────────────────────────────

    async def _run_udp(self):
        """Receive Skydio JSON telemetry packets over UDP."""
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)

        class _Protocol(asyncio.DatagramProtocol):
            def datagram_received(self, data, addr):
                try:
                    queue.put_nowait(data)
                except asyncio.QueueFull:
                    pass  # drop if lagging

        try:
            transport, _ = await loop.create_datagram_endpoint(
                _Protocol,
                local_addr=(_UDP_HOST, _UDP_PORT),
            )
        except OSError as e:
            logger.error(f"Skydio UDP bind failed: {e} — falling back to simulation")
            await self._run_simulated()
            return

        logger.info(f"Skydio UDP listening on {_UDP_HOST}:{_UDP_PORT}")
        try:
            while not self.stopped:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=2.0)
                    payload = json.loads(data.decode("utf-8"))
                    vehicle_id = payload.get("serial", "skydio-udp")
                    entity = self._state_to_entity(payload, vehicle_id)
                    self.publish(entity, qos=0)
                except asyncio.TimeoutError:
                    pass
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Skydio UDP parse error: {e}")
        finally:
            transport.close()

    # ── Entity mapping ───────────────────────────────────────────────────────

    def _state_to_entity(self, state: Dict[str, Any], vehicle_id: str) -> Dict[str, Any]:
        lat = float(state.get("lat", 0.0))
        lon = float(state.get("lon", 0.0))
        alt = float(state.get("alt_meters", state.get("altitude", 0.0)))
        heading = float(state.get("heading", 0.0))
        speed = float(state.get("speed_mps", state.get("groundspeed", 0.0)))
        battery_pct = float(state.get("battery_percent", state.get("battery", 0.0)))
        flight_state = str(state.get("flight_state", "UNKNOWN")).upper()
        signal = str(state.get("signal_quality", "good")).lower()

        b = (
            EntityBuilder(f"skydio-{vehicle_id}", state.get("name", vehicle_id))
            .track()
            .aerial()
            .at(lat, lon, alt)
            .moving(heading, speed, 0.0)
            .label("quadcopter")
            .source("skydio", f"skydio-{vehicle_id}")
            .org(self.org_id)
            .ttl(15)
            .aerial_telemetry(
                flight_mode=flight_state,
                battery_pct=battery_pct,
                airspeed_mps=speed,
                link_quality=signal,
            )
            .meta_dict({
                "vehicle_id": vehicle_id,
                "flight_state": flight_state,
                "battery_pct": str(round(battery_pct, 1)),
                "protocol": "skydio",
                "connection_mode": self._mode,
            })
        )

        if battery_pct < 10.0:
            b = b.critical()
        elif battery_pct < 20.0:
            b = b.warning()
        elif flight_state in ("FLYING", "HOVERING", "RETURNING"):
            b = b.active()
        else:
            b = b.standby()

        return b.build()

    # ── Simulation fallback ──────────────────────────────────────────────────

    async def _run_simulated(self):
        logger.info("Skydio adapter: running in simulation mode")
        t = 0.0
        while not self.stopped:
            t += self._poll_interval
            state = {
                "lat": 37.7749 + 0.001 * math.sin(t * 0.04),
                "lon": -122.4194 + 0.001 * math.cos(t * 0.04),
                "alt_meters": 80.0 + 15.0 * math.sin(t * 0.02),
                "heading": (t * 3.0) % 360.0,
                "speed_mps": 8.0 + 2.0 * math.sin(t * 0.1),
                "battery_percent": max(0, 95 - int(t * 0.04)),
                "flight_state": "FLYING",
                "signal_quality": "excellent",
            }
            entity = self._state_to_entity(state, "skydio-sim-01")
            entity["metadata"]["simulated"] = "true"
            self.publish(entity, qos=0)
            await self.sleep(self._poll_interval)
