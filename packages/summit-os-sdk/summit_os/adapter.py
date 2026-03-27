"""
Summit.OS Unified Device Adapter

Single base class that a hardware integrator subclasses to connect
any device (drone, robot, tower, sensor) into Summit.OS.

Combines:
  - HAL abstract interface (connect, telemetry, commands)
  - MQTT transport (heartbeat, telemetry publish, command subscribe)
  - Entity protocol (registers as Entity in WorldStore)
  - Gateway registration (automatic on start)

Minimal integration:
    class MyDrone(SummitAdapter):
        async def get_telemetry(self) -> dict:
            return {"lat": ..., "lon": ..., "alt": ..., "battery": ...}

        async def handle_command(self, cmd: str, params: dict) -> bool:
            if cmd == "goto":
                ...
                return True
            return False

    adapter = MyDrone(device_id="drone-01", device_type="DRONE")
    asyncio.run(adapter.start())
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import abstractmethod
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("summit.adapter")

try:
    import paho.mqtt.client as mqtt  # type: ignore
except ImportError:
    mqtt = None  # type: ignore


class SummitAdapter:
    """
    Base class for all Summit.OS device integrations.

    Subclass this and implement:
      - get_telemetry() → dict with at least {lat, lon, alt}
      - handle_command(cmd, params) → bool
    Optionally override:
      - on_connect() — called after MQTT + registration
      - on_disconnect() — called on shutdown
      - get_capabilities() → list of strings
    """

    def __init__(
        self,
        device_id: str,
        device_type: str = "GENERIC",
        org_id: Optional[str] = None,
        # MQTT
        mqtt_host: str = os.getenv("MQTT_HOST", "localhost"),
        mqtt_port: int = int(os.getenv("MQTT_PORT", "1883")),
        mqtt_username: Optional[str] = os.getenv("MQTT_USERNAME"),
        mqtt_password: Optional[str] = os.getenv("MQTT_PASSWORD"),
        # API gateway
        api_base: str = os.getenv("SUMMIT_API_URL", "http://localhost:8000"),
        # Intervals
        heartbeat_interval: float = 30.0,
        telemetry_interval: float = 5.0,
    ):
        self.device_id = device_id
        self.device_type = device_type
        self.org_id = org_id
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.mqtt_username = mqtt_username
        self.mqtt_password = mqtt_password
        self.api_base = api_base.rstrip("/")
        self.heartbeat_interval = heartbeat_interval
        self.telemetry_interval = telemetry_interval

        self._mqtt: Optional[mqtt.Client] = None
        self._stop = asyncio.Event()
        self._token: Optional[str] = None
        self._registered = False
        self._command_handlers: Dict[str, Callable] = {}

    # ── Abstract interface (MUST implement) ────────────────

    @abstractmethod
    async def get_telemetry(self) -> Dict[str, Any]:
        """
        Return current device telemetry.
        Must include at minimum: {"lat": float, "lon": float, "alt": float}
        Recommended: {"battery": float, "status": str, "sensors": dict}
        """
        ...

    @abstractmethod
    async def handle_command(self, cmd: str, params: Dict[str, Any]) -> bool:
        """
        Handle an incoming command from the platform.
        Return True if the command was handled successfully.

        Common commands:
          "goto"      → params: {lat, lon, alt, speed}
          "rtl"       → return to launch
          "land"      → immediate land
          "hold"      → hold position
          "set_mode"  → params: {mode: str}
        """
        ...

    # ── Optional overrides ─────────────────────────────────

    async def on_connect(self):
        """Called after successful MQTT connect + gateway registration."""
        pass

    async def on_disconnect(self):
        """Called before shutdown."""
        pass

    def get_capabilities(self) -> List[str]:
        """Return list of device capabilities (e.g., ['thermal', 'rgb_camera', 'lidar'])."""
        return []

    def get_comm_protocols(self) -> List[str]:
        """Return list of communication protocols (e.g., ['mqtt', 'mavlink'])."""
        return ["mqtt"]

    # ── Lifecycle ──────────────────────────────────────────

    async def start(self):
        """Start the adapter: connect MQTT, register, begin heartbeat + telemetry loops."""
        logger.info(f"Starting adapter for {self.device_id} ({self.device_type})")

        # Connect MQTT
        await self._connect_mqtt()

        # Register with gateway
        await self._register()

        # Subscribe to command topics
        self._subscribe_commands()

        # Notify subclass
        await self.on_connect()

        # Start background loops
        tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._telemetry_loop()),
        ]

        logger.info(f"Adapter running: {self.device_id}")

        # Wait until stop is requested
        await self._stop.wait()

        # Cleanup
        for t in tasks:
            t.cancel()
        await self.on_disconnect()
        self._disconnect_mqtt()

        logger.info(f"Adapter stopped: {self.device_id}")

    async def stop(self):
        """Signal the adapter to stop."""
        self._stop.set()

    # ── MQTT ───────────────────────────────────────────────

    async def _connect_mqtt(self):
        if mqtt is None:
            raise RuntimeError("paho-mqtt not installed; pip install paho-mqtt")

        self._mqtt = mqtt.Client(client_id=f"summit-{self.device_id}")

        if self.mqtt_username and self.mqtt_password:
            self._mqtt.username_pw_set(self.mqtt_username, self.mqtt_password)

        # Command handler callback
        def _on_message(_client, _userdata, msg):
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
            except Exception:
                payload = {"raw": msg.payload.decode("utf-8", errors="ignore")}

            # Schedule async handling
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(
                    asyncio.ensure_future,
                    self._dispatch_command(msg.topic, payload),
                )
            except Exception as e:
                logger.error(f"Command dispatch error: {e}")

        self._mqtt.on_message = _on_message
        self._mqtt.connect(self.mqtt_host, self.mqtt_port, 60)
        self._mqtt.loop_start()
        logger.info(f"MQTT connected: {self.mqtt_host}:{self.mqtt_port}")

    def _disconnect_mqtt(self):
        if self._mqtt:
            try:
                self._mqtt.loop_stop()
                self._mqtt.disconnect()
            except Exception:
                pass

    def _subscribe_commands(self):
        """Subscribe to command/task topics for this device."""
        if not self._mqtt:
            return
        topics = [
            f"tasks/{self.device_id}/#",
            f"control/{self.device_id}/#",
            f"commands/{self.device_id}",
        ]
        for t in topics:
            self._mqtt.subscribe(t, qos=1)
        logger.info(f"Subscribed to command topics for {self.device_id}")

    async def _dispatch_command(self, topic: str, payload: Dict[str, Any]):
        """Route incoming MQTT messages to handle_command."""
        cmd = (
            payload.get("action")
            or payload.get("command")
            or payload.get("cmd")
            or "unknown"
        )
        params = payload.get("params") or payload.get("waypoints") or payload
        try:
            ok = await self.handle_command(cmd, params)
            logger.info(f"Command {cmd} → {'OK' if ok else 'FAILED'}")
        except Exception as e:
            logger.error(f"Command handler error: {e}")

    # ── Registration ───────────────────────────────────────

    async def _register(self):
        """Register device with the API Gateway → Fabric node registry."""
        try:
            import requests

            payload = {
                "id": self.device_id,
                "type": self.device_type,
                "pubkey": None,
                "fw_version": None,
                "location": None,
                "capabilities": self.get_capabilities(),
                "comm": self.get_comm_protocols(),
            }
            headers = {}
            if self.org_id:
                headers["X-Org-ID"] = self.org_id

            r = requests.post(
                f"{self.api_base}/api/v1/nodes/register",
                json=payload,
                headers=headers,
                timeout=5,
            )
            r.raise_for_status()
            data = r.json()
            self._token = data.get("token") if isinstance(data, dict) else None
            self._registered = True
            logger.info(f"Registered with gateway: {self.device_id}")
        except Exception as e:
            logger.warning(f"Gateway registration failed (will retry): {e}")
            self._registered = False

    # ── Background loops ───────────────────────────────────

    async def _heartbeat_loop(self):
        """Publish heartbeat at regular intervals."""
        while not self._stop.is_set():
            try:
                if self._mqtt:
                    payload = {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "status": "OK",
                        "device_id": self.device_id,
                    }
                    self._mqtt.publish(
                        f"health/{self.device_id}/heartbeat",
                        json.dumps(payload),
                        qos=0,
                    )
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
            await asyncio.sleep(self.heartbeat_interval)

    async def _telemetry_loop(self):
        """Read telemetry from subclass and publish to MQTT + entity topics."""
        while not self._stop.is_set():
            try:
                telem = await self.get_telemetry()

                if self._mqtt and telem:
                    # Publish to device telemetry topic
                    topic = f"devices/{self.device_id}/telemetry"
                    payload = {
                        "device_id": self.device_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "location": {
                            "lat": telem.get("lat", 0),
                            "lon": telem.get("lon", 0),
                            "alt": telem.get("alt", 0),
                        },
                        "sensors": telem.get("sensors", {}),
                        "status": telem.get("status", "ACTIVE"),
                        "battery": telem.get("battery"),
                    }
                    self._mqtt.publish(topic, json.dumps(payload), qos=0)

                    # Also publish as entity update for WorldStore ingestion
                    entity_topic = f"entities/{self.device_id}/update"
                    entity_payload = {
                        "entity_id": self.device_id,
                        "entity_type": "ASSET",
                        "domain": self.device_type.upper(),
                        "state": telem.get("status", "ACTIVE"),
                        "kinematics": {
                            "latitude": telem.get("lat", 0),
                            "longitude": telem.get("lon", 0),
                            "altitude": telem.get("alt", 0),
                        },
                        "metadata": {
                            "battery": telem.get("battery"),
                            "type": self.device_type,
                            **{
                                k: v
                                for k, v in telem.items()
                                if k
                                not in (
                                    "lat",
                                    "lon",
                                    "alt",
                                    "battery",
                                    "status",
                                    "sensors",
                                )
                            },
                        },
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    self._mqtt.publish(entity_topic, json.dumps(entity_payload), qos=0)

            except Exception as e:
                logger.error(f"Telemetry error: {e}")

            await asyncio.sleep(self.telemetry_interval)

    # ── Utility ────────────────────────────────────────────

    def publish(self, topic: str, payload: Dict[str, Any], qos: int = 0):
        """Publish a custom MQTT message."""
        if self._mqtt:
            self._mqtt.publish(topic, json.dumps(payload), qos=qos)

    def publish_alert(
        self,
        alert_id: str,
        severity: str,
        description: str,
        lat: float = 0,
        lon: float = 0,
    ):
        """Convenience: publish an alert."""
        if self._mqtt:
            payload = {
                "alert_id": alert_id,
                "severity": severity,
                "description": description,
                "source": self.device_id,
                "location": {"lat": lat, "lon": lon},
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            self._mqtt.publish(f"alerts/{self.device_id}", json.dumps(payload), qos=1)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
