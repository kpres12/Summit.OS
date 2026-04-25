"""
Heli.OS — DJI Cloud API Adapter
=================================

Integrates DJI enterprise drones (Matrice 300/350, Mavic 3E, M30 series)
via the DJI Cloud API protocol — the same protocol used by DJI FlightHub 2.

How DJI Cloud API works
-----------------------
DJI drones running DJI Pilot 2 (or DJI Dock controller) connect to a
user-supplied MQTT broker. Once connected:

  Device → MQTT → Heli.OS fabric   (telemetry, status, events)
  Heli.OS fabric → MQTT → Device   (commands: takeoff, waypoint, RTH)

This is fundamentally different from MAVLink (which is a binary protocol
over serial/UDP). DJI Cloud API is JSON over MQTT — making it a natural
fit for Heli.OS's existing MQTT infrastructure.

Topics (DJI Cloud API spec)
----------------------------
Upstream (device → cloud):
  sys/product/{sn}/status               — device status / heartbeat
  thing/product/{sn}/osd                — real-time OSD telemetry
  thing/product/{sn}/events             — discrete events (takeoff, land, alert)

Downstream (cloud → device):
  thing/product/{sn}/services           — command delivery
  thing/product/{sn}/services_reply     — command ACK

Config extras
-------------
  broker_host      : str   — MQTT broker IP/hostname (default "localhost")
  broker_port      : int   — MQTT broker port (default 1883)
  broker_user      : str   — optional MQTT username
  broker_pass      : str   — optional MQTT password
  serial_numbers   : list  — DJI device serial numbers to track ([] = all)
  detect_persons   : bool  — run YOLOv8 on transmitted JPEG frames (default False)
  detect_conf      : float — YOLO confidence threshold (default 0.45)

DJI Pilot 2 setup (on the drone controller tablet)
---------------------------------------------------
  Settings → Cloud Services → select "Private" → enter your MQTT broker URL.
  The drone will begin publishing telemetry automatically.

Dependencies
------------
  pip install asyncio-mqtt          # for DJI MQTT broker connection
  pip install ultralytics opencv-python   # optional — enables frame detection
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, List, Optional, Set

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("heli.adapters.dji")

try:
    from asyncio_mqtt import Client as MQTTClient, MqttError
    _ASYNCIO_MQTT_AVAILABLE = True
except ImportError:
    MQTTClient = None   # type: ignore
    MqttError = Exception
    _ASYNCIO_MQTT_AVAILABLE = False

try:
    from ultralytics import YOLO as _YOLO
    import cv2 as _cv2
    import numpy as _np
    _DETECTION_AVAILABLE = True
except ImportError:
    _YOLO = None    # type: ignore
    _cv2 = None     # type: ignore
    _np = None      # type: ignore
    _DETECTION_AVAILABLE = False


# ---------------------------------------------------------------------------
# DJI Cloud API topic helpers
# ---------------------------------------------------------------------------

def _osd_topic(sn: str) -> str:
    return f"thing/product/{sn}/osd"

def _events_topic(sn: str) -> str:
    return f"thing/product/{sn}/events"

def _status_topic(sn: str) -> str:
    return f"sys/product/{sn}/status"

def _services_topic(sn: str) -> str:
    return f"thing/product/{sn}/services"

def _wildcard_osd() -> str:
    return "thing/product/+/osd"

def _wildcard_events() -> str:
    return "thing/product/+/events"

def _wildcard_status() -> str:
    return "sys/product/+/status"


# ---------------------------------------------------------------------------
# DJI command builders
# ---------------------------------------------------------------------------

def _build_takeoff_cmd() -> dict:
    return {
        "bid": str(uuid.uuid4()),
        "tid": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "method": "takeoff_to_point",
        "data": {"security_code": ""},
    }

def _build_rth_cmd() -> dict:
    return {
        "bid": str(uuid.uuid4()),
        "tid": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "method": "return_home",
        "data": {},
    }

def _build_land_cmd() -> dict:
    return {
        "bid": str(uuid.uuid4()),
        "tid": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "method": "land",
        "data": {},
    }

def _build_goto_cmd(lat: float, lon: float, alt_m: float) -> dict:
    return {
        "bid": str(uuid.uuid4()),
        "tid": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "method": "fly_to_point",
        "data": {
            "max_speed": 10.0,
            "points": [{"latitude": lat, "longitude": lon, "height": alt_m}],
        },
    }

def _build_waypoint_mission(waypoints: List[Dict], speed_mps: float = 8.0) -> dict:
    return {
        "bid": str(uuid.uuid4()),
        "tid": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "method": "waylines_upload",
        "data": {
            "flight_speed": speed_mps,
            "waypoints": [
                {
                    "latitude": wp["lat"],
                    "longitude": wp["lon"],
                    "height": wp.get("alt", 50.0),
                    "use_global_speed": True,
                    "actions": [],
                }
                for wp in waypoints
            ],
        },
    }


# ---------------------------------------------------------------------------
# DJI Adapter
# ---------------------------------------------------------------------------

class DJIAdapter(BaseAdapter):
    """
    Integrates DJI enterprise drones via DJI Cloud API (MQTT JSON protocol).

    Each drone identified by serial number becomes an entity in WorldStore.
    Telemetry is emitted every OSD tick (~1 Hz from Pilot 2).
    Commands flow back over the services topic.
    """

    adapter_type = "dji"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra

        self._broker_host: str = ex.get("broker_host", "localhost")
        self._broker_port: int = int(ex.get("broker_port", 1883))
        self._broker_user: Optional[str] = ex.get("broker_user")
        self._broker_pass: Optional[str] = ex.get("broker_pass")
        self._allowed_sns: Set[str] = set(ex.get("serial_numbers", []))
        self._detect_persons: bool = bool(ex.get("detect_persons", False))
        self._detect_conf: float = float(ex.get("detect_conf", 0.45))

        self._dji_client: Optional[MQTTClient] = None
        self._obs_queue: asyncio.Queue = asyncio.Queue(maxsize=256)

        # State per serial number
        self._device_state: Dict[str, dict] = {}

        self._yolo = None
        if self._detect_persons and _DETECTION_AVAILABLE:
            try:
                self._yolo = _YOLO("yolov8n.pt")
                logger.info("DJI adapter detection enabled (YOLOv8n)")
            except Exception as e:
                logger.warning("YOLOv8 load failed: %s", e)

    @classmethod
    def required_extra_fields(cls) -> list[str]:
        return ["broker_host"]

    async def connect(self) -> None:
        if not _ASYNCIO_MQTT_AVAILABLE:
            raise RuntimeError(
                "asyncio-mqtt not installed. Run: pip install asyncio-mqtt"
            )

        kwargs = {
            "hostname": self._broker_host,
            "port": self._broker_port,
            "keepalive": 30,
        }
        if self._broker_user:
            kwargs["username"] = self._broker_user
        if self._broker_pass:
            kwargs["password"] = self._broker_pass

        self._dji_client = MQTTClient(**kwargs)
        await self._dji_client.__aenter__()

        # Subscribe to all DJI upstream topics
        await self._dji_client.subscribe(_wildcard_osd())
        await self._dji_client.subscribe(_wildcard_events())
        await self._dji_client.subscribe(_wildcard_status())

        logger.info(
            "DJI adapter connected to broker %s:%d — listening for drones",
            self._broker_host, self._broker_port,
        )

    async def disconnect(self) -> None:
        if self._dji_client:
            try:
                await self._dji_client.__aexit__(None, None, None)
            except Exception:
                pass
            self._dji_client = None

    async def stream_observations(self) -> AsyncIterator[dict]:
        if not self._dji_client:
            return

        # Pump incoming DJI messages into obs_queue in a background task
        pump_task = asyncio.create_task(self._pump_messages())
        try:
            while not self._stop_event.is_set():
                try:
                    obs = await asyncio.wait_for(self._obs_queue.get(), timeout=1.0)
                    yield obs
                except asyncio.TimeoutError:
                    continue
        finally:
            pump_task.cancel()
            try:
                await pump_task
            except asyncio.CancelledError:
                pass

    async def _pump_messages(self) -> None:
        """Read from DJI MQTT broker and put parsed observations on the queue."""
        try:
            async with self._dji_client.messages() as messages:
                async for msg in messages:
                    if self._stop_event.is_set():
                        break
                    try:
                        topic = str(msg.topic)
                        payload = json.loads(msg.payload.decode())
                        obs = self._route_message(topic, payload)
                        if obs:
                            if self._obs_queue.full():
                                self._obs_queue.get_nowait()  # drop oldest
                            await self._obs_queue.put(obs)
                    except Exception as e:
                        logger.debug("DJI message parse error: %s", e)
        except MqttError as e:
            logger.error("DJI MQTT error: %s", e)
            raise

    def _route_message(self, topic: str, payload: dict) -> Optional[dict]:
        """Dispatch incoming DJI message to the right parser."""
        parts = topic.split("/")

        # Extract serial number from topic
        if len(parts) >= 3 and parts[0] in ("thing", "sys"):
            sn = parts[2]
        else:
            return None

        if self._allowed_sns and sn not in self._allowed_sns:
            return None

        if "osd" in topic:
            return self._parse_osd(sn, payload)
        elif "events" in topic:
            return self._parse_event(sn, payload)
        elif "status" in topic:
            return self._parse_status(sn, payload)

        return None

    def _parse_osd(self, sn: str, payload: dict) -> Optional[dict]:
        """Parse DJI OSD (On-Screen Display) telemetry message."""
        now = datetime.now(timezone.utc)
        data = payload.get("data", {})

        # DJI OSD fields (Cloud API spec)
        lat = data.get("latitude")
        lon = data.get("longitude")
        alt = data.get("height")           # meters above ground
        abs_alt = data.get("elevation")    # meters above sea level
        heading = data.get("attitude_head", 0.0)
        speed_h = data.get("horizontal_speed", 0.0)
        speed_v = data.get("vertical_speed", 0.0)
        battery = data.get("battery", {}).get("capacity_percent", 100)
        mode = data.get("mode_code", 0)
        home_lat = data.get("home_latitude")
        home_lon = data.get("home_longitude")
        signal = data.get("wireless_link", {}).get("quality", 100)

        if lat is None or lon is None:
            return None

        # Update local state for command routing
        self._device_state[sn] = {
            "lat": lat, "lon": lon, "alt": alt,
            "battery": battery, "heading": heading,
            "ts": now.isoformat(),
        }

        entity_id = f"dji-{sn}"
        obs = {
            "source_id": f"{entity_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": entity_id,
            "callsign": sn,
            "entity_type": "UAV",
            "classification": "dji_drone",
            "position": {
                "lat": lat,
                "lon": lon,
                "alt_m": alt,
            },
            "velocity": {
                "heading_deg": heading,
                "speed_mps": speed_h,
                "vertical_mps": speed_v,
            },
            "ts_iso": now.isoformat(),
            "metadata": {
                "serial_number": sn,
                "battery_remaining": battery,
                "flight_mode_code": mode,
                "link_quality": signal,
                "home_lat": home_lat,
                "home_lon": home_lon,
                "abs_alt_m": abs_alt,
                "adapter_type": "dji",
            },
        }

        # Battery warnings → emit c2_intel-compatible event observations
        if battery is not None:
            if battery <= 15:
                obs["event_type"] = "battery_critical"
            elif battery <= 25:
                obs["event_type"] = "battery_low"

        return obs

    def _parse_event(self, sn: str, payload: dict) -> Optional[dict]:
        """Parse discrete DJI event messages."""
        now = datetime.now(timezone.utc)
        method = payload.get("method", "")
        data = payload.get("data", {})
        entity_id = f"dji-{sn}"
        state = self._device_state.get(sn, {})

        # Map DJI event methods → c2_intel event types
        EVENT_MAP = {
            "device_online": "asset_online",
            "device_offline": "asset_offline",
            "flight_task_progress": None,       # handled below
            "return_home": "mission_aborted",
            "device_reboot": "node_recovered",
            "low_battery_go_home": "battery_critical",
            "obstacle_avoidance_warning": "threat_identified",
            "mission_finish": "mission_completed",
        }

        event_type = EVENT_MAP.get(method)
        if not event_type and method != "flight_task_progress":
            return None

        if method == "flight_task_progress":
            progress = data.get("progress", {}).get("percent", 0)
            if progress >= 100:
                event_type = "mission_completed"
            else:
                return None  # intermediate progress — not emitted

        obs = {
            "source_id": f"{entity_id}:event:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": entity_id,
            "event_type": event_type,
            "callsign": sn,
            "entity_type": "UAV",
            "lat": state.get("lat"),
            "lon": state.get("lon"),
            "position": {
                "lat": state.get("lat"),
                "lon": state.get("lon"),
                "alt_m": state.get("alt"),
            },
            "ts_iso": now.isoformat(),
            "metadata": {"serial_number": sn, "dji_method": method, "raw": data},
        }
        return obs

    def _parse_status(self, sn: str, payload: dict) -> Optional[dict]:
        """Parse DJI device online/offline status heartbeat."""
        now = datetime.now(timezone.utc)
        entity_id = f"dji-{sn}"
        domain = payload.get("domain", "")
        online = domain == "1" or payload.get("online", False)

        return {
            "source_id": f"{entity_id}:status:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": entity_id,
            "event_type": "asset_online" if online else "asset_offline",
            "callsign": sn,
            "entity_type": "UAV",
            "ts_iso": now.isoformat(),
            "metadata": {"serial_number": sn, "online": online},
        }

    # ── Command interface ────────────────────────────────────────────────────

    async def send_command(self, entity_id: str, command: dict) -> None:
        """
        Send a command to a DJI drone.

        Supported command types
        -----------------------
        GOTO       : {"type": "GOTO", "lat": float, "lon": float, "alt": float}
        WAYPOINTS  : {"type": "WAYPOINTS", "waypoints": [{lat, lon, alt}, ...]}
        RTL / RTH  : {"type": "RTL"}
        LAND       : {"type": "LAND"}
        TAKEOFF    : {"type": "TAKEOFF"}
        """
        if not self._dji_client:
            logger.warning("DJI send_command: not connected")
            return

        # Extract serial number from entity_id (format: "dji-{SN}")
        sn = entity_id.replace("dji-", "")
        if not sn:
            logger.warning("DJI send_command: could not extract SN from %s", entity_id)
            return

        cmd_type = command.get("type", "").upper()
        topic = _services_topic(sn)

        if cmd_type == "GOTO":
            payload = _build_goto_cmd(
                lat=float(command["lat"]),
                lon=float(command["lon"]),
                alt_m=float(command.get("alt", 50.0)),
            )
        elif cmd_type in ("WAYPOINTS", "MISSION"):
            wps = command.get("waypoints", [])
            payload = _build_waypoint_mission(wps, speed_mps=command.get("speed_mps", 8.0))
        elif cmd_type in ("RTL", "RTH", "RETURN_HOME"):
            payload = _build_rth_cmd()
        elif cmd_type == "LAND":
            payload = _build_land_cmd()
        elif cmd_type == "TAKEOFF":
            payload = _build_takeoff_cmd()
        else:
            logger.warning("DJI: unsupported command type %s", cmd_type)
            return

        try:
            await self._dji_client.publish(topic, json.dumps(payload), qos=1)
            logger.info("DJI command sent: %s → %s (SN: %s)", cmd_type, topic, sn)
        except Exception as e:
            logger.error("DJI command publish failed: %s", e)
