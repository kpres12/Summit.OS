"""
Heli.OS — LoRaWAN Adapter (ChirpStack)
==========================================

Ingests sensor data from LoRaWAN networks via ChirpStack (open-source
LoRaWAN Network Server). Subscribes to ChirpStack's MQTT integration.

LoRaWAN is widely used for long-range, low-power sensor networks:
- Wildfire sensor arrays (temperature, humidity, smoke)
- Flood / water level sensors
- Wildlife tracking collars
- Perimeter sensors (PIR, vibration, magnetic)
- Environmental monitoring buoys
- Soil moisture / agricultural sensors

ChirpStack publishes uplinks to MQTT topic:
    application/{app_id}/device/{dev_eui}/event/up

Dependencies
------------
    pip install paho-mqtt  (already a Heli.OS dependency)

Config extras
-------------
mqtt_host       : str   — ChirpStack MQTT broker host (default "localhost")
mqtt_port       : int   — MQTT port (default 1883)
mqtt_username   : str   — MQTT username
mqtt_password   : str   — MQTT password
application_id  : str   — ChirpStack application ID to subscribe to ("+" for all)
decoder         : str   — payload decoder: "cayenne" | "json" | "raw" (default "json")
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("heli.adapters.lorawan")

try:
    import paho.mqtt.client as _mqtt
    _PAHO_AVAILABLE = True
except ImportError:
    _mqtt = None  # type: ignore
    _PAHO_AVAILABLE = False


def _decode_cayenne(raw_bytes: bytes) -> dict:
    """Minimal CayenneLPP decoder for common channels."""
    result: dict = {}
    i = 0
    while i + 2 <= len(raw_bytes):
        channel = raw_bytes[i]
        lpp_type = raw_bytes[i + 1]
        i += 2
        if lpp_type == 0 and i + 1 <= len(raw_bytes):  # digital input
            result[f"digital_{channel}"] = raw_bytes[i]; i += 1
        elif lpp_type == 2 and i + 2 <= len(raw_bytes):  # analog input
            val = int.from_bytes(raw_bytes[i:i+2], "big", signed=True) / 100
            result[f"analog_{channel}"] = val; i += 2
        elif lpp_type == 103 and i + 6 <= len(raw_bytes):  # GPS
            lat = int.from_bytes(raw_bytes[i:i+3], "big", signed=True) / 10000
            lon = int.from_bytes(raw_bytes[i+3:i+6], "big", signed=True) / 10000
            result["lat"] = lat; result["lon"] = lon; i += 6
        elif lpp_type == 104 and i + 2 <= len(raw_bytes):  # barometer
            result[f"pressure_hpa_{channel}"] = int.from_bytes(raw_bytes[i:i+2], "big") / 10; i += 2
        elif lpp_type == 115 and i + 2 <= len(raw_bytes):  # temperature
            result[f"temp_c_{channel}"] = int.from_bytes(raw_bytes[i:i+2], "big", signed=True) / 10; i += 2
        elif lpp_type == 104 and i + 1 <= len(raw_bytes):  # humidity
            result[f"humidity_pct_{channel}"] = raw_bytes[i] / 2; i += 1
        else:
            break  # unknown type — stop
    return result


class LoRaWANAdapter(BaseAdapter):
    """
    Subscribes to ChirpStack MQTT and emits IOT_SENSOR observations.

    Handles GPS-equipped nodes (position updates) and static sensors
    (position from ChirpStack gateway metadata or config).
    """

    adapter_type = "lorawan"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra
        self._host: str = ex.get("mqtt_host", "localhost")
        self._port: int = int(ex.get("mqtt_port", 1883))
        self._username: Optional[str] = ex.get("mqtt_username")
        self._password: Optional[str] = ex.get("mqtt_password")
        self._app_id: str = str(ex.get("application_id", "+"))
        self._decoder: str = ex.get("decoder", "json")
        self._client = None
        self._obs_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

    async def connect(self) -> None:
        if not _PAHO_AVAILABLE:
            raise RuntimeError("paho-mqtt not installed")

        loop = asyncio.get_event_loop()
        topic = f"application/{self._app_id}/device/+/event/up"

        def _on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode())
                obs = self._chirpstack_to_obs(payload)
                if obs:
                    loop.call_soon_threadsafe(self._obs_queue.put_nowait, obs)
            except Exception as e:
                logger.debug("LoRaWAN decode error: %s", e)

        self._client = _mqtt.Client()
        if self._username:
            self._client.username_pw_set(self._username, self._password)
        self._client.on_message = _on_message
        self._client.connect(self._host, self._port, 60)
        self._client.subscribe(topic, qos=1)
        self._client.loop_start()
        logger.info("LoRaWAN subscribed: %s:%d topic=%s", self._host, self._port, topic)

    async def disconnect(self) -> None:
        if self._client:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._client = None

    async def stream_observations(self) -> AsyncIterator[dict]:
        while not self._stop_event.is_set():
            try:
                obs = await asyncio.wait_for(self._obs_queue.get(), timeout=5.0)
                yield obs
            except asyncio.TimeoutError:
                pass

    def _chirpstack_to_obs(self, payload: dict) -> Optional[dict]:
        now = datetime.now(timezone.utc)
        dev_eui = payload.get("devEUI") or payload.get("deviceInfo", {}).get("devEui", "unknown")
        device_name = (
            payload.get("deviceName")
            or payload.get("deviceInfo", {}).get("deviceName")
            or dev_eui
        )

        # Decode uplink payload
        data_b64 = payload.get("data", "")
        metadata: dict = {}
        lat = lon = None

        if data_b64:
            try:
                raw = base64.b64decode(data_b64)
                if self._decoder == "cayenne":
                    decoded = _decode_cayenne(raw)
                    lat = decoded.pop("lat", None)
                    lon = decoded.pop("lon", None)
                    metadata.update(decoded)
                elif self._decoder == "json":
                    metadata.update(json.loads(raw.decode()))
                    lat = metadata.pop("lat", metadata.pop("latitude", None))
                    lon = metadata.pop("lon", metadata.pop("longitude", None))
                else:
                    metadata["raw_hex"] = raw.hex()
            except Exception:
                pass

        # ChirpStack also includes rxInfo with gateway location
        if lat is None:
            rx_info = payload.get("rxInfo") or []
            if rx_info and isinstance(rx_info, list):
                loc = rx_info[0].get("location", {})
                lat = loc.get("latitude")
                lon = loc.get("longitude")

        # Object data (for newer ChirpStack v4)
        obj_data = payload.get("object", {})
        if isinstance(obj_data, dict):
            metadata.update(obj_data)
            if lat is None:
                lat = obj_data.get("latitude") or obj_data.get("lat")
                lon = obj_data.get("longitude") or obj_data.get("lon")

        obs: dict = {
            "source_id": f"{dev_eui}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": f"lorawan-{dev_eui}",
            "callsign": device_name,
            "entity_type": "IOT_SENSOR",
            "classification": "lorawan_node",
            "ts_iso": now.isoformat(),
            "metadata": metadata,
        }
        if lat is not None and lon is not None:
            obs["position"] = {"lat": float(lat), "lon": float(lon), "alt_m": None}

        return obs
