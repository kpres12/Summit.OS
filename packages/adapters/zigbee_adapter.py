"""
Heli.OS — Zigbee2MQTT Adapter
==================================

Ingests Zigbee sensor data via Zigbee2MQTT — the industry-standard open-source
Zigbee bridge. Subscribes to the Zigbee2MQTT MQTT topics.

Zigbee2MQTT supports 3,000+ devices from Aqara, IKEA, Philips Hue, Sonoff,
Tuya, Xiaomi, and many others — motion sensors, door contacts, temperature
probes, smoke detectors, water leak sensors, vibration sensors.

Relevant for Heli.OS
-----------------------
- Perimeter intrusion sensors (door/window/motion)
- Environmental monitoring (temp, humidity, CO2, VOC)
- Smoke / CO detectors → auto-alert on Heli.OS
- Vibration sensors on critical infrastructure
- Flood sensors in low-lying areas

Zigbee2MQTT publishes each device to: zigbee2mqtt/{friendly_name}

Dependencies
------------
    pip install paho-mqtt  (already a Heli.OS dependency)
    Requires Zigbee2MQTT running separately: https://www.zigbee2mqtt.io

Config extras
-------------
mqtt_host       : str   — MQTT broker host (default "localhost")
mqtt_port       : int   — MQTT port (default 1883)
mqtt_username   : str   — optional MQTT username
mqtt_password   : str   — optional MQTT password
z2m_base_topic  : str   — Zigbee2MQTT base topic (default "zigbee2mqtt")
device_locations: dict  — map device friendly_name → {lat, lon} for positioning
alert_on        : list  — list of field/value pairs that trigger ALERT entity type
                          e.g. [{"field": "smoke", "value": true},
                                {"field": "occupancy", "value": true}]
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("heli.adapters.zigbee")

try:
    import paho.mqtt.client as _mqtt
    _PAHO_AVAILABLE = True
except ImportError:
    _mqtt = None  # type: ignore
    _PAHO_AVAILABLE = False


class ZigbeeAdapter(BaseAdapter):
    """
    Subscribes to Zigbee2MQTT and emits IOT_SENSOR (or ALERT) observations.
    """

    adapter_type = "zigbee"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra
        self._host: str = ex.get("mqtt_host", "localhost")
        self._port: int = int(ex.get("mqtt_port", 1883))
        self._username: Optional[str] = ex.get("mqtt_username")
        self._password: Optional[str] = ex.get("mqtt_password")
        self._base_topic: str = ex.get("z2m_base_topic", "zigbee2mqtt").rstrip("/")
        self._device_locations: dict = ex.get("device_locations", {})
        self._alert_on: list = ex.get("alert_on", [
            {"field": "smoke", "value": True},
            {"field": "carbon_monoxide", "value": True},
            {"field": "water_leak", "value": True},
            {"field": "gas", "value": True},
        ])
        self._client = None
        self._obs_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

    async def connect(self) -> None:
        if not _PAHO_AVAILABLE:
            raise RuntimeError("paho-mqtt not installed")

        loop = asyncio.get_event_loop()

        def _on_message(client, userdata, msg):
            # Skip Zigbee2MQTT bridge status topics
            topic = msg.topic
            if "/bridge/" in topic:
                return
            try:
                device_name = topic.replace(f"{self._base_topic}/", "", 1)
                payload = json.loads(msg.payload.decode())
                obs = self._to_obs(device_name, payload)
                if obs:
                    loop.call_soon_threadsafe(self._obs_queue.put_nowait, obs)
            except Exception as e:
                logger.debug("Zigbee2MQTT decode error: %s", e)

        self._client = _mqtt.Client()
        if self._username:
            self._client.username_pw_set(self._username, self._password)
        self._client.on_message = _on_message
        self._client.connect(self._host, self._port, 60)
        self._client.subscribe(f"{self._base_topic}/#", qos=1)
        self._client.loop_start()
        logger.info("Zigbee2MQTT subscribed: %s:%d", self._host, self._port)

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

    def _to_obs(self, device_name: str, payload: dict) -> Optional[dict]:
        now = datetime.now(timezone.utc)
        entity_id = f"zigbee-{device_name.replace(' ', '_').replace('/', '_')}"

        # Determine if this triggers an alert
        is_alert = False
        for rule in self._alert_on:
            field = rule.get("field", "")
            expected = rule.get("value")
            if field in payload and payload[field] == expected:
                is_alert = True
                break

        entity_type = "ALERT" if is_alert else "IOT_SENSOR"

        obs: dict = {
            "source_id": f"{entity_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": entity_id,
            "callsign": device_name,
            "entity_type": entity_type,
            "classification": "zigbee_sensor",
            "ts_iso": now.isoformat(),
            "metadata": dict(payload),
        }

        loc = self._device_locations.get(device_name)
        if loc:
            obs["position"] = {
                "lat": loc.get("lat", 0),
                "lon": loc.get("lon", 0),
                "alt_m": loc.get("alt"),
            }

        return obs
