"""
Summit.OS — MQTT Relay Adapter
================================

Subscribes to topics on an external MQTT broker and relays observations into
Summit.OS. Covers the IoT ecosystem — any device publishing MQTT becomes a
signal source.

Dependencies
------------
    pip install asyncio-mqtt
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

try:
    import asyncio_mqtt as mqtt
except ImportError:
    raise ImportError(
        "asyncio-mqtt is required for MQTTRelayAdapter. "
        "Install with: pip install asyncio-mqtt>=0.16.2"
    )

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("summit.adapters.mqtt_relay")

_LAT_ALIASES = ("lat", "latitude", "y", "LAT", "LATITUDE")
_LON_ALIASES = ("lon", "lng", "longitude", "x", "LON", "LNG", "LONGITUDE")
_ALT_ALIASES = ("alt_m", "alt", "altitude", "elevation", "elev")


def _extract_float(payload: dict, *keys) -> Optional[float]:
    for k in keys:
        v = payload.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


class MQTTRelayAdapter(BaseAdapter):
    """
    Subscribes to an MQTT broker and relays messages as observations.

    Config extras
    -------------
    broker_host          : str
    broker_port          : int              (default 1883)
    username             : str              (default "")
    password             : str              (default "")
    topics               : list[str]        — MQTT topics (supports wildcards)
    entity_type_default  : str              (default "IOT_SENSOR")
    payload_format       : "json"|"nmea"|"raw"  (default "json")
    """

    adapter_type = "mqtt_relay"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra

        self._broker_host: str = ex.get("broker_host", "")
        if not self._broker_host:
            raise ValueError("broker_host must be set in adapter extra config")
        self._broker_port: int = int(ex.get("broker_port", 1883))
        self._username: str = ex.get("username", "")
        self._password: str = ex.get("password", "")
        self._topics: list[str] = ex.get("topics", [])
        if not self._topics:
            raise ValueError("At least one topic must be configured")
        self._entity_type_default: str = ex.get("entity_type_default", "IOT_SENSOR")
        self._payload_format: str = ex.get("payload_format", "json")

        self._mqtt_client: Optional[mqtt.Client] = None
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=2000)

    async def connect(self) -> None:
        # Client is initialised in stream_observations to keep it within the
        # async context manager. Just validate config here.
        self._log.info(
            "MQTT relay ready — broker=%s:%d, topics=%s",
            self._broker_host,
            self._broker_port,
            self._topics,
        )

    async def disconnect(self) -> None:
        # asyncio-mqtt cleans up via context manager; nothing explicit needed here
        pass

    async def stream_observations(self) -> AsyncIterator[dict]:
        connect_kwargs: dict = {
            "hostname": self._broker_host,
            "port": self._broker_port,
        }
        if self._username:
            connect_kwargs["username"] = self._username
        if self._password:
            connect_kwargs["password"] = self._password

        async with mqtt.Client(**connect_kwargs) as client:
            # Subscribe to all configured topics
            for topic in self._topics:
                await client.subscribe(topic)
                self._log.info("Subscribed to MQTT topic: %s", topic)

            async with client.messages() as messages:
                async for message in messages:
                    if self._stop_event.is_set():
                        break
                    obs = self._message_to_obs(message)
                    if obs is not None:
                        yield obs

    def _message_to_obs(self, message) -> Optional[dict]:
        topic: str = str(message.topic)
        payload_bytes: bytes = message.payload

        # Derive entity_id from topic path
        entity_id = topic.replace("/", "-")

        now = datetime.now(timezone.utc)

        if self._payload_format == "json":
            return self._parse_json_payload(topic, entity_id, payload_bytes, now)
        elif self._payload_format == "nmea":
            return self._parse_nmea_payload(topic, entity_id, payload_bytes, now)
        else:
            return self._parse_raw_payload(topic, entity_id, payload_bytes, now)

    def _parse_json_payload(
        self,
        topic: str,
        entity_id: str,
        payload_bytes: bytes,
        now: datetime,
    ) -> Optional[dict]:
        try:
            import json

            data = json.loads(payload_bytes.decode("utf-8"))
        except Exception as exc:
            self._log.debug("MQTT JSON parse error on topic %s: %s", topic, exc)
            return None

        if not isinstance(data, dict):
            return None

        # Allow entity_id override from payload
        entity_id = str(data.get("entity_id", data.get("id", entity_id)))
        callsign = data.get("callsign") or data.get("name") or entity_id
        entity_type = (
            data.get("entity_type") or data.get("type") or self._entity_type_default
        )

        lat = _extract_float(data, *_LAT_ALIASES)
        lon = _extract_float(data, *_LON_ALIASES)
        alt_m = _extract_float(data, *_ALT_ALIASES)

        position = None
        if lat is not None and lon is not None:
            position = {"lat": lat, "lon": lon, "alt_m": alt_m}

        heading = _extract_float(data, "heading", "heading_deg", "course", "cog")
        speed = _extract_float(data, "speed", "speed_mps", "sog")
        velocity = None
        if heading is not None or speed is not None:
            velocity = {
                "heading_deg": heading,
                "speed_mps": speed,
                "vertical_mps": _extract_float(data, "vertical_mps", "climb_rate"),
            }

        metadata = {
            k: v
            for k, v in data.items()
            if k
            not in {
                "entity_id",
                "id",
                "callsign",
                "name",
                "entity_type",
                "type",
                *_LAT_ALIASES,
                *_LON_ALIASES,
                *_ALT_ALIASES,
                "heading",
                "heading_deg",
                "course",
                "cog",
                "speed",
                "speed_mps",
                "sog",
                "vertical_mps",
                "climb_rate",
            }
        }
        metadata["mqtt_topic"] = topic

        return {
            "source_id": f"{entity_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": entity_id,
            "callsign": callsign,
            "position": position,
            "velocity": velocity,
            "entity_type": entity_type,
            "classification": None,
            "metadata": metadata,
            "ts_iso": now.isoformat(),
        }

    def _parse_nmea_payload(
        self,
        topic: str,
        entity_id: str,
        payload_bytes: bytes,
        now: datetime,
    ) -> Optional[dict]:
        """
        Parse NMEA sentences published to an MQTT topic.
        Uses pynmea2 if available; falls back to raw if not.
        """
        sentence = payload_bytes.decode("ascii", errors="ignore").strip()
        try:
            import pynmea2

            msg = pynmea2.parse(sentence)
            lat = getattr(msg, "latitude", None)
            lon = getattr(msg, "longitude", None)
            position = None
            if lat and lon:
                position = {"lat": lat, "lon": lon, "alt_m": None}
            return {
                "source_id": f"{entity_id}:{now.timestamp():.3f}",
                "adapter_id": self.config.adapter_id,
                "adapter_type": self.adapter_type,
                "entity_id": entity_id,
                "callsign": entity_id,
                "position": position,
                "velocity": None,
                "entity_type": self._entity_type_default,
                "classification": None,
                "metadata": {
                    "nmea_sentence": sentence,
                    "sentence_type": msg.sentence_type,
                    "mqtt_topic": topic,
                },
                "ts_iso": now.isoformat(),
            }
        except Exception:
            # Fall back to raw
            return self._parse_raw_payload(topic, entity_id, payload_bytes, now)

    def _parse_raw_payload(
        self,
        topic: str,
        entity_id: str,
        payload_bytes: bytes,
        now: datetime,
    ) -> dict:
        raw_str = payload_bytes.decode("utf-8", errors="replace")
        return {
            "source_id": f"{entity_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": entity_id,
            "callsign": entity_id,
            "position": None,
            "velocity": None,
            "entity_type": self._entity_type_default,
            "classification": None,
            "metadata": {
                "raw_payload": raw_str,
                "mqtt_topic": topic,
            },
            "ts_iso": now.isoformat(),
        }
