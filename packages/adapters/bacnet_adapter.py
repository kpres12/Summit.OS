"""
Heli.OS — BACnet Adapter
============================

Integrates building automation systems via BACnet/IP.

BACnet is the ASHRAE standard for building automation: elevators, HVAC,
access control, fire suppression, lighting, and energy management.
Relevant for emergency response scenarios where operators need to control
building systems (unlock doors, control elevators, activate ventilation).

Reads any BACnet object (AI, AO, BI, BO, AV, BV, MSV) and emits
BUILDING_SENSOR observations. With WRITE capability, can write to
BACnet output/value objects.

Dependencies
------------
    pip install BAC0

Config extras
-------------
ip              : str   — BACnet device IP (e.g. "192.168.1.100")
port            : int   — BACnet UDP port (default 47808)
device_id       : int   — BACnet device instance number
objects         : list  — list of BACnet object dicts to read, each:
    type    : str   — "analogInput", "binaryInput", "analogValue", etc.
    instance: int   — object instance number
    name    : str   — human name for this point
    unit    : str   — optional engineering unit string
entity_lat      : float — building latitude (static)
entity_lon      : float — building longitude (static)
building_name   : str   — display name
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("heli.adapters.bacnet")

try:
    import BAC0
    from BAC0.core.devices.local.models import analog_input, binary_input
    _BAC0_AVAILABLE = True
except ImportError:
    BAC0 = None  # type: ignore
    _BAC0_AVAILABLE = False


class BACnetAdapter(BaseAdapter):
    """
    Polls BACnet building systems and emits BUILDING_SENSOR observations.

    Each configured BACnet device becomes an entity. Object values are
    surfaced as metadata. Battery (UPS) and fault states trigger alerts.
    """

    adapter_type = "bacnet"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra
        self._ip: str = ex.get("ip", "")
        self._port: int = int(ex.get("port", 47808))
        self._device_id: int = int(ex.get("device_id", 1))
        self._objects: list = ex.get("objects", [])
        self._lat: Optional[float] = ex.get("entity_lat")
        self._lon: Optional[float] = ex.get("entity_lon")
        self._building_name: str = ex.get("building_name", config.display_name)
        self._bacnet = None
        self._device = None

    async def connect(self) -> None:
        if not _BAC0_AVAILABLE:
            raise RuntimeError("BAC0 not installed. Run: pip install BAC0")
        if not self._ip:
            raise ValueError("BACnet adapter requires 'ip' in config.extra")

        loop = asyncio.get_event_loop()

        def _connect():
            bacnet = BAC0.lite()
            device = BAC0.device(
                address=f"{self._ip}:{self._port}",
                device_id=self._device_id,
                network=bacnet,
                poll=60,
            )
            return bacnet, device

        self._bacnet, self._device = await loop.run_in_executor(None, _connect)
        logger.info("BACnet connected: %s (device %d)", self._ip, self._device_id)

    async def disconnect(self) -> None:
        if self._bacnet:
            try:
                self._bacnet.disconnect()
            except Exception:
                pass
        self._bacnet = None
        self._device = None

    async def stream_observations(self) -> AsyncIterator[dict]:
        loop = asyncio.get_event_loop()
        while not self._stop_event.is_set():
            try:
                obs = await loop.run_in_executor(None, self._read_all_objects)
                if obs:
                    yield obs
            except Exception as e:
                logger.warning("BACnet read failed: %s", e)
                raise
            await asyncio.sleep(self.config.poll_interval_seconds)

    def _read_all_objects(self) -> Optional[dict]:
        now = datetime.now(timezone.utc)
        entity_id = f"bacnet-{self._ip.replace('.', '-')}-{self._device_id}"
        metadata: dict = {}
        alert = False

        for obj_cfg in self._objects:
            obj_type = obj_cfg.get("type", "analogInput")
            instance = obj_cfg.get("instance", 0)
            name = obj_cfg.get("name", f"{obj_type}:{instance}")
            try:
                point_name = f"{obj_type}:{instance}"
                value = self._device[point_name].lastValue
                if value is not None:
                    metadata[name] = value
                    # Flag faults and fire alarms as alert
                    if "fault" in name.lower() or "fire" in name.lower() or "alarm" in name.lower():
                        if value not in (0, False, "normal", "inactive"):
                            alert = True
            except Exception:
                pass

        obs: dict = {
            "source_id": f"{entity_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": entity_id,
            "callsign": self._building_name,
            "entity_type": "ALERT" if alert else "BUILDING_SENSOR",
            "classification": "bacnet_device",
            "ts_iso": now.isoformat(),
            "metadata": metadata,
        }
        if self._lat is not None:
            obs["position"] = {"lat": self._lat, "lon": self._lon, "alt_m": None}

        return obs
