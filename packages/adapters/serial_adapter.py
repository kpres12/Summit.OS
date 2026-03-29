"""
Summit.OS — Generic Serial Adapter
=====================================

Connects to any device over RS-232 or RS-485 serial and emits observations
from parsed responses.

Covers the enormous long tail of hardware that speaks nothing fancy:
- Custom sensor arrays
- Legacy industrial PLCs not on Modbus
- Research robots with serial command interfaces
- Winch / crane controllers
- Older marine instruments
- Custom UAV ground stations
- Pan-tilt-zoom camera controllers

Two operation modes
-------------------
1. **poll** — Send a configurable command string on a timer, parse the response
2. **stream** — Listen continuously for newline-delimited messages

Response parsing
----------------
Supports three parse strategies via ``extra.parser``:
- ``json``    — response is JSON, fields mapped directly to observation metadata
- ``csv``     — comma-separated values, mapped by position to named fields
- ``regex``   — named capture groups map to observation fields
                (lat, lon, alt, speed, heading, or metadata.{name})

Dependencies
------------
    pip install pyserial

Config extras
-------------
port            : str   — serial port, e.g. "/dev/ttyUSB0", "COM3"
baud_rate       : int   — baud rate (default 9600)
mode            : str   — "poll" | "stream" (default "stream")
poll_command    : str   — bytes to send each poll (hex-escaped ok: "\\x02STATUS\\x03")
poll_interval_seconds: float — polling rate (default 1.0)
parser          : str   — "json" | "csv" | "regex" (default "json")
csv_fields      : list  — ordered field names for CSV parser
regex_pattern   : str   — regex with named groups for regex parser
entity_id       : str   — fixed entity ID for this device
entity_type     : str   — Summit.OS entity type (default SERIAL_DEVICE)
entity_lat      : float — static latitude (if device doesn't report position)
entity_lon      : float — static longitude
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("summit.adapters.serial")

try:
    import serial
    import serial.threaded
    _SERIAL_AVAILABLE = True
except ImportError:
    serial = None  # type: ignore
    _SERIAL_AVAILABLE = False


class SerialAdapter(BaseAdapter):
    """
    Generic serial adapter. Parses responses into Summit.OS observations.
    """

    adapter_type = "serial"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra
        self._port: str = ex.get("port", "/dev/ttyUSB0")
        self._baud: int = int(ex.get("baud_rate", 9600))
        self._mode: str = ex.get("mode", "stream")
        self._poll_cmd: Optional[str] = ex.get("poll_command")
        self._parser: str = ex.get("parser", "json")
        self._csv_fields: list = ex.get("csv_fields", [])
        self._regex: Optional[str] = ex.get("regex_pattern")
        self._entity_id: str = ex.get("entity_id", config.adapter_id)
        self._entity_type: str = ex.get("entity_type", "SERIAL_DEVICE")
        self._static_lat: Optional[float] = ex.get("entity_lat")
        self._static_lon: Optional[float] = ex.get("entity_lon")
        self._ser = None
        self._obs_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._compiled_regex = re.compile(self._regex) if self._regex else None

    async def connect(self) -> None:
        if not _SERIAL_AVAILABLE:
            raise RuntimeError("pyserial not installed. Run: pip install pyserial")
        loop = asyncio.get_event_loop()

        def _open():
            self._ser = serial.Serial(
                port=self._port,
                baudrate=self._baud,
                timeout=2.0,
            )
            logger.info("Serial opened: %s @ %d baud", self._port, self._baud)

            while not self._stop_event.is_set():
                if self._mode == "poll" and self._poll_cmd:
                    cmd = self._poll_cmd.encode().decode("unicode_escape").encode()
                    self._ser.write(cmd)
                try:
                    line = self._ser.readline()
                    if line:
                        obs = self._parse_line(line)
                        if obs:
                            loop.call_soon_threadsafe(self._obs_queue.put_nowait, obs)
                except serial.SerialTimeoutException:
                    pass

        asyncio.get_event_loop().run_in_executor(None, _open)
        await asyncio.sleep(0.5)

    async def disconnect(self) -> None:
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None

    async def stream_observations(self) -> AsyncIterator[dict]:
        while not self._stop_event.is_set():
            try:
                obs = await asyncio.wait_for(self._obs_queue.get(), timeout=5.0)
                yield obs
            except asyncio.TimeoutError:
                pass

    def _parse_line(self, raw: bytes) -> Optional[dict]:
        text = raw.strip().decode("utf-8", errors="replace")
        if not text:
            return None

        fields: dict = {}
        try:
            if self._parser == "json":
                fields = json.loads(text)
            elif self._parser == "csv" and self._csv_fields:
                parts = text.split(",")
                fields = dict(zip(self._csv_fields, parts))
            elif self._parser == "regex" and self._compiled_regex:
                m = self._compiled_regex.search(text)
                if m:
                    fields = m.groupdict()
        except Exception as e:
            logger.debug("Serial parse error: %s — raw: %s", e, text[:80])
            return None

        return self._fields_to_obs(fields, text)

    def _fields_to_obs(self, fields: dict, raw_text: str) -> dict:
        now = datetime.now(timezone.utc)
        lat = _float(fields.pop("lat", fields.pop("latitude", self._static_lat)))
        lon = _float(fields.pop("lon", fields.pop("longitude", self._static_lon)))
        alt = _float(fields.pop("alt", fields.pop("altitude", None)))
        speed = _float(fields.pop("speed", fields.pop("speed_mps", None)))
        heading = _float(fields.pop("heading", fields.pop("heading_deg", None)))

        obs: dict = {
            "source_id": f"{self._entity_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": self._entity_id,
            "callsign": self.config.display_name or self._entity_id,
            "entity_type": self._entity_type,
            "classification": "serial_device",
            "ts_iso": now.isoformat(),
            "metadata": dict(fields),
        }
        if lat is not None:
            obs["position"] = {"lat": lat, "lon": lon, "alt_m": alt}
        if speed is not None or heading is not None:
            obs["velocity"] = {"heading_deg": heading, "speed_mps": speed, "vertical_mps": None}

        return obs


def _float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
