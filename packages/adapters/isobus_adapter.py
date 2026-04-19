"""
Heli.OS — ISOBUS (ISO 11783) Adapter
=========================================

Integrates agricultural and off-highway machinery via ISOBUS.

ISOBUS (ISO 11783) is the CAN-based protocol for modern farm equipment:
John Deere, CNH (Case / New Holland), AGCO (Fendt, Massey Ferguson),
Claas, KUHN, LEMKEN, and virtually every precision agriculture system.

A superset of J1939 extended with agricultural-specific PGNs for:
- Machine position and speed
- PTO (power take-off) state
- Implement section control
- Variable-rate prescription maps
- Task Controller data (field operations logging)

Uses the same python-can backend as the J1939 adapter.

Dependencies
------------
    pip install python-can

Config extras
-------------
interface       : str   — CAN interface (e.g. "can0", "vcan0")
bustype         : str   — python-can bustype (default "socketcan")
machine_id      : str   — unique identifier for this machine
machine_type    : str   — "TRACTOR", "COMBINE", "SPRAYER", "PLANTER", "HARVESTER"
machine_lat     : float — static lat fallback
machine_lon     : float — static lon fallback
bitrate         : int   — CAN bitrate (default 250000)
"""
from __future__ import annotations

import asyncio
import logging
import struct
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("heli.adapters.isobus")

try:
    import can
    _CAN_AVAILABLE = True
except ImportError:
    can = None  # type: ignore
    _CAN_AVAILABLE = False

# ISOBUS / J1939 PGNs relevant to agricultural equipment
PGN_VEHICLE_POSITION        = 65267   # lat/lon (shared with J1939)
PGN_GROUND_BASED_SPEED_DIST = 65097   # GBSD — ground speed from radar
PGN_WHEEL_BASED_SPEED_DIST  = 65096   # WBSD — wheel speed
PGN_ENGINE_SPEED            = 61444   # EEC1
PGN_PTO_OUTPUT_SHAFT        = 65091   # PTO speed
PGN_LIGHTING_DATA           = 65280   # field lights on/off
PGN_MACHINE_SELECTED_SPEED  = 0xFE48  # ISOBUS machine selected speed

_LAT_SCALE = 1e-7
_LON_SCALE = 1e-7
_SPEED_SCALE = 1 / 256  # km/h


def _pgn_from_id(can_id: int) -> int:
    pf = (can_id >> 16) & 0xFF
    if pf >= 240:
        return (can_id >> 8) & 0x3FFFF
    return (can_id >> 8) & 0x3FF00


class ISOBUSAdapter(BaseAdapter):
    """
    Reads ISOBUS CAN frames from agricultural machinery and emits
    GROUND_VEHICLE observations with farm-specific metadata.
    """

    adapter_type = "isobus"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra
        self._interface: str = ex.get("interface", "can0")
        self._bustype: str = ex.get("bustype", "socketcan")
        self._bitrate: int = int(ex.get("bitrate", 250000))
        self._machine_id: str = ex.get("machine_id", config.adapter_id)
        self._machine_type: str = ex.get("machine_type", "AGRICULTURAL_MACHINE")
        self._static_lat: Optional[float] = ex.get("machine_lat")
        self._static_lon: Optional[float] = ex.get("machine_lon")
        self._bus = None
        self._state: dict = {}
        self._obs_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

    async def connect(self) -> None:
        if not _CAN_AVAILABLE:
            raise RuntimeError("python-can not installed. Run: pip install python-can")

        loop = asyncio.get_event_loop()

        def _run():
            self._bus = can.interface.Bus(
                channel=self._interface,
                bustype=self._bustype,
                bitrate=self._bitrate,
            )
            reader = can.BufferedReader()
            can.Notifier(self._bus, [reader])
            while not self._stop_event.is_set():
                msg = reader.get_message(timeout=1.0)
                if msg is None:
                    continue
                obs = self._process(msg)
                if obs:
                    loop.call_soon_threadsafe(self._obs_queue.put_nowait, obs)

        asyncio.get_event_loop().run_in_executor(None, _run)
        await asyncio.sleep(0.5)

    async def disconnect(self) -> None:
        if self._bus:
            try:
                self._bus.shutdown()
            except Exception:
                pass
            self._bus = None

    async def stream_observations(self) -> AsyncIterator[dict]:
        while not self._stop_event.is_set():
            try:
                obs = await asyncio.wait_for(self._obs_queue.get(), timeout=2.0)
                yield obs
            except asyncio.TimeoutError:
                if self._state:
                    yield self._build_obs()

    def _process(self, msg) -> Optional[dict]:
        if not msg.is_extended_id:
            return None
        pgn = _pgn_from_id(msg.arbitration_id)
        data = msg.data
        updated = False

        if pgn == PGN_VEHICLE_POSITION and len(data) >= 8:
            lat_raw = struct.unpack_from('<i', data, 0)[0]
            lon_raw = struct.unpack_from('<i', data, 4)[0]
            if lat_raw != -2147483648:
                self._state["lat"] = lat_raw * _LAT_SCALE
                self._state["lon"] = lon_raw * _LON_SCALE
                updated = True

        elif pgn == PGN_GROUND_BASED_SPEED_DIST and len(data) >= 2:
            spd_raw = struct.unpack_from('<H', data, 0)[0]
            if spd_raw != 0xFFFF:
                self._state["speed_mps"] = spd_raw * (1 / 1000)  # mm/s → m/s per ISOBUS
                updated = True

        elif pgn == PGN_ENGINE_SPEED and len(data) >= 4:
            rpm_raw = struct.unpack_from('<H', data, 3)[0]
            if rpm_raw != 0xFFFF:
                self._state["engine_rpm"] = rpm_raw * 0.125

        elif pgn == PGN_PTO_OUTPUT_SHAFT and len(data) >= 2:
            pto_raw = struct.unpack_from('<H', data, 0)[0]
            if pto_raw != 0xFFFF:
                self._state["pto_rpm"] = pto_raw * 0.125

        if updated and (self._state.get("lat") or self._static_lat):
            return self._build_obs()
        return None

    def _build_obs(self) -> dict:
        now = datetime.now(timezone.utc)
        s = self._state
        obs: dict = {
            "source_id": f"{self._machine_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": self._machine_id,
            "callsign": self.config.display_name or self._machine_id,
            "entity_type": self._machine_type,
            "classification": "agricultural_machine",
            "ts_iso": now.isoformat(),
            "metadata": {},
        }
        lat = s.get("lat", self._static_lat)
        lon = s.get("lon", self._static_lon)
        if lat is not None:
            obs["position"] = {"lat": lat, "lon": lon, "alt_m": None}
        if s.get("speed_mps") is not None:
            obs["velocity"] = {
                "heading_deg": s.get("heading_deg"),
                "speed_mps": round(s["speed_mps"], 3),
                "vertical_mps": None,
            }
        for k in ("engine_rpm", "pto_rpm"):
            if s.get(k) is not None:
                obs["metadata"][k] = round(s[k], 1)
        return obs
