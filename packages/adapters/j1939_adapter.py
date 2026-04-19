"""
Heli.OS — J1939 / CAN Bus Adapter
=====================================

Reads SAE J1939 messages from heavy vehicles and construction equipment
over CAN bus (or virtual CAN for testing).

J1939 is the protocol spoken by: Caterpillar, Komatsu, Hitachi,
John Deere, Volvo/Mack/Freightliner trucks, agricultural equipment,
mining machinery, and any ISO-11992-compliant vehicle.

PGNs decoded
------------
- 65267 (0xFF13)  VEHICLE_POSITION        — lat/lon
- 65265 (0xFF11)  CRUISE_CONTROL_VEH_SPEED — speed
- 65271 (0xFF17)  VEHICLE_HOURS           — engine hours
- 65262 (0xFF0E)  ENGINE_TEMPERATURE_1    — coolant temp
- 61444 (0xF004)  EL_CONT_1              — engine RPM
- 65263 (0xFF0F)  FUEL_ECONOMY           — fuel rate

Dependencies
------------
    pip install python-can j1939

Config extras
-------------
interface       : str   — CAN interface, e.g. "can0", "vcan0", "socketcan"
bustype         : str   — python-can bustype, e.g. "socketcan", "kvaser", "pcan"
vehicle_id      : str   — unique identifier for this vehicle
vehicle_type    : str   — "TRUCK", "EXCAVATOR", "DOZER", "GRADER", "HAUL_TRUCK"
vehicle_lat     : float — static lat if GPS PGN not available
vehicle_lon     : float — static lon if GPS PGN not available
bitrate         : int   — CAN bus bitrate (default 250000 for J1939)
"""
from __future__ import annotations

import asyncio
import logging
import struct
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("heli.adapters.j1939")

try:
    import can
    _CAN_AVAILABLE = True
except ImportError:
    can = None  # type: ignore
    _CAN_AVAILABLE = False

# J1939 PGN constants
PGN_VEHICLE_POSITION = 65267
PGN_VEHICLE_SPEED    = 65265
PGN_ENGINE_SPEED     = 61444
PGN_ENGINE_TEMP      = 65262
PGN_FUEL_ECONOMY     = 65263
PGN_VEHICLE_HOURS    = 65271

# Scaling per J1939 spec
_POS_SCALE = 1e-7      # raw int32 → degrees
_SPEED_SCALE = 1/256   # raw uint16 → km/h
_RPM_SCALE = 0.125     # raw uint16 → RPM


def _extract_pgn(can_id: int) -> int:
    """Extract 18-bit PGN from a 29-bit J1939 CAN ID."""
    pf = (can_id >> 16) & 0xFF
    if pf >= 240:  # PDU2 format: peer-to-peer
        return (can_id >> 8) & 0x3FFFF
    else:          # PDU1 format: destination-specific
        return (can_id >> 8) & 0x3FF00


class J1939Adapter(BaseAdapter):
    """
    Reads J1939 frames from a CAN bus and emits GROUND_VEHICLE observations.

    Decodes GPS position (PGN 65267), speed (65265), and engine telemetry,
    merging them into a unified observation per vehicle per cycle.
    """

    adapter_type = "j1939"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra
        self._interface: str = ex.get("interface", "can0")
        self._bustype: str = ex.get("bustype", "socketcan")
        self._bitrate: int = int(ex.get("bitrate", 250000))
        self._vehicle_id: str = ex.get("vehicle_id", config.adapter_id)
        self._vehicle_type: str = ex.get("vehicle_type", "GROUND_VEHICLE")
        self._static_lat: Optional[float] = ex.get("vehicle_lat")
        self._static_lon: Optional[float] = ex.get("vehicle_lon")
        self._bus = None
        self._state: dict = {}
        self._obs_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

    async def connect(self) -> None:
        if not _CAN_AVAILABLE:
            raise RuntimeError(
                "python-can not installed. Run: pip install python-can j1939"
            )
        loop = asyncio.get_event_loop()

        def _start_bus():
            self._bus = can.interface.Bus(
                channel=self._interface,
                bustype=self._bustype,
                bitrate=self._bitrate,
            )
            reader = can.BufferedReader()
            notifier = can.Notifier(self._bus, [reader])
            while not self._stop_event.is_set():
                msg = reader.get_message(timeout=1.0)
                if msg is None:
                    continue
                obs = self._process_frame(msg)
                if obs:
                    loop.call_soon_threadsafe(self._obs_queue.put_nowait, obs)

        asyncio.get_event_loop().run_in_executor(None, _start_bus)
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

    def _process_frame(self, msg) -> Optional[dict]:
        """Decode a CAN frame and update vehicle state. Returns obs if position changed."""
        if not msg.is_extended_id:
            return None
        pgn = _extract_pgn(msg.arbitration_id)
        data = msg.data
        updated = False

        if pgn == PGN_VEHICLE_POSITION and len(data) >= 8:
            lat_raw = struct.unpack_from('<i', data, 0)[0]
            lon_raw = struct.unpack_from('<i', data, 4)[0]
            if lat_raw != -2147483648 and lon_raw != -2147483648:  # J1939 error indicator
                self._state["lat"] = lat_raw * _POS_SCALE
                self._state["lon"] = lon_raw * _POS_SCALE
                updated = True

        elif pgn == PGN_VEHICLE_SPEED and len(data) >= 2:
            spd_raw = struct.unpack_from('<H', data, 1)[0]
            if spd_raw != 0xFFFF:
                self._state["speed_kmh"] = spd_raw * _SPEED_SCALE
                self._state["speed_mps"] = self._state["speed_kmh"] / 3.6
                updated = True

        elif pgn == PGN_ENGINE_SPEED and len(data) >= 4:
            rpm_raw = struct.unpack_from('<H', data, 3)[0]
            if rpm_raw != 0xFFFF:
                self._state["engine_rpm"] = rpm_raw * _RPM_SCALE

        elif pgn == PGN_ENGINE_TEMP and len(data) >= 1:
            temp = data[0] - 40  # offset 40 per J1939 spec
            self._state["coolant_temp_c"] = temp

        elif pgn == PGN_FUEL_ECONOMY and len(data) >= 4:
            fuel_raw = struct.unpack_from('<H', data, 2)[0]
            if fuel_raw != 0xFFFF:
                self._state["fuel_rate_l_h"] = fuel_raw * 0.05  # L/h

        if updated and (self._state.get("lat") or self._static_lat):
            return self._build_obs()
        return None

    def _build_obs(self) -> dict:
        now = datetime.now(timezone.utc)
        s = self._state
        lat = s.get("lat", self._static_lat)
        lon = s.get("lon", self._static_lon)
        obs: dict = {
            "source_id": f"{self._vehicle_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": self._vehicle_id,
            "callsign": self.config.display_name or self._vehicle_id,
            "entity_type": self._vehicle_type,
            "classification": "heavy_vehicle",
            "ts_iso": now.isoformat(),
            "metadata": {},
        }
        if lat is not None:
            obs["position"] = {"lat": lat, "lon": lon, "alt_m": None}
        if s.get("speed_mps") is not None:
            obs["velocity"] = {
                "heading_deg": s.get("heading_deg"),
                "speed_mps": round(s["speed_mps"], 2),
                "vertical_mps": None,
            }
        for key in ("engine_rpm", "coolant_temp_c", "fuel_rate_l_h"):
            if s.get(key) is not None:
                obs["metadata"][key] = round(s[key], 1)
        return obs
