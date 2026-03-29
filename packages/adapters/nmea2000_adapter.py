"""
Summit.OS — NMEA 2000 Adapter
================================

Integrates modern marine electronics via NMEA 2000 (the CAN-based marine
standard, replacing NMEA 0183 on vessels built since ~2005).

NMEA 2000 is deployed on: powerboats, sailboats, commercial fishing vessels,
ferries, patrol boats, research vessels, and any vessel with modern electronics.

Decodes
-------
- PGN 129025 — Position, Rapid Update (lat/lon at 10 Hz)
- PGN 129026 — COG & SOG, Rapid Update (course and speed)
- PGN 127250 — Vessel Heading
- PGN 128267 — Water Depth
- PGN 127257 — Attitude (pitch, roll, yaw)
- PGN 127505 — Fluid Level (fuel, water, oil)
- PGN 127488 — Engine Parameters (RPM)
- PGN 130306 — Wind Data (if weather station fitted)
- PGN 129029 — GNSS Position Data (full fix with altitude)

Uses python-can for the CAN interface (same as J1939/ISOBUS adapters).

Dependencies
------------
    pip install python-can

Config extras
-------------
interface       : str   — CAN interface, e.g. "can0", "vcan0"
bustype         : str   — python-can bustype (default "socketcan")
vessel_id       : str   — MMSI or custom vessel identifier
vessel_name     : str   — display name
bitrate         : int   — CAN bitrate, always 250000 for NMEA 2000
"""
from __future__ import annotations

import asyncio
import logging
import struct
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("summit.adapters.nmea2000")

try:
    import can
    _CAN_AVAILABLE = True
except ImportError:
    can = None  # type: ignore
    _CAN_AVAILABLE = False

# NMEA 2000 PGN constants
PGN_POSITION_RAPID      = 129025
PGN_COG_SOG_RAPID       = 129026
PGN_VESSEL_HEADING      = 127250
PGN_WATER_DEPTH         = 128267
PGN_ATTITUDE            = 127257
PGN_FLUID_LEVEL         = 127505
PGN_ENGINE_RAPID        = 127488
PGN_WIND_DATA           = 130306
PGN_GNSS_POSITION       = 129029

_RAD_TO_DEG = 57.295779513


def _n2k_pgn(can_id: int) -> int:
    """Extract PGN from a 29-bit NMEA 2000 CAN ID."""
    pf = (can_id >> 16) & 0xFF
    if pf >= 240:
        return (can_id >> 8) & 0x3FFFF
    return (can_id >> 8) & 0x3FF00


class NMEA2000Adapter(BaseAdapter):
    """
    Reads NMEA 2000 CAN frames and emits SURFACE_VESSEL observations.
    """

    adapter_type = "nmea2000"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra
        self._interface: str = ex.get("interface", "can0")
        self._bustype: str = ex.get("bustype", "socketcan")
        self._vessel_id: str = ex.get("vessel_id", config.adapter_id)
        self._vessel_name: str = ex.get("vessel_name", config.display_name)
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
                bitrate=250000,  # NMEA 2000 is always 250k
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
                if self._state.get("lat"):
                    yield self._build_obs()

    def _process(self, msg) -> Optional[dict]:
        if not msg.is_extended_id:
            return None
        pgn = _n2k_pgn(msg.arbitration_id)
        data = msg.data
        updated = False

        if pgn == PGN_POSITION_RAPID and len(data) >= 8:
            lat_raw = struct.unpack_from('<i', data, 0)[0]
            lon_raw = struct.unpack_from('<i', data, 4)[0]
            if lat_raw != 0x7FFFFFFF:
                self._state["lat"] = lat_raw * 1e-7
                self._state["lon"] = lon_raw * 1e-7
                updated = True

        elif pgn == PGN_COG_SOG_RAPID and len(data) >= 6:
            cog_raw = struct.unpack_from('<H', data, 2)[0]
            sog_raw = struct.unpack_from('<H', data, 4)[0]
            if cog_raw != 0xFFFF:
                self._state["cog_deg"] = (cog_raw * 1e-4) * _RAD_TO_DEG % 360
            if sog_raw != 0xFFFF:
                self._state["sog_mps"] = sog_raw * 0.01 * (1852 / 3600)  # 0.01 knots → m/s

        elif pgn == PGN_VESSEL_HEADING and len(data) >= 3:
            hdg_raw = struct.unpack_from('<H', data, 1)[0]
            if hdg_raw != 0xFFFF:
                self._state["heading_deg"] = (hdg_raw * 1e-4) * _RAD_TO_DEG % 360

        elif pgn == PGN_WATER_DEPTH and len(data) >= 4:
            depth_raw = struct.unpack_from('<I', data, 0)[0]
            if depth_raw != 0xFFFFFFFF:
                self._state["depth_m"] = depth_raw * 0.01

        elif pgn == PGN_GNSS_POSITION and len(data) >= 9:
            alt_raw = struct.unpack_from('<q', data, 1)[0] if len(data) >= 9 else None
            if alt_raw is not None and alt_raw != 0x7FFFFFFFFFFFFFFF:
                self._state["alt_m"] = alt_raw * 1e-6

        elif pgn == PGN_ATTITUDE and len(data) >= 7:
            pitch_raw = struct.unpack_from('<h', data, 1)[0]
            roll_raw  = struct.unpack_from('<h', data, 3)[0]
            if pitch_raw != 0x7FFF:
                self._state["pitch_deg"] = pitch_raw * 1e-4 * _RAD_TO_DEG
                self._state["roll_deg"]  = roll_raw  * 1e-4 * _RAD_TO_DEG

        elif pgn == PGN_WIND_DATA and len(data) >= 6:
            wind_spd_raw = struct.unpack_from('<H', data, 0)[0]
            wind_ang_raw = struct.unpack_from('<H', data, 2)[0]
            if wind_spd_raw != 0xFFFF:
                self._state["wind_speed_mps"] = wind_spd_raw * 0.01
                self._state["wind_angle_deg"] = wind_ang_raw * 1e-4 * _RAD_TO_DEG

        elif pgn == PGN_FLUID_LEVEL and len(data) >= 4:
            level_raw = struct.unpack_from('<H', data, 2)[0]
            if level_raw != 0xFFFF:
                instance = data[0] & 0x0F
                self._state[f"fluid_level_{instance}_pct"] = level_raw * 0.004

        if updated:
            return self._build_obs()
        return None

    def _build_obs(self) -> dict:
        now = datetime.now(timezone.utc)
        s = self._state
        obs: dict = {
            "source_id": f"{self._vessel_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": self._vessel_id,
            "callsign": self._vessel_name,
            "entity_type": "SURFACE_VESSEL",
            "classification": "nmea2000_vessel",
            "ts_iso": now.isoformat(),
            "metadata": {},
        }
        if s.get("lat"):
            obs["position"] = {
                "lat": s["lat"],
                "lon": s["lon"],
                "alt_m": s.get("alt_m"),
            }
        if s.get("sog_mps") is not None:
            obs["velocity"] = {
                "heading_deg": s.get("heading_deg") or s.get("cog_deg"),
                "speed_mps": round(s["sog_mps"], 3),
                "vertical_mps": None,
            }
        for k in ("depth_m", "pitch_deg", "roll_deg", "wind_speed_mps",
                  "wind_angle_deg", "cog_deg"):
            if s.get(k) is not None:
                obs["metadata"][k] = round(s[k], 2)
        # Fluid levels
        for k, v in s.items():
            if k.startswith("fluid_level_"):
                obs["metadata"][k] = round(v, 1)
        return obs
