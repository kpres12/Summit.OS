"""
Heli.OS MAVLink Adapter
==========================

Bidirectional adapter for MAVLink-speaking vehicles (ArduPilot, PX4).

Covers the two dominant open-source drone autopilots and handles the vast
majority of serious UAV platforms (quadrotors, fixed-wing, rovers, boats).

Capabilities
------------
- Connects to any MAVLink endpoint: UDP, TCP, or serial
- Tracks multiple vehicles on a single connection (by sysid)
- Requests telemetry streams at a configurable rate on connect
- Builds per-vehicle state from multiple message types, merges into one
  observation per vehicle per cycle
- Sends commands back to vehicles (GOTO, RTL, LAND, ARM, DISARM, SET_MODE)
- Runs synchronous pymavlink I/O in a thread executor so the async event
  loop is never blocked

Usage example
-------------
::

    from adapters.mavlink_adapter import MAVLinkAdapter, MAVLinkConfig

    cfg = MAVLinkConfig(
        adapter_id="mavlink-sitl",
        adapter_type="mavlink",
        display_name="ArduCopter SITL",
        connection_string="udpin:0.0.0.0:14550",
        system_ids=[],          # [] = track all
        request_streams=True,
        stream_rate_hz=4,
    )
    adapter = MAVLinkAdapter(cfg)
    await adapter.start()
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import datetime, timezone
from typing import AsyncIterator

from .base import AdapterConfig, BaseAdapter

# ---------------------------------------------------------------------------
# Optional pymavlink import — fail loudly at connect time, not import time
# ---------------------------------------------------------------------------
try:
    from pymavlink import mavutil

    _MAVLINK_AVAILABLE = True
except ImportError:
    mavutil = None  # type: ignore[assignment]
    _MAVLINK_AVAILABLE = False

logger = logging.getLogger("heli.adapters.mavlink")


# ---------------------------------------------------------------------------
# MAV_TYPE integer constants (subset used for entity-type mapping)
# These mirror pymavlink enums but are defined here so the module can be
# imported without pymavlink installed.
# ---------------------------------------------------------------------------
MAV_TYPE_FIXED_WING = 1
MAV_TYPE_QUADROTOR = 2
MAV_TYPE_GROUND_ROVER = 10
MAV_TYPE_SURFACE_BOAT = 11
MAV_TYPE_SUBMARINE = 12
MAV_TYPE_HEXAROTOR = 13
MAV_TYPE_OCTOROTOR = 14

# MAV_MODE_FLAG bitmask
MAV_MODE_FLAG_SAFETY_ARMED = 0x80  # bit 7

# System status strings (MAV_STATE)
_SYS_STATUS = {
    0: "UNINIT",
    1: "BOOT",
    2: "CALIBRATING",
    3: "STANDBY",
    4: "ACTIVE",
    5: "CRITICAL",
    6: "EMERGENCY",
    7: "POWEROFF",
    8: "FLIGHT_TERMINATION",
}

# ArduCopter flight modes (custom_mode → human string)
_ARDUCOPTER_MODES: dict[int, str] = {
    0: "STABILIZE",
    2: "ALT_HOLD",
    3: "AUTO",
    4: "GUIDED",
    5: "LOITER",
    6: "RTL",
    9: "LAND",
    16: "POSHOLD",
    17: "BRAKE",
    20: "THROW",
    21: "AVOID_ADSB",
    22: "GUIDED_NOGPS",
}

# Entity type mapping from MAV_TYPE
_ENTITY_TYPE_MAP: dict[int, str] = {
    MAV_TYPE_QUADROTOR: "ROTARY_WING",
    MAV_TYPE_HEXAROTOR: "ROTARY_WING",
    MAV_TYPE_OCTOROTOR: "ROTARY_WING",
    MAV_TYPE_FIXED_WING: "FIXED_WING",
    MAV_TYPE_GROUND_ROVER: "GROUND_VEHICLE",
    MAV_TYPE_SURFACE_BOAT: "SURFACE_VESSEL",
    MAV_TYPE_SUBMARINE: "SUBSURFACE",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class MAVLinkConfig(AdapterConfig):
    """
    Configuration for the MAVLink adapter.

    ``connection_string`` follows pymavlink conventions::

        "udpin:0.0.0.0:14550"          # listen for incoming UDP
        "udpout:192.168.1.100:14550"   # send UDP to GCS
        "tcp:192.168.1.100:5760"       # TCP client
        "serial:/dev/ttyUSB0:57600"    # serial port

    ``system_ids``: list of MAVLink sysids to track.  Empty list = track all.
    """

    connection_string: str = "udpin:0.0.0.0:14550"
    """MAVLink connection URI passed directly to mavutil.mavlink_connection()."""

    system_ids: list[int] = []
    """Sysids of vehicles to track. Empty = track all."""

    request_streams: bool = True
    """Send REQUEST_DATA_STREAM messages to autopilot on connect."""

    stream_rate_hz: int = 4
    """Telemetry rate to request (Hz). Typical range 1–10."""

    class Config:
        extra = "allow"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class MAVLinkAdapter(BaseAdapter):
    """
    MAVLink bidirectional adapter.

    Reads telemetry from ArduPilot/PX4 autopilots and sends commands back.
    Supports any connection type accepted by pymavlink (UDP, TCP, serial).
    Handles a swarm on a single connection — each sysid is tracked separately
    and emitted as an independent observation stream.
    """

    adapter_type = "mavlink"

    def __init__(self, config: MAVLinkConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        self.config: MAVLinkConfig = config

        # pymavlink connection object (synchronous)
        self._conn = None

        # Per-vehicle state: sysid → dict of latest field values
        # Fields accumulate across multiple message types and are merged
        # into a single observation each cycle.
        self._vehicle_state: dict[int, dict] = {}

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the MAVLink connection and optionally request telemetry streams."""
        if not _MAVLINK_AVAILABLE:
            raise RuntimeError(
                "pymavlink is not installed. "
                "Install it with: pip install pymavlink>=2.4.41"
            )

        conn_str = self.config.connection_string
        self._log.info("Connecting to MAVLink endpoint: %s", conn_str)

        loop = asyncio.get_event_loop()
        self._conn = await loop.run_in_executor(
            None,
            lambda: mavutil.mavlink_connection(conn_str, autoreconnect=True),
        )

        if self.config.request_streams:
            await self._request_data_streams()

        self._log.info("MAVLink connection established: %s", conn_str)

    async def disconnect(self) -> None:
        """Close the MAVLink connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception as exc:
                self._log.debug("Error closing MAVLink connection: %s", exc)
            finally:
                self._conn = None
        self._vehicle_state.clear()

    # -------------------------------------------------------------------------
    # Stream
    # -------------------------------------------------------------------------

    async def stream_observations(self) -> AsyncIterator[dict]:
        """
        Continuously read MAVLink messages and yield per-vehicle observations.

        Uses run_in_executor to avoid blocking the event loop on the
        synchronous pymavlink recv_match() call.  A 0.1-second timeout
        keeps the blocking call short so stop signals are handled promptly.

        Only vehicles with a known GPS position are emitted.
        """
        loop = asyncio.get_event_loop()
        conn = self._conn

        while not self._stop_event.is_set():
            # Offload blocking I/O to thread pool
            msg = await loop.run_in_executor(
                None,
                lambda: conn.recv_match(blocking=True, timeout=0.1),
            )

            if msg is None:
                # Timeout — no message received; keep looping
                continue

            msg_type = msg.get_type()
            if msg_type == "BAD_DATA":
                continue

            sysid: int = msg.get_srcSystem()

            # Filter by system_ids if a whitelist is configured
            if self.config.system_ids and sysid not in self.config.system_ids:
                continue

            # Initialise state bucket for new vehicles
            if sysid not in self._vehicle_state:
                self._vehicle_state[sysid] = {}

            state = self._vehicle_state[sysid]
            updated = self._handle_message(msg_type, msg, state)

            if not updated:
                continue

            # Only emit when we have a valid GPS position
            if not self._has_position(state):
                continue

            obs = self._build_observation(sysid, state)
            yield obs

    # -------------------------------------------------------------------------
    # Message handling
    # -------------------------------------------------------------------------

    def _handle_message(self, msg_type: str, msg, state: dict) -> bool:
        """
        Merge relevant MAVLink message fields into the vehicle state dict.

        Returns True if any fields were updated (triggers an observation).
        """
        if msg_type == "GLOBAL_POSITION_INT":
            state["lat"] = msg.lat / 1e7
            state["lon"] = msg.lon / 1e7
            state["alt_m"] = msg.alt / 1000.0
            state["relative_alt_m"] = msg.relative_alt / 1000.0
            state["vx_mps"] = msg.vx / 100.0
            state["vy_mps"] = msg.vy / 100.0
            state["vz_mps"] = msg.vz / 100.0
            # hdg is in cdeg (0–36000), 0xFFFF = unknown
            if msg.hdg != 0xFFFF:
                state["heading_deg"] = msg.hdg / 100.0
            return True

        if msg_type == "VFR_HUD":
            state["airspeed_mps"] = msg.airspeed
            state["groundspeed_mps"] = msg.groundspeed
            if "heading_deg" not in state:
                state["heading_deg"] = float(msg.heading)
            state["throttle_pct"] = msg.throttle
            state["climb_mps"] = msg.climb
            return True

        if msg_type == "BATTERY_STATUS":
            # battery_remaining is -1 when not supported
            if msg.battery_remaining >= 0:
                state["battery_pct"] = msg.battery_remaining
            return True

        if msg_type == "HEARTBEAT":
            state["base_mode"] = msg.base_mode
            state["custom_mode"] = msg.custom_mode
            state["system_status"] = msg.system_status
            state["mav_type"] = msg.type
            return True

        if msg_type == "SYS_STATUS":
            state["voltage_battery"] = msg.voltage_battery  # mV
            state["current_battery"] = msg.current_battery  # cA
            # Prefer BATTERY_STATUS if already set; SYS_STATUS as fallback
            if "battery_pct" not in state and msg.battery_remaining >= 0:
                state["battery_pct"] = msg.battery_remaining
            return True

        if msg_type == "GPS_RAW_INT":
            state["gps_fix_type"] = msg.fix_type
            state["satellites_visible"] = msg.satellites_visible
            state["eph"] = msg.eph  # HDOP * 100
            state["epv"] = msg.epv  # VDOP * 100
            return True

        if msg_type == "ATTITUDE":
            state["roll_deg"] = math.degrees(msg.roll)
            state["pitch_deg"] = math.degrees(msg.pitch)
            state["yaw_deg"] = math.degrees(msg.yaw)
            return True

        return False

    # -------------------------------------------------------------------------
    # Observation builder
    # -------------------------------------------------------------------------

    def _build_observation(self, sysid: int, state: dict) -> dict:
        """Construct a normalised Heli.OS observation from vehicle state."""
        ts_ms = int(time.time() * 1000)
        utcnow_iso = datetime.now(timezone.utc).isoformat()

        # Entity type from MAV_TYPE
        mav_type = state.get("mav_type", -1)
        entity_type = _ENTITY_TYPE_MAP.get(mav_type, "AIRCRAFT")

        # Flight mode string
        custom_mode = state.get("custom_mode")
        mode_string = (
            _ARDUCOPTER_MODES.get(custom_mode, f"MODE_{custom_mode}")
            if custom_mode is not None
            else "UNKNOWN"
        )

        # System status string
        sys_status_int = state.get("system_status")
        system_status_string = (
            _SYS_STATUS.get(sys_status_int, f"STATUS_{sys_status_int}")
            if sys_status_int is not None
            else "UNKNOWN"
        )

        # ARM flag from base_mode bitmask
        base_mode = state.get("base_mode", 0)
        armed = bool(base_mode & MAV_MODE_FLAG_SAFETY_ARMED)

        # Battery voltage (mV → V)
        voltage_battery = state.get("voltage_battery")
        voltage_v = voltage_battery / 1000.0 if voltage_battery is not None else None

        # HDOP from eph (eph = HDOP * 100)
        eph = state.get("eph")
        hdop = eph / 100.0 if eph is not None else None

        return {
            "source_id": f"mavlink-{sysid}-{ts_ms}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": "mavlink",
            "entity_id": f"mavlink-{self.config.adapter_id}-{sysid}",
            "callsign": f"UAV-{sysid}",
            "entity_type": entity_type,
            "position": {
                "lat": state.get("lat"),
                "lon": state.get("lon"),
                "alt_m": state.get("alt_m"),
                "relative_alt_m": state.get("relative_alt_m"),
            },
            "velocity": {
                "vx_mps": state.get("vx_mps"),
                "vy_mps": state.get("vy_mps"),
                "vz_mps": state.get("vz_mps"),
                "groundspeed_mps": state.get("groundspeed_mps"),
                "airspeed_mps": state.get("airspeed_mps"),
                "heading_deg": state.get("heading_deg"),
                "climb_mps": state.get("climb_mps"),
            },
            "attitude": {
                "roll_deg": state.get("roll_deg"),
                "pitch_deg": state.get("pitch_deg"),
                "yaw_deg": state.get("yaw_deg"),
            },
            "metadata": {
                "battery_pct": state.get("battery_pct"),
                "armed": armed,
                "mode": mode_string,
                "system_status": system_status_string,
                "gps_fix": state.get("gps_fix_type"),
                "gps_satellites": state.get("satellites_visible"),
                "hdop": hdop,
                "sysid": sysid,
                "throttle_pct": state.get("throttle_pct"),
                "voltage_v": voltage_v,
            },
            "ts_iso": utcnow_iso,
        }

    # -------------------------------------------------------------------------
    # Command sending
    # -------------------------------------------------------------------------

    async def send_command(self, entity_id: str, command: dict) -> None:
        """
        Send a command to a vehicle identified by its entity_id.

        Supported command types::

            {"type": "GOTO", "lat": float, "lon": float, "alt": float}
            {"type": "RTL"}
            {"type": "LAND"}
            {"type": "ARM"}
            {"type": "DISARM"}
            {"type": "SET_MODE", "mode": str}

        The sysid is extracted from the last segment of the entity_id
        (e.g. ``"mavlink-sitl-1"`` → sysid 1).
        """
        if self._conn is None:
            raise RuntimeError("Adapter not connected — cannot send command.")

        # Extract sysid from entity_id
        try:
            sysid = int(entity_id.rsplit("-", 1)[-1])
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Cannot extract sysid from entity_id '{entity_id}'. "
                "Expected format: 'mavlink-<adapter_id>-<sysid>'"
            ) from exc

        cmd_type = command.get("type", "").upper()
        loop = asyncio.get_event_loop()

        if cmd_type == "GOTO":
            lat = float(command["lat"])
            lon = float(command["lon"])
            alt = float(command["alt"])
            await loop.run_in_executor(
                None,
                lambda: self._send_mavlink_command(
                    sysid=sysid,
                    command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                    param1=0,  # hold time (s)
                    param2=0,  # accept radius (m)
                    param3=0,  # pass radius (m, 0 = loiter)
                    param4=0,  # yaw (NaN = unchanged)
                    param5=lat,
                    param6=lon,
                    param7=alt,
                ),
            )
            self._log.info(
                "Sent GOTO to sysid=%d: lat=%.6f lon=%.6f alt=%.1f",
                sysid,
                lat,
                lon,
                alt,
            )

        elif cmd_type == "RTL":
            await loop.run_in_executor(
                None,
                lambda: self._send_mavlink_command(
                    sysid=sysid,
                    command=mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
                ),
            )
            self._log.info("Sent RTL to sysid=%d", sysid)

        elif cmd_type == "LAND":
            await loop.run_in_executor(
                None,
                lambda: self._send_mavlink_command(
                    sysid=sysid,
                    command=mavutil.mavlink.MAV_CMD_NAV_LAND,
                ),
            )
            self._log.info("Sent LAND to sysid=%d", sysid)

        elif cmd_type == "ARM":
            await loop.run_in_executor(
                None,
                lambda: self._send_mavlink_command(
                    sysid=sysid,
                    command=mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    param1=1,  # 1 = arm
                ),
            )
            self._log.info("Sent ARM to sysid=%d", sysid)

        elif cmd_type == "DISARM":
            await loop.run_in_executor(
                None,
                lambda: self._send_mavlink_command(
                    sysid=sysid,
                    command=mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    param1=0,  # 0 = disarm
                ),
            )
            self._log.info("Sent DISARM to sysid=%d", sysid)

        elif cmd_type == "SET_MODE":
            mode_str = command.get("mode", "")
            # Find custom_mode integer from mode string
            custom_mode_int = None
            for k, v in _ARDUCOPTER_MODES.items():
                if v == mode_str.upper():
                    custom_mode_int = k
                    break
            if custom_mode_int is None:
                raise ValueError(
                    f"Unknown mode '{mode_str}'. "
                    f"Known modes: {list(_ARDUCOPTER_MODES.values())}"
                )
            await loop.run_in_executor(
                None,
                lambda: self._send_mavlink_command(
                    sysid=sysid,
                    command=mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                    param1=mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    param2=float(custom_mode_int),
                ),
            )
            self._log.info(
                "Sent SET_MODE=%s (%d) to sysid=%d", mode_str, custom_mode_int, sysid
            )

        else:
            raise ValueError(
                f"Unknown command type '{cmd_type}'. "
                "Supported: GOTO, RTL, LAND, ARM, DISARM, SET_MODE"
            )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _send_mavlink_command(
        self,
        sysid: int,
        command: int,
        param1: float = 0,
        param2: float = 0,
        param3: float = 0,
        param4: float = 0,
        param5: float = 0,
        param6: float = 0,
        param7: float = 0,
    ) -> None:
        """
        Send a MAV_CMD via COMMAND_LONG to the specified sysid.

        This is a synchronous call and must be invoked via run_in_executor.
        """
        self._conn.mav.command_long_send(
            sysid,  # target_system
            1,  # target_component (autopilot)
            command,  # command
            0,  # confirmation
            param1,
            param2,
            param3,
            param4,
            param5,
            param6,
            param7,
        )

    async def _request_data_streams(self) -> None:
        """
        Request standard telemetry streams from the autopilot.

        Sends REQUEST_DATA_STREAM for the four stream groups that provide
        the fields this adapter consumes.  This is best-effort; ArduPilot
        may ignore individual requests without warning.
        """
        if not _MAVLINK_AVAILABLE:
            return

        conn = self._conn
        rate = self.config.stream_rate_hz

        streams = [
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,  # GLOBAL_POSITION_INT
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,  # ATTITUDE
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA2,  # VFR_HUD
            mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS,  # SYS_STATUS, GPS_RAW_INT
        ]

        loop = asyncio.get_event_loop()

        for stream_id in streams:
            await loop.run_in_executor(
                None,
                lambda sid=stream_id: conn.mav.request_data_stream_send(
                    conn.target_system,
                    conn.target_component,
                    sid,
                    rate,
                    1,  # start_stop: 1 = start
                ),
            )
            self._log.debug("Requested MAVLink stream id=%d at %d Hz", stream_id, rate)

    @staticmethod
    def _has_position(state: dict) -> bool:
        """Return True only when the vehicle has reported a GPS position."""
        return state.get("lat") is not None and state.get("lon") is not None
