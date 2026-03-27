"""
MAVLink Vehicle Driver for Summit.OS

Interfaces with MAVLink-compatible autopilots (ArduPilot, PX4) via:
- pymavlink (when installed)
- Direct serial/UDP connection
- Telemetry ingestion and command execution

Implements the VehicleDriver interface from packages/hal/base.py.
"""

from __future__ import annotations

import asyncio
import time
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger("hal.mavlink")


class MAVFlightMode(str, Enum):
    STABILIZE = "STABILIZE"
    ALT_HOLD = "ALT_HOLD"
    LOITER = "LOITER"
    RTL = "RTL"
    AUTO = "AUTO"
    GUIDED = "GUIDED"
    LAND = "LAND"
    TAKEOFF = "TAKEOFF"
    OFFBOARD = "OFFBOARD"


@dataclass
class MAVTelemetry:
    """Telemetry data from a MAVLink vehicle."""

    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    relative_alt: float = 0.0
    heading: float = 0.0
    groundspeed: float = 0.0
    airspeed: float = 0.0
    climb_rate: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    battery_voltage: float = 0.0
    battery_remaining: int = 100
    gps_fix_type: int = 0
    satellites_visible: int = 0
    flight_mode: str = "UNKNOWN"
    armed: bool = False
    system_status: str = "UNINIT"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "position": {"lat": self.lat, "lon": self.lon, "alt": self.alt},
            "attitude": {"roll": self.roll, "pitch": self.pitch, "yaw": self.yaw},
            "speed": {
                "ground": self.groundspeed,
                "air": self.airspeed,
                "climb": self.climb_rate,
            },
            "heading": self.heading,
            "battery": {
                "voltage": self.battery_voltage,
                "remaining": self.battery_remaining,
            },
            "gps": {"fix": self.gps_fix_type, "sats": self.satellites_visible},
            "mode": self.flight_mode,
            "armed": self.armed,
        }


class MAVLinkDriver:
    """
    MAVLink vehicle driver.

    Connects to autopilot via UDP or serial and provides:
    - Telemetry streaming
    - Waypoint navigation
    - Mode changes
    - Arm/disarm
    - Guided mode goto commands

    Falls back to simulation when pymavlink is not available.
    """

    def __init__(
        self,
        connection_string: str = "udp:127.0.0.1:14550",
        system_id: int = 1,
        component_id: int = 1,
        heartbeat_interval: float = 1.0,
    ):
        self.connection_string = connection_string
        self.system_id = system_id
        self.component_id = component_id
        self.heartbeat_interval = heartbeat_interval

        self._mavlink = None
        self._connected = False
        self._telemetry = MAVTelemetry()
        self._running = False
        self._pymavlink_available = self._check_pymavlink()

        # Waypoint queue
        self._waypoints: List[Tuple[float, float, float]] = []
        self._current_wp_idx = 0

        # Callbacks
        self._telemetry_callbacks: List[Any] = []

    @staticmethod
    def _check_pymavlink() -> bool:
        try:
            from pymavlink import mavutil

            return True
        except ImportError:
            return False

    async def connect(self) -> bool:
        """Connect to the autopilot."""
        if self._pymavlink_available:
            try:
                from pymavlink import mavutil

                self._mavlink = mavutil.mavlink_connection(
                    self.connection_string,
                    source_system=self.system_id,
                    source_component=self.component_id,
                )
                # Wait for heartbeat
                self._mavlink.wait_heartbeat(timeout=10)
                self._connected = True
                logger.info(f"Connected to MAVLink vehicle at {self.connection_string}")
                return True
            except Exception as e:
                logger.error(f"MAVLink connection failed: {e}")
                return False
        else:
            logger.warning("pymavlink not installed — using simulated MAVLink driver")
            self._connected = True
            return True

    async def disconnect(self) -> None:
        """Disconnect from the autopilot."""
        self._running = False
        self._connected = False
        if self._mavlink:
            self._mavlink.close()
        logger.info("MAVLink disconnected")

    async def start_telemetry(self) -> None:
        """Start receiving telemetry."""
        self._running = True
        if self._pymavlink_available and self._mavlink:
            asyncio.ensure_future(self._telemetry_loop())
        else:
            asyncio.ensure_future(self._sim_telemetry_loop())

    async def _telemetry_loop(self) -> None:
        """Read telemetry from real MAVLink connection."""
        while self._running:
            try:
                msg = self._mavlink.recv_match(blocking=False)
                if msg:
                    self._process_message(msg)
            except Exception as e:
                logger.error(f"Telemetry error: {e}")
            await asyncio.sleep(0.05)

    async def _sim_telemetry_loop(self) -> None:
        """Generate simulated telemetry."""
        t = 0
        while self._running:
            t += 1
            self._telemetry.lat = 34.0 + 0.0001 * math.sin(t * 0.01)
            self._telemetry.lon = -118.0 + 0.0001 * math.cos(t * 0.01)
            self._telemetry.alt = 100 + 10 * math.sin(t * 0.02)
            self._telemetry.heading = (t * 0.5) % 360
            self._telemetry.groundspeed = 15 + 2 * math.sin(t * 0.03)
            self._telemetry.battery_remaining = max(0, 100 - t // 100)
            self._telemetry.armed = True
            self._telemetry.flight_mode = "GUIDED"
            self._telemetry.gps_fix_type = 3
            self._telemetry.satellites_visible = 12
            self._telemetry.timestamp = time.time()

            for cb in self._telemetry_callbacks:
                cb(self._telemetry)

            await asyncio.sleep(0.1)

    def _process_message(self, msg):
        """Process a MAVLink message."""
        msg_type = msg.get_type()

        if msg_type == "GLOBAL_POSITION_INT":
            self._telemetry.lat = msg.lat / 1e7
            self._telemetry.lon = msg.lon / 1e7
            self._telemetry.alt = msg.alt / 1000
            self._telemetry.relative_alt = msg.relative_alt / 1000
            self._telemetry.heading = msg.hdg / 100
        elif msg_type == "VFR_HUD":
            self._telemetry.airspeed = msg.airspeed
            self._telemetry.groundspeed = msg.groundspeed
            self._telemetry.heading = msg.heading
            self._telemetry.climb_rate = msg.climb
        elif msg_type == "ATTITUDE":
            self._telemetry.roll = math.degrees(msg.roll)
            self._telemetry.pitch = math.degrees(msg.pitch)
            self._telemetry.yaw = math.degrees(msg.yaw)
        elif msg_type == "SYS_STATUS":
            self._telemetry.battery_voltage = msg.voltage_battery / 1000
            self._telemetry.battery_remaining = msg.battery_remaining
        elif msg_type == "HEARTBEAT":
            from pymavlink import mavutil

            self._telemetry.armed = (
                msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            ) != 0
        elif msg_type == "GPS_RAW_INT":
            self._telemetry.gps_fix_type = msg.fix_type
            self._telemetry.satellites_visible = msg.satellites_visible

        self._telemetry.timestamp = time.time()
        for cb in self._telemetry_callbacks:
            cb(self._telemetry)

    # ── Commands ────────────────────────────────────────────

    async def arm(self) -> bool:
        """Arm the vehicle."""
        if self._mavlink:
            self._mavlink.arducopter_arm()
            return True
        self._telemetry.armed = True
        return True

    async def disarm(self) -> bool:
        if self._mavlink:
            self._mavlink.arducopter_disarm()
            return True
        self._telemetry.armed = False
        return True

    async def set_mode(self, mode: str) -> bool:
        """Set flight mode."""
        if self._mavlink:
            self._mavlink.set_mode_apm(mode)
        self._telemetry.flight_mode = mode
        return True

    async def goto(self, lat: float, lon: float, alt: float) -> bool:
        """Navigate to a waypoint in GUIDED mode."""
        if self._mavlink:
            from pymavlink import mavutil

            self._mavlink.mav.set_position_target_global_int_send(
                0,
                self.system_id,
                self.component_id,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                0b0000111111111000,
                int(lat * 1e7),
                int(lon * 1e7),
                alt,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )
        logger.info(f"Goto: ({lat:.6f}, {lon:.6f}, {alt:.1f}m)")
        return True

    async def takeoff(self, altitude: float = 10.0) -> bool:
        """Takeoff to specified altitude."""
        await self.set_mode("GUIDED")
        await self.arm()
        if self._mavlink:
            self._mavlink.mav.command_long_send(
                self.system_id,
                self.component_id,
                22,  # MAV_CMD_NAV_TAKEOFF
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                altitude,
            )
        self._telemetry.alt = altitude
        return True

    async def land(self) -> bool:
        return await self.set_mode("LAND")

    async def rtl(self) -> bool:
        return await self.set_mode("RTL")

    def on_telemetry(self, callback) -> None:
        self._telemetry_callbacks.append(callback)

    @property
    def telemetry(self) -> MAVTelemetry:
        return self._telemetry

    @property
    def is_connected(self) -> bool:
        return self._connected
