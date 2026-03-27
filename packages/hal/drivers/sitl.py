"""
SITL (Software-In-The-Loop) Vehicle Driver

A simulated vehicle driver for testing and development.
Simulates realistic position, velocity, battery drain, and flight modes
without requiring physical hardware or external simulators.
"""

from __future__ import annotations

import math
import time
import random
import logging
from typing import Any, Dict, Optional

from packages.hal.base import (
    VehicleDriver,
    HardwareInfo,
    HardwareState,
    Position,
    Velocity,
)

logger = logging.getLogger("hal.sitl")


class SITLDriver(VehicleDriver):
    """
    Software-in-the-loop vehicle simulator.

    Simulates:
    - GPS position with configurable noise
    - Velocity based on waypoint navigation
    - Battery drain based on motor usage
    - Flight mode transitions
    """

    def __init__(
        self,
        vehicle_id: str = "sitl-001",
        start_lat: float = 34.0522,
        start_lon: float = -118.2437,
        start_alt: float = 0.0,
        max_speed_mps: float = 15.0,
        battery_capacity_mah: float = 5000.0,
        gps_noise_m: float = 1.5,
    ):
        info = HardwareInfo(
            hardware_id=vehicle_id,
            hardware_type="vehicle",
            make="Summit",
            model="SITL-Quad",
            firmware_version="1.0.0-sim",
            capabilities=["fly", "hover", "land", "rtl", "guided", "auto"],
        )
        super().__init__(info)

        self._pos = Position(lat=start_lat, lon=start_lon, alt_m=start_alt)
        self._vel = Velocity()
        self._target: Optional[Position] = None
        self._target_speed: float = 5.0
        self._mode: str = "STABILIZE"
        self._armed: bool = False

        # Battery sim
        self._battery_mah = battery_capacity_mah
        self._battery_remaining = battery_capacity_mah
        self._battery_voltage = 16.8  # 4S fully charged

        # Sim params
        self.max_speed_mps = max_speed_mps
        self.gps_noise_m = gps_noise_m
        self._last_update = time.time()
        self._home = Position(lat=start_lat, lon=start_lon, alt_m=0.0)

    async def connect(self) -> bool:
        self.state = HardwareState.CONNECTED
        self._last_update = time.time()
        logger.info(f"SITL {self.info.hardware_id} connected")
        return True

    async def disconnect(self):
        self.state = HardwareState.DISCONNECTED
        self._armed = False

    async def health_check(self) -> Dict[str, Any]:
        return {
            "status": "ok" if self.state != HardwareState.ERROR else "error",
            "armed": self._armed,
            "mode": self._mode,
            "battery_pct": self._get_battery_pct(),
            "gps_fix": True,
            "satellites": 12 + random.randint(-2, 2),
        }

    async def arm(self) -> bool:
        if self.state == HardwareState.DISCONNECTED:
            return False
        self._armed = True
        self.state = HardwareState.ARMED
        logger.info(f"SITL {self.info.hardware_id} armed")
        return True

    async def disarm(self) -> bool:
        self._armed = False
        self.state = HardwareState.CONNECTED
        return True

    async def takeoff(self, altitude_m: float) -> bool:
        if not self._armed:
            return False
        self._target = Position(lat=self._pos.lat, lon=self._pos.lon, alt_m=altitude_m)
        self._mode = "GUIDED"
        self.state = HardwareState.ACTIVE
        return True

    async def land(self) -> bool:
        self._target = Position(lat=self._pos.lat, lon=self._pos.lon, alt_m=0.0)
        self._mode = "LAND"
        return True

    async def goto(
        self, lat: float, lon: float, alt_m: float, speed_mps: float = 5.0
    ) -> bool:
        if not self._armed:
            return False
        self._target = Position(lat=lat, lon=lon, alt_m=alt_m)
        self._target_speed = min(speed_mps, self.max_speed_mps)
        self._mode = "GUIDED"
        return True

    async def set_mode(self, mode: str) -> bool:
        if mode == "RTL":
            self._target = self._home
            self._mode = "RTL"
        else:
            self._mode = mode
        return True

    async def get_position(self) -> Position:
        self._update_sim()
        # Add GPS noise
        noise_lat = random.gauss(0, self.gps_noise_m / 111320)
        noise_lon = random.gauss(
            0,
            self.gps_noise_m
            / (111320 * max(0.01, math.cos(math.radians(self._pos.lat)))),
        )
        return Position(
            lat=self._pos.lat + noise_lat,
            lon=self._pos.lon + noise_lon,
            alt_m=self._pos.alt_m + random.gauss(0, self.gps_noise_m * 0.5),
            heading_deg=self._pos.heading_deg,
        )

    async def get_velocity(self) -> Velocity:
        self._update_sim()
        return Velocity(
            north_mps=self._vel.north_mps,
            east_mps=self._vel.east_mps,
            down_mps=self._vel.down_mps,
            ground_speed_mps=math.sqrt(self._vel.north_mps**2 + self._vel.east_mps**2),
        )

    async def get_battery(self) -> Dict[str, float]:
        self._update_sim()
        pct = self._get_battery_pct()
        # Voltage sags with discharge (simple linear model)
        voltage = 14.0 + (16.8 - 14.0) * (pct / 100)
        return {
            "voltage": round(voltage, 2),
            "current": round(random.uniform(5, 15) if self._armed else 0.5, 2),
            "percent": round(pct, 1),
        }

    # ── Simulation Engine ──────────────────────────────────

    def _update_sim(self):
        """Advance simulation by elapsed time."""
        now = time.time()
        dt = now - self._last_update
        if dt < 0.01:
            return
        self._last_update = now

        # Battery drain
        if self._armed:
            drain_rate = 10.0 if self._pos.alt_m > 1 else 3.0  # mAh/sec (rough)
            self._battery_remaining = max(0, self._battery_remaining - drain_rate * dt)

        # Navigation
        if self._target and self._armed:
            dlat = (self._target.lat - self._pos.lat) * 111320
            dlon = (
                (self._target.lon - self._pos.lon)
                * 111320
                * math.cos(math.radians(self._pos.lat))
            )
            dalt = self._target.alt_m - self._pos.alt_m
            dist_h = math.sqrt(dlat**2 + dlon**2)
            dist_3d = math.sqrt(dist_h**2 + dalt**2)

            if dist_3d < 0.5:
                # Arrived
                self._vel = Velocity()
                if self._mode == "LAND" and self._pos.alt_m < 0.5:
                    self._armed = False
                    self.state = HardwareState.CONNECTED
                    self._target = None
            else:
                speed = min(self._target_speed, dist_3d)
                # Heading
                self._pos.heading_deg = math.degrees(math.atan2(dlon, dlat)) % 360
                # Move
                ratio = speed * dt / dist_3d
                ratio = min(ratio, 1.0)
                self._pos.lat += (self._target.lat - self._pos.lat) * ratio
                self._pos.lon += (self._target.lon - self._pos.lon) * ratio
                self._pos.alt_m += dalt * ratio
                # Velocity
                self._vel.north_mps = dlat / dist_3d * speed
                self._vel.east_mps = dlon / dist_3d * speed
                self._vel.down_mps = -dalt / dist_3d * speed

        self.last_heartbeat = now

    def _get_battery_pct(self) -> float:
        if self._battery_mah <= 0:
            return 0.0
        return (self._battery_remaining / self._battery_mah) * 100.0
