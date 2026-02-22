"""
Hardware Abstraction Layer (HAL) for Summit.OS

Defines abstract interfaces for hardware interaction:
- Vehicle: movement, navigation, status
- Sensor: data acquisition, mode control
- Actuator: physical actions (payload, effectors)

Concrete drivers (MAVLink, SITL, ROS2, etc.) implement these interfaces.
The HAL registry provides runtime discovery and lifecycle management.
"""
from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("hal")


# ── Common Types ────────────────────────────────────────────


class HardwareState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ARMED = "armed"
    ACTIVE = "active"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class Position:
    lat: float = 0.0
    lon: float = 0.0
    alt_m: float = 0.0
    heading_deg: float = 0.0

    def to_dict(self) -> Dict:
        return {"lat": self.lat, "lon": self.lon, "alt_m": self.alt_m,
                "heading_deg": self.heading_deg}


@dataclass
class Velocity:
    north_mps: float = 0.0
    east_mps: float = 0.0
    down_mps: float = 0.0
    ground_speed_mps: float = 0.0

    def to_dict(self) -> Dict:
        return {"north": self.north_mps, "east": self.east_mps, "down": self.down_mps,
                "ground_speed": self.ground_speed_mps}


@dataclass
class HardwareInfo:
    """Static hardware identification."""
    hardware_id: str
    hardware_type: str  # "vehicle", "sensor", "actuator"
    make: str = ""
    model: str = ""
    firmware_version: str = ""
    serial_number: str = ""
    capabilities: List[str] = field(default_factory=list)


# ── Abstract Interfaces ────────────────────────────────────


class HardwareDriver(ABC):
    """Base class for all hardware drivers."""

    def __init__(self, info: HardwareInfo):
        self.info = info
        self.state = HardwareState.DISCONNECTED
        self.last_heartbeat: float = 0.0
        self._callbacks: Dict[str, List[Callable]] = {}

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to hardware."""
        ...

    @abstractmethod
    async def disconnect(self):
        """Disconnect from hardware."""
        ...

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Return health status."""
        ...

    def on(self, event: str, callback: Callable):
        """Register event callback."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def _emit(self, event: str, data: Any = None):
        """Emit event to registered callbacks."""
        for cb in self._callbacks.get(event, []):
            try:
                cb(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    def get_status(self) -> Dict:
        return {
            "hardware_id": self.info.hardware_id,
            "type": self.info.hardware_type,
            "state": self.state.value,
            "last_heartbeat": self.last_heartbeat,
        }


class VehicleDriver(HardwareDriver):
    """
    Abstract vehicle interface.

    Supports:
    - Arm/disarm
    - Takeoff/land
    - Navigate to waypoints
    - Set flight mode
    - Read position/velocity/battery
    """

    @abstractmethod
    async def arm(self) -> bool:
        ...

    @abstractmethod
    async def disarm(self) -> bool:
        ...

    @abstractmethod
    async def takeoff(self, altitude_m: float) -> bool:
        ...

    @abstractmethod
    async def land(self) -> bool:
        ...

    @abstractmethod
    async def goto(self, lat: float, lon: float, alt_m: float,
                   speed_mps: float = 5.0) -> bool:
        """Command vehicle to navigate to position."""
        ...

    @abstractmethod
    async def set_mode(self, mode: str) -> bool:
        """Set flight/drive mode (e.g., GUIDED, AUTO, LOITER, RTL)."""
        ...

    @abstractmethod
    async def get_position(self) -> Position:
        ...

    @abstractmethod
    async def get_velocity(self) -> Velocity:
        ...

    @abstractmethod
    async def get_battery(self) -> Dict[str, float]:
        """Return {"voltage": v, "current": a, "percent": pct}."""
        ...

    async def get_telemetry(self) -> Dict:
        """Get full vehicle telemetry snapshot."""
        pos = await self.get_position()
        vel = await self.get_velocity()
        bat = await self.get_battery()
        return {
            "position": pos.to_dict(),
            "velocity": vel.to_dict(),
            "battery": bat,
            "state": self.state.value,
        }


class SensorDriver(HardwareDriver):
    """
    Abstract sensor interface.

    Supports:
    - Start/stop data acquisition
    - Set operating mode
    - Read latest data frame
    """

    @abstractmethod
    async def start(self) -> bool:
        """Start data acquisition."""
        ...

    @abstractmethod
    async def stop(self) -> bool:
        """Stop data acquisition."""
        ...

    @abstractmethod
    async def set_mode(self, mode: str) -> bool:
        """Set sensor operating mode."""
        ...

    @abstractmethod
    async def get_frame(self) -> Dict[str, Any]:
        """Read latest sensor data frame."""
        ...

    @abstractmethod
    async def get_fov(self) -> Dict[str, float]:
        """Get current field of view parameters."""
        ...


class ActuatorDriver(HardwareDriver):
    """
    Abstract actuator interface.

    Supports:
    - Activate/deactivate
    - Set parameter
    - Get status
    """

    @abstractmethod
    async def activate(self) -> bool:
        ...

    @abstractmethod
    async def deactivate(self) -> bool:
        ...

    @abstractmethod
    async def set_parameter(self, name: str, value: Any) -> bool:
        ...

    @abstractmethod
    async def get_parameters(self) -> Dict[str, Any]:
        ...


# ── HAL Registry ───────────────────────────────────────────


class HALRegistry:
    """
    Central registry for all hardware drivers.

    Provides discovery, lifecycle management, and health monitoring.
    """

    def __init__(self):
        self.vehicles: Dict[str, VehicleDriver] = {}
        self.sensors: Dict[str, SensorDriver] = {}
        self.actuators: Dict[str, ActuatorDriver] = {}

    def register_vehicle(self, driver: VehicleDriver):
        self.vehicles[driver.info.hardware_id] = driver
        logger.info(f"Vehicle registered: {driver.info.hardware_id} ({driver.info.model})")

    def register_sensor(self, driver: SensorDriver):
        self.sensors[driver.info.hardware_id] = driver
        logger.info(f"Sensor registered: {driver.info.hardware_id} ({driver.info.model})")

    def register_actuator(self, driver: ActuatorDriver):
        self.actuators[driver.info.hardware_id] = driver
        logger.info(f"Actuator registered: {driver.info.hardware_id} ({driver.info.model})")

    def get_vehicle(self, hardware_id: str) -> Optional[VehicleDriver]:
        return self.vehicles.get(hardware_id)

    def get_sensor(self, hardware_id: str) -> Optional[SensorDriver]:
        return self.sensors.get(hardware_id)

    def get_actuator(self, hardware_id: str) -> Optional[ActuatorDriver]:
        return self.actuators.get(hardware_id)

    def all_drivers(self) -> List[HardwareDriver]:
        return (
            list(self.vehicles.values()) +
            list(self.sensors.values()) +
            list(self.actuators.values())
        )

    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all registered hardware."""
        results = {}
        for driver in self.all_drivers():
            try:
                ok = await driver.connect()
                results[driver.info.hardware_id] = ok
            except Exception as e:
                logger.error(f"Connect failed for {driver.info.hardware_id}: {e}")
                results[driver.info.hardware_id] = False
        return results

    async def disconnect_all(self):
        for driver in self.all_drivers():
            try:
                await driver.disconnect()
            except Exception as e:
                logger.error(f"Disconnect failed for {driver.info.hardware_id}: {e}")

    async def health_check_all(self) -> Dict[str, Dict]:
        results = {}
        for driver in self.all_drivers():
            try:
                results[driver.info.hardware_id] = await driver.health_check()
            except Exception as e:
                results[driver.info.hardware_id] = {"error": str(e)}
        return results

    def get_status(self) -> Dict:
        return {
            "vehicles": {k: v.get_status() for k, v in self.vehicles.items()},
            "sensors": {k: v.get_status() for k, v in self.sensors.items()},
            "actuators": {k: v.get_status() for k, v in self.actuators.items()},
        }
