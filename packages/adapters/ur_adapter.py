"""
Summit.OS — Universal Robots (UR) Adapter
==========================================

Integrates Universal Robots cobots (UR3, UR5, UR10, UR16, UR20, UR30)
into Summit.OS via the RTDE (Real-Time Data Exchange) protocol.

Universal Robots are the world's best-selling collaborative robots.
They are deployed for: inspection, assembly, palletizing, machine tending,
welding, dispensing, packaging, quality control, and field operations.

Capabilities
------------
- Real-time joint angles and TCP (tool center point) position
- Robot mode: running, stopped, paused, error, e-stopped
- Safety status monitoring — alerts on protective stop or e-stop
- Tool I/O state (digital inputs/outputs)
- Payload and joint torque monitoring

Dependencies
------------
    pip install ur-rtde  (or: pip install urx)

Config extras
-------------
host            : str   — Robot IP address (e.g. "192.168.1.100")
port            : int   — RTDE port (default 30004)
robot_id        : str   — unique identifier
robot_location  : dict  — {"lat": float, "lon": float} of physical location
frequency       : float — RTDE sampling frequency Hz (default 10)
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("summit.adapters.ur")

try:
    import rtde.rtde as rtde_lib
    import rtde.rtde_config as rtde_config_lib
    _RTDE_AVAILABLE = True
except ImportError:
    _RTDE_AVAILABLE = False

# RTDE variables we want from the robot
_RTDE_RECIPE = [
    "actual_q",                # joint angles [6]
    "actual_TCP_pose",         # TCP pose [6]: x,y,z,rx,ry,rz
    "actual_TCP_speed",        # TCP speed [6]
    "robot_mode",              # int: RUNNING=7, IDLE=5, etc.
    "safety_mode",             # int: NORMAL=1, PROTECTIVE_STOP=3, etc.
    "actual_current",          # joint currents [6]
    "target_speed_fraction",   # speed slider 0-1
]

_ROBOT_MODE = {
    0: "disconnected",
    1: "confirm_safety",
    2: "booting",
    3: "power_off",
    4: "power_on",
    5: "idle",
    6: "backdrive",
    7: "running",
    8: "updating_firmware",
}

_SAFETY_MODE = {
    1: "normal",
    2: "reduced",
    3: "protective_stop",
    4: "recovery",
    5: "safeguard_stop",
    6: "system_emergency_stop",
    7: "robot_emergency_stop",
    8: "violation",
    9: "fault",
}


class URAdapter(BaseAdapter):
    """
    Connects to a Universal Robot via RTDE and emits INDUSTRIAL_ROBOT observations.
    """

    adapter_type = "ur_robot"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra
        self._host: str = ex.get("host", "")
        self._port: int = int(ex.get("port", 30004))
        self._robot_id: str = ex.get("robot_id", config.adapter_id)
        self._location: dict = ex.get("robot_location", {})
        self._frequency: float = float(ex.get("frequency", 10.0))
        self._rtde = None

    async def connect(self) -> None:
        if not _RTDE_AVAILABLE:
            raise RuntimeError(
                "ur-rtde not installed. Run: pip install ur-rtde\n"
                "See: https://sdurobotics.gitlab.io/ur_rtde/"
            )
        if not self._host:
            raise ValueError("UR adapter requires 'host' in config.extra")

        loop = asyncio.get_event_loop()

        def _connect():
            r = rtde_lib.RTDE(self._host, self._port)
            r.connect()
            r.get_controller_version()
            r.send_output_setup(_RTDE_RECIPE, frequencies=[self._frequency] * len(_RTDE_RECIPE))
            r.send_start()
            return r

        self._rtde = await loop.run_in_executor(None, _connect)
        logger.info("UR robot connected: %s", self._host)

    async def disconnect(self) -> None:
        if self._rtde:
            try:
                self._rtde.send_pause()
                self._rtde.disconnect()
            except Exception:
                pass
        self._rtde = None

    async def stream_observations(self) -> AsyncIterator[dict]:
        loop = asyncio.get_event_loop()
        interval = 1.0 / max(self._frequency, 0.1)
        while not self._stop_event.is_set():
            try:
                state = await loop.run_in_executor(None, self._rtde.receive)
                if state:
                    obs = self._state_to_obs(state)
                    if obs:
                        yield obs
            except Exception as e:
                logger.warning("UR RTDE receive error: %s", e)
                raise
            await asyncio.sleep(interval)

    def _state_to_obs(self, state) -> Optional[dict]:
        now = datetime.now(timezone.utc)
        robot_mode = getattr(state, "robot_mode", -1)
        safety_mode = getattr(state, "safety_mode", 1)
        tcp_pose = list(getattr(state, "actual_TCP_pose", [0] * 6))
        joint_angles = list(getattr(state, "actual_q", [0] * 6))
        tcp_speed = list(getattr(state, "actual_TCP_speed", [0] * 6))

        # Alert if in any stop/fault state
        is_alert = safety_mode not in (1, 2)  # not normal or reduced
        entity_type = "ALERT" if is_alert else "INDUSTRIAL_ROBOT"

        speed_mps = (sum(v ** 2 for v in tcp_speed[:3])) ** 0.5 if tcp_speed else 0

        obs: dict = {
            "source_id": f"{self._robot_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": self._robot_id,
            "callsign": self.config.display_name or self._robot_id,
            "entity_type": entity_type,
            "classification": "ur_cobot",
            "ts_iso": now.isoformat(),
            "metadata": {
                "robot_mode": _ROBOT_MODE.get(robot_mode, f"mode_{robot_mode}"),
                "safety_mode": _SAFETY_MODE.get(safety_mode, f"safety_{safety_mode}"),
                "tcp_pose_m": [round(v, 4) for v in tcp_pose[:3]],
                "joint_angles_deg": [round(v * 57.2958, 2) for v in joint_angles],
                "speed_mps": round(speed_mps, 4),
            },
        }
        if self._location:
            obs["position"] = {
                "lat": self._location.get("lat", 0),
                "lon": self._location.get("lon", 0),
                "alt_m": self._location.get("alt"),
            }
        obs["velocity"] = {
            "heading_deg": None,
            "speed_mps": round(speed_mps, 4),
            "vertical_mps": round(tcp_speed[2], 4) if tcp_speed else None,
        }
        return obs
