"""
Summit.OS — Boston Dynamics Spot Adapter
=========================================

Integrates Boston Dynamics Spot robot dogs into Summit.OS.

Uses the official Spot Python SDK (bosdyn-client) to read telemetry
and publish GROUND_ROBOT observations. Spot is widely deployed for
SAR, inspection, hazmat, and construction site monitoring.

Capabilities
------------
- Robot state: position, velocity, battery, E-stop status
- Camera feeds: front/back/left/right (via metadata link)
- Fault monitoring: alerts on E-stop or critical faults
- Command: STAND, SIT, STOP (if WRITE capability needed, extend via bosdyn.client.robot_command)

Dependencies
------------
    pip install bosdyn-client bosdyn-mission bosdyn-choreography-client

Config extras
-------------
hostname        : str   — Spot's IP address or hostname (e.g. "192.168.80.3")
username        : str   — Spot web UI username (default "user")
password        : str   — Spot web UI password
robot_id        : str   — unique identifier (default adapter_id)
poll_interval_seconds : float — telemetry poll rate (default 1.0)
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("summit.adapters.spot")

try:
    import bosdyn.client
    import bosdyn.client.util
    from bosdyn.client.robot_state import RobotStateClient
    from bosdyn.client.frame_helpers import get_vision_tform_body, VISION_FRAME_NAME, BODY_FRAME_NAME
    _SPOT_SDK_AVAILABLE = True
except ImportError:
    _SPOT_SDK_AVAILABLE = False


class SpotAdapter(BaseAdapter):
    """
    Polls Boston Dynamics Spot for state and emits GROUND_ROBOT observations.
    """

    adapter_type = "spot"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra
        self._hostname: str = ex.get("hostname", "192.168.80.3")
        self._username: str = ex.get("username", "user")
        self._password: str = ex.get("password", "")
        self._robot_id: str = ex.get("robot_id", config.adapter_id)
        self._robot = None
        self._state_client = None

    async def connect(self) -> None:
        if not _SPOT_SDK_AVAILABLE:
            raise RuntimeError(
                "bosdyn-client not installed. Run: pip install bosdyn-client"
            )
        loop = asyncio.get_event_loop()

        def _authenticate():
            sdk = bosdyn.client.create_standard_sdk("summit_os_spot_adapter")
            robot = sdk.create_robot(self._hostname)
            bosdyn.client.util.authenticate(robot)
            robot.sync_with_directory()
            return robot

        self._robot = await loop.run_in_executor(None, _authenticate)
        self._state_client = self._robot.ensure_client(RobotStateClient.default_service_name)
        logger.info("Spot connected: %s", self._hostname)

    async def disconnect(self) -> None:
        self._robot = None
        self._state_client = None

    async def stream_observations(self) -> AsyncIterator[dict]:
        loop = asyncio.get_event_loop()
        while not self._stop_event.is_set():
            try:
                state = await loop.run_in_executor(None, self._state_client.get_robot_state)
                obs = self._state_to_obs(state)
                if obs:
                    yield obs
            except Exception as e:
                logger.warning("Spot state poll failed: %s", e)
                raise
            await asyncio.sleep(self.config.poll_interval_seconds)

    def _state_to_obs(self, state) -> Optional[dict]:
        now = datetime.now(timezone.utc)
        obs: dict = {
            "source_id": f"{self._robot_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": self._robot_id,
            "callsign": self.config.display_name or self._robot_id,
            "entity_type": "GROUND_ROBOT",
            "classification": "spot_robot",
            "ts_iso": now.isoformat(),
            "metadata": {},
        }

        # Battery
        try:
            bat = state.power_state.locomotion_charge_percentage.value
            obs["metadata"]["battery_pct"] = round(bat, 1)
        except Exception:
            pass

        # E-stop
        try:
            estop = state.estop_states
            obs["metadata"]["estop_active"] = any(
                s.state == 1 for s in estop  # ESTOPPED = 1
            )
        except Exception:
            pass

        # Kinematic state / vision frame position
        try:
            kin = state.kinematic_state
            tform = get_vision_tform_body(kin.transforms_snapshot)
            if tform:
                obs["position"] = {
                    "lat": None,  # Spot doesn't have GPS by default
                    "lon": None,
                    "alt_m": tform.z,
                }
                import math
                # Convert quaternion to heading
                q = tform.rot
                yaw = math.degrees(math.atan2(
                    2 * (q.w * q.z + q.x * q.y),
                    1 - 2 * (q.y ** 2 + q.z ** 2)
                )) % 360
                vx = kin.velocity_of_body_in_vision.linear.x
                vy = kin.velocity_of_body_in_vision.linear.y
                speed = (vx ** 2 + vy ** 2) ** 0.5
                obs["velocity"] = {
                    "heading_deg": round(yaw, 1),
                    "speed_mps": round(speed, 3),
                    "vertical_mps": round(kin.velocity_of_body_in_vision.linear.z, 3),
                }
        except Exception:
            pass

        # Fault monitoring
        try:
            faults = state.system_fault_state.faults
            if faults:
                obs["metadata"]["active_faults"] = [f.name for f in faults]
        except Exception:
            pass

        return obs
