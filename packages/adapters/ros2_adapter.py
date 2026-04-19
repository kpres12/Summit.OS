"""
Heli.OS — ROS 2 Adapter
==========================

Bridges ROS 2 robots into the Heli.OS entity stream.

Subscribes to standard ROS 2 topics and translates them into Heli.OS
observations. Covers any robot running ROS 2: Boston Dynamics Spot (via
spot_ros2), Clearpath Husky/Jackal, AgileX, Unitree, custom AMRs, and
any research platform.

Standard topics consumed
------------------------
- /fix                  sensor_msgs/NavSatFix          GPS position
- /odom                 nav_msgs/Odometry               velocity + pose
- /battery_state        sensor_msgs/BatteryState        battery %
- /joint_states         sensor_msgs/JointState          robot joints (metadata)
- /robot_description    std_msgs/String                 URDF (metadata)

Custom topic override
---------------------
Set ``extra.topics`` to a dict mapping field → topic path to override defaults.

Dependencies
------------
    pip install rclpy  (requires a ROS 2 installation)

Config extras
-------------
robot_id        : str   — unique identifier for this robot instance
robot_type      : str   — e.g. "SPOT", "HUSKY", "CUSTOM" (default GROUND_ROBOT)
namespace       : str   — ROS 2 node namespace prefix (default "")
topics          : dict  — override default topic paths
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("heli.adapters.ros2")

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    _ROS2_AVAILABLE = True
except ImportError:
    rclpy = None  # type: ignore
    _ROS2_AVAILABLE = False


class ROS2Adapter(BaseAdapter):
    """
    Bridges any ROS 2 robot into the Heli.OS entity stream.

    Runs a minimal ROS 2 node in a background thread, subscribing to
    standard topics and enqueuing observations for the async stream.
    """

    adapter_type = "ros2"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra
        self._robot_id: str = ex.get("robot_id", config.adapter_id)
        self._robot_type: str = ex.get("robot_type", "GROUND_ROBOT")
        self._ns: str = ex.get("namespace", "").rstrip("/")
        self._topic_overrides: dict = ex.get("topics", {})
        self._obs_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._node = None
        self._executor = None
        self._spin_task = None
        self._state: dict = {}  # latest state merged from multiple topics

    def _topic(self, default: str) -> str:
        key = default.lstrip("/")
        override = self._topic_overrides.get(key, default)
        if self._ns:
            return f"{self._ns}/{override.lstrip('/')}"
        return override

    async def connect(self) -> None:
        if not _ROS2_AVAILABLE:
            raise RuntimeError(
                "rclpy not available. Install ROS 2 and source the setup script, "
                "then: pip install rclpy"
            )
        loop = asyncio.get_event_loop()

        def _spin():
            rclpy.init(args=None)
            self._node = rclpy.create_node(
                f"summit_{self._robot_id.replace('-', '_')}"
            )
            # Import message types lazily
            try:
                from sensor_msgs.msg import NavSatFix, BatteryState
                from nav_msgs.msg import Odometry

                self._node.create_subscription(
                    NavSatFix, self._topic("/fix"),
                    lambda msg: loop.call_soon_threadsafe(
                        self._obs_queue.put_nowait,
                        self._from_navsatfix(msg)
                    ), 10
                )
                self._node.create_subscription(
                    Odometry, self._topic("/odom"),
                    lambda msg: self._merge_odom(msg), 10
                )
                self._node.create_subscription(
                    BatteryState, self._topic("/battery_state"),
                    lambda msg: self._merge_battery(msg), 10
                )
            except ImportError as e:
                logger.warning("ROS 2 message type not available: %s", e)

            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            while not self._stop_event.is_set():
                self._executor.spin_once(timeout_sec=0.1)

        self._spin_task = asyncio.get_event_loop().run_in_executor(None, _spin)
        await asyncio.sleep(1.0)  # give spin thread time to initialise

    async def disconnect(self) -> None:
        if self._executor:
            try:
                self._executor.shutdown()
            except Exception:
                pass
        if self._node:
            try:
                self._node.destroy_node()
            except Exception:
                pass
        if _ROS2_AVAILABLE:
            try:
                rclpy.shutdown()
            except Exception:
                pass

    async def stream_observations(self) -> AsyncIterator[dict]:
        while not self._stop_event.is_set():
            try:
                obs = await asyncio.wait_for(self._obs_queue.get(), timeout=5.0)
                yield obs
            except asyncio.TimeoutError:
                # Emit a heartbeat with last known state if we have position
                if self._state.get("lat") is not None:
                    yield self._build_obs()

    def _from_navsatfix(self, msg) -> dict:
        self._state["lat"] = msg.latitude
        self._state["lon"] = msg.longitude
        self._state["alt"] = msg.altitude
        return self._build_obs()

    def _merge_odom(self, msg) -> None:
        twist = msg.twist.twist
        vx = twist.linear.x
        vy = twist.linear.y
        speed = (vx ** 2 + vy ** 2) ** 0.5
        import math
        heading = math.degrees(math.atan2(vy, vx)) % 360
        self._state["speed_mps"] = speed
        self._state["heading_deg"] = heading
        self._state["vertical_mps"] = twist.linear.z

    def _merge_battery(self, msg) -> None:
        if msg.percentage >= 0:
            self._state["battery_pct"] = msg.percentage * 100

    def _build_obs(self) -> dict:
        now = datetime.now(timezone.utc)
        s = self._state
        obs: dict = {
            "source_id": f"{self._robot_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": self._robot_id,
            "callsign": self.config.display_name or self._robot_id,
            "entity_type": self._robot_type,
            "classification": "ros2_robot",
            "ts_iso": now.isoformat(),
            "metadata": {},
        }
        if s.get("lat") is not None:
            obs["position"] = {
                "lat": s["lat"],
                "lon": s["lon"],
                "alt_m": s.get("alt"),
            }
        if s.get("speed_mps") is not None:
            obs["velocity"] = {
                "heading_deg": s.get("heading_deg"),
                "speed_mps": s.get("speed_mps"),
                "vertical_mps": s.get("vertical_mps"),
            }
        if s.get("battery_pct") is not None:
            obs["metadata"]["battery_pct"] = round(s["battery_pct"], 1)
        return obs
