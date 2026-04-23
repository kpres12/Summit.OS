"""
Heli.OS — Boston Dynamics Spot Adapter
=========================================

Integrates Boston Dynamics Spot robot dogs into Heli.OS.

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
import math
from datetime import datetime, timezone
from typing import AsyncIterator, List, Optional

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("heli.adapters.spot")

try:
    import bosdyn.client
    import bosdyn.client.util
    from bosdyn.client.robot_state import RobotStateClient
    from bosdyn.client.image import ImageClient, build_image_request
    from bosdyn.client.frame_helpers import get_vision_tform_body, VISION_FRAME_NAME, BODY_FRAME_NAME
    _SPOT_SDK_AVAILABLE = True
except ImportError:
    _SPOT_SDK_AVAILABLE = False

try:
    from ultralytics import YOLO as _YOLO
    import cv2 as _cv2
    import numpy as _np
    _DETECTION_AVAILABLE = True
except ImportError:
    _YOLO = None      # type: ignore
    _cv2 = None       # type: ignore
    _np = None        # type: ignore
    _DETECTION_AVAILABLE = False

# Spot front camera source name
_SPOT_FRONT_CAM = "frontleft_fisheye_image"
_DETECT_CONF = 0.45
_DETECT_INTERVAL = 3.0   # seconds between detection frames


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
        self._image_client = None
        self._yolo = None
        self._detect_interval: float = float(ex.get("detect_interval_sec", _DETECT_INTERVAL))
        self._current_lat: float = 0.0
        self._current_lon: float = 0.0
        self._current_heading_deg: float = 0.0

    async def connect(self) -> None:
        if not _SPOT_SDK_AVAILABLE:
            raise RuntimeError(
                "bosdyn-client not installed. Run: pip install bosdyn-client"
            )
        loop = asyncio.get_event_loop()

        def _authenticate():
            sdk = bosdyn.client.create_standard_sdk("heli_os_spot_adapter")
            robot = sdk.create_robot(self._hostname)
            bosdyn.client.util.authenticate(robot)
            robot.sync_with_directory()
            return robot

        self._robot = await loop.run_in_executor(None, _authenticate)
        self._state_client = self._robot.ensure_client(RobotStateClient.default_service_name)
        if _SPOT_SDK_AVAILABLE:
            try:
                self._image_client = self._robot.ensure_client(ImageClient.default_service_name)
            except Exception:
                self._image_client = None
        if _DETECTION_AVAILABLE:
            try:
                self._yolo = _YOLO("yolov8n.pt")
                logger.info("Spot detection enabled (YOLOv8n)")
            except Exception as e:
                logger.warning("YOLOv8 load failed: %s", e)
        logger.info("Spot connected: %s", self._hostname)

    async def disconnect(self) -> None:
        self._robot = None
        self._state_client = None
        self._image_client = None

    async def stream_observations(self) -> AsyncIterator[dict]:
        import time as _time
        loop = asyncio.get_event_loop()
        last_detect = 0.0
        while not self._stop_event.is_set():
            try:
                state = await loop.run_in_executor(None, self._state_client.get_robot_state)
                obs = self._state_to_obs(state)
                if obs:
                    yield obs

                # Run detection on interval
                if self._yolo and self._image_client:
                    now = _time.monotonic()
                    if now - last_detect >= self._detect_interval:
                        last_detect = now
                        detections = await self._run_detection(loop)
                        for det in detections:
                            yield det
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
                # Convert quaternion to heading
                q = tform.rot
                yaw = math.degrees(math.atan2(
                    2 * (q.w * q.z + q.x * q.y),
                    1 - 2 * (q.y ** 2 + q.z ** 2)
                )) % 360
                self._current_heading_deg = yaw
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

        # GPS requires Boston Dynamics RTK GPS payload (spot-gps accessory) or
        # an external fix injected into metadata by the operator (key: gps_lat/gps_lon).
        # Without it, lat/lon remain None and detection positions will be unreliable.
        # Do NOT use Spot for geo-referenced detection without a confirmed GPS fix.
        gps_lat = obs.get("metadata", {}).get("gps_lat")
        gps_lon = obs.get("metadata", {}).get("gps_lon")
        if gps_lat and gps_lon:
            self._current_lat = float(gps_lat)
            self._current_lon = float(gps_lon)
            if obs.get("position"):
                obs["position"]["lat"] = self._current_lat
                obs["position"]["lon"] = self._current_lon

        # Fault monitoring
        try:
            faults = state.system_fault_state.faults
            if faults:
                obs["metadata"]["active_faults"] = [f.name for f in faults]
        except Exception:
            pass

        return obs

    async def _run_detection(self, loop) -> List[dict]:
        """Grab front camera frame from Spot SDK and run YOLOv8 person detection."""

        def _grab():
            req = build_image_request(_SPOT_FRONT_CAM, quality_percent=75)
            responses = self._image_client.get_image([req])
            if not responses:
                return None
            img_resp = responses[0]
            data = _np.frombuffer(img_resp.shot.image.data, dtype=_np.uint8)
            frame = _cv2.imdecode(data, _cv2.IMREAD_COLOR)
            return frame

        try:
            frame = await loop.run_in_executor(None, _grab)
        except Exception as e:
            logger.debug("Spot image grab failed: %s", e)
            return []

        if frame is None:
            return []

        try:
            results = self._yolo(frame, classes=[0], conf=_DETECT_CONF, verbose=False)
        except Exception as e:
            logger.debug("Spot YOLO inference failed: %s", e)
            return []

        now = datetime.now(timezone.utc)
        detections = []
        h, w = frame.shape[:2]

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                # Estimate bearing from pixel x offset
                cx = float(box.xywh[0][0])
                offset_ratio = (cx - w / 2) / (w / 2)
                bearing_offset = offset_ratio * 30.0  # ~30° half-FOV for front cam
                bearing = (self._current_heading_deg + bearing_offset) % 360

                # Project ~10m ahead in bearing direction (Spot can't see far)
                det_lat, det_lon = self._project_latlon(
                    self._current_lat, self._current_lon, bearing, 10.0
                )

                det = {
                    "source_id": f"{self._robot_id}:detect:{now.timestamp():.3f}",
                    "adapter_id": self.config.adapter_id,
                    "adapter_type": self.adapter_type,
                    "entity_id": self._robot_id,
                    "event_type": "entity_detected",
                    "classification": "person",
                    "confidence": round(conf, 3),
                    "lat": det_lat,
                    "lon": det_lon,
                    "position": {"lat": det_lat, "lon": det_lon, "alt_m": 0.0},
                    "ts_iso": now.isoformat(),
                    "metadata": {
                        "bearing_deg": round(bearing, 1),
                        "robot_heading_deg": round(self._current_heading_deg, 1),
                        "detection_source": "yolov8n",
                        "camera": _SPOT_FRONT_CAM,
                    },
                }
                detections.append(det)
                logger.info(
                    "Spot person detected — conf=%.2f bearing=%.1f°",
                    conf, bearing,
                )

        return detections

    @staticmethod
    def _project_latlon(lat: float, lon: float, bearing_deg: float, dist_m: float):
        R = 6_371_000.0
        b = math.radians(bearing_deg)
        lat_r = math.radians(lat)
        lon_r = math.radians(lon)
        d = dist_m / R
        new_lat = math.asin(
            math.sin(lat_r) * math.cos(d) +
            math.cos(lat_r) * math.sin(d) * math.cos(b)
        )
        new_lon = lon_r + math.atan2(
            math.sin(b) * math.sin(d) * math.cos(lat_r),
            math.cos(d) - math.sin(lat_r) * math.sin(new_lat),
        )
        return math.degrees(new_lat), math.degrees(new_lon)
