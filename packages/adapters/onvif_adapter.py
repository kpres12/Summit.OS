"""
Heli.OS — ONVIF IP Camera Adapter
=====================================

Connects to ONVIF-compliant IP cameras. Fetches device info, media profiles,
and PTZ capabilities. Also supports PTZ control via send_command().

Detection
---------
When ultralytics (YOLOv8) and opencv-python are installed, the adapter
optionally pulls frames from the camera's RTSP stream and runs person
detection. Each detection emits an entity_detected observation alongside the
normal heartbeat, feeding the mission orchestrator's _check_c2intel loop.

Set rtsp_url in config extras to enable detection. Without it, the adapter
still emits heartbeat observations as before.

Dependencies
------------
    pip install onvif-zeep-async
    pip install ultralytics opencv-python   # optional — enables detection
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import AsyncIterator, List, Optional

try:
    from onvif import ONVIFCamera
except ImportError:
    raise ImportError(
        "onvif-zeep-async is required for ONVIFAdapter. "
        "Install with: pip install onvif-zeep-async"
    )

try:
    from ultralytics import YOLO as _YOLO
    import cv2 as _cv2
    _DETECTION_AVAILABLE = True
except ImportError:
    _YOLO = None  # type: ignore
    _cv2 = None   # type: ignore
    _DETECTION_AVAILABLE = False

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("heli.adapters.onvif")

_UPDATE_INTERVAL = 60.0  # seconds between heartbeat observations


class ONVIFAdapter(BaseAdapter):
    """
    Connects to an ONVIF IP camera and emits CAMERA entity observations.

    Config extras
    -------------
    host           : str
    port           : int    (default 80)
    username       : str
    password       : str
    camera_lat     : float  (default 0.0)
    camera_lon     : float  (default 0.0)
    camera_alt_m   : float  (default 0.0)
    """

    adapter_type = "onvif"

    # Detection config defaults
    _DETECT_INTERVAL = 5.0    # seconds between detection frames
    _DETECT_CONF = 0.45        # minimum YOLO confidence to emit
    _DETECT_CLASS_PERSON = 0   # COCO class 0 = person

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra

        self._host: str = ex.get("host", "")
        if not self._host:
            raise ValueError("host must be set in adapter extra config")
        self._port: int = int(ex.get("port", 80))
        self._username: str = ex.get("username", "admin")
        self._password: str = ex.get("password", "")
        self._camera_lat: float = float(ex.get("camera_lat", 0.0))
        self._camera_lon: float = float(ex.get("camera_lon", 0.0))
        self._camera_alt_m: float = float(ex.get("camera_alt_m", 0.0))
        self._camera_fov_deg: float = float(ex.get("camera_fov_deg", 60.0))
        self._camera_height_m: float = float(ex.get("camera_height_m", 4.0))

        # Optional RTSP URL for detection frames
        self._rtsp_url: Optional[str] = ex.get("rtsp_url")
        self._detect_interval: float = float(ex.get("detect_interval_sec", self._DETECT_INTERVAL))
        self._detect_conf: float = float(ex.get("detect_conf", self._DETECT_CONF))

        self._camera: Optional[ONVIFCamera] = None
        self._ptz_service = None
        self._media_service = None
        self._profiles: list[str] = []
        self._ptz_supported: bool = False
        self._yolo = None
        self._current_pan_deg: float = 0.0   # updated after PTZ_MOVE commands

        # Device info populated on connect
        self._manufacturer: Optional[str] = None
        self._model: Optional[str] = None
        self._firmware_version: Optional[str] = None
        self._serial_number: Optional[str] = None

    async def connect(self) -> None:
        self._camera = ONVIFCamera(
            self._host,
            self._port,
            self._username,
            self._password,
        )
        await self._camera.update_xaddrs()

        # Fetch device information
        device_service = self._camera.create_devicemgmt_service()
        info = await device_service.GetDeviceInformation()
        self._manufacturer = getattr(info, "Manufacturer", None)
        self._model = getattr(info, "Model", None)
        self._firmware_version = getattr(info, "FirmwareVersion", None)
        self._serial_number = getattr(info, "SerialNumber", None)

        # Fetch media profiles
        self._media_service = self._camera.create_media_service()
        try:
            profiles = await self._media_service.GetProfiles()
            self._profiles = [
                getattr(p, "Name", str(i)) for i, p in enumerate(profiles)
            ]
        except Exception as exc:
            self._log.warning("Could not fetch media profiles: %s", exc)
            self._profiles = []

        # Check PTZ support
        try:
            self._ptz_service = self._camera.create_ptz_service()
            ptz_nodes = await self._ptz_service.GetNodes()
            self._ptz_supported = bool(ptz_nodes)
        except Exception:
            self._ptz_supported = False
            self._ptz_service = None

        self._log.info(
            "ONVIF camera connected: %s %s (PTZ=%s, profiles=%d)",
            self._manufacturer,
            self._model,
            self._ptz_supported,
            len(self._profiles),
        )

        # Load YOLOv8 nano if detection is configured
        if self._rtsp_url and _DETECTION_AVAILABLE:
            try:
                self._yolo = _YOLO("yolov8n.pt")
                self._log.info("ONVIF detection enabled (YOLOv8n, RTSP: %s)", self._rtsp_url)
            except Exception as e:
                self._log.warning("YOLOv8 load failed — detection disabled: %s", e)
        elif self._rtsp_url and not _DETECTION_AVAILABLE:
            self._log.warning(
                "rtsp_url configured but ultralytics/opencv not installed. "
                "Run: pip install ultralytics opencv-python"
            )

    async def disconnect(self) -> None:
        # onvif-zeep-async doesn't require explicit close
        self._camera = None
        self._ptz_service = None
        self._media_service = None

    async def stream_observations(self) -> AsyncIterator[dict]:
        last_detect = 0.0
        import time as _time
        while not self._stop_event.is_set():
            # Always emit heartbeat
            yield self._build_observation()

            # Run detection if configured and interval elapsed
            if self._yolo and self._rtsp_url:
                now = _time.monotonic()
                if now - last_detect >= self._detect_interval:
                    last_detect = now
                    detections = await self._run_detection()
                    for det in detections:
                        yield det

            await self._interruptible_sleep(_UPDATE_INTERVAL)

    def _build_observation(self) -> dict:
        now = datetime.now(timezone.utc)
        entity_id = self.config.adapter_id
        return {
            "source_id": f"{entity_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": entity_id,
            "callsign": self.config.display_name or entity_id,
            "position": {
                "lat": self._camera_lat,
                "lon": self._camera_lon,
                "alt_m": self._camera_alt_m,
            },
            "velocity": None,
            "entity_type": "CAMERA",
            "classification": None,
            "metadata": {
                "manufacturer": self._manufacturer,
                "model": self._model,
                "firmware_version": self._firmware_version,
                "serial_number": self._serial_number,
                "profiles": self._profiles,
                "ptz_supported": self._ptz_supported,
                "host": self._host,
                "port": self._port,
            },
            "ts_iso": now.isoformat(),
        }

    async def _run_detection(self) -> List[dict]:
        """
        Pull one frame from RTSP and run YOLOv8 person detection.

        Person pixel position → approximate real-world bearing → lat/lon
        using camera mount position, height, and current pan angle.
        """
        loop = asyncio.get_event_loop()

        def _grab():
            cap = _cv2.VideoCapture(self._rtsp_url)
            cap.set(_cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None

        try:
            frame = await loop.run_in_executor(None, _grab)
        except Exception as e:
            self._log.debug("RTSP frame grab failed: %s", e)
            return []

        if frame is None:
            return []

        try:
            results = self._yolo(frame, classes=[self._DETECT_CLASS_PERSON],
                                 conf=self._detect_conf, verbose=False)
        except Exception as e:
            self._log.debug("YOLO inference failed: %s", e)
            return []

        now = datetime.now(timezone.utc)
        entity_id = self.config.adapter_id
        detections = []

        h, w = frame.shape[:2]
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                # Box center x → horizontal offset from frame center
                cx = float(box.xywh[0][0])
                offset_ratio = (cx - w / 2) / (w / 2)  # -1 … +1

                # Approximate bearing to detection from camera position
                half_fov = self._camera_fov_deg / 2.0
                bearing_offset = offset_ratio * half_fov
                absolute_bearing = (self._current_pan_deg + bearing_offset) % 360

                # Approximate ground distance using camera height and tilt
                det_lat, det_lon = self._bearing_to_latlon(
                    self._camera_lat, self._camera_lon,
                    absolute_bearing, self._camera_height_m
                )

                det = {
                    "source_id": f"{entity_id}:detect:{now.timestamp():.3f}",
                    "adapter_id": self.config.adapter_id,
                    "adapter_type": self.adapter_type,
                    "entity_id": entity_id,
                    "event_type": "entity_detected",
                    "classification": "person",
                    "confidence": round(conf, 3),
                    "lat": det_lat,
                    "lon": det_lon,
                    "position": {"lat": det_lat, "lon": det_lon, "alt_m": 0.0},
                    "ts_iso": now.isoformat(),
                    "metadata": {
                        "bearing_deg": round(absolute_bearing, 1),
                        "camera_pan_deg": round(self._current_pan_deg, 1),
                        "detection_source": "yolov8n",
                    },
                }
                detections.append(det)
                self._log.info(
                    "Person detected — conf=%.2f bearing=%.1f° (%.6f, %.6f)",
                    conf, absolute_bearing, det_lat, det_lon,
                )

        return detections

    @staticmethod
    def _bearing_to_latlon(lat: float, lon: float, bearing_deg: float, dist_m: float):
        """Project a point dist_m away in bearing_deg direction from lat/lon."""
        R = 6_371_000.0
        bearing = math.radians(bearing_deg)
        lat_r = math.radians(lat)
        lon_r = math.radians(lon)
        d = dist_m / R
        new_lat = math.asin(
            math.sin(lat_r) * math.cos(d) +
            math.cos(lat_r) * math.sin(d) * math.cos(bearing)
        )
        new_lon = lon_r + math.atan2(
            math.sin(bearing) * math.sin(d) * math.cos(lat_r),
            math.cos(d) - math.sin(lat_r) * math.sin(new_lat),
        )
        return math.degrees(new_lat), math.degrees(new_lon)

    async def send_command(self, entity_id: str, command: dict) -> None:
        """
        Execute a PTZ command on this camera.

        Supported command types
        -----------------------
        PTZ_MOVE  : {"type": "PTZ_MOVE", "pan": float, "tilt": float, "zoom": float}
        PTZ_STOP  : {"type": "PTZ_STOP"}
        PTZ_PRESET: {"type": "PTZ_PRESET", "preset_token": str}
        """
        if not self._ptz_supported or self._ptz_service is None:
            self._log.warning("PTZ command received but PTZ not supported")
            return

        cmd_type = command.get("type", "")

        # Obtain the first profile token for PTZ operations
        try:
            profiles = await self._media_service.GetProfiles()
            profile_token = profiles[0].token if profiles else None
        except Exception as exc:
            self._log.error("Could not fetch profile token: %s", exc)
            return

        if profile_token is None:
            self._log.error("No media profile available for PTZ command")
            return

        try:
            if cmd_type == "PTZ_MOVE":
                request = self._ptz_service.create_type("ContinuousMove")
                request.ProfileToken = profile_token
                request.Velocity = {
                    "PanTilt": {
                        "x": float(command.get("pan", 0.0)),
                        "y": float(command.get("tilt", 0.0)),
                    },
                    "Zoom": {"x": float(command.get("zoom", 0.0))},
                }
                await self._ptz_service.ContinuousMove(request)
                # Track pan angle for detection bearing calculations
                self._current_pan_deg = (
                    self._current_pan_deg + float(command.get("pan", 0.0)) * self._camera_fov_deg
                ) % 360
                self._log.info(
                    "PTZ_MOVE executed: pan=%.2f tilt=%.2f zoom=%.2f (abs_pan=%.1f°)",
                    command.get("pan", 0.0),
                    command.get("tilt", 0.0),
                    command.get("zoom", 0.0),
                    self._current_pan_deg,
                )

            elif cmd_type == "PTZ_STOP":
                request = self._ptz_service.create_type("Stop")
                request.ProfileToken = profile_token
                request.PanTilt = True
                request.Zoom = True
                await self._ptz_service.Stop(request)
                self._log.info("PTZ_STOP executed")

            elif cmd_type == "PTZ_PRESET":
                preset_token = command.get("preset_token")
                if not preset_token:
                    self._log.error("PTZ_PRESET requires preset_token")
                    return
                request = self._ptz_service.create_type("GotoPreset")
                request.ProfileToken = profile_token
                request.PresetToken = preset_token
                await self._ptz_service.GotoPreset(request)
                self._log.info("PTZ_PRESET executed: token=%s", preset_token)

            else:
                self._log.warning("Unknown PTZ command type: %s", cmd_type)

        except Exception as exc:
            self._log.error("PTZ command %s failed: %s", cmd_type, exc)
