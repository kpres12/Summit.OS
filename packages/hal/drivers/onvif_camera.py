"""
ONVIF IP Camera Driver for Heli.OS

Interfaces with ONVIF-compatible IP cameras for:
- RTSP stream discovery and management
- PTZ (Pan-Tilt-Zoom) control
- Snapshot capture
- Camera health monitoring

Falls back to simulation when onvif-zeep is not available.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger("hal.onvif")


@dataclass
class CameraProfile:
    """Camera media profile."""

    name: str
    token: str
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 30
    encoding: str = "H264"
    rtsp_uri: str = ""


@dataclass
class PTZPosition:
    """PTZ position."""

    pan: float = 0.0  # -1.0 to 1.0
    tilt: float = 0.0  # -1.0 to 1.0
    zoom: float = 0.0  # 0.0 to 1.0


@dataclass
class CameraStatus:
    """Camera status."""

    online: bool = False
    recording: bool = False
    ptz_position: PTZPosition = field(default_factory=PTZPosition)
    active_profile: str = ""
    uptime_seconds: float = 0.0
    temperature_c: float = 0.0
    last_seen: float = field(default_factory=time.time)


class ONVIFCameraDriver:
    """
    ONVIF IP camera driver.

    Supports discovery, streaming, and PTZ control.
    """

    def __init__(
        self, host: str, port: int = 80, username: str = "admin", password: str = ""
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password

        self._camera = None
        self._connected = False
        self._profiles: List[CameraProfile] = []
        self._status = CameraStatus()
        self._onvif_available = self._check_onvif()

    @staticmethod
    def _check_onvif() -> bool:
        try:
            from onvif import ONVIFCamera

            return True
        except ImportError:
            return False

    async def connect(self) -> bool:
        """Connect to the ONVIF camera."""
        if self._onvif_available:
            try:
                from onvif import ONVIFCamera

                self._camera = ONVIFCamera(
                    self.host, self.port, self.username, self.password
                )
                await self._camera.update_xaddrs()
                self._connected = True
                await self._discover_profiles()
                logger.info(f"Connected to ONVIF camera at {self.host}:{self.port}")
                return True
            except Exception as e:
                logger.error(f"ONVIF connection failed: {e}")
                return False
        else:
            logger.warning("onvif-zeep not installed — using simulated camera driver")
            self._connected = True
            self._profiles = [
                CameraProfile(
                    name="MainStream",
                    token="main",
                    resolution=(1920, 1080),
                    fps=30,
                    rtsp_uri=f"rtsp://{self.host}:554/stream1",
                ),
                CameraProfile(
                    name="SubStream",
                    token="sub",
                    resolution=(640, 480),
                    fps=15,
                    rtsp_uri=f"rtsp://{self.host}:554/stream2",
                ),
            ]
            self._status.online = True
            return True

    async def disconnect(self) -> None:
        self._connected = False
        self._camera = None

    async def _discover_profiles(self):
        """Discover media profiles from camera."""
        if not self._camera:
            return
        try:
            media_service = self._camera.create_media_service()
            profiles = await media_service.GetProfiles()
            for p in profiles:
                profile = CameraProfile(
                    name=p.Name,
                    token=p.token,
                )
                if hasattr(p, "VideoEncoderConfiguration"):
                    vec = p.VideoEncoderConfiguration
                    profile.resolution = (vec.Resolution.Width, vec.Resolution.Height)
                    profile.fps = vec.RateControl.FrameRateLimit
                    profile.encoding = vec.Encoding

                # Get stream URI
                stream_setup = {
                    "Stream": "RTP-Unicast",
                    "Transport": {"Protocol": "RTSP"},
                }
                uri_response = await media_service.GetStreamUri(stream_setup, p.token)
                profile.rtsp_uri = uri_response.Uri

                self._profiles.append(profile)
        except Exception as e:
            logger.error(f"Profile discovery failed: {e}")

    async def ptz_move(
        self, pan: float = 0.0, tilt: float = 0.0, zoom: float = 0.0, speed: float = 0.5
    ) -> bool:
        """Move PTZ camera (continuous)."""
        if self._camera and self._onvif_available:
            try:
                ptz_service = self._camera.create_ptz_service()
                request = ptz_service.create_type("ContinuousMove")
                request.ProfileToken = (
                    self._profiles[0].token if self._profiles else "main"
                )
                request.Velocity = {
                    "PanTilt": {"x": pan * speed, "y": tilt * speed},
                    "Zoom": {"x": zoom * speed},
                }
                await ptz_service.ContinuousMove(request)
                return True
            except Exception as e:
                logger.error(f"PTZ move failed: {e}")
                return False

        self._status.ptz_position.pan = max(
            -1, min(1, self._status.ptz_position.pan + pan * 0.1)
        )
        self._status.ptz_position.tilt = max(
            -1, min(1, self._status.ptz_position.tilt + tilt * 0.1)
        )
        self._status.ptz_position.zoom = max(
            0, min(1, self._status.ptz_position.zoom + zoom * 0.1)
        )
        return True

    async def ptz_stop(self) -> bool:
        if self._camera and self._onvif_available:
            try:
                ptz_service = self._camera.create_ptz_service()
                token = self._profiles[0].token if self._profiles else "main"
                await ptz_service.Stop({"ProfileToken": token})
                return True
            except Exception:
                return False
        return True

    async def ptz_goto_preset(self, preset_token: str) -> bool:
        if self._camera and self._onvif_available:
            try:
                ptz_service = self._camera.create_ptz_service()
                token = self._profiles[0].token if self._profiles else "main"
                await ptz_service.GotoPreset(
                    {
                        "ProfileToken": token,
                        "PresetToken": preset_token,
                    }
                )
                return True
            except Exception:
                return False
        return True

    def get_stream_uri(self, profile_idx: int = 0) -> str:
        if profile_idx < len(self._profiles):
            return self._profiles[profile_idx].rtsp_uri
        return ""

    @property
    def profiles(self) -> List[CameraProfile]:
        return self._profiles

    @property
    def status(self) -> CameraStatus:
        return self._status

    @property
    def is_connected(self) -> bool:
        return self._connected
