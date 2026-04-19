"""
Heli.OS — ONVIF IP Camera Adapter
=====================================

Connects to ONVIF-compliant IP cameras. Fetches device info, media profiles,
and PTZ capabilities. Also supports PTZ control via send_command().

Dependencies
------------
    pip install onvif-zeep-async
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

try:
    from onvif import ONVIFCamera
except ImportError:
    raise ImportError(
        "onvif-zeep-async is required for ONVIFAdapter. "
        "Install with: pip install onvif-zeep-async"
    )

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

        self._camera: Optional[ONVIFCamera] = None
        self._ptz_service = None
        self._media_service = None
        self._profiles: list[str] = []
        self._ptz_supported: bool = False

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

    async def disconnect(self) -> None:
        # onvif-zeep-async doesn't require explicit close
        self._camera = None
        self._ptz_service = None
        self._media_service = None

    async def stream_observations(self) -> AsyncIterator[dict]:
        while not self._stop_event.is_set():
            yield self._build_observation()
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
                self._log.info(
                    "PTZ_MOVE executed: pan=%.2f tilt=%.2f zoom=%.2f",
                    command.get("pan", 0.0),
                    command.get("tilt", 0.0),
                    command.get("zoom", 0.0),
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
