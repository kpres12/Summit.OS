"""
Heli.OS — RTSP Camera Adapter
=================================

Represents an RTSP camera as a fixed entity in the world model. Does NOT
process video frames — that is the inference service's job. Periodically
probes the RTSP URL to determine whether the stream is alive, and emits
CAMERA entity observations.

Dependencies
------------
    pip install aiohttp
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import AsyncIterator, Optional
from urllib.parse import urlparse, urlunparse

try:
    import aiohttp
except ImportError:
    raise ImportError(
        "aiohttp is required for RTSPAdapter. Install with: pip install aiohttp>=3.9.0"
    )

from .base import AdapterConfig, BaseAdapter

logger = logging.getLogger("heli.adapters.rtsp")


def _redact_credentials(url: str) -> str:
    """Strip username/password from a URL for safe storage."""
    try:
        parsed = urlparse(url)
        if parsed.username or parsed.password:
            netloc = parsed.hostname or ""
            if parsed.port:
                netloc = f"{netloc}:{parsed.port}"
            redacted = parsed._replace(netloc=netloc)
            return urlunparse(redacted)
    except Exception:
        pass
    return re.sub(r"//[^@]+@", "//", url)


class RTSPAdapter(BaseAdapter):
    """
    Monitors an RTSP camera and emits periodic CAMERA entity observations.

    Config extras
    -------------
    rtsp_url              : str
    camera_lat            : float   (default 0.0)
    camera_lon            : float   (default 0.0)
    camera_alt_m          : float   (default 0.0)
    fov_degrees           : float   (default 90.0)
    pan_degrees           : float | None
    tilt_degrees          : float | None
    check_interval_seconds: float   (default 30.0)
    """

    adapter_type = "rtsp"

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        ex = config.extra

        self._rtsp_url: str = ex.get("rtsp_url", "")
        if not self._rtsp_url:
            raise ValueError("rtsp_url must be set in adapter extra config")

        self._camera_lat: float = float(ex.get("camera_lat", 0.0))
        self._camera_lon: float = float(ex.get("camera_lon", 0.0))
        self._camera_alt_m: float = float(ex.get("camera_alt_m", 0.0))
        self._fov_degrees: float = float(ex.get("fov_degrees", 90.0))
        self._pan_degrees: Optional[float] = (
            float(ex["pan_degrees"]) if ex.get("pan_degrees") is not None else None
        )
        self._tilt_degrees: Optional[float] = (
            float(ex["tilt_degrees"]) if ex.get("tilt_degrees") is not None else None
        )
        self._check_interval: float = float(ex.get("check_interval_seconds", 30.0))

        # Redacted URL for safe storage in metadata
        self._safe_url: str = _redact_credentials(self._rtsp_url)
        self._entity_id: str = config.adapter_id

        self._session: Optional[aiohttp.ClientSession] = None
        self._stream_alive: bool = False
        self._last_checked: Optional[str] = None

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession()
        self._log.info("RTSP adapter ready for %s", self._safe_url)

    async def disconnect(self) -> None:
        try:
            if self._session is not None:
                await self._session.close()
                self._session = None
        except Exception:
            pass

    async def stream_observations(self) -> AsyncIterator[dict]:
        while not self._stop_event.is_set():
            alive = await self._probe_stream()
            self._stream_alive = alive
            self._last_checked = datetime.now(timezone.utc).isoformat()
            yield self._build_observation()
            await self._interruptible_sleep(self._check_interval)

    async def _probe_stream(self) -> bool:
        """
        Send an OPTIONS request to the RTSP URL's HTTP equivalent to check
        liveness. Falls back gracefully if the server doesn't respond.

        Note: A true RTSP OPTIONS probe would require a raw TCP connection;
        this uses HTTP OPTIONS as a lightweight connectivity check. For full
        RTSP probing, replace with asyncio.open_connection + RTSP handshake.
        """
        # Convert rtsp:// to http:// for the OPTIONS probe
        probe_url = re.sub(r"^rtsp://", "http://", self._rtsp_url)
        try:
            async with self._session.options(
                probe_url,
                timeout=aiohttp.ClientTimeout(total=5.0),
                allow_redirects=False,
            ) as resp:
                # Any response (including 4xx) means the server is reachable
                return resp.status < 500
        except Exception as exc:
            self._log.debug("RTSP probe failed: %s", exc)
            return False

    def _build_observation(self) -> dict:
        now = datetime.now(timezone.utc)
        return {
            "source_id": f"{self._entity_id}:{now.timestamp():.3f}",
            "adapter_id": self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id": self._entity_id,
            "callsign": self.config.display_name or self._entity_id,
            "position": {
                "lat": self._camera_lat,
                "lon": self._camera_lon,
                "alt_m": self._camera_alt_m,
            },
            "velocity": None,
            "entity_type": "CAMERA",
            "classification": None,
            "metadata": {
                "rtsp_url": self._safe_url,
                "fov_degrees": self._fov_degrees,
                "pan_degrees": self._pan_degrees,
                "tilt_degrees": self._tilt_degrees,
                "stream_alive": self._stream_alive,
                "last_checked": self._last_checked,
            },
            "ts_iso": now.isoformat(),
        }
