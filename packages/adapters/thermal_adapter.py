"""
Thermal / FLIR Camera Adapter

Connects to thermal imaging cameras (FLIR, Seek, Teledyne, HIKMICRO) via:
  - RTSP stream with thermal metadata (most FLIR pan-tilt units)
  - FLIR Nexus SDK HTTP API (Atlas, Vue Pro R via Nexus middleware)
  - Generic HTTP MJPEG + temperature overlay JSON endpoint

Emits THERMAL_DETECTION observations with:
  - Bounding box + peak temperature for each detected heat signature
  - Classification: PERSON / VEHICLE / FIRE / HOTSPOT / UNKNOWN
  - Palette info (Ironbow, Rainbow, White-Hot) for downstream annotation

Usage:
    config = AdapterConfig(
        adapter_id="flir-roof-1",
        adapter_type="thermal",
        display_name="Roof FLIR",
        extra={
            "host": "192.168.1.50",
            "port": 8080,
            "stream_type": "nexus",   # nexus | rtsp | mjpeg_json
            "rtsp_path": "/stream1",  # for rtsp mode
            "temp_threshold_c": 35.0, # min peak temp to emit detection
            "fov_deg": 45.0,
        },
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from .base import BaseAdapter, AdapterConfig

logger = logging.getLogger("heli.adapters.thermal")

_CLASSIFICATION_TEMP = {
    "FIRE":    80.0,
    "VEHICLE": 45.0,
    "PERSON":  32.0,
}


class ThermalAdapter(BaseAdapter):
    """
    Thermal camera adapter for FLIR and compatible sensors.

    Supports three integration modes:
      - nexus: FLIR Nexus HTTP API (JSON detections endpoint)
      - rtsp:  RTSP stream; temperature data parsed from SEI NAL units or
               companion JSON sidecar at /api/detections
      - mjpeg_json: Generic HTTP endpoint returning JSON frames with
                    bounding boxes and temperature overlays
    """

    adapter_type = "thermal"

    @classmethod
    def required_extra_fields(cls) -> list[str]:
        return ["host"]

    def __init__(self, config: AdapterConfig, mqtt_client=None) -> None:
        super().__init__(config, mqtt_client)
        extra = config.extra
        self._host = extra.get("host", "localhost")
        self._port = int(extra.get("port", 8080))
        self._stream_type = extra.get("stream_type", "nexus")
        self._rtsp_path = extra.get("rtsp_path", "/stream1")
        self._temp_threshold = float(extra.get("temp_threshold_c", 32.0))
        self._fov_deg = float(extra.get("fov_deg", 45.0))
        self._lat = float(extra.get("lat", 0.0))
        self._lon = float(extra.get("lon", 0.0))
        self._alt_m = float(extra.get("alt_m", 0.0))
        self._session: Optional[object] = None

    async def connect(self) -> None:
        try:
            import aiohttp
            connector = aiohttp.TCPConnector(ssl=False)
            self._session = aiohttp.ClientSession(connector=connector)
        except ImportError:
            self._log.warning("aiohttp not installed — thermal adapter in stub mode")
            self._session = None

    async def disconnect(self) -> None:
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None

    async def stream_observations(self) -> AsyncIterator[dict]:
        if self._stream_type == "nexus":
            async for obs in self._stream_nexus():
                yield obs
        elif self._stream_type == "mjpeg_json":
            async for obs in self._stream_mjpeg_json():
                yield obs
        else:
            # RTSP: poll companion JSON sidecar
            async for obs in self._stream_rtsp_sidecar():
                yield obs

    # -------------------------------------------------------------------------
    # FLIR Nexus HTTP API
    # -------------------------------------------------------------------------

    async def _stream_nexus(self) -> AsyncIterator[dict]:
        url = f"http://{self._host}:{self._port}/api/v1/detections"
        while not self._stop_event.is_set():
            try:
                detections = await self._get_json(url)
                if isinstance(detections, list):
                    for det in detections:
                        obs = self._nexus_detection_to_obs(det)
                        if obs:
                            yield obs
            except Exception as exc:
                self._log.debug("Nexus poll error: %s", exc)
            await asyncio.sleep(self.config.poll_interval_seconds)

    def _nexus_detection_to_obs(self, det: dict) -> Optional[dict]:
        peak_c = float(det.get("peak_temp_c", det.get("peakTempC", 0)))
        if peak_c < self._temp_threshold:
            return None
        classification = self._classify_temp(peak_c, det.get("label", ""))
        bbox = det.get("bbox", det.get("boundingBox", {}))
        return self._build_obs(
            entity_id=f"thermal-{self._host}-{det.get('track_id', int(time.time()))}",
            classification=classification,
            peak_temp_c=peak_c,
            avg_temp_c=float(det.get("avg_temp_c", det.get("avgTempC", peak_c * 0.9))),
            bbox=bbox,
            confidence=float(det.get("confidence", 0.8)),
            palette=det.get("palette", "ironbow"),
        )

    # -------------------------------------------------------------------------
    # Generic MJPEG + JSON detections endpoint
    # -------------------------------------------------------------------------

    async def _stream_mjpeg_json(self) -> AsyncIterator[dict]:
        url = f"http://{self._host}:{self._port}/api/detections"
        while not self._stop_event.is_set():
            try:
                data = await self._get_json(url)
                frames = data if isinstance(data, list) else data.get("detections", [])
                for det in frames:
                    peak_c = float(det.get("peak_temp_c", 0))
                    if peak_c < self._temp_threshold:
                        continue
                    classification = self._classify_temp(peak_c, det.get("class", ""))
                    obs = self._build_obs(
                        entity_id=f"thermal-{self._host}-{det.get('id', int(time.time()))}",
                        classification=classification,
                        peak_temp_c=peak_c,
                        avg_temp_c=float(det.get("avg_temp_c", peak_c * 0.9)),
                        bbox=det.get("bbox", {}),
                        confidence=float(det.get("confidence", 0.75)),
                        palette=det.get("palette", "white_hot"),
                    )
                    yield obs
            except Exception as exc:
                self._log.debug("MJPEG-JSON poll error: %s", exc)
            await asyncio.sleep(self.config.poll_interval_seconds)

    # -------------------------------------------------------------------------
    # RTSP companion JSON sidecar
    # -------------------------------------------------------------------------

    async def _stream_rtsp_sidecar(self) -> AsyncIterator[dict]:
        url = f"http://{self._host}:{self._port}/api/detections"
        while not self._stop_event.is_set():
            try:
                data = await self._get_json(url)
                dets = data if isinstance(data, list) else data.get("detections", [])
                for det in dets:
                    peak_c = float(det.get("peak_temp_c", 0))
                    if peak_c < self._temp_threshold:
                        continue
                    obs = self._build_obs(
                        entity_id=f"thermal-rtsp-{int(time.time())}",
                        classification=self._classify_temp(peak_c, ""),
                        peak_temp_c=peak_c,
                        avg_temp_c=float(det.get("avg_temp_c", peak_c * 0.9)),
                        bbox=det.get("bbox", {}),
                        confidence=float(det.get("confidence", 0.7)),
                        palette="ironbow",
                    )
                    yield obs
            except Exception as exc:
                self._log.debug("RTSP sidecar error: %s", exc)
            await asyncio.sleep(self.config.poll_interval_seconds)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    async def _get_json(self, url: str) -> object:
        if self._session is None:
            await asyncio.sleep(5)
            return []
        async with self._session.get(url, timeout=5) as resp:
            return await resp.json(content_type=None)

    def _classify_temp(self, peak_c: float, label: str) -> str:
        label_upper = label.upper()
        if "PERSON" in label_upper or "HUMAN" in label_upper:
            return "PERSON"
        if "VEHICLE" in label_upper or "CAR" in label_upper or "TRUCK" in label_upper:
            return "VEHICLE"
        if "FIRE" in label_upper or "FLAME" in label_upper:
            return "FIRE"
        if peak_c >= _CLASSIFICATION_TEMP["FIRE"]:
            return "FIRE"
        if peak_c >= _CLASSIFICATION_TEMP["VEHICLE"]:
            return "VEHICLE"
        if peak_c >= _CLASSIFICATION_TEMP["PERSON"]:
            return "PERSON"
        return "HOTSPOT"

    def _build_obs(
        self,
        entity_id: str,
        classification: str,
        peak_temp_c: float,
        avg_temp_c: float,
        bbox: dict,
        confidence: float,
        palette: str,
    ) -> dict:
        entity_type = "FIRE" if classification == "FIRE" else "GROUND"
        return {
            "source_id":    f"{entity_id}-{int(time.time())}",
            "adapter_id":   self.config.adapter_id,
            "adapter_type": self.adapter_type,
            "entity_id":    entity_id,
            "callsign":     f"THERMAL-{classification}",
            "position": {
                "lat":   self._lat,
                "lon":   self._lon,
                "alt_m": self._alt_m,
            } if self._lat or self._lon else None,
            "velocity":     None,
            "entity_type":  entity_type,
            "classification": classification,
            "metadata": {
                "peak_temp_c": peak_temp_c,
                "avg_temp_c":  avg_temp_c,
                "temp_threshold_c": self._temp_threshold,
                "confidence":  confidence,
                "palette":     palette,
                "bbox":        bbox,
                "fov_deg":     self._fov_deg,
                "stream_type": self._stream_type,
                "camera_host": self._host,
            },
            "ts_iso": datetime.now(timezone.utc).isoformat(),
        }
