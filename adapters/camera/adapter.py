"""
Summit.OS Camera Adapter — ONVIF/RTSP

Connects to IP cameras via RTSP, streams frames to the Inference service
for YOLO detection, and publishes detected objects as Summit.OS TRACK
entities using ByteTrack IDs for persistent cross-frame identity.

Simulation mode (CAMERA_ENABLED=true, no actual camera reachable):
  Generates synthetic TRACK entities with randomised positions so the
  adapter can be tested and the pipeline exercised without hardware.

Environment variables:
    CAMERA_ENABLED          - "true" to enable
    CAMERA_STREAMS          - path to JSON file listing streams
    CAMERA_STREAM_URL       - single-stream RTSP URL
    CAMERA_STREAM_ID        - ID for single stream (default: "cam-01")
    CAMERA_FPS              - frames per second to process (default: 5)
    CAMERA_INFERENCE_URL    - Inference service URL (default: http://localhost:8006)
    CAMERA_ORG_ID           - org_id tag on published entities
    CAMERA_LAT / CAMERA_LON - camera mounting location (for geo-projection)
    MQTT_HOST / MQTT_PORT   - broker connection
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from sdk import BaseAdapter, AdapterManifest, EntityBuilder, Protocol, Capability

logger = logging.getLogger("summit.adapter.camera")

# Optional imports
try:
    import httpx
    _HTTPX = True
except ImportError:
    _HTTPX = False

try:
    import cv2  # type: ignore
    _CV2 = True
except ImportError:
    _CV2 = False


def _load_stream_config(path: Optional[str]) -> List[Dict[str, Any]]:
    if path:
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load stream config from {path}: {e}")
    return [{
        "stream_id": os.getenv("CAMERA_STREAM_ID", "cam-01"),
        "url": os.getenv("CAMERA_STREAM_URL", ""),
        "lat": float(os.getenv("CAMERA_LAT", "0.0")),
        "lon": float(os.getenv("CAMERA_LON", "0.0")),
        "fov_deg": float(os.getenv("CAMERA_FOV_DEG", "90.0")),
        "tilt_deg": float(os.getenv("CAMERA_TILT_DEG", "45.0")),
        "altitude_m": float(os.getenv("CAMERA_ALTITUDE_M", "10.0")),
        "description": "Camera",
    }]


def _pixel_to_latlon(
    px: float, py: float,
    frame_w: int, frame_h: int,
    cam_lat: float, cam_lon: float,
    cam_alt_m: float,
    fov_deg: float,
    tilt_deg: float,
) -> Tuple[float, float]:
    """
    Approximate ground-plane geo-projection of a pixel coordinate.

    Assumes flat ground and nadir/forward-tilted camera.
    Returns (lat, lon) estimate — accuracy depends on camera geometry.
    """
    # Normalised image coords (-1 to 1)
    nx = (px - frame_w / 2.0) / (frame_w / 2.0)
    ny = (py - frame_h / 2.0) / (frame_h / 2.0)

    # Angular offset from boresight
    hfov = math.radians(fov_deg / 2.0)
    vfov = hfov * (frame_h / max(1, frame_w))
    tilt = math.radians(tilt_deg)

    # Ground range approximation
    az_offset = nx * hfov
    el_offset = tilt + ny * vfov
    if abs(math.cos(el_offset)) < 1e-6:
        ground_range = cam_alt_m
    else:
        ground_range = cam_alt_m * math.tan(el_offset)

    # Convert to lat/lon offset (simple flat-earth)
    lat_deg_per_m = 1.0 / 111_320.0
    lon_deg_per_m = 1.0 / (111_320.0 * math.cos(math.radians(cam_lat)) + 1e-10)

    bearing_offset = az_offset
    delta_north = ground_range * math.cos(bearing_offset)
    delta_east = ground_range * math.sin(bearing_offset)

    est_lat = cam_lat + delta_north * lat_deg_per_m
    est_lon = cam_lon + delta_east * lon_deg_per_m
    return est_lat, est_lon


class CameraAdapter(BaseAdapter):
    """RTSP camera adapter with YOLO inference and ByteTrack IDs."""

    MANIFEST = AdapterManifest(
        name="camera",
        version="1.0.0",
        protocol=Protocol.RTSP,
        capabilities=[Capability.READ, Capability.STREAM],
        entity_types=["TRACK"],
        description="RTSP/ONVIF camera with YOLO detection and ByteTrack multi-object tracking",
        optional_env=["CAMERA_STREAM_URL", "CAMERA_STREAMS", "CAMERA_INFERENCE_URL"],
    )

    def __init__(self, **kwargs):
        super().__init__(device_id="camera", **kwargs)
        self.fps = min(float(os.getenv("CAMERA_FPS", "5")), 30.0)
        self.inference_url = os.getenv("CAMERA_INFERENCE_URL", "http://localhost:8006")
        self.org_id = os.getenv("CAMERA_ORG_ID", "")
        self.streams = _load_stream_config(os.getenv("CAMERA_STREAMS"))
        self._frame_interval = 1.0 / max(self.fps, 0.5)
        self._simulation = True  # Flipped to False if camera opens OK
        self._bytetrackers: Dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return os.getenv("CAMERA_ENABLED", "false").lower() == "true"

    # ── ByteTracker per stream ────────────────────────────────────────────────

    def _get_tracker(self, stream_id: str):
        if stream_id not in self._bytetrackers:
            # Import inline — ByteTrack lives in apps/fusion, not adapters/
            # When running in the full stack the PYTHONPATH will include apps/
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "apps", "fusion"))
                from bytetrack import ByteTracker
                self._bytetrackers[stream_id] = ByteTracker()
                logger.info(f"ByteTracker initialised for stream {stream_id}")
            except ImportError:
                logger.warning("ByteTracker not found — using simple dict tracker")
                self._bytetrackers[stream_id] = None
        return self._bytetrackers[stream_id]

    # ── Inference call ────────────────────────────────────────────────────────

    async def _detect(self, jpeg_bytes: bytes, stream_id: str) -> List[Dict]:
        """Send JPEG frame to Inference service, return detections list."""
        if not _HTTPX:
            return []
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.post(
                    f"{self.inference_url}/detect",
                    content=jpeg_bytes,
                    headers={"Content-Type": "image/jpeg",
                             "X-Source-Id": stream_id},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("detections", [])
        except Exception as e:
            logger.debug(f"Inference call failed for {stream_id}: {e}")
        return []

    # ── Entity construction ───────────────────────────────────────────────────

    def _detection_to_entity(
        self,
        det: Dict,
        stream_cfg: Dict,
        frame_w: int,
        frame_h: int,
        track_id: int,
    ) -> Optional[Dict]:
        bbox = det.get("bbox", [0, 0, 0, 0])
        if not bbox or len(bbox) < 4:
            return None

        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        label = det.get("class_label", det.get("label", "object"))
        score = det.get("score", det.get("confidence", 1.0))

        lat, lon = _pixel_to_latlon(
            cx, cy, frame_w, frame_h,
            stream_cfg.get("lat", 0.0),
            stream_cfg.get("lon", 0.0),
            stream_cfg.get("altitude_m", 10.0),
            stream_cfg.get("fov_deg", 90.0),
            stream_cfg.get("tilt_deg", 45.0),
        )

        stream_id = stream_cfg.get("stream_id", "cam")
        entity_id = f"cam-track-{stream_id}-{track_id}"

        return (
            EntityBuilder(entity_id, "TRACK", "GROUND")
            .position(lat, lon)
            .metadata({
                "source_camera": stream_id,
                "track_id": str(track_id),
                "class_label": label,
                "confidence": str(round(score, 3)),
                "bbox": json.dumps([round(v, 1) for v in bbox]),
            })
            .provenance(source_type="camera", source_id=stream_id)
            .org(self.org_id)
            .build()
        )

    # ── Single-stream processing loop ────────────────────────────────────────

    async def _run_stream(self, stream_cfg: Dict) -> None:
        stream_id = stream_cfg.get("stream_id", "cam-01")
        url = stream_cfg.get("url", "")
        tracker = self._get_tracker(stream_id)

        logger.info(f"[{stream_id}] starting — url={'<sim>' if not url else url}")

        cap = None
        if url and _CV2:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                self._simulation = False
                logger.info(f"[{stream_id}] RTSP stream opened")
            else:
                logger.warning(f"[{stream_id}] RTSP open failed — simulation mode")
                cap = None

        while True:
            t_start = time.monotonic()

            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"[{stream_id}] Frame read failed — retrying in 5s")
                    await asyncio.sleep(5)
                    cap = cv2.VideoCapture(url)
                    continue

                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                jpeg_bytes = buf.tobytes()
                fh, fw = frame.shape[:2]

                raw_dets = await self._detect(jpeg_bytes, stream_id)
            else:
                # Simulation: generate synthetic detections
                raw_dets = self._simulate_detections(stream_cfg)
                fw, fh = 1920, 1080

            # Run through ByteTracker
            if tracker is not None:
                tracked = tracker.update(raw_dets)
            else:
                # Fallback: assign sequential fake track IDs
                tracked = [{**d, "track_id": i + 1} for i, d in enumerate(raw_dets)]

            # Publish each tracked object as a TRACK entity
            for tdet in tracked:
                tid = tdet.get("track_id", 0)
                entity = self._detection_to_entity(tdet, stream_cfg, fw, fh, tid)
                if entity:
                    self.publish(entity)

            elapsed = time.monotonic() - t_start
            await asyncio.sleep(max(0.0, self._frame_interval - elapsed))

    def _simulate_detections(self, stream_cfg: Dict) -> List[Dict]:
        """Return 0–3 randomised synthetic detections for testing."""
        labels = ["person", "vehicle", "animal", "drone"]
        n = random.randint(0, 3)
        dets = []
        for i in range(n):
            dets.append({
                "bbox": [
                    random.uniform(0, 1600), random.uniform(0, 900),
                    random.uniform(40, 200), random.uniform(40, 200),
                ],
                "score": random.uniform(0.65, 0.99),
                "class_id": i % len(labels),
                "class_label": labels[i % len(labels)],
            })
        return dets

    # ── Adapter lifecycle ─────────────────────────────────────────────────────

    async def run(self) -> None:
        """Start all configured camera streams concurrently."""
        tasks = [
            asyncio.create_task(self._run_stream(cfg))
            for cfg in self.streams
        ]
        await asyncio.gather(*tasks)
