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
                              Set to "" to skip remote inference entirely (edge mode)
    CAMERA_LOCAL_MODEL      - path to local ONNX/PT model file for on-device inference
                              Falls back automatically when remote inference is unreachable
    CAMERA_LOCAL_BACKEND    - "onnx", "yolo", or "auto" (default: "auto")
    CAMERA_EDGE_MODE        - "true" to disable remote inference, run fully offline
    CAMERA_BUFFER_PATH      - path for offline SQLite buffer (default: /tmp/summit_camera_buffer.db)
    CAMERA_BUFFER_MAX       - max buffered entities before oldest are dropped (default: 5000)
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
import sqlite3
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


# ── Offline entity buffer ──────────────────────────────────────────────────────

class OfflineBuffer:
    """
    SQLite-backed buffer for entities that couldn't be published (MQTT offline).

    Persists across process restarts. Entities are flushed in FIFO order
    when connectivity returns. Capped at max_size to protect disk space on
    constrained edge hardware.
    """

    def __init__(self, db_path: str = "/tmp/summit_camera_buffer.db", max_size: int = 5_000):
        self._db_path = db_path
        self._max_size = max_size
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS buffer "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, payload TEXT)"
        )
        self._conn.commit()
        existing = self._conn.execute("SELECT COUNT(*) FROM buffer").fetchone()[0]
        if existing:
            logger.info(f"OfflineBuffer: {existing} entities pending flush from previous run")

    def store(self, entity: Dict[str, Any]) -> None:
        """Buffer one entity. Drops the oldest if over capacity."""
        cur = self._conn.execute("SELECT COUNT(*) FROM buffer").fetchone()[0]
        if cur >= self._max_size:
            self._conn.execute(
                "DELETE FROM buffer WHERE id = (SELECT MIN(id) FROM buffer)"
            )
        self._conn.execute(
            "INSERT INTO buffer (ts, payload) VALUES (?, ?)",
            (time.time(), json.dumps(entity)),
        )
        self._conn.commit()

    def flush(self, publish_fn) -> int:
        """
        Attempt to publish all buffered entities via publish_fn.
        Removes successfully published entries. Returns count flushed.
        """
        rows = self._conn.execute(
            "SELECT id, payload FROM buffer ORDER BY id LIMIT 200"
        ).fetchall()
        flushed = 0
        for row_id, payload in rows:
            try:
                publish_fn(json.loads(payload), qos=1)
                self._conn.execute("DELETE FROM buffer WHERE id = ?", (row_id,))
                flushed += 1
            except Exception:
                break  # stop on first failure — still offline
        if flushed:
            self._conn.commit()
        return flushed

    @property
    def size(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM buffer").fetchone()[0]


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


class CameraModel:
    """
    Pinhole camera model with ray-casting geo-projection.

    Projects a pixel (u, v) to a real-world (lat, lon) by:
      1. Undistorting the pixel using lens distortion coefficients (if provided)
      2. Building a ray in camera space using intrinsic matrix K
      3. Rotating the ray into ENU (East-North-Up) coordinates using
         the camera's azimuth, tilt, and roll
      4. Intersecting the ray with the ground plane (or using LiDAR depth)
      5. Converting the ENU ground offset to lat/lon

    Stream config keys consumed:
      lat, lon, altitude_m       — camera mounting position
      azimuth_deg                — pan direction (0=North, clockwise)
      tilt_deg                   — tilt below horizontal (45 = 45° down)
      roll_deg                   — roll (0 = level)
      fx, fy                     — focal length in pixels (optional)
      cx, cy                     — principal point (default: image centre)
      fov_deg                    — horizontal FOV; used to derive fx when
                                   fx/fy are not supplied
      distortion                 — list [k1, k2, p1, p2[, k3]] (optional)
    """

    def __init__(self, stream_cfg: Dict[str, Any], frame_w: int = 1920, frame_h: int = 1080):
        self.lat = stream_cfg.get("lat", 0.0)
        self.lon = stream_cfg.get("lon", 0.0)
        self.alt = stream_cfg.get("altitude_m", stream_cfg.get("altitude", 10.0))
        self.frame_w = frame_w
        self.frame_h = frame_h

        az = math.radians(stream_cfg.get("azimuth_deg", 0.0))
        tilt = math.radians(stream_cfg.get("tilt_deg", 45.0))
        roll = math.radians(stream_cfg.get("roll_deg", 0.0))

        # Intrinsics: use explicit fx/fy if given, else derive from FOV
        cx = stream_cfg.get("cx", frame_w / 2.0)
        cy = stream_cfg.get("cy", frame_h / 2.0)
        if "fx" in stream_cfg:
            fx = float(stream_cfg["fx"])
            fy = float(stream_cfg.get("fy", fx))
        else:
            hfov = math.radians(stream_cfg.get("fov_deg", 90.0))
            fx = (frame_w / 2.0) / math.tan(hfov / 2.0)
            fy = fx
        self._fx, self._fy, self._cx, self._cy = fx, fy, cx, cy
        self._dist = stream_cfg.get("distortion")  # [k1, k2, p1, p2[, k3]] or None

        # Rotation matrix: camera axes expressed in ENU
        #   ENU: x=East, y=North, z=Up
        #   Camera: x=right, y=down, z=forward (into scene)
        #
        #   boresight (camera +z) in ENU:
        #     pointing azimuth deg clockwise from North, tilt deg below horizontal
        fwd = _enu(math.sin(az) * math.cos(tilt),
                   math.cos(az) * math.cos(tilt),
                   -math.sin(tilt))

        # camera +x (right) in ENU: 90° clockwise from azimuth in horizontal plane
        right = _enu(math.cos(az), -math.sin(az), 0.0)

        # Apply roll around boresight axis
        if abs(roll) > 1e-6:
            right = _rot_axis(right, fwd, roll)

        # camera +y (down) = fwd × right
        down = _cross(fwd, right)

        # R columns: cam_x→right, cam_y→down, cam_z→fwd (all in ENU)
        self._R = (right, down, fwd)

    def project(self, u: float, v: float, depth_m: Optional[float] = None) -> Tuple[float, float]:
        """
        Project pixel (u, v) to (lat, lon).

        depth_m — if provided (e.g. from co-located LiDAR), use as the
                  slant range along the ray instead of ground-plane intersection.
        """
        # 1. Undistort using OpenCV if distortion coefficients are available
        nu, nv = self._undistort(u, v)

        # 2. Build normalised direction in camera space (z=1 forward)
        dx = (nu - self._cx) / self._fx
        dy = (nv - self._cy) / self._fy
        dz = 1.0
        ray_len = math.sqrt(dx * dx + dy * dy + dz * dz)

        # 3. Rotate ray into ENU
        rx, ry, rz = self._R
        east  = rx[0] * dx + ry[0] * dy + rz[0] * dz
        north = rx[1] * dx + ry[1] * dy + rz[1] * dz
        up    = rx[2] * dx + ry[2] * dy + rz[2] * dz

        # 4. Find ground intersection
        if depth_m is not None and depth_m > 0.0:
            # LiDAR depth: scale ray by depth / |ray|
            t = depth_m / ray_len
        elif up < -1e-6:
            # Ray hits ground: camera at ENU z=alt_m, ground at z=0
            t = self.alt / (-up)
        else:
            # Ray pointing up or horizontal — clamp to a 200 m max range
            t = 200.0

        delta_east  = t * east
        delta_north = t * north

        # 5. ENU offset → lat/lon
        lat_deg_per_m = 1.0 / 111_320.0
        lon_deg_per_m = 1.0 / (111_320.0 * math.cos(math.radians(self.lat)) + 1e-10)
        return (
            self.lat + delta_north * lat_deg_per_m,
            self.lon + delta_east  * lon_deg_per_m,
        )

    def update_frame_size(self, w: int, h: int):
        """Call when the actual frame dimensions are known (adjusts default cx/cy)."""
        if w != self.frame_w or h != self.frame_h:
            self.frame_w, self.frame_h = w, h
            # Re-centre principal point only if it was at the default centre
            if abs(self._cx - self.frame_w / 2.0) < 2 and abs(self._cy - self.frame_h / 2.0) < 2:
                self._cx = w / 2.0
                self._cy = h / 2.0

    def _undistort(self, u: float, v: float) -> Tuple[float, float]:
        if not self._dist or not _CV2:
            return u, v
        try:
            import numpy as np
            pts = np.array([[[u, v]]], dtype=np.float32)
            K = np.array([[self._fx, 0, self._cx],
                          [0, self._fy, self._cy],
                          [0, 0, 1]], dtype=np.float64)
            d = np.array(self._dist, dtype=np.float64)
            out = cv2.undistortPoints(pts, K, d, P=K)
            return float(out[0, 0, 0]), float(out[0, 0, 1])
        except Exception:
            return u, v


# ── Small vector helpers used by CameraModel ─────────────────────────────────

def _enu(e: float, n: float, u: float) -> Tuple[float, float, float]:
    return (e, n, u)

def _cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )

def _rot_axis(v: Tuple[float, float, float],
              axis: Tuple[float, float, float],
              angle: float) -> Tuple[float, float, float]:
    """Rotate vector v around unit axis by angle (radians) — Rodrigues' formula."""
    c, s = math.cos(angle), math.sin(angle)
    dot = v[0]*axis[0] + v[1]*axis[1] + v[2]*axis[2]
    cx = _cross(axis, v)
    return (
        c * v[0] + s * cx[0] + (1 - c) * dot * axis[0],
        c * v[1] + s * cx[1] + (1 - c) * dot * axis[1],
        c * v[2] + s * cx[2] + (1 - c) * dot * axis[2],
    )


# Keep the old function as a thin wrapper so nothing else breaks
def _pixel_to_latlon(
    px: float, py: float,
    frame_w: int, frame_h: int,
    cam_lat: float, cam_lon: float,
    cam_alt_m: float,
    fov_deg: float,
    tilt_deg: float,
) -> Tuple[float, float]:
    model = CameraModel({
        "lat": cam_lat, "lon": cam_lon, "altitude_m": cam_alt_m,
        "fov_deg": fov_deg, "tilt_deg": tilt_deg,
    }, frame_w=frame_w, frame_h=frame_h)
    return model.project(px, py)


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

        # Edge mode: run inference locally when remote is unavailable or disabled
        self._edge_mode = os.getenv("CAMERA_EDGE_MODE", "false").lower() == "true"
        self._local_detector = None
        local_model = os.getenv("CAMERA_LOCAL_MODEL", "")
        local_backend = os.getenv("CAMERA_LOCAL_BACKEND", "auto")
        try:
            packages_path = os.path.join(os.path.dirname(__file__), "..", "..", "packages")
            if packages_path not in sys.path:
                sys.path.insert(0, packages_path)
            from ai.detection import create_detector
            self._local_detector = create_detector(
                backend=local_backend,
                **({"model_path": local_model} if local_model else {}),
            )
            logger.info(
                f"Local detector ready: {self._local_detector.__class__.__name__} "
                f"({'edge-only' if self._edge_mode else 'fallback'})"
            )
        except Exception as e:
            logger.warning(f"Local detector unavailable: {e}")

        # Offline entity buffer — persists detections when MQTT is down
        self._offline_buffer = OfflineBuffer(
            db_path=os.getenv("CAMERA_BUFFER_PATH", "/tmp/summit_camera_buffer.db"),
            max_size=int(os.getenv("CAMERA_BUFFER_MAX", "5000")),
        )
        self._mqtt_online = True  # tracks connectivity state

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

    # ── Inference ────────────────────────────────────────────────────────────

    async def _detect(self, jpeg_bytes: bytes, stream_id: str,
                      frame: Optional[Any] = None) -> List[Dict]:
        """
        Run object detection on a frame.

        Priority:
          1. Remote inference service (when CAMERA_EDGE_MODE != true and reachable)
          2. Local detector (ONNX/YOLO running on this device)
          3. Empty list (no inference available)

        frame — raw numpy array; passed to local detector for best performance.
                Falls back to jpeg_bytes when frame is None.
        """
        # Remote inference (skipped in edge mode)
        if not self._edge_mode and self.inference_url and _HTTPX:
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
                logger.debug(
                    f"[{stream_id}] Remote inference unavailable ({e}), "
                    f"{'using local detector' if self._local_detector else 'no fallback'}"
                )

        # Local inference fallback
        if self._local_detector is not None:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._run_local_detect, frame if frame is not None else jpeg_bytes, stream_id
            )

        return []

    def _run_local_detect(self, image: Any, stream_id: str) -> List[Dict]:
        """Run detection synchronously in a thread pool (keeps async loop free)."""
        try:
            result = self._local_detector.detect(image, confidence_threshold=0.45)
            dets = []
            for d in result.detections:
                dets.append({
                    "bbox": [d.bbox.x1, d.bbox.y1, d.bbox.width, d.bbox.height],
                    "score": d.confidence,
                    "class_id": d.class_id,
                    "class_label": d.class_name,
                })
            return dets
        except Exception as e:
            logger.error(f"[{stream_id}] Local inference error: {e}")
            return []

    # ── Entity construction ───────────────────────────────────────────────────

    def _detection_to_entity(
        self,
        det: Dict,
        stream_cfg: Dict,
        frame_w: int,
        frame_h: int,
        track_id: int,
        cam_model: Optional["CameraModel"] = None,
    ) -> Optional[Dict]:
        bbox = det.get("bbox", [0, 0, 0, 0])
        if not bbox or len(bbox) < 4:
            return None

        px = bbox[0] + bbox[2] / 2.0
        py = bbox[1] + bbox[3] / 2.0
        label = det.get("class_label", det.get("label", "object"))
        score = det.get("score", det.get("confidence", 1.0))
        # LiDAR depth may be injected by a co-located depth sensor adapter
        depth_m = det.get("depth_m")

        if cam_model is not None:
            cam_model.update_frame_size(frame_w, frame_h)
            lat, lon = cam_model.project(px, py, depth_m=depth_m)
        else:
            lat, lon = _pixel_to_latlon(
                px, py, frame_w, frame_h,
                stream_cfg.get("lat", 0.0),
                stream_cfg.get("lon", 0.0),
                stream_cfg.get("altitude_m", 10.0),
                stream_cfg.get("fov_deg", 90.0),
                stream_cfg.get("tilt_deg", 45.0),
            )

        stream_id = stream_cfg.get("stream_id", "cam")
        entity_id = f"cam-track-{stream_id}-{track_id}"

        meta: Dict[str, str] = {
            "source_camera": stream_id,
            "track_id": str(track_id),
            "class_label": label,
            "confidence": str(round(score, 3)),
            "bbox": json.dumps([round(v, 1) for v in bbox]),
            "geo_method": "ray_cast" if cam_model is not None else "flat_earth",
        }
        if depth_m is not None:
            meta["depth_m"] = str(round(depth_m, 2))

        return (
            EntityBuilder(entity_id, "TRACK", "GROUND")
            .position(lat, lon)
            .metadata(meta)
            .provenance(source_type="camera", source_id=stream_id)
            .org(self.org_id)
            .build()
        )

    # ── Single-stream processing loop ────────────────────────────────────────

    async def _run_stream(self, stream_cfg: Dict) -> None:
        stream_id = stream_cfg.get("stream_id", "cam-01")
        url = stream_cfg.get("url", "")
        tracker = self._get_tracker(stream_id)

        # Build camera model once — used for every frame in this stream
        cam_model = CameraModel(stream_cfg)
        logger.info(
            f"[{stream_id}] starting — url={'<sim>' if not url else url} "
            f"geo=ray_cast az={stream_cfg.get('azimuth_deg', 0)}° "
            f"tilt={stream_cfg.get('tilt_deg', 45)}°"
        )

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
                cam_model.update_frame_size(fw, fh)

                raw_dets = await self._detect(jpeg_bytes, stream_id, frame=frame)
            else:
                # Simulation: generate synthetic detections
                raw_dets = self._simulate_detections(stream_cfg)
                fw, fh = 1920, 1080
                frame = None

            # Run through ByteTracker
            if tracker is not None:
                tracked = tracker.update(raw_dets)
            else:
                # Fallback: assign sequential fake track IDs
                tracked = [{**d, "track_id": i + 1} for i, d in enumerate(raw_dets)]

            # Attempt to flush buffered entities first (if MQTT just came back)
            if not self._mqtt_online and self._offline_buffer.size > 0:
                flushed = self._offline_buffer.flush(self.publish)
                if flushed > 0:
                    self._mqtt_online = True
                    logger.info(
                        f"[{stream_id}] MQTT reconnected — flushed {flushed} buffered entities "
                        f"({self._offline_buffer.size} remaining)"
                    )

            # Publish each tracked object, buffering locally if MQTT is offline
            for tdet in tracked:
                tid = tdet.get("track_id", 0)
                entity = self._detection_to_entity(tdet, stream_cfg, fw, fh, tid, cam_model)
                if not entity:
                    continue
                try:
                    self.publish(entity, qos=1)
                    self._mqtt_online = True
                except Exception:
                    self._mqtt_online = False
                    self._offline_buffer.store(entity)
                    if self._offline_buffer.size % 100 == 1:
                        logger.warning(
                            f"[{stream_id}] MQTT offline — {self._offline_buffer.size} entities buffered locally"
                        )

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
