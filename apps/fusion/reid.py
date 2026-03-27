"""
Cross-Camera Re-Identification for Summit.OS Fusion Service

Maintains a gallery of appearance embeddings per track. When a new detection
arrives from a different camera with no spatial match, compare its appearance
embedding against the gallery to find a candidate identity match.

Two embedding backends (in priority order):
  1. ONNX Re-ID model (e.g. OSNet, LightMBN) — when REID_MODEL_PATH is set
  2. Color histogram (L*a*b* in 8×8×8 bins) — always available as fallback

Usage:
    reid = AppearanceReID()
    # Register crop from camera-A
    reid.update("track-1", crop_bgr_array, camera_id="cam-a")
    # Query from camera-B — returns (track_id, score) or (None, 0.0)
    match_id, score = reid.query(crop_bgr_array, camera_id="cam-b", exclude_cameras={"cam-b"})
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("fusion.reid")

try:
    import cv2

    _CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    _CV2 = False

try:
    import onnxruntime as ort

    _ORT = True
except ImportError:
    ort = None  # type: ignore
    _ORT = False


# ── Histogram embedding (always available) ───────────────────────────────────


def _color_histogram(bgr: np.ndarray, bins: int = 8) -> np.ndarray:
    """
    Compute a normalised L*a*b* colour histogram.
    Returns a flat float32 vector of length bins**3.
    """
    if not _CV2 or bgr is None or bgr.size == 0:
        return np.zeros(bins**3, dtype=np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    hist = cv2.calcHist(
        [lab],
        [0, 1, 2],
        None,
        [bins, bins, bins],
        [0, 256, 0, 256, 0, 256],
    )
    hist = hist.flatten().astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm
    return hist


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── ONNX embedding (optional) ────────────────────────────────────────────────


class _ONNXEmbedder:
    """Thin wrapper around an ONNX Re-ID model (e.g. OSNet-x0.25)."""

    INPUT_H = 256
    INPUT_W = 128

    def __init__(self, model_path: str):
        if not _ORT:
            raise ImportError("onnxruntime not installed")
        self._sess = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_name = self._sess.get_inputs()[0].name

    def embed(self, bgr: np.ndarray) -> np.ndarray:
        if not _CV2 or bgr is None or bgr.size == 0:
            return np.zeros(512, dtype=np.float32)
        img = cv2.resize(bgr, (self.INPUT_W, self.INPUT_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # ImageNet normalisation
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)[np.newaxis]  # NCHW
        emb = self._sess.run(None, {self._input_name: img})[0][0]
        norm = np.linalg.norm(emb)
        return (emb / norm) if norm > 1e-9 else emb


# ── Gallery entry ─────────────────────────────────────────────────────────────


class _GalleryEntry:
    MAX_HISTORY = 5  # keep last N embeddings per track per camera

    def __init__(self, track_id: str):
        self.track_id = track_id
        # camera_id → deque of embeddings
        self._embeds: Dict[str, deque] = {}
        self.last_seen: float = time.time()

    def add(self, camera_id: str, emb: np.ndarray):
        if camera_id not in self._embeds:
            self._embeds[camera_id] = deque(maxlen=self.MAX_HISTORY)
        self._embeds[camera_id].append(emb)
        self.last_seen = time.time()

    def mean_embedding(self, exclude_cameras: set) -> Optional[np.ndarray]:
        """Return the mean embedding across cameras NOT in exclude_cameras."""
        vecs = []
        for cam, dq in self._embeds.items():
            if cam not in exclude_cameras:
                vecs.extend(list(dq))
        if not vecs:
            return None
        return np.mean(vecs, axis=0)

    @property
    def cameras(self) -> set:
        return set(self._embeds.keys())


# ── Main Re-ID class ──────────────────────────────────────────────────────────


class AppearanceReID:
    """
    Cross-camera appearance re-identification gallery.

    Thread-safe for read access; writes should be serialised by the caller
    (single asyncio thread is fine).
    """

    MATCH_THRESHOLD_ONNX = float(os.getenv("REID_MATCH_THRESHOLD_ONNX", "0.75"))
    MATCH_THRESHOLD_HIST = float(os.getenv("REID_MATCH_THRESHOLD_HIST", "0.85"))
    MAX_GALLERY_AGE_S = float(os.getenv("REID_MAX_GALLERY_AGE_S", "300"))  # 5 min

    def __init__(self):
        self._gallery: Dict[str, _GalleryEntry] = {}
        self._embedder: Optional[_ONNXEmbedder] = None
        self._use_onnx = False

        model_path = os.getenv("REID_MODEL_PATH")
        if model_path and os.path.isfile(model_path):
            try:
                self._embedder = _ONNXEmbedder(model_path)
                self._use_onnx = True
                logger.info(f"ReID ONNX model loaded: {model_path}")
            except Exception as e:
                logger.warning(
                    f"ReID ONNX model failed to load ({e}) — using histogram fallback"
                )
        else:
            logger.info(
                "ReID using colour-histogram embeddings (set REID_MODEL_PATH for ONNX)"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, track_id: str, crop: np.ndarray, camera_id: str = "default"):
        """Register a crop for track_id seen on camera_id."""
        emb = self._embed(crop)
        if track_id not in self._gallery:
            self._gallery[track_id] = _GalleryEntry(track_id)
        self._gallery[track_id].add(camera_id, emb)
        self._evict_stale()

    def query(
        self,
        crop: np.ndarray,
        camera_id: str = "default",
        exclude_cameras: Optional[set] = None,
        min_cross_camera: bool = True,
    ) -> Tuple[Optional[str], float]:
        """
        Find the best matching track_id for a crop.

        Args:
            crop: BGR image crop of the detected object.
            camera_id: Which camera this crop comes from.
            exclude_cameras: Only match against tracks NOT seen from these cameras.
                             Defaults to {camera_id} (cross-camera only).
            min_cross_camera: If True (default), require the gallery entry to have
                              been seen from at least one other camera.

        Returns:
            (track_id, score) — track_id is None if no match exceeds threshold.
        """
        if exclude_cameras is None:
            exclude_cameras = {camera_id}

        emb = self._embed(crop)
        threshold = (
            self.MATCH_THRESHOLD_ONNX if self._use_onnx else self.MATCH_THRESHOLD_HIST
        )

        best_id: Optional[str] = None
        best_score: float = 0.0

        for tid, entry in self._gallery.items():
            if min_cross_camera and not (entry.cameras - exclude_cameras):
                continue  # no cross-camera observations yet
            gallery_emb = entry.mean_embedding(exclude_cameras)
            if gallery_emb is None:
                continue
            score = _cosine_sim(emb, gallery_emb)
            if score > best_score:
                best_score = score
                best_id = tid

        if best_score >= threshold:
            return best_id, best_score
        return None, best_score

    def get_track_cameras(self, track_id: str) -> List[str]:
        """Return list of cameras that have seen this track."""
        entry = self._gallery.get(track_id)
        return list(entry.cameras) if entry else []

    def gallery_size(self) -> int:
        return len(self._gallery)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _embed(self, crop: np.ndarray) -> np.ndarray:
        if self._use_onnx and self._embedder is not None:
            try:
                return self._embedder.embed(crop)
            except Exception as e:
                logger.debug(f"ONNX embed failed, falling back to histogram: {e}")
        return _color_histogram(crop)

    def _evict_stale(self):
        now = time.time()
        stale = [
            tid
            for tid, e in self._gallery.items()
            if now - e.last_seen > self.MAX_GALLERY_AGE_S
        ]
        for tid in stale:
            del self._gallery[tid]


# ── Module-level singleton ────────────────────────────────────────────────────

_default_reid: Optional[AppearanceReID] = None


def get_reid() -> AppearanceReID:
    global _default_reid
    if _default_reid is None:
        _default_reid = AppearanceReID()
    return _default_reid
