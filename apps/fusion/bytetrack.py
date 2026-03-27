"""
ByteTrack — multi-object tracking via two-stage association.

Combines high-confidence detection matching (IOU) with low-confidence
recovery and unconfirmed track promotion. Pure Python/NumPy — no GPU needed.

Reference: ByteTrack: Multi-Object Tracking by Associating Every Detection Box
           (Zhang et al., 2022)

Usage:
    tracker = ByteTracker()
    for frame_dets in detections_per_frame:
        tracks = tracker.update(frame_dets)
        # Each track: {track_id, bbox, score, class_id, state, ...}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np

    _NUMPY = True
except ImportError:
    _NUMPY = False


class TrackState(Enum):
    NEW = "new"  # Seen once, not yet confirmed
    TRACKED = "tracked"  # Active confirmed track
    LOST = "lost"  # Not matched recently, still buffered
    REMOVED = "removed"  # Expired — removed from tracker


@dataclass
class KalmanState:
    """Lightweight Kalman filter state for 2D bounding-box tracking.

    State vector: [cx, cy, aspect, height, dcx, dcy, daspect, dheight]
    Observation:  [cx, cy, aspect, height]
    """

    # Mean and covariance
    mean: List[float] = field(default_factory=lambda: [0.0] * 8)
    cov: List[List[float]] = field(
        default_factory=lambda: [[0.0] * 8 for _ in range(8)]
    )

    def predict(self) -> None:
        """Constant-velocity predict step."""
        # Transition: cx += dcx, etc.
        for i in range(4):
            self.mean[i] += self.mean[i + 4]
        # Inflate uncertainty
        for i in range(8):
            self.cov[i][i] = max(1.0, self.cov[i][i] * 1.1 + 1.0)

    def update(self, obs: Tuple[float, float, float, float]) -> None:
        """Update with [cx, cy, aspect, height] observation."""
        for i in range(4):
            innovation = obs[i] - self.mean[i]
            k = self.cov[i][i] / max(1e-6, self.cov[i][i] + 10.0)
            self.mean[i] += k * innovation
            self.cov[i][i] *= 1.0 - k


def _bbox_xywh_to_obs(
    bbox: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    aspect = w / max(1e-6, h)
    return cx, cy, aspect, h


def _obs_to_bbox_xywh(mean: List[float]) -> Tuple[float, float, float, float]:
    cx, cy, aspect, h = mean[0], mean[1], mean[2], mean[3]
    w = aspect * h
    x = cx - w / 2.0
    y = cy - h / 2.0
    return x, y, w, h


def _iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / max(1e-6, aw * ah + bw * bh - inter)


def _cost_matrix(
    tracks_bbox: List[Tuple[float, float, float, float]],
    dets_bbox: List[Tuple[float, float, float, float]],
) -> List[List[float]]:
    """Returns 1 - IOU cost matrix (lower = better match)."""
    mat = []
    for tb in tracks_bbox:
        row = [1.0 - _iou(tb, db) for db in dets_bbox]
        mat.append(row)
    return mat


def _hungarian(cost: List[List[float]], threshold: float) -> Dict[int, int]:
    """
    Greedy O(n*m) approximation of the Hungarian algorithm.
    Returns {row_idx: col_idx} assignments where cost < threshold.
    """
    if not cost or not cost[0]:
        return {}

    assignments: Dict[int, int] = {}
    used_cols: set = set()
    flat = []
    for r, row in enumerate(cost):
        for c, v in enumerate(row):
            if v < threshold:
                flat.append((v, r, c))
    flat.sort()

    used_rows: set = set()
    for v, r, c in flat:
        if r not in used_rows and c not in used_cols:
            assignments[r] = c
            used_rows.add(r)
            used_cols.add(c)

    return assignments


_NEXT_ID = 1


def _new_id() -> int:
    global _NEXT_ID
    i = _NEXT_ID
    _NEXT_ID += 1
    return i


@dataclass
class Track:
    track_id: int
    state: TrackState
    bbox: Tuple[float, float, float, float]  # xywh
    score: float
    class_id: int
    class_label: str
    frame_id: int
    hit_streak: int = 0
    frames_since_update: int = 0
    kalman: KalmanState = field(default_factory=KalmanState)

    def predict(self) -> None:
        self.kalman.predict()
        self.bbox = _obs_to_bbox_xywh(self.kalman.mean)
        self.frames_since_update += 1

    def update(self, det: Dict, frame_id: int) -> None:
        bbox = tuple(det.get("bbox", [0, 0, 0, 0]))
        obs = _bbox_xywh_to_obs(bbox)
        self.kalman.update(obs)
        self.bbox = _obs_to_bbox_xywh(self.kalman.mean)
        self.score = det.get("score", det.get("confidence", 1.0))
        self.class_id = int(det.get("class_id", 0))
        self.class_label = det.get("class_label", det.get("label", str(self.class_id)))
        self.frame_id = frame_id
        self.hit_streak += 1
        self.frames_since_update = 0
        self.state = TrackState.TRACKED

    @property
    def predicted_bbox(self) -> Tuple[float, float, float, float]:
        return _obs_to_bbox_xywh(self.kalman.mean)

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "score": self.score,
            "class_id": self.class_id,
            "class_label": self.class_label,
            "state": self.state.value,
            "frame_id": self.frame_id,
            "hit_streak": self.hit_streak,
            "frames_since_update": self.frames_since_update,
        }


class ByteTracker:
    """
    ByteTrack implementation.

    Two-stage matching:
      Stage 1 — high-confidence detections ↔ tracked + lost tracks (IOU)
      Stage 2 — low-confidence detections ↔ unmatched tracks (IOU)

    New detections not matched to any track become NEW tracks.
    NEW tracks are promoted to TRACKED after min_hits frames.
    TRACKED tracks become LOST after max_age frames without a match.
    LOST tracks become REMOVED after lost_age frames.
    """

    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        iou_threshold: float = 0.3,
        min_hits: int = 3,
        max_age: int = 30,
        lost_age: int = 90,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_age = max_age
        self.lost_age = lost_age

        self._tracked: Dict[int, Track] = {}
        self._lost: Dict[int, Track] = {}
        self._frame_id = 0

    def _new_track(self, det: Dict) -> Track:
        bbox = tuple(det.get("bbox", [0, 0, 0, 0]))
        score = det.get("score", det.get("confidence", 1.0))
        class_id = int(det.get("class_id", 0))
        label = det.get("class_label", det.get("label", str(class_id)))

        ks = KalmanState()
        obs = _bbox_xywh_to_obs(bbox)
        # Initialise mean with observation, zero velocity
        ks.mean = [obs[0], obs[1], obs[2], obs[3], 0.0, 0.0, 0.0, 0.0]
        ks.cov = [[100.0 if i == j else 0.0 for j in range(8)] for i in range(8)]

        return Track(
            track_id=_new_id(),
            state=TrackState.NEW,
            bbox=bbox,
            score=score,
            class_id=class_id,
            class_label=label,
            frame_id=self._frame_id,
            kalman=ks,
        )

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Process one frame of detections.

        Args:
            detections: list of dicts with keys:
                bbox      — [x, y, w, h] in pixels
                score / confidence — float 0-1
                class_id  — int
                class_label / label — str

        Returns:
            List of dicts from active Track.to_dict() for confirmed tracks only
            (state == TRACKED with hit_streak >= min_hits).
        """
        self._frame_id += 1

        high_dets = [
            d
            for d in detections
            if d.get("score", d.get("confidence", 1.0)) >= self.high_thresh
        ]
        low_dets = [
            d
            for d in detections
            if self.low_thresh
            <= d.get("score", d.get("confidence", 1.0))
            < self.high_thresh
        ]

        # Predict all existing tracks
        for tr in list(self._tracked.values()) + list(self._lost.values()):
            tr.predict()

        # --- Stage 1: high-confidence dets vs tracked + lost tracks ---
        all_tracks = list(self._tracked.values()) + list(self._lost.values())
        track_bboxes = [tr.predicted_bbox for tr in all_tracks]
        det_bboxes = [tuple(d.get("bbox", [0, 0, 0, 0])) for d in high_dets]

        cost = _cost_matrix(track_bboxes, det_bboxes)
        matches_1 = _hungarian(cost, 1.0 - self.iou_threshold)

        matched_track_ids: set = set()
        matched_det_ids: set = set()
        for t_idx, d_idx in matches_1.items():
            tr = all_tracks[t_idx]
            tr.update(high_dets[d_idx], self._frame_id)
            self._tracked[tr.track_id] = tr
            self._lost.pop(tr.track_id, None)
            matched_track_ids.add(tr.track_id)
            matched_det_ids.add(d_idx)

        # --- Stage 2: low-confidence dets vs unmatched tracked tracks ---
        unmatched_tracks = [
            tr for tr in all_tracks if tr.track_id not in matched_track_ids
        ]
        unmatched_high_dets = [
            d for i, d in enumerate(high_dets) if i not in matched_det_ids
        ]
        low_det_bboxes = [tuple(d.get("bbox", [0, 0, 0, 0])) for d in low_dets]

        if unmatched_tracks and low_dets:
            cost2 = _cost_matrix(
                [tr.predicted_bbox for tr in unmatched_tracks], low_det_bboxes
            )
            matches_2 = _hungarian(cost2, 1.0 - self.iou_threshold)
            for t_idx, d_idx in matches_2.items():
                tr = unmatched_tracks[t_idx]
                tr.update(low_dets[d_idx], self._frame_id)
                self._tracked[tr.track_id] = tr
                self._lost.pop(tr.track_id, None)
                matched_track_ids.add(tr.track_id)

        # --- New tracks from unmatched high-confidence detections ---
        for i, det in enumerate(high_dets):
            if i not in matched_det_ids:
                nt = self._new_track(det)
                self._tracked[nt.track_id] = nt

        # Also create from unmatched_high_dets that went through stage 2 path
        # (already handled above since we only skipped matched_det_ids in high_dets)

        # --- Promote/age tracked tracks ---
        for tid in list(self._tracked.keys()):
            tr = self._tracked[tid]
            if tr.frames_since_update > self.max_age:
                self._lost[tid] = tr
                del self._tracked[tid]

        # --- Expire lost tracks ---
        for tid in list(self._lost.keys()):
            tr = self._lost[tid]
            if tr.frames_since_update > self.lost_age:
                tr.state = TrackState.REMOVED
                del self._lost[tid]

        # Return only confirmed active tracks
        result = []
        for tr in self._tracked.values():
            if tr.hit_streak >= self.min_hits or self._frame_id <= self.min_hits:
                tr.state = TrackState.TRACKED
                result.append(tr.to_dict())

        return result

    @property
    def active_track_count(self) -> int:
        return len(self._tracked)

    @property
    def lost_track_count(self) -> int:
        return len(self._lost)

    def reset(self) -> None:
        self._tracked.clear()
        self._lost.clear()
        self._frame_id = 0
