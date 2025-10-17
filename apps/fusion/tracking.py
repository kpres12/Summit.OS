"""
Simple IOU-based multi-object tracker for Fusion service.

- Adds track_id to detections using greedy IOU matching.
- Optional velocity estimation in pixels/second based on bbox centers.
- Designed as a lightweight fallback when ByteTrack/DeepSORT are unavailable.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from time import time
import math


def _iou_xywh(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = aw * ah
    area_b = bw * bh
    return inter / max(1e-6, (area_a + area_b - inter))


@dataclass
class _Track:
    track_id: int
    bbox: Tuple[float, float, float, float]
    last_ts: float
    class_id: Optional[int] = None
    vx_px_s: float = 0.0
    vy_px_s: float = 0.0


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age_s: float = 1.0):
        self.iou_threshold = iou_threshold
        self.max_age_s = max_age_s
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 1

    def update(self, detections: List[Dict[str, Any]], timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        now = timestamp or time()
        # Remove stale tracks
        self._tracks = {tid: tr for tid, tr in self._tracks.items() if (now - tr.last_ts) <= self.max_age_s}

        # Build matching between current detections and existing tracks
        unmatched_dets = list(range(len(detections)))
        assignments: Dict[int, int] = {}  # det_idx -> track_id

        # Greedy matching by IOU
        for det_idx in list(unmatched_dets):
            det = detections[det_idx]
            bbox = tuple(det.get("bbox", [0, 0, 0, 0]))  # xywh
            best_iou = 0.0
            best_tid = None
            for tid, tr in self._tracks.items():
                iou = _iou_xywh(bbox, tr.bbox)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None:
                assignments[det_idx] = best_tid
                unmatched_dets.remove(det_idx)

        # Update matched tracks
        for det_idx, tid in assignments.items():
            det = detections[det_idx]
            new_bbox = tuple(det.get("bbox", [0, 0, 0, 0]))
            tr = self._tracks[tid]
            dt = max(1e-3, now - tr.last_ts)
            # Velocity in pixels/sec using bbox center
            cx_prev = tr.bbox[0] + tr.bbox[2] / 2.0
            cy_prev = tr.bbox[1] + tr.bbox[3] / 2.0
            cx = new_bbox[0] + new_bbox[2] / 2.0
            cy = new_bbox[1] + new_bbox[3] / 2.0
            tr.vx_px_s = (cx - cx_prev) / dt
            tr.vy_px_s = (cy - cy_prev) / dt
            tr.bbox = new_bbox  # type: ignore
            tr.last_ts = now
            det["track_id"] = tid
            det.setdefault("velocity_px_per_s", math.hypot(tr.vx_px_s, tr.vy_px_s))

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            bbox = tuple(det.get("bbox", [0, 0, 0, 0]))
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _Track(track_id=tid, bbox=bbox, last_ts=now, class_id=int(det.get("class_id", 0)))
            det["track_id"] = tid
            det.setdefault("velocity_px_per_s", 0.0)

        return detections
