"""
Track Lifecycle Manager for Summit.OS Fusion Service

Manages track creation, confirmation, coasting, and deletion.
Uses M-of-N confirmation logic and integrates with the EKF and correlator.

Track lifecycle:
  TENTATIVE → CONFIRMED → COASTING → DELETED
       ↑          ↑          |
       └──────────┘          └─ (no updates for max_coast ticks)
"""
from __future__ import annotations

import uuid
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from apps.fusion.filters.kalman import EKFState, ExtendedKalmanFilter
from apps.fusion.correlation.track_correlator import TrackCorrelator

logger = logging.getLogger("fusion.track_manager")


@dataclass
class ManagedTrack:
    """A track with lifecycle metadata."""
    track_id: str
    ekf_state: EKFState
    class_label: str = ""
    confidence: float = 0.0

    # Lifecycle
    state: str = "TENTATIVE"  # TENTATIVE, CONFIRMED, COASTING, DELETED
    hits: int = 1
    misses: int = 0
    age_ticks: int = 0

    # M-of-N confirmation
    recent_detections: List[bool] = field(default_factory=list)

    # Source info
    contributing_sensor_ids: List[str] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # Organization
    org_id: str = ""


class TrackManager:
    """
    Manages the full lifecycle of tracks in the fusion engine.

    Each fusion tick:
    1. Receive new observations
    2. Correlate observations to existing tracks
    3. Update matched tracks with EKF
    4. Create new tentative tracks for unmatched observations
    5. Coast unmatched tracks (predict without update)
    6. Promote tentative → confirmed (M-of-N)
    7. Delete stale tracks
    """

    def __init__(
        self,
        # M-of-N confirmation: need M detections in last N scans
        confirm_m: int = 3,
        confirm_n: int = 5,
        # Coasting: max ticks without observation before deletion
        max_coast_ticks: int = 10,
        # EKF parameters
        process_noise_pos: float = 0.5,
        process_noise_vel: float = 2.0,
        # Correlator parameters
        gate_threshold: float = 9.21,
    ):
        self.confirm_m = confirm_m
        self.confirm_n = confirm_n
        self.max_coast_ticks = max_coast_ticks

        self.ekf = ExtendedKalmanFilter(
            process_noise_pos=process_noise_pos,
            process_noise_vel=process_noise_vel,
        )
        self.correlator = TrackCorrelator(gate_threshold=gate_threshold)

        self.tracks: Dict[str, ManagedTrack] = {}
        self.tick_count: int = 0

    def process_observations(
        self,
        observations: List[Dict],
        t: Optional[float] = None,
    ) -> List[ManagedTrack]:
        """
        Process a batch of observations through the full fusion pipeline.

        Each observation dict must have:
          - lat, lon, alt: position
          - sigma_m: measurement noise in meters
          - sensor_id: source sensor
          - class_label (optional): classification

        Returns list of all active tracks after processing.
        """
        if t is None:
            t = time.time()
        self.tick_count += 1

        # Prepare observations for correlator
        obs_tuples = [
            (o["lat"], o["lon"], o["alt"], o.get("sigma_m", 10.0), o.get("sensor_id", "unknown"))
            for o in observations
        ]

        # Get EKF states for correlation
        ekf_states = {tid: tr.ekf_state for tid, tr in self.tracks.items()
                      if tr.state != "DELETED"}

        # Correlate
        matched, unmatched_obs, unmatched_trks = self.correlator.correlate(
            obs_tuples, ekf_states, t, self.ekf
        )

        # 1. Update matched tracks
        for obs_idx, track_id, distance in matched:
            obs = observations[obs_idx]
            track = self.tracks[track_id]

            # EKF update
            track.ekf_state = self.ekf.update_position(
                track.ekf_state,
                obs["lat"], obs["lon"], obs["alt"],
                t, sigma_pos_m=obs.get("sigma_m", 10.0),
            )

            # Update metadata
            track.hits += 1
            track.misses = 0
            track.last_seen = t
            track.recent_detections.append(True)
            if len(track.recent_detections) > self.confirm_n:
                track.recent_detections = track.recent_detections[-self.confirm_n:]

            # Update class label if provided (majority vote would be better)
            if obs.get("class_label"):
                track.class_label = obs["class_label"]

            # Add sensor
            sid = obs.get("sensor_id", "unknown")
            if sid not in track.contributing_sensor_ids:
                track.contributing_sensor_ids.append(sid)

            # Update confidence based on hits and uncertainty
            uncertainty = self.ekf.get_position_uncertainty(track.ekf_state)
            track.confidence = min(1.0, track.hits / 10.0) * max(0.1, 1.0 - uncertainty / 100.0)

        # 2. Create new tracks for unmatched observations
        for obs_idx in unmatched_obs:
            obs = observations[obs_idx]
            track_id = str(uuid.uuid4())
            ekf_state = self.ekf.initialize(obs["lat"], obs["lon"], obs["alt"], t)

            track = ManagedTrack(
                track_id=track_id,
                ekf_state=ekf_state,
                class_label=obs.get("class_label", ""),
                confidence=0.1,
                state="TENTATIVE",
                hits=1,
                misses=0,
                age_ticks=0,
                recent_detections=[True],
                contributing_sensor_ids=[obs.get("sensor_id", "unknown")],
                first_seen=t,
                last_seen=t,
                org_id=obs.get("org_id", ""),
            )
            self.tracks[track_id] = track
            logger.debug(f"New tentative track: {track_id}")

        # 3. Coast unmatched tracks
        for track_id in unmatched_trks:
            if track_id in self.tracks:
                track = self.tracks[track_id]
                track.ekf_state = self.ekf.predict(track.ekf_state, t)
                track.misses += 1
                track.recent_detections.append(False)
                if len(track.recent_detections) > self.confirm_n:
                    track.recent_detections = track.recent_detections[-self.confirm_n:]

                if track.state == "CONFIRMED":
                    track.state = "COASTING"
                    logger.debug(f"Track {track_id} now COASTING (misses={track.misses})")

        # 4. Lifecycle transitions
        to_delete = []
        for tid, track in self.tracks.items():
            track.age_ticks += 1

            # Tentative → Confirmed (M-of-N)
            if track.state == "TENTATIVE":
                detections = sum(track.recent_detections[-self.confirm_n:])
                if detections >= self.confirm_m:
                    track.state = "CONFIRMED"
                    logger.info(f"Track {tid} CONFIRMED ({detections}/{self.confirm_n} detections)")
                elif track.age_ticks > self.confirm_n and detections < self.confirm_m:
                    # Failed to confirm
                    to_delete.append(tid)

            # Coasting → Deleted
            if track.state == "COASTING" and track.misses >= self.max_coast_ticks:
                to_delete.append(tid)
                logger.info(f"Track {tid} DELETED (coasted {track.misses} ticks)")

        for tid in to_delete:
            self.tracks[tid].state = "DELETED"

        # Return active tracks
        return [t for t in self.tracks.values() if t.state != "DELETED"]

    def get_confirmed_tracks(self) -> List[ManagedTrack]:
        """Get only confirmed tracks."""
        return [t for t in self.tracks.values() if t.state == "CONFIRMED"]

    def get_all_active_tracks(self) -> List[ManagedTrack]:
        """Get all non-deleted tracks."""
        return [t for t in self.tracks.values() if t.state != "DELETED"]

    def get_track(self, track_id: str) -> Optional[ManagedTrack]:
        """Get a specific track."""
        return self.tracks.get(track_id)

    def cleanup(self):
        """Remove deleted tracks from memory."""
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.state != "DELETED"}

    def get_stats(self) -> Dict:
        """Get track manager statistics."""
        states = {}
        for t in self.tracks.values():
            states[t.state] = states.get(t.state, 0) + 1
        return {
            "tick_count": self.tick_count,
            "total_tracks": len(self.tracks),
            "by_state": states,
        }
