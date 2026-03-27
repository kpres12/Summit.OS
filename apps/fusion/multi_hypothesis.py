"""
Multi-Hypothesis Tracker (MHT) for Summit.OS

Maintains multiple association hypotheses when sensor measurements
are ambiguous. Uses hypothesis tree pruning to keep computation bounded.

This is critical for dense multi-target environments where the
Hungarian algorithm may produce suboptimal associations.

Complements the existing track correlator (which uses single-hypothesis
Hungarian assignment) by providing an alternative for contested scenarios.
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger("fusion.mht")


@dataclass
class Hypothesis:
    """A single association hypothesis."""

    hypothesis_id: int
    parent_id: int = -1
    # Track-to-measurement associations: {track_id: measurement_idx}
    associations: Dict[str, int] = field(default_factory=dict)
    # New tracks from unassociated measurements
    new_tracks: List[int] = field(default_factory=list)
    # Probability of this hypothesis
    log_likelihood: float = 0.0
    score: float = 0.0
    depth: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def probability(self) -> float:
        return math.exp(min(self.log_likelihood, 0))


@dataclass
class MHTTrack:
    """A track maintained by the MHT."""

    track_id: str
    positions: List[Tuple[float, float, float]] = field(
        default_factory=list
    )  # (lat, lon, alt)
    timestamps: List[float] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    confirmed: bool = False
    miss_count: int = 0
    hit_count: int = 0

    @property
    def last_position(self) -> Optional[Tuple[float, float, float]]:
        return self.positions[-1] if self.positions else None


@dataclass
class Measurement:
    """A sensor measurement to be associated."""

    idx: int
    lat: float
    lon: float
    alt: float = 0.0
    source: str = ""
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)


class MultiHypothesisTracker:
    """
    Multi-Hypothesis Tracker.

    Parameters:
        max_hypotheses: Maximum hypotheses to maintain (pruning limit)
        max_depth: Maximum tree depth before forced pruning
        gate_distance_m: Gating distance for valid associations (meters)
        new_track_score: Initial score for new tracks from unassociated measurements
        confirm_threshold: Hits needed to confirm a track
        delete_threshold: Misses before track deletion
    """

    EARTH_R = 6_371_000.0

    def __init__(
        self,
        max_hypotheses: int = 100,
        max_depth: int = 5,
        gate_distance_m: float = 1000.0,
        new_track_score: float = 0.3,
        confirm_threshold: int = 3,
        delete_threshold: int = 5,
    ):
        self.max_hypotheses = max_hypotheses
        self.max_depth = max_depth
        self.gate_distance_m = gate_distance_m
        self.new_track_score = new_track_score
        self.confirm_threshold = confirm_threshold
        self.delete_threshold = delete_threshold

        self._hypotheses: List[Hypothesis] = []
        self._tracks: Dict[str, MHTTrack] = {}
        self._next_hyp_id = 0
        self._next_track_id = 0

        # Initialize root hypothesis
        root = Hypothesis(hypothesis_id=self._next_hyp_id, score=1.0)
        self._hypotheses.append(root)
        self._next_hyp_id += 1

    def process_scan(self, measurements: List[Measurement]) -> Dict[str, Any]:
        """
        Process a scan of measurements and update hypotheses.

        Returns dict with best hypothesis associations and track updates.
        """
        if not measurements:
            self._miss_all_tracks()
            return {"tracks": {}, "new_tracks": [], "hypotheses": len(self._hypotheses)}

        new_hypotheses = []
        active_tracks = [
            tid
            for tid, t in self._tracks.items()
            if t.miss_count < self.delete_threshold
        ]

        for parent in self._hypotheses:
            # Generate child hypotheses
            children = self._generate_children(parent, measurements, active_tracks)
            new_hypotheses.extend(children)

        # Score and prune
        self._hypotheses = self._prune(new_hypotheses)

        # Extract best hypothesis
        best = (
            max(self._hypotheses, key=lambda h: h.score) if self._hypotheses else None
        )

        if best:
            return self._apply_hypothesis(best, measurements)

        return {"tracks": {}, "new_tracks": [], "hypotheses": len(self._hypotheses)}

    def _generate_children(
        self, parent: Hypothesis, measurements: List[Measurement], track_ids: List[str]
    ) -> List[Hypothesis]:
        """Generate child hypotheses from a parent."""
        children = []

        # Compute gated associations: which measurements can associate to which tracks
        gated: Dict[str, List[int]] = defaultdict(list)
        for tid in track_ids:
            track = self._tracks.get(tid)
            if not track or not track.last_position:
                continue
            for m in measurements:
                dist = self._haversine(
                    track.last_position[0],
                    track.last_position[1],
                    m.lat,
                    m.lon,
                )
                if dist < self.gate_distance_m:
                    gated[tid].append(m.idx)

        # Generate association hypotheses
        # 1. All tracks miss (all measurements are new)
        hyp_miss = Hypothesis(
            hypothesis_id=self._next_hyp_id,
            parent_id=parent.hypothesis_id,
            associations={},
            new_tracks=[m.idx for m in measurements],
            log_likelihood=parent.log_likelihood - 1.0,
            score=parent.score * 0.5,
            depth=parent.depth + 1,
        )
        self._next_hyp_id += 1
        children.append(hyp_miss)

        # 2. Greedy best associations
        used_measurements: set = set()
        associations: Dict[str, int] = {}

        # Sort by closest distance
        scored_pairs: List[Tuple[float, str, int]] = []
        for tid, m_idxs in gated.items():
            track = self._tracks[tid]
            for m_idx in m_idxs:
                m = measurements[m_idx]
                dist = self._haversine(
                    track.last_position[0],
                    track.last_position[1],
                    m.lat,
                    m.lon,
                )
                scored_pairs.append((dist, tid, m_idx))

        scored_pairs.sort()

        used_tracks: set = set()
        for dist, tid, m_idx in scored_pairs:
            if tid in used_tracks or m_idx in used_measurements:
                continue
            associations[tid] = m_idx
            used_tracks.add(tid)
            used_measurements.add(m_idx)

        new_tracks = [m.idx for m in measurements if m.idx not in used_measurements]

        # Compute likelihood
        ll = parent.log_likelihood
        for tid, m_idx in associations.items():
            m = measurements[m_idx]
            track = self._tracks[tid]
            dist = self._haversine(
                track.last_position[0],
                track.last_position[1],
                m.lat,
                m.lon,
            )
            ll += -0.5 * (dist / self.gate_distance_m) ** 2

        hyp_best = Hypothesis(
            hypothesis_id=self._next_hyp_id,
            parent_id=parent.hypothesis_id,
            associations=associations,
            new_tracks=new_tracks,
            log_likelihood=ll,
            score=parent.score * math.exp(min(ll - parent.log_likelihood, 0)),
            depth=parent.depth + 1,
        )
        self._next_hyp_id += 1
        children.append(hyp_best)

        return children

    def _prune(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Prune hypotheses to keep bounded."""
        if not hypotheses:
            return hypotheses

        # Normalize scores
        total = sum(h.score for h in hypotheses)
        if total > 0:
            for h in hypotheses:
                h.score /= total

        # Sort by score descending and keep top N
        hypotheses.sort(key=lambda h: h.score, reverse=True)
        pruned = hypotheses[: self.max_hypotheses]

        # Depth pruning
        pruned = [h for h in pruned if h.depth <= self.max_depth]

        return pruned if pruned else [hypotheses[0]]

    def _apply_hypothesis(
        self, hyp: Hypothesis, measurements: List[Measurement]
    ) -> Dict:
        """Apply the best hypothesis to update tracks."""
        results = {"tracks": {}, "new_tracks": [], "hypotheses": len(self._hypotheses)}

        # Update associated tracks
        for tid, m_idx in hyp.associations.items():
            m = measurements[m_idx]
            track = self._tracks.get(tid)
            if track:
                track.positions.append((m.lat, m.lon, m.alt))
                track.timestamps.append(m.timestamp)
                track.hit_count += 1
                track.miss_count = 0
                if track.hit_count >= self.confirm_threshold:
                    track.confirmed = True
                results["tracks"][tid] = track.last_position

        # Increment miss count for unassociated tracks
        associated = set(hyp.associations.keys())
        for tid, track in self._tracks.items():
            if tid not in associated:
                track.miss_count += 1

        # Create new tracks
        for m_idx in hyp.new_tracks:
            m = measurements[m_idx]
            tid = f"mht_{self._next_track_id:04d}"
            self._next_track_id += 1
            track = MHTTrack(
                track_id=tid,
                positions=[(m.lat, m.lon, m.alt)],
                timestamps=[m.timestamp],
                scores=[self.new_track_score],
            )
            self._tracks[tid] = track
            results["new_tracks"].append(tid)

        # Delete dead tracks
        dead = [
            tid
            for tid, t in self._tracks.items()
            if t.miss_count >= self.delete_threshold
        ]
        for tid in dead:
            del self._tracks[tid]

        return results

    def _miss_all_tracks(self):
        for track in self._tracks.values():
            track.miss_count += 1

    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        la1, lo1 = math.radians(lat1), math.radians(lon1)
        la2, lo2 = math.radians(lat2), math.radians(lon2)
        dlat = la2 - la1
        dlon = lo2 - lo1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
        )
        return 2 * self.EARTH_R * math.asin(math.sqrt(min(a, 1.0)))

    def get_confirmed_tracks(self) -> List[MHTTrack]:
        return [t for t in self._tracks.values() if t.confirmed]

    def get_all_tracks(self) -> Dict[str, MHTTrack]:
        return dict(self._tracks)

    def stats(self) -> Dict:
        return {
            "total_tracks": len(self._tracks),
            "confirmed": sum(1 for t in self._tracks.values() if t.confirmed),
            "tentative": sum(1 for t in self._tracks.values() if not t.confirmed),
            "hypotheses": len(self._hypotheses),
        }
