"""
Cross-Modal Identity Registry — Summit.OS Fusion

Associates observations from multiple sensor modalities (ADS-B, AIS, camera,
radar, sonar, Meshtastic mesh) to canonical entity IDs. Uses a configurable
fusion score to decide when two descriptors are "same entity":

  score = w_pos * position_score + w_id * identifier_score + w_type * type_score

Where:
  position_score = 1 - min(haversine(a,b) / MAX_ASSOC_DIST_M, 1.0)
  identifier_score = 1.0 if shared callsign/MMSI/reid matches else 0.0
  type_score = 1.0 if compatible modality types else 0.5

Association threshold: SCORE >= 0.6 (configurable via CROSS_MODAL_THRESHOLD env).
"""

from __future__ import annotations

import math
import os
import threading
import uuid
from typing import Dict, List, Optional

from .modality_adapters import TrackDescriptor

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_ASSOC_DIST_M: float = float(os.getenv("CROSS_MODAL_MAX_DIST_M", "500"))
DEFAULT_THRESHOLD: float = 0.6

# Fusion score weights
W_POS: float = 0.5
W_ID: float = 0.4
W_TYPE: float = 0.1

# Modalities that can reasonably co-observe the same physical entity
_COMPATIBLE_PAIRS = {
    frozenset({"ADSB", "RADAR"}),
    frozenset({"ADSB", "CAMERA"}),
    frozenset({"AIS", "RADAR"}),
    frozenset({"AIS", "SONAR"}),
    frozenset({"AIS", "CAMERA"}),
    frozenset({"RADAR", "CAMERA"}),
    frozenset({"RADAR", "SONAR"}),
    frozenset({"MESH", "CAMERA"}),
    frozenset({"MESH", "ADSB"}),
    frozenset({"SONAR", "CAMERA"}),
}


class CrossModalRegistry:
    """Associates sensor observations to canonical entity IDs."""

    def __init__(self, threshold: float = None):
        env_thresh = os.getenv("CROSS_MODAL_THRESHOLD")
        self._threshold = (
            float(env_thresh) if env_thresh is not None
            else (threshold if threshold is not None else DEFAULT_THRESHOLD)
        )
        # source_id → canonical_id
        self._source_to_canonical: Dict[str, str] = {}
        # canonical_id → list[source_id]
        self._canonical_to_sources: Dict[str, List[str]] = {}
        # canonical_id → list[TrackDescriptor]  (all observations for fusing)
        self._canonical_to_descriptors: Dict[str, List[TrackDescriptor]] = {}
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def associate(self, descriptor: TrackDescriptor) -> str:
        """Return canonical entity_id for descriptor (creates new if no match)."""
        with self._lock:
            # If we've seen this source before, return existing canonical
            existing = self._source_to_canonical.get(descriptor.source_id)
            if existing:
                # Update descriptor list
                descs = self._canonical_to_descriptors.get(existing, [])
                # Replace descriptor for this source
                self._canonical_to_descriptors[existing] = [
                    d for d in descs if d.source_id != descriptor.source_id
                ] + [descriptor]
                return existing

            # Compare against all known canonical entities (most recent descriptor each)
            best_score = 0.0
            best_canonical: Optional[str] = None
            for canonical_id, descs in self._canonical_to_descriptors.items():
                for existing_desc in descs:
                    s = self.score(descriptor, existing_desc)
                    if s > best_score:
                        best_score = s
                        best_canonical = canonical_id

            if best_score >= self._threshold and best_canonical is not None:
                # Associate with existing canonical
                self._source_to_canonical[descriptor.source_id] = best_canonical
                if descriptor.source_id not in self._canonical_to_sources[best_canonical]:
                    self._canonical_to_sources[best_canonical].append(descriptor.source_id)
                self._canonical_to_descriptors[best_canonical].append(descriptor)
                return best_canonical

            # Create new canonical entity
            new_id = str(uuid.uuid4())
            self._source_to_canonical[descriptor.source_id] = new_id
            self._canonical_to_sources[new_id] = [descriptor.source_id]
            self._canonical_to_descriptors[new_id] = [descriptor]
            return new_id

    def get_canonical(self, source_id: str) -> Optional[str]:
        """Resolve a source observation ID to its canonical entity ID."""
        with self._lock:
            return self._source_to_canonical.get(source_id)

    def get_sources(self, canonical_id: str) -> List[str]:
        """Return all source IDs associated with a canonical entity."""
        with self._lock:
            return list(self._canonical_to_sources.get(canonical_id, []))

    def fuse(self, canonical_id: str) -> dict:
        """Merge all descriptors for a canonical entity into a fused entity dict.

        Position is confidence-weighted average. Identifiers are unioned.
        """
        with self._lock:
            descs = self._canonical_to_descriptors.get(canonical_id, [])

        if not descs:
            return {}

        # Confidence-weighted position average
        total_weight = 0.0
        lat_sum = lon_sum = alt_sum = 0.0
        lat_w = lon_w = alt_w = 0.0

        callsigns: set[str] = set()
        mmsis: set[str] = set()
        visual_ids: set[str] = set()
        modalities: set[str] = set()

        speed_vals: list[float] = []
        heading_vals: list[float] = []
        ts_latest = 0.0

        for d in descs:
            w = d.confidence
            total_weight += w
            if d.lat is not None and d.lon is not None:
                lat_sum += d.lat * w
                lon_sum += d.lon * w
                lat_w += w
                lon_w += w
            if d.alt_m is not None:
                alt_sum += d.alt_m * w
                alt_w += w
            if d.callsign:
                callsigns.add(d.callsign)
            if d.mmsi:
                mmsis.add(d.mmsi)
            if d.visual_id:
                visual_ids.add(d.visual_id)
            modalities.add(d.modality)
            if d.speed_mps is not None:
                speed_vals.append(d.speed_mps)
            if d.heading_deg is not None:
                heading_vals.append(d.heading_deg)
            if d.ts > ts_latest:
                ts_latest = d.ts

        fused: dict = {
            "canonical_id": canonical_id,
            "modalities": sorted(modalities),
            "source_ids": [d.source_id for d in descs],
            "source_count": len(descs),
            "ts": ts_latest,
            "confidence": total_weight / len(descs) if descs else 0.0,
        }

        if lat_w > 0:
            fused["lat"] = lat_sum / lat_w
            fused["lon"] = lon_sum / lon_w
        if alt_w > 0:
            fused["alt_m"] = alt_sum / alt_w
        if callsigns:
            fused["callsign"] = sorted(callsigns)[0]
        if mmsis:
            fused["mmsi"] = sorted(mmsis)[0]
        if visual_ids:
            fused["visual_id"] = sorted(visual_ids)[0]
        if speed_vals:
            fused["speed_mps"] = sum(speed_vals) / len(speed_vals)
        if heading_vals:
            fused["heading_deg"] = sum(heading_vals) / len(heading_vals)

        return fused

    def score(self, a: TrackDescriptor, b: TrackDescriptor) -> float:
        """Compute association score between two TrackDescriptors (0–1)."""
        # ── Position score ────────────────────────────────────────────────────
        if (a.lat is not None and a.lon is not None
                and b.lat is not None and b.lon is not None):
            dist_m = self._haversine_m(a.lat, a.lon, b.lat, b.lon)
            pos_score = 1.0 - min(dist_m / MAX_ASSOC_DIST_M, 1.0)
        else:
            pos_score = 0.5  # unknown position — neutral

        # ── Identifier score ─────────────────────────────────────────────────
        id_score = 0.0
        if a.callsign and b.callsign and a.callsign == b.callsign:
            id_score = 1.0
        elif a.mmsi and b.mmsi and a.mmsi == b.mmsi:
            id_score = 1.0
        elif a.visual_id and b.visual_id and a.visual_id == b.visual_id:
            id_score = 1.0

        # ── Type/modality compatibility score ────────────────────────────────
        if a.modality == b.modality:
            # Same sensor type — could be same track (high) or duplicate (medium)
            type_score = 0.8
        elif frozenset({a.modality, b.modality}) in _COMPATIBLE_PAIRS:
            type_score = 1.0
        else:
            type_score = 0.5

        return W_POS * pos_score + W_ID * id_score + W_TYPE * type_score

    def _haversine_m(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Great-circle distance in metres."""
        R = 6_371_000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        )
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Singleton ─────────────────────────────────────────────────────────────────

_registry: Optional[CrossModalRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> CrossModalRegistry:
    """Return (or create) the module-level CrossModalRegistry singleton."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = CrossModalRegistry()
    return _registry
