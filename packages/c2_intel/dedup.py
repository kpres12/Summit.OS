"""
C2 Observation Deduplication Engine

Ported from Mira Signals' deduplication.py.
Domain remapped: buying signal dedup → C2 observation dedup.

Prevents duplicate observations when the same physical event is reported
by multiple sensor sources simultaneously:
  - ADS-B + Radar both detect the same aircraft
  - Two mesh peers both report the same asset going offline
  - MQTT heartbeat + WorldStore poll both flag the same battery event

Fingerprint components (same approach as Mira, C2 domain):
  - Normalized entity ID
  - Event type
  - Time bucket (configurable window, default 30s)
  - Key terms from observation title
"""

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from .models import C2Observation, SensorSource


# Sensor source quality ranking (higher = more authoritative)
SENSOR_QUALITY_RANK: Dict[SensorSource, int] = {
    # Tier 1: Authoritative direct sensors
    SensorSource.IFF:         100,   # IFF is definitive identification
    SensorSource.MAVLINK:      95,   # Direct MAVLink telemetry
    SensorSource.RADAR:        90,   # Primary sensor
    SensorSource.EO_IR:        85,   # Electro-optical/IR
    SensorSource.ADS_B:        80,   # Cooperative, but spoofable
    SensorSource.SIGINT:       80,

    # Tier 2: Platform / network
    SensorSource.FUSED:        75,   # Multi-sensor fusion
    SensorSource.MESH_PEER:    70,   # Peer-reported
    SensorSource.WORLD_STORE:  70,
    SensorSource.MQTT:         65,

    # Tier 3: External / degraded
    SensorSource.OPENSKY:      60,   # Public ADS-B
    SensorSource.MANUAL:       50,   # Human-entered
    SensorSource.UNKNOWN:      30,
}


def normalize_entity_id(entity_id: str) -> str:
    """
    Normalize entity ID for fingerprint comparison.
    Strips sensor-adapter prefixes so "opensky-abc123" == "radar-abc123" fingerprint.
    """
    if not entity_id:
        return ""
    name = entity_id.lower()
    prefixes = ["opensky-", "mavlink-", "radar-track-", "iff-", "ads-b-", "mesh-"]
    for pfx in prefixes:
        if name.startswith(pfx):
            name = name[len(pfx):]
    name = re.sub(r"[^a-z0-9\s\-]", "", name)
    return " ".join(name.split()).strip()


def extract_key_terms(text: str, max_terms: int = 5) -> List[str]:
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    stopwords = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of",
        "with", "by", "from", "is", "are", "was", "were", "has", "have", "had",
        "this", "that", "it", "its", "they", "we", "you", "node", "asset",
        "entity", "signal", "detected", "alert", "update", "status",
    }
    words = text.split()
    meaningful = []
    seen: Set[str] = set()
    for word in words:
        if len(word) > 2 and word not in stopwords and word not in seen:
            meaningful.append(word)
            seen.add(word)
    return meaningful[:max_terms]


def generate_observation_fingerprint(
    obs: C2Observation,
    window_seconds: float = 30.0,
) -> str:
    """
    Generate a fingerprint for a C2 observation to detect duplicates.

    Fingerprint components:
      - Normalized entity ID
      - Event type
      - Time bucket (window_seconds granularity)
      - Key terms from title
    """
    entity = normalize_entity_id(obs.node_id or "")
    evt_type = (obs.event_type.value if hasattr(obs.event_type, "value")
                else str(obs.event_type))

    obs_time = obs.event_time or obs.detected_at or datetime.now(timezone.utc)
    if isinstance(obs_time, str):
        try:
            obs_time = datetime.fromisoformat(obs_time.replace("Z", "+00:00"))
        except ValueError:
            obs_time = datetime.now(timezone.utc)

    # Round to nearest window bucket
    bucket = int(obs_time.timestamp() / window_seconds)

    terms = extract_key_terms(obs.title or "", max_terms=3)
    terms_str = "_".join(sorted(terms))

    fingerprint_data = f"{entity}|{evt_type}|{bucket}|{terms_str}"
    # MD5 used as non-cryptographic event-dedup fingerprint, not for security.
    return hashlib.md5(fingerprint_data.encode(), usedforsecurity=False).hexdigest()[:16]


def get_sensor_quality(source: SensorSource) -> int:
    return SENSOR_QUALITY_RANK.get(source, 30)


def select_best_observation(observations: List[C2Observation]) -> C2Observation:
    """Select the highest-quality observation from a group of duplicates."""
    if not observations:
        raise ValueError("No observations provided")
    if len(observations) == 1:
        return observations[0]

    def sort_key(obs: C2Observation) -> Tuple:
        quality = get_sensor_quality(obs.source)
        confidence = obs.confidence or 0.0
        score = obs.score or 0
        detected = obs.detected_at or datetime.min.replace(tzinfo=timezone.utc)
        return (quality, confidence, score, detected)

    return sorted(observations, key=sort_key, reverse=True)[0]


class ObservationDeduplicator:
    """
    Deduplication engine for C2 observation streams.

    Usage:
        deduper = ObservationDeduplicator(window_seconds=30.0)
        unique = deduper.deduplicate(batch_of_observations)

        # Streaming: check one at a time
        if not deduper.is_duplicate(obs):
            deduper.mark_seen(obs)
            process(obs)
    """

    def __init__(
        self,
        window_seconds: float = 30.0,
        persist_file: Optional[str] = None,
    ):
        self.window_seconds = window_seconds
        self.persist_file = persist_file
        self._seen: Dict[str, datetime] = {}
        self._load_persisted()

    def _load_persisted(self):
        if not self.persist_file:
            return
        try:
            import json
            with open(self.persist_file, "r") as f:
                data = json.load(f)
                for fp, ts in data.items():
                    self._seen[fp] = datetime.fromisoformat(ts)
        except Exception:
            pass

    def _save_persisted(self):
        if not self.persist_file:
            return
        try:
            import json
            with open(self.persist_file, "w") as f:
                json.dump({fp: ts.isoformat() for fp, ts in self._seen.items()}, f)
        except Exception:
            pass

    def _cleanup(self):
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.window_seconds * 10)
        self._seen = {fp: ts for fp, ts in self._seen.items() if ts > cutoff}

    def fingerprint(self, obs: C2Observation) -> str:
        return generate_observation_fingerprint(obs, self.window_seconds)

    def is_duplicate(self, obs: C2Observation) -> bool:
        return self.fingerprint(obs) in self._seen

    def mark_seen(self, obs: C2Observation):
        self._seen[self.fingerprint(obs)] = datetime.now(timezone.utc)

    def deduplicate(
        self, observations: List[C2Observation], mark_seen: bool = True
    ) -> List[C2Observation]:
        """Deduplicate a batch. Returns best version of each unique observation."""
        if not observations:
            return []

        self._cleanup()
        groups: Dict[str, List[C2Observation]] = defaultdict(list)
        for obs in observations:
            groups[self.fingerprint(obs)].append(obs)

        unique = []
        for fp, group in groups.items():
            if fp in self._seen:
                continue
            best = select_best_observation(group)
            unique.append(best)
            if mark_seen:
                self._seen[fp] = datetime.now(timezone.utc)

        if mark_seen:
            self._save_persisted()

        return unique

    def deduplicate_by_node(
        self, observations: List[C2Observation]
    ) -> Dict[str, List[C2Observation]]:
        """Deduplicate and group by node_id."""
        unique = self.deduplicate(observations)
        by_node: Dict[str, List[C2Observation]] = defaultdict(list)
        for obs in unique:
            by_node[normalize_entity_id(obs.node_id or "unknown")].append(obs)
        return dict(by_node)

    def stats(self) -> Dict[str, Any]:
        return {
            "fingerprints_seen": len(self._seen),
            "window_seconds": self.window_seconds,
            "oldest": min(self._seen.values()).isoformat() if self._seen else None,
            "newest": max(self._seen.values()).isoformat() if self._seen else None,
        }


__all__ = [
    "ObservationDeduplicator",
    "generate_observation_fingerprint",
    "normalize_entity_id",
    "get_sensor_quality",
    "SENSOR_QUALITY_RANK",
]
