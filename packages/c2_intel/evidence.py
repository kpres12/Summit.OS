"""
C2 Evidence Aggregation Engine

Instead of showing 5 separate "COMMS_DEGRADED" observations, roll them up into:

  OBSERVATION CLUSTER: Multiple Comms Degradation Events

  "Is Node-Bravo experiencing a systemic comms failure, not just a transient event?"

  Yes — Node-Bravo has reported 4 comms degradation events in the last 8 minutes
  including MAVLINK link drops and RADAR contact intermittence. This represents a
  3x increase over baseline activity rate for this node.

  Supporting evidence:
  • 14:22:03 — COMMS_DEGRADED (MAVLINK, confidence 0.92)
  • 14:24:18 — LINK_DEGRADED (RADAR, confidence 0.87)
  • 14:26:45 — COMMS_DEGRADED (MESH_PEER, confidence 0.75)
  • 14:28:01 — COMMS_DEGRADED (MAVLINK, confidence 0.90)

This separates noise from intelligence in the Situation Feed.
"""

from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from .models import C2Observation


# Broad categories for grouping observations into evidence clusters
OBSERVATION_CATEGORIES = {
    "comms":       ["COMMS_DEGRADED", "COMMS_RESTORED", "LINK_DEGRADED", "LINK_LOST"],
    "threat":      ["THREAT_IDENTIFIED", "THREAT_NEUTRALIZED", "GEOFENCE_BREACH", "GEOFENCE_CLEARED"],
    "asset_health": ["BATTERY_CRITICAL", "BATTERY_LOW", "ASSET_OFFLINE", "ASSET_ONLINE",
                     "NODE_DEGRADED", "NODE_FAILED", "NODE_RECOVERED"],
    "sensor":      ["SENSOR_LOSS", "SENSOR_RESTORED"],
    "authority":   ["AUTHORITY_DELEGATED", "AUTHORITY_REVOKED", "HANDOFF_INITIATED", "HANDOFF_COMPLETE"],
    "mission":     ["MISSION_STARTED", "MISSION_COMPLETED", "MISSION_ABORTED",
                    "ENGAGEMENT_AUTHORIZED", "ENGAGEMENT_DENIED", "ENGAGEMENT_COMPLETE"],
    "environment": ["WEATHER_ALERT", "AIRSPACE_CONFLICT"],
    "peer":        ["PEER_OBSERVATION"],
}

_OBS_TO_CATEGORY: Dict[str, str] = {}
for _cat, _types in OBSERVATION_CATEGORIES.items():
    for _t in _types:
        _OBS_TO_CATEGORY[_t] = _cat


def _get_obs_category(event_type: str) -> str:
    evt = event_type.value if hasattr(event_type, "value") else str(event_type)
    return _OBS_TO_CATEGORY.get(evt.upper(), _OBS_TO_CATEGORY.get(evt, "other"))


HYPOTHESIS_TEMPLATES = {
    "comms": {
        "single":   "Is {entity_id} experiencing a transient comms event?",
        "multiple": "Is {entity_id} experiencing a systemic comms failure, not just a transient event?",
        "generic":  "Does {entity_id} have a comms issue that warrants authority delegation?",
    },
    "threat": {
        "single":   "Has a threat been identified near {entity_id}?",
        "multiple": "Is {entity_id} in a contested threat environment with multiple active indicators?",
        "generic":  "Is there credible threat activity associated with {entity_id}?",
    },
    "asset_health": {
        "single":   "Is {entity_id} showing a single asset health warning?",
        "multiple": "Is {entity_id} experiencing compounding asset degradation?",
        "generic":  "Is {entity_id}'s operational readiness at risk?",
    },
    "sensor": {
        "single":   "Has {entity_id} lost sensor coverage?",
        "multiple": "Is {entity_id} experiencing a multi-sensor outage?",
        "generic":  "Is the sensor picture for {entity_id} reliable?",
    },
    "authority": {
        "single":   "Has a C2 authority change been initiated for {entity_id}?",
        "multiple": "Is {entity_id} undergoing rapid C2 topology changes?",
        "generic":  "Is the command authority for {entity_id} stable?",
    },
    "mission": {
        "single":   "Is {entity_id}'s mission status changing?",
        "multiple": "Is {entity_id} experiencing multiple mission state transitions?",
        "generic":  "What is the current mission status for {entity_id}?",
    },
}


@dataclass
class ObservationEvidence:
    """One piece of evidence backing an insight cluster."""
    event_type:    str
    sensor_source: str
    timestamp:     Optional[datetime]
    entity_id:     str
    confidence:    float
    score:         int
    detail:        Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type":    self.event_type,
            "sensor_source": self.sensor_source,
            "timestamp":     self.timestamp.isoformat() if self.timestamp else None,
            "entity_id":     self.entity_id,
            "confidence":    self.confidence,
            "score":         self.score,
            "detail":        self.detail,
        }


@dataclass
class C2EvidenceCluster:
    """
    A rolled-up insight backed by multiple C2 observations.

    Instead of N separate observations, this is ONE insight with N supporting events.
    Displayed in the Situation Feed as a compound intelligence item.
    """
    hypothesis:          str
    answer:              str
    evidence:            List[ObservationEvidence]
    obs_category:        str
    entity_id:           str
    confidence:          float
    combined_score:      int      = 0
    observation_count:   int      = 0
    combination_context: Optional[str] = None  # Cross-cluster insight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis":        self.hypothesis,
            "answer":            self.answer,
            "evidence":          [e.to_dict() for e in self.evidence],
            "obs_category":      self.obs_category,
            "entity_id":         self.entity_id,
            "confidence":        self.confidence,
            "combined_score":    self.combined_score,
            "observation_count": self.observation_count,
            "combination_context": self.combination_context,
        }


def _build_answer(entity_id: str, category: str, observations: List[C2Observation]) -> str:
    n = len(observations)
    event_types = list({
        (o.event_type.value if hasattr(o.event_type, "value") else str(o.event_type))
        for o in observations
    })
    sources = list({
        (o.source.value if hasattr(o.source, "value") else str(o.source))
        for o in observations
    })

    source_str = ", ".join(sources[:3])
    event_str  = ", ".join(event_types[:3])

    if category == "comms":
        if n > 1:
            return (
                f"Yes — {entity_id} has reported {n} comms events ({event_str}) "
                f"from sources: {source_str}. Pattern suggests systemic degradation, not transient noise."
            )
        return f"Yes — {entity_id} has a single {event_str} event from {source_str}. Monitor for recurrence."

    elif category == "threat":
        if n > 1:
            return (
                f"Yes — {entity_id} is associated with {n} threat indicators ({event_str}) "
                f"across {source_str}. Multi-source confirmation increases reliability."
            )
        return f"Yes — {entity_id} has a single threat indicator: {event_str} via {source_str}."

    elif category == "asset_health":
        if n > 1:
            return (
                f"Yes — {entity_id} is showing {n} compounding health warnings: {event_str}. "
                f"Immediate operator assessment required."
            )
        return f"Yes — {entity_id} has a health warning: {event_str} via {source_str}."

    elif category == "sensor":
        return (
            f"{'Multiple sensor losses' if n > 1 else 'Sensor loss'} detected for {entity_id} "
            f"({event_str}, {source_str}). Area coverage may be degraded."
        )

    else:
        return (
            f"{entity_id} shows {n} {category} observation(s): {event_str} "
            f"from {source_str}."
        )


class C2EvidenceAggregator:
    """
    Aggregates raw C2 observations into evidence-backed insight clusters.

    Usage:
        aggregator = C2EvidenceAggregator()
        clusters = aggregator.aggregate(entity_id, observations)
    """

    def aggregate(
        self,
        observations: List[C2Observation],
        entity_id: str,
    ) -> List[C2EvidenceCluster]:
        if not observations:
            return []

        groups: Dict[str, List[C2Observation]] = defaultdict(list)
        for obs in observations:
            cat = _get_obs_category(obs.event_type)
            groups[cat].append(obs)

        clusters = []
        for category, cat_obs in groups.items():
            if category == "other":
                continue
            cluster = self._build_cluster(entity_id, category, cat_obs)
            if cluster:
                clusters.append(cluster)

        self._apply_combinations(clusters)
        clusters.sort(key=lambda c: c.combined_score, reverse=True)
        return clusters

    def _apply_combinations(self, clusters: List[C2EvidenceCluster]) -> None:
        categories = {c.obs_category for c in clusters}

        COMBINATIONS = [
            ({"comms", "asset_health"}, "Comms + health degradation — possible systemic failure or jamming", 12),
            ({"comms", "threat"},       "Comms disruption + active threat — hostile action suspected", 15),
            ({"threat", "sensor"},      "Threat + sensor loss — possible sensor denial attack", 12),
            ({"asset_health", "mission"}, "Asset degradation + mission state change — operational continuity at risk", 8),
            ({"authority", "comms"},    "Authority change + comms event — verify C2 topology is intact", 10),
        ]

        matched = []
        for combo_cats, label, boost in COMBINATIONS:
            if combo_cats.issubset(categories):
                matched.append((combo_cats, label, boost))

        if not matched:
            return

        matched.sort(key=lambda x: x[2], reverse=True)

        for combo_cats, label, boost in matched:
            for cluster in clusters:
                if cluster.obs_category in combo_cats:
                    if not cluster.combination_context:
                        cluster.combination_context = label
                    cluster.combined_score = min(cluster.combined_score + boost, 100)

    def _build_cluster(
        self,
        entity_id: str,
        category: str,
        observations: List[C2Observation],
    ) -> Optional[C2EvidenceCluster]:
        if not observations:
            return None

        templates = HYPOTHESIS_TEMPLATES.get(category, HYPOTHESIS_TEMPLATES.get("peer", {}))
        if len(observations) > 1:
            template = templates.get("multiple", templates.get("generic", "{entity_id} — {category} events"))
        else:
            template = templates.get("single", templates.get("generic", "{entity_id} — {category} event"))

        hypothesis = template.format(entity_id=entity_id, category=category)
        answer     = _build_answer(entity_id, category, observations)

        evidence = []
        seen_fingerprints = set()
        for obs in observations:
            et_str  = obs.event_type.value if hasattr(obs.event_type, "value") else str(obs.event_type)
            src_str = obs.source.value if hasattr(obs.source, "value") else str(obs.source)
            fp = f"{et_str}:{src_str}"
            if fp in seen_fingerprints:
                continue
            seen_fingerprints.add(fp)
            evidence.append(ObservationEvidence(
                event_type=et_str,
                sensor_source=src_str,
                timestamp=getattr(obs, "detected_at", None),
                entity_id=getattr(obs, "node_id", entity_id),
                confidence=getattr(obs, "confidence", 0.5),
                score=getattr(obs, "score", 50),
                detail=getattr(obs, "description", None),
            ))

        avg_score   = sum(o.score for o in observations if hasattr(o, "score")) / len(observations) if observations else 50
        count_bonus = min(len(observations) * 3, 15)
        combined_score = min(int(avg_score + count_bonus), 100)

        avg_confidence = (
            sum(o.confidence for o in observations if hasattr(o, "confidence")) / len(observations)
            if observations else 0.5
        )
        multi_src_bonus = 0.1 if len(evidence) > 1 else 0
        confidence = min(avg_confidence + multi_src_bonus, 1.0)

        return C2EvidenceCluster(
            hypothesis=hypothesis,
            answer=answer,
            evidence=evidence[:10],
            obs_category=category,
            entity_id=entity_id,
            confidence=round(confidence, 2),
            combined_score=combined_score,
            observation_count=len(observations),
        )
