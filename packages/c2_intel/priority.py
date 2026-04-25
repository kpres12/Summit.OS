"""
C2 Priority Matrix

Ported from Mira Signals' priority_matrix.py.
Domain remapped: buying signal priority → C2 condition/authority priority.

Composite promotion logic (unchanged from Mira):
  P3 + P3 (same node, same window) → P2
  P3 + P2 (same node)              → P1
  Multiple P2s (diverse types)     → P1 CRITICAL

In C2 context this means:
  COMMS_DEGRADED (P2) + BATTERY_CRITICAL (P2) on same node → P1 (node about to be lost)
  ENTITY_DETECTED (P2) + GEOFENCE_BREACH (P1) → P1 CRITICAL (confirmed intruder)
  Multiple ASSET_OFFLINE events in same sector → escalate to command

This composite logic is the core of TA1 simulation:
  "Given current conditions, what priority level would this authority delegation trigger?"
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple
from collections import defaultdict

from .models import (
    C2Observation, C2EventType, ObservationPriority, C2ActionType
)


# Base priority for each event type
CONDITION_BASE_PRIORITY: Dict[C2EventType, ObservationPriority] = {
    # P1 CRITICAL — immediate operational impact
    C2EventType.COMMS_DENIED:         ObservationPriority.CRITICAL,
    C2EventType.THREAT_IDENTIFIED:    ObservationPriority.CRITICAL,
    C2EventType.AUTHORITY_DELEGATED:  ObservationPriority.CRITICAL,
    C2EventType.AUTHORITY_REVOKED:    ObservationPriority.CRITICAL,
    C2EventType.NODE_FAILED:          ObservationPriority.CRITICAL,
    C2EventType.GEOFENCE_BREACH:      ObservationPriority.CRITICAL,
    C2EventType.MISSION_ABORTED:      ObservationPriority.CRITICAL,

    # P2 HIGH — urgent attention
    C2EventType.COMMS_DEGRADED:       ObservationPriority.HIGH,
    C2EventType.BATTERY_CRITICAL:     ObservationPriority.HIGH,
    C2EventType.ASSET_OFFLINE:        ObservationPriority.HIGH,
    C2EventType.ASSET_DEGRADED:       ObservationPriority.HIGH,
    C2EventType.NODE_DEGRADED:        ObservationPriority.HIGH,
    C2EventType.TASK_FAILED:          ObservationPriority.HIGH,
    C2EventType.ENTITY_DETECTED:      ObservationPriority.HIGH,
    C2EventType.RESOURCE_DEPLETED:    ObservationPriority.HIGH,

    # P3 MEDIUM — monitor
    C2EventType.BATTERY_WARNING:      ObservationPriority.MEDIUM,
    C2EventType.COMMS_RESTORED:       ObservationPriority.MEDIUM,
    C2EventType.NODE_RECOVERED:       ObservationPriority.MEDIUM,
    C2EventType.ASSET_ONLINE:         ObservationPriority.MEDIUM,
    C2EventType.MISSION_COMPLETED:    ObservationPriority.MEDIUM,
    C2EventType.ENTITY_LOST:          ObservationPriority.MEDIUM,
    C2EventType.ENTITY_REACQUIRED:    ObservationPriority.MEDIUM,

    # P4 LOW — ambient
    C2EventType.TASK_CREATED:         ObservationPriority.LOW,
    C2EventType.TASK_APPROVED:        ObservationPriority.LOW,
    C2EventType.MISSION_ASSIGNED:     ObservationPriority.LOW,
    C2EventType.MISSION_STARTED:      ObservationPriority.LOW,
    C2EventType.NODE_JOINED_MESH:     ObservationPriority.LOW,
    C2EventType.NODE_LEFT_MESH:       ObservationPriority.LOW,
}

# Score thresholds for priority override
SCORE_THRESHOLDS = {
    "critical": 85,
    "high": 70,
    "medium": 50,
}

# High-volume event types that need composite confirmation before reaching P1
BULK_EVENT_TYPES = {
    C2EventType.ENTITY_DETECTED,
    C2EventType.NODE_JOINED_MESH,
    C2EventType.NODE_LEFT_MESH,
    C2EventType.TASK_CREATED,
    C2EventType.TASK_APPROVED,
}

_PRIORITY_RANK = {
    ObservationPriority.CRITICAL: 0,
    ObservationPriority.HIGH: 1,
    ObservationPriority.MEDIUM: 2,
    ObservationPriority.LOW: 3,
}


class C2PriorityMatrix:
    """
    C2 condition priority and composite scoring engine.

    Single observation → priority + recommended actions.
    Multiple observations on same node → composite priority with promotion rules.

    Also serves as the TA1 simulation layer:
    call score_node_observations() with a hypothetical condition set
    to preview what priority level would result before committing.

    Usage:
        matrix = C2PriorityMatrix()

        # Score a single observation
        priority, actions = matrix.score_observation(obs)

        # Score all observations for a node (composite logic)
        result = matrix.score_node_observations("node-alpha", observations)
        print(result["composite_priority"])
        print(result["promotion_reason"])

        # Preview: "what if comms also degrades?"
        hypothetical = observations + [simulated_comms_obs]
        preview = matrix.score_node_observations("node-alpha", hypothetical)
    """

    def __init__(
        self,
        composite_window_seconds: float = 300.0,  # 5-minute window (vs Mira's 7-day)
        p1_promotion_threshold: int = 2,
    ):
        self.composite_window = timedelta(seconds=composite_window_seconds)
        self.p1_threshold = p1_promotion_threshold

    def get_base_priority(self, obs: C2Observation) -> ObservationPriority:
        evt = obs.event_type
        if isinstance(evt, str):
            try:
                evt = C2EventType(evt)
            except ValueError:
                return ObservationPriority.LOW
        return CONDITION_BASE_PRIORITY.get(evt, ObservationPriority.LOW)

    def score_observation(
        self, obs: C2Observation
    ) -> Tuple[ObservationPriority, List[C2ActionType]]:
        """Score a single observation and return (priority, actions)."""
        priority = self.get_base_priority(obs)

        evt = obs.event_type
        if isinstance(evt, str):
            try:
                evt = C2EventType(evt)
            except ValueError:
                evt = None

        # High-volume types need composite confirmation for P1
        if obs.score >= SCORE_THRESHOLDS["critical"]:
            if evt in BULK_EVENT_TYPES:
                priority = ObservationPriority.HIGH
            else:
                priority = ObservationPriority.CRITICAL
        elif obs.score >= SCORE_THRESHOLDS["high"]:
            if priority == ObservationPriority.LOW:
                priority = ObservationPriority.MEDIUM

        return priority, self._actions_for_priority(priority)

    def _actions_for_priority(self, priority: ObservationPriority) -> List[C2ActionType]:
        if priority == ObservationPriority.CRITICAL:
            return [
                C2ActionType.ESCALATE_COMMAND,
                C2ActionType.SURFACE_TO_OPERATOR,
                C2ActionType.BROADCAST_MESH,
                C2ActionType.AUTO_TASK,
            ]
        elif priority == ObservationPriority.HIGH:
            return [
                C2ActionType.SURFACE_TO_OPERATOR,
                C2ActionType.GENERATE_BRIEF,
                C2ActionType.BROADCAST_MESH,
            ]
        elif priority == ObservationPriority.MEDIUM:
            return [C2ActionType.LOG_ONLY]
        else:
            return [C2ActionType.LOG_ONLY]

    def score_node_observations(
        self,
        node_id: str,
        observations: List[C2Observation],
    ) -> Dict[str, Any]:
        """
        Score all observations for a node with composite promotion logic.

        This is also the TA1 simulation entry point — pass a hypothetical
        observation set to preview what priority level results before
        pushing any authority changes to the decentralized framework.

        Returns:
            {
                node_id, composite_priority, observations (scored),
                actions, composite_score, is_critical_compound,
                promotion_reason, event_counts
            }
        """
        if not observations:
            return {
                "node_id": node_id,
                "composite_priority": ObservationPriority.LOW,
                "observations": [],
                "actions": [],
                "composite_score": 0,
                "is_critical_compound": False,
                "promotion_reason": None,
            }

        now = datetime.now(timezone.utc)
        scored = []
        p1_count = p2_count = p3_count = 0
        recent: List[Dict] = []

        for obs in observations:
            priority, actions = self.score_observation(obs)
            scored.append({"observation": obs, "priority": priority, "actions": actions})

            if priority == ObservationPriority.CRITICAL:
                p1_count += 1
            elif priority == ObservationPriority.HIGH:
                p2_count += 1
            else:
                p3_count += 1

            obs_time = obs.event_time or obs.detected_at
            if obs_time:
                if obs_time.tzinfo is None:
                    obs_time = obs_time.replace(tzinfo=timezone.utc)
                if (now - obs_time) <= self.composite_window:
                    recent.append({"observation": obs, "priority": priority})

        # --- Composite promotion (same logic as Mira, C2 domain) ---
        composite_priority = ObservationPriority.LOW
        promotion_reason = None
        is_critical_compound = False

        if p1_count > 0:
            composite_priority = ObservationPriority.CRITICAL
            promotion_reason = f"{p1_count} critical condition(s) active"

        elif p2_count >= self.p1_threshold:
            unique_p2_types = {
                (s["observation"].event_type.value
                 if hasattr(s["observation"].event_type, "value")
                 else str(s["observation"].event_type))
                for s in scored if s["priority"] == ObservationPriority.HIGH
            }
            if len(unique_p2_types) >= 2:
                composite_priority = ObservationPriority.CRITICAL
                is_critical_compound = True
                promotion_reason = (
                    f"Compound critical: {p2_count} high-priority conditions across "
                    f"{len(unique_p2_types)} types ({', '.join(sorted(unique_p2_types))})"
                )
            else:
                composite_priority = ObservationPriority.HIGH
                promotion_reason = f"{p2_count} high conditions (same type — needs diversity for compound critical)"

        elif p2_count > 0 and p3_count > 0:
            recent_p2 = any(s["priority"] == ObservationPriority.HIGH for s in recent)
            recent_p3 = any(s["priority"] in (ObservationPriority.MEDIUM, ObservationPriority.LOW) for s in recent)
            p2_types = {s["observation"].event_type for s in scored if s["priority"] == ObservationPriority.HIGH}
            p3_types = {s["observation"].event_type for s in scored
                        if s["priority"] in (ObservationPriority.MEDIUM, ObservationPriority.LOW)}
            cross_diverse = bool(p2_types) and bool(p3_types - p2_types)

            if recent_p2 and recent_p3 and cross_diverse:
                composite_priority = ObservationPriority.CRITICAL
                promotion_reason = "High + medium conditions within window (e.g., comms degraded + battery warning = imminent loss)"
            else:
                composite_priority = ObservationPriority.HIGH

        elif p3_count >= 2 and len([s for s in recent
                                    if s["priority"] in (ObservationPriority.MEDIUM, ObservationPriority.LOW)]) >= 2:
            composite_priority = ObservationPriority.HIGH
            promotion_reason = f"Multiple ambient conditions ({p3_count}) within window"

        elif p2_count > 0:
            composite_priority = ObservationPriority.HIGH

        elif p3_count > 0:
            composite_priority = ObservationPriority.MEDIUM if p3_count >= 2 else ObservationPriority.LOW

        composite_score = self._composite_score(observations, composite_priority)
        actions = self._actions_for_priority(composite_priority)

        return {
            "node_id": node_id,
            "composite_priority": composite_priority,
            "observations": scored,
            "actions": actions,
            "composite_score": composite_score,
            "is_critical_compound": is_critical_compound,
            "promotion_reason": promotion_reason,
            "event_counts": {"p1": p1_count, "p2": p2_count, "p3": p3_count},
        }

    def _composite_score(
        self, observations: List[C2Observation], composite_priority: ObservationPriority
    ) -> int:
        if not observations:
            return 0
        base = sum(o.score for o in observations) / len(observations)
        priority_bonus = {
            ObservationPriority.CRITICAL: 30,
            ObservationPriority.HIGH: 15,
            ObservationPriority.MEDIUM: 5,
            ObservationPriority.LOW: 0,
        }
        unique_types = len(set(o.event_type for o in observations))
        diversity_bonus = min(unique_types * 5, 20)
        now = datetime.now(timezone.utc)
        recent_count = sum(
            1 for o in observations
            if o.detected_at and (now - o.detected_at).total_seconds() <= 300
        )
        recency_bonus = min(recent_count * 3, 15)
        return min(int(base + priority_bonus.get(composite_priority, 0) + diversity_bonus + recency_bonus), 100)

    def get_urgent_nodes(
        self,
        all_observations: List[C2Observation],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return top N most urgent nodes sorted by composite priority."""
        by_node: Dict[str, List[C2Observation]] = defaultdict(list)
        for obs in all_observations:
            if obs.node_id:
                by_node[obs.node_id].append(obs)

        scored = []
        for node_id, obs_list in by_node.items():
            scored.append(self.score_node_observations(node_id, obs_list))

        scored.sort(key=lambda x: (
            _PRIORITY_RANK.get(x["composite_priority"], 99),
            not x["is_critical_compound"],
            -x["composite_score"],
        ))
        return scored[:limit]

    def simulate(
        self,
        node_id: str,
        current_observations: List[C2Observation],
        hypothetical: List[C2Observation],
    ) -> Dict[str, Any]:
        """
        TA1 simulation: preview composite priority before committing a change.

        Returns both current and projected states so the operator can
        compare before pushing to the decentralized framework.
        """
        current = self.score_node_observations(node_id, current_observations)
        projected = self.score_node_observations(node_id, current_observations + hypothetical)
        priority_changed = current["composite_priority"] != projected["composite_priority"]
        return {
            "node_id": node_id,
            "current": current,
            "projected": projected,
            "priority_changed": priority_changed,
            "escalates": _PRIORITY_RANK.get(projected["composite_priority"], 99) < _PRIORITY_RANK.get(current["composite_priority"], 99),
        }


__all__ = [
    "C2PriorityMatrix",
    "CONDITION_BASE_PRIORITY",
    "SCORE_THRESHOLDS",
]
