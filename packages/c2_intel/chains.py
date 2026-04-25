"""
C2 Signal Chain Detector

Detects where an asset/node is in a known event chain and predicts what's next.

When COMMS_DEGRADED fires, this predicts NODE_FAILED in 2-8 minutes with 80% confidence.
When BATTERY_CRITICAL fires, this predicts ASSET_OFFLINE in 3-10 minutes with 90% confidence.

This is the foundation of the CANVAS TA1 preview layer — predict the cascade
before committing authority changes to the decentralized framework.

Usage:
    from c2_intel.chains import C2ChainDetector, get_chain_detector

    detector = get_chain_detector()
    predictions = detector.predict(observations)
    window = detector.get_action_window(predictions)
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .models import C2Observation, C2EventType


def _et(val) -> str:
    """Normalize C2EventType to string."""
    return val.value if hasattr(val, "value") else str(val)


@dataclass
class C2PredictedEvent:
    """A predicted future event based on observed C2 signal chain."""
    event:          str         # What we predict will happen
    event_type:     str         # C2EventType string of predicted event
    minutes_from_now: tuple     # (min_minutes, max_minutes) expected window
    confidence:     float       # 0-1 how confident we are
    consequence:    str         # What this means for the mission
    triggered_by:   List[str]   # Which observed event types triggered this


# Known C2 event chains: when we see event A, predict event B in N minutes
C2_SIGNAL_CHAINS: List[Dict[str, Any]] = [
    {
        "name": "Comms Degradation → Node Failure",
        "trigger_types": [C2EventType.COMMS_DEGRADED, C2EventType.LINK_DEGRADED],
        "predictions": [
            {
                "event": "Node failure or isolation",
                "event_type": "NODE_FAILED",
                "minutes": (2, 8),
                "confidence": 0.80,
                "consequence": "Node may lose two-way comms. Initiate authority delegation or fallback procedure.",
            },
            {
                "event": "Mission continuity risk",
                "event_type": "MISSION_ABORTED",
                "minutes": (5, 20),
                "confidence": 0.55,
                "consequence": "If node is mission-critical, abort or re-route mission assets.",
            },
        ],
    },
    {
        "name": "Link Loss → Authority Cascade",
        "trigger_types": [C2EventType.LINK_LOST, C2EventType.COMMS_DEGRADED],
        "min_observations": 2,
        "predictions": [
            {
                "event": "Autonomous fallback activation",
                "event_type": "AUTHORITY_DELEGATED",
                "minutes": (0, 2),
                "confidence": 0.90,
                "consequence": "Asset transitions to pre-authorized autonomous behavior. Review ROE compliance.",
            },
            {
                "event": "Node enters degraded-comms mode",
                "event_type": "NODE_DEGRADED",
                "minutes": (1, 5),
                "confidence": 0.85,
                "consequence": "Node operates on last received orders. Restore link or issue RTB.",
            },
        ],
    },
    {
        "name": "Battery Critical → Asset Offline",
        "trigger_types": [C2EventType.BATTERY_CRITICAL],
        "predictions": [
            {
                "event": "Forced RTB or emergency landing",
                "event_type": "ASSET_OFFLINE",
                "minutes": (3, 10),
                "confidence": 0.90,
                "consequence": "Asset will go offline. Issue RTB command now or accept coverage gap.",
            },
            {
                "event": "Coverage gap at current position",
                "event_type": "SENSOR_LOSS",
                "minutes": (5, 15),
                "confidence": 0.75,
                "consequence": "Sensor coverage lost at asset position. Re-task nearby asset if available.",
            },
        ],
    },
    {
        "name": "Battery Low → Pre-RTB Window",
        "trigger_types": [C2EventType.BATTERY_LOW],
        "predictions": [
            {
                "event": "Battery critical threshold",
                "event_type": "BATTERY_CRITICAL",
                "minutes": (5, 15),
                "confidence": 0.85,
                "consequence": "Battery critical incoming. Plan RTB or handoff now while asset is still controllable.",
            },
        ],
    },
    {
        "name": "Threat Identified → Engagement Decision Window",
        "trigger_types": [C2EventType.THREAT_IDENTIFIED],
        "predictions": [
            {
                "event": "Engagement authorization request",
                "event_type": "ENGAGEMENT_AUTHORIZED",
                "minutes": (0, 5),
                "confidence": 0.70,
                "consequence": "Operator decision required within OODA window. Delay risks threat escalation.",
            },
            {
                "event": "Threat escalation or movement",
                "event_type": "THREAT_IDENTIFIED",
                "minutes": (2, 10),
                "confidence": 0.60,
                "consequence": "Unengaged threats typically reposition. Update track continuously.",
            },
        ],
    },
    {
        "name": "Sensor Loss → Intelligence Gap",
        "trigger_types": [C2EventType.SENSOR_LOSS, C2EventType.ASSET_OFFLINE],
        "predictions": [
            {
                "event": "Blind spot in operational area",
                "event_type": "SENSOR_LOSS",
                "minutes": (0, 2),
                "confidence": 0.95,
                "consequence": "Area now unmonitored. Re-task adjacent sensor or accept intelligence gap.",
            },
            {
                "event": "Threat maneuver during sensor gap",
                "event_type": "THREAT_IDENTIFIED",
                "minutes": (5, 30),
                "confidence": 0.40,
                "consequence": "Threats exploit sensor gaps. Restore coverage or notify upstream.",
            },
        ],
    },
    {
        "name": "Geofence Breach → Intercept Window",
        "trigger_types": [C2EventType.GEOFENCE_BREACH],
        "predictions": [
            {
                "event": "Engagement authorization window",
                "event_type": "ENGAGEMENT_AUTHORIZED",
                "minutes": (0, 3),
                "confidence": 0.75,
                "consequence": "Breaching entity in restricted airspace. ROE decision required immediately.",
            },
            {
                "event": "Escalation to higher authority",
                "event_type": "HANDOFF_INITIATED",
                "minutes": (1, 5),
                "confidence": 0.50,
                "consequence": "If engagement not authorized, escalate to next authority tier.",
            },
        ],
    },
    {
        "name": "Node Failure → Mission Re-planning",
        "trigger_types": [C2EventType.NODE_FAILED, C2EventType.NODE_DEGRADED],
        "predictions": [
            {
                "event": "Mission task reassignment required",
                "event_type": "MISSION_ABORTED",
                "minutes": (0, 5),
                "confidence": 0.70,
                "consequence": "Affected mission tasks must be redistributed to remaining capable nodes.",
            },
            {
                "event": "Authority delegation to peer node",
                "event_type": "AUTHORITY_DELEGATED",
                "minutes": (1, 8),
                "confidence": 0.65,
                "consequence": "Surviving nodes may inherit authority. Validate new C2 topology.",
            },
        ],
    },
    {
        "name": "Authority Delegated → Handoff Completion",
        "trigger_types": [C2EventType.AUTHORITY_DELEGATED],
        "predictions": [
            {
                "event": "Handoff acknowledgment",
                "event_type": "HANDOFF_COMPLETE",
                "minutes": (0, 3),
                "confidence": 0.80,
                "consequence": "Receiving node must confirm authority transfer. Watch for acknowledgment timeout.",
            },
            {
                "event": "Peer mesh update propagation",
                "event_type": "PEER_OBSERVATION",
                "minutes": (0, 1),
                "confidence": 0.90,
                "consequence": "CRDT mesh will propagate authority change. Verify convergence.",
            },
        ],
    },
    {
        "name": "Multiple Asset Offline → Force Degradation",
        "trigger_types": [C2EventType.ASSET_OFFLINE, C2EventType.NODE_FAILED],
        "min_observations": 2,
        "predictions": [
            {
                "event": "Force structure below minimum viable threshold",
                "event_type": "MISSION_ABORTED",
                "minutes": (0, 10),
                "confidence": 0.75,
                "consequence": "Multiple simultaneous losses suggest systemic failure or adversarial jamming. Escalate.",
            },
            {
                "event": "Comms check on surviving assets",
                "event_type": "COMMS_RESTORED",
                "minutes": (2, 15),
                "confidence": 0.60,
                "consequence": "Surviving assets may be comms-degraded too. Issue check-in across all nodes.",
            },
        ],
    },
]


class C2ChainDetector:
    """
    Detects where a node/asset is in a known C2 event chain and predicts what's next.
    """

    def predict(self, observations: List[C2Observation]) -> List[C2PredictedEvent]:
        """Predict future C2 events based on observed observations."""
        predictions: List[C2PredictedEvent] = []
        seen_events: set = set()

        type_strs = set(_et(o.event_type) for o in observations)
        type_counts: Dict[str, int] = {}
        for o in observations:
            et = _et(o.event_type)
            type_counts[et] = type_counts.get(et, 0) + 1

        for chain in C2_SIGNAL_CHAINS:
            trigger_strs = set(_et(t) for t in chain["trigger_types"])
            if not (type_strs & trigger_strs):
                continue

            min_obs = chain.get("min_observations", 1)
            trigger_count = sum(type_counts.get(_et(t), 0) for t in chain["trigger_types"])
            if trigger_count < min_obs:
                continue

            trigger_titles = [
                _et(o.event_type) for o in observations
                if _et(o.event_type) in trigger_strs
            ][:3]

            for pred in chain["predictions"]:
                event_key = pred["event"]
                if event_key in seen_events:
                    continue
                seen_events.add(event_key)

                predictions.append(C2PredictedEvent(
                    event=pred["event"],
                    event_type=pred["event_type"],
                    minutes_from_now=pred["minutes"],
                    confidence=pred["confidence"],
                    consequence=pred["consequence"],
                    triggered_by=trigger_titles,
                ))

        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions

    def get_action_window(self, predictions: List[C2PredictedEvent]) -> Optional[str]:
        """Return human-readable action window based on highest-confidence prediction."""
        if not predictions:
            return None

        best = None
        for pred in predictions:
            if pred.confidence >= 0.70:
                if best is None or pred.minutes_from_now[0] < best.minutes_from_now[0]:
                    best = pred

        if not best:
            best = predictions[0]

        min_m, max_m = best.minutes_from_now
        if min_m == 0:
            return f"Act immediately — {best.event.lower()} imminent (within {max_m} minutes)"
        elif min_m <= 3:
            return f"Act within {min_m}-{max_m} minutes — {best.event.lower()} expected"
        else:
            return f"Act within {min_m} minutes — {best.event.lower()} expected by minute {max_m}"

    def chain_summary(self, observations: List[C2Observation]) -> Dict[str, Any]:
        """Return a structured summary of detected chains and predictions."""
        predictions = self.predict(observations)
        return {
            "chain_count":   len(predictions),
            "top_prediction": predictions[0].__dict__ if predictions else None,
            "action_window":  self.get_action_window(predictions),
            "all_predictions": [p.__dict__ for p in predictions],
        }


_detector: Optional[C2ChainDetector] = None


def get_chain_detector() -> C2ChainDetector:
    global _detector
    if _detector is None:
        _detector = C2ChainDetector()
    return _detector
