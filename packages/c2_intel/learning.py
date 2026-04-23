"""
C2 Observation Feedback Learning System

Learns from operator behavior to improve observation relevance:
1. Track which observations operators act on (acknowledged, dispatched)
2. Track which observations operators dismiss (and why)
3. Adjust scoring weights based on engagement patterns
4. Per-operator and global learning

Creates a feedback loop that continuously improves signal accuracy.
Persists to a local JSON file (no external DB dependency).
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR  = Path(__file__).parent / "data"
LEARN_PATH = DATA_DIR / "learning.json"


class FeedbackType(str, Enum):
    """Types of operator feedback on C2 observations."""
    # Positive — operator found this useful
    ACKNOWLEDGED = "acknowledged"   # Operator opened/read the observation
    ACTIONED     = "actioned"       # Operator issued a command in response
    RESOLVED     = "resolved"       # Observation led to resolved situation
    FLAGGED      = "flagged"        # Operator flagged as high priority

    # Negative — operator did not find this useful
    DISMISSED         = "dismissed"          # Dismissed without reason
    IRRELEVANT        = "irrelevant"         # Marked as not relevant to current op
    WRONG_ENTITY      = "wrong_entity"       # Observation linked to wrong entity
    STALE             = "stale"              # Information was outdated
    FALSE_POSITIVE    = "false_positive"     # Observation type was incorrect


class DismissReason(str, Enum):
    OUTSIDE_MISSION    = "outside_mission"
    WRONG_TIMING       = "wrong_timing"
    ALREADY_AWARE      = "already_aware"
    NOT_ACTIONABLE     = "not_actionable"
    LOW_CONFIDENCE     = "low_confidence"
    DUPLICATE          = "duplicate"
    OTHER              = "other"


@dataclass
class ObservationFeedback:
    """A single piece of operator feedback on a C2 observation."""
    observation_id: str
    operator_id: str
    feedback_type: FeedbackType
    dismiss_reason: Optional[DismissReason] = None

    # Observation attributes (for learning)
    event_type:    str   = ""
    sensor_source: str   = ""
    entity_id:     str   = ""
    confidence:    float = 0.0
    score:         int   = 0

    notes: Optional[str]  = None
    timestamp: datetime   = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class LearningMetrics:
    """Metrics for an event type or sensor source."""
    total_observations: int = 0
    actioned_count:     int = 0
    dismissed_count:    int = 0
    resolved_count:     int = 0
    false_positive_count: int = 0

    @property
    def engagement_rate(self) -> float:
        if self.total_observations == 0:
            return 0.0
        return self.actioned_count / self.total_observations

    @property
    def dismissal_rate(self) -> float:
        if self.total_observations == 0:
            return 0.0
        return self.dismissed_count / self.total_observations

    @property
    def resolution_rate(self) -> float:
        if self.total_observations == 0:
            return 0.0
        return self.resolved_count / self.total_observations

    @property
    def false_positive_rate(self) -> float:
        if self.total_observations == 0:
            return 0.0
        return self.false_positive_count / self.total_observations

    @property
    def quality_score(self) -> float:
        if self.total_observations < 5:
            return 0.5
        score = (
            self.engagement_rate  * 0.4 +
            self.resolution_rate  * 0.4 +
            (1 - self.false_positive_rate) * 0.2
        )
        return min(max(score, 0.0), 1.0)


class ObservationFeedbackLearner:
    """
    Learns from operator feedback to improve C2 observation scoring.

    Usage:
        learner = ObservationFeedbackLearner()
        learner.record_feedback(ObservationFeedback(
            observation_id="obs-123",
            operator_id="op-alpha",
            feedback_type=FeedbackType.ACTIONED,
            event_type="THREAT_IDENTIFIED",
            sensor_source="RADAR",
        ))
        adj = learner.get_score_adjustment("THREAT_IDENTIFIED", "RADAR", "op-alpha")
    """

    def __init__(self, min_samples: int = 10):
        self.min_samples = min_samples
        self.hydrated: bool = False

        self.feedback_history: List[ObservationFeedback] = []

        self.type_metrics:   Dict[str, LearningMetrics] = defaultdict(LearningMetrics)
        self.source_metrics: Dict[str, LearningMetrics] = defaultdict(LearningMetrics)

        self.operator_type_metrics:   Dict[str, Dict[str, LearningMetrics]] = defaultdict(
            lambda: defaultdict(LearningMetrics)
        )
        self.operator_source_metrics: Dict[str, Dict[str, LearningMetrics]] = defaultdict(
            lambda: defaultdict(LearningMetrics)
        )

        self.confidence_buckets: Dict[str, Dict[str, List[bool]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def record_feedback(self, feedback: ObservationFeedback):
        self.feedback_history.append(feedback)

        type_m   = self.type_metrics[feedback.event_type]
        source_m = self.source_metrics[feedback.sensor_source]
        op_type  = self.operator_type_metrics[feedback.operator_id][feedback.event_type]
        op_src   = self.operator_source_metrics[feedback.operator_id][feedback.sensor_source]

        for m in (type_m, source_m, op_type, op_src):
            m.total_observations += 1

        if feedback.feedback_type in (FeedbackType.ACKNOWLEDGED, FeedbackType.ACTIONED, FeedbackType.FLAGGED):
            for m in (type_m, source_m, op_type, op_src):
                m.actioned_count += 1

        elif feedback.feedback_type == FeedbackType.RESOLVED:
            for m in (type_m, source_m, op_type, op_src):
                m.actioned_count += 1
                m.resolved_count += 1

        elif feedback.feedback_type in (FeedbackType.DISMISSED, FeedbackType.IRRELEVANT, FeedbackType.STALE):
            for m in (type_m, source_m, op_type, op_src):
                m.dismissed_count += 1

        elif feedback.feedback_type in (FeedbackType.WRONG_ENTITY, FeedbackType.FALSE_POSITIVE):
            for m in (type_m, source_m, op_type, op_src):
                m.false_positive_count += 1

        is_positive = feedback.feedback_type in (
            FeedbackType.ACKNOWLEDGED, FeedbackType.ACTIONED,
            FeedbackType.RESOLVED, FeedbackType.FLAGGED,
        )
        bucket = str(int(feedback.confidence * 10) * 10)
        self.confidence_buckets[feedback.event_type][bucket].append(is_positive)

    def get_score_adjustment(
        self,
        event_type: str,
        sensor_source: str,
        operator_id: str = None,
        confidence: float = None,
    ) -> float:
        """
        Returns multiplier: 1.0 = no change, >1.0 = boost, <1.0 = penalty.
        """
        adjustments = []

        type_m = self.type_metrics.get(event_type)
        if type_m and type_m.total_observations >= self.min_samples:
            adjustments.append(0.5 + type_m.quality_score)

        src_m = self.source_metrics.get(sensor_source)
        if src_m and src_m.total_observations >= self.min_samples:
            adjustments.append(0.5 + src_m.quality_score)

        if operator_id:
            op_type = self.operator_type_metrics.get(operator_id, {}).get(event_type)
            if op_type and op_type.total_observations >= self.min_samples // 2:
                adj = 0.5 + op_type.quality_score
                adjustments.extend([adj, adj])  # user preference weighted 2x

            op_src = self.operator_source_metrics.get(operator_id, {}).get(sensor_source)
            if op_src and op_src.total_observations >= self.min_samples // 2:
                adj = 0.5 + op_src.quality_score
                adjustments.extend([adj, adj])

        if confidence is not None and event_type:
            cal = self._get_confidence_calibration(event_type, confidence)
            if cal is not None:
                adjustments.append(cal)

        if not adjustments:
            return 1.0
        return sum(adjustments) / len(adjustments)

    def _get_confidence_calibration(self, event_type: str, confidence: float) -> Optional[float]:
        bucket   = str(int(confidence * 10) * 10)
        outcomes = self.confidence_buckets.get(event_type, {}).get(bucket, [])
        if len(outcomes) < self.min_samples:
            return None
        actual   = sum(outcomes) / len(outcomes)
        expected = confidence
        cal      = actual / expected if expected > 0 else 1.0
        return max(0.5, min(1.5, cal))

    def apply_learned_adjustments(self, observation: Dict[str, Any], operator_id: str = None) -> Dict[str, Any]:
        event_type     = observation.get("event_type", "")
        sensor_source  = observation.get("sensor_source", "")
        original_score = observation.get("score", 50)
        confidence     = observation.get("confidence", 0.5)

        adjustment = self.get_score_adjustment(
            event_type=event_type,
            sensor_source=sensor_source,
            operator_id=operator_id,
            confidence=confidence,
        )

        adjusted_score = int(original_score * adjustment)
        adjusted_score = max(0, min(100, adjusted_score))

        observation["original_score"]   = original_score
        observation["score"]            = adjusted_score
        observation["learning_adjustment"] = round(adjustment, 3)
        observation["learning_applied"] = adjustment != 1.0

        if adjustment > 1.1:
            observation["learning_note"] = "Score boosted based on operator engagement patterns"
        elif adjustment < 0.9:
            observation["learning_note"] = "Score reduced based on dismissal patterns"

        return observation

    def get_event_type_insights(self, event_type: str) -> Dict[str, Any]:
        m = self.type_metrics.get(event_type, LearningMetrics())
        return {
            "event_type":            event_type,
            "total_observations":    m.total_observations,
            "engagement_rate":       round(m.engagement_rate, 3),
            "dismissal_rate":        round(m.dismissal_rate, 3),
            "resolution_rate":       round(m.resolution_rate, 3),
            "false_positive_rate":   round(m.false_positive_rate, 3),
            "quality_score":         round(m.quality_score, 3),
            "recommended_adjustment": round(self.get_score_adjustment(event_type, ""), 3),
        }

    def export_model(self) -> Dict[str, Any]:
        return {
            "type_metrics": {
                k: {
                    "total":         v.total_observations,
                    "actioned":      v.actioned_count,
                    "dismissed":     v.dismissed_count,
                    "resolved":      v.resolved_count,
                    "false_positives": v.false_positive_count,
                }
                for k, v in self.type_metrics.items()
            },
            "source_metrics": {
                k: {
                    "total":         v.total_observations,
                    "actioned":      v.actioned_count,
                    "dismissed":     v.dismissed_count,
                    "resolved":      v.resolved_count,
                    "false_positives": v.false_positive_count,
                }
                for k, v in self.source_metrics.items()
            },
            "confidence_calibration": {
                evt: {
                    bucket: sum(outcomes) / len(outcomes) if outcomes else 0
                    for bucket, outcomes in buckets.items()
                }
                for evt, buckets in self.confidence_buckets.items()
            },
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

    def import_model(self, model_data: Dict[str, Any]):
        for event_type, data in model_data.get("type_metrics", {}).items():
            m = self.type_metrics[event_type]
            m.total_observations    = data.get("total", 0)
            m.actioned_count        = data.get("actioned", 0)
            m.dismissed_count       = data.get("dismissed", 0)
            m.resolved_count        = data.get("resolved", 0)
            m.false_positive_count  = data.get("false_positives", 0)

        for source, data in model_data.get("source_metrics", {}).items():
            m = self.source_metrics[source]
            m.total_observations    = data.get("total", 0)
            m.actioned_count        = data.get("actioned", 0)
            m.dismissed_count       = data.get("dismissed", 0)
            m.resolved_count        = data.get("resolved", 0)
            m.false_positive_count  = data.get("false_positives", 0)

    def persist(self) -> bool:
        try:
            DATA_DIR.mkdir(exist_ok=True)
            LEARN_PATH.write_text(json.dumps(self.export_model(), indent=2))
            return True
        except Exception as e:
            logger.error("[C2Learner] Persist failed: %s", e)
            return False

    def hydrate(self) -> bool:
        if not LEARN_PATH.exists():
            return False
        try:
            data = json.loads(LEARN_PATH.read_text())
            self.import_model(data)
            self.hydrated = True
            logger.info("[C2Learner] Hydrated from %s", LEARN_PATH)
            return True
        except Exception as e:
            logger.error("[C2Learner] Hydrate failed: %s", e)
            return False


_global_learner: Optional[ObservationFeedbackLearner] = None


def get_learner() -> ObservationFeedbackLearner:
    global _global_learner
    if _global_learner is None:
        _global_learner = ObservationFeedbackLearner()
        _global_learner.hydrate()
    return _global_learner


def record_observation_feedback(
    observation_id: str,
    operator_id: str,
    feedback_type: FeedbackType,
    event_type: str = "",
    sensor_source: str = "",
    dismiss_reason: DismissReason = None,
    **kwargs,
):
    feedback = ObservationFeedback(
        observation_id=observation_id,
        operator_id=operator_id,
        feedback_type=feedback_type,
        event_type=event_type,
        sensor_source=sensor_source,
        dismiss_reason=dismiss_reason,
        **kwargs,
    )
    get_learner().record_feedback(feedback)


__all__ = [
    "ObservationFeedbackLearner",
    "ObservationFeedback",
    "FeedbackType",
    "DismissReason",
    "LearningMetrics",
    "get_learner",
    "record_observation_feedback",
]
