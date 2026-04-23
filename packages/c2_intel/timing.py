"""
C2 Action Timing Predictor

Predicts how many minutes typically pass between an observation firing and a
command being issued, broken down by event type and operational context.

Returns the action window: [p25_minutes, median_minutes, p75_minutes]
Augments seeded priors in C2TimingEngine with ML-learned predictions.

Usage:
    from c2_intel.timing import get_timing_predictor

    predictor = get_timing_predictor()
    window = predictor.predict_window("COMMS_DEGRADED", context="wildfire", score=75)
    # window = {"p25": 2, "median": 5, "p75": 12, "recommendation": "..."}
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"

# C2EventType string values
C2_EVENT_TYPES = [
    "COMMS_DEGRADED", "COMMS_RESTORED", "THREAT_IDENTIFIED", "THREAT_NEUTRALIZED",
    "ASSET_OFFLINE", "ASSET_ONLINE", "AUTHORITY_DELEGATED", "AUTHORITY_REVOKED",
    "MISSION_STARTED", "MISSION_COMPLETED", "MISSION_ABORTED",
    "SENSOR_LOSS", "SENSOR_RESTORED", "GEOFENCE_BREACH", "GEOFENCE_CLEARED",
    "ENGAGEMENT_AUTHORIZED", "ENGAGEMENT_DENIED", "ENGAGEMENT_COMPLETE",
    "BATTERY_CRITICAL", "BATTERY_LOW", "HANDOFF_INITIATED", "HANDOFF_COMPLETE",
    "NODE_DEGRADED", "NODE_FAILED", "NODE_RECOVERED",
    "WEATHER_ALERT", "AIRSPACE_CONFLICT", "LINK_DEGRADED", "LINK_LOST",
    "PEER_OBSERVATION",
]

# Operational contexts (replaces Mira's VERTICALS)
C2_CONTEXTS = [
    "urban_sar",           # Urban search and rescue
    "wildfire",            # Wildfire operations
    "disaster_response",   # General disaster response
    "military_ace",        # Agile Combat Employment
    "border_patrol",       # Border/perimeter surveillance
    "other",
]


class C2TimingPredictor:
    """
    Predicts operator action timing windows for C2 observations.

    When loaded with a trained model: returns learned predictions.
    When not trained: falls back to None (caller uses seeded priors from C2TimingEngine).
    """

    def __init__(self):
        self._median = None
        self._p25 = None
        self._p75 = None
        self._meta = None
        self._load_attempted = False

    def _try_load(self):
        if self._load_attempted:
            return
        self._load_attempted = True

        median_path = MODELS_DIR / "c2_timing_predictor.joblib"
        if not median_path.exists():
            return

        try:
            import joblib
            self._median = joblib.load(median_path)
            self._p25    = joblib.load(MODELS_DIR / "c2_timing_predictor_p25.joblib")
            self._p75    = joblib.load(MODELS_DIR / "c2_timing_predictor_p75.joblib")

            meta_path = MODELS_DIR / "c2_timing_predictor_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    self._meta = json.load(f)

            mae = self._meta.get("cv_mae_mean") if self._meta else None
            mae_str = f" (MAE: {mae:.1f}min)" if mae else ""
            logger.info("[C2TimingPredictor] Loaded timing models%s", mae_str)
        except Exception as e:
            logger.warning("[C2TimingPredictor] Failed to load: %s", e)

    @property
    def is_loaded(self) -> bool:
        self._try_load()
        return self._median is not None

    def _build_features(self, event_type: str, context: str, score: int, n_obs: int, urgency_tier: int):
        type_vec = [1.0 if t == event_type else 0.0 for t in C2_EVENT_TYPES]
        ctx_vec  = [1.0 if c == context else 0.0 for c in C2_CONTEXTS]
        return type_vec + ctx_vec + [float(score), float(n_obs), float(urgency_tier)]

    def predict_window(
        self,
        event_type: str,
        context: str = "other",
        score: int = 50,
        n_obs: int = 1,
        urgency_tier: int = 1,
    ) -> Optional[Dict]:
        """
        Predict operator action timing window in minutes.

        Args:
            event_type: C2EventType string
            context: Operational context
            score: Composite observation score (0-100)
            n_obs: Number of concurrent observations of this type
            urgency_tier: 1=routine, 2=elevated, 3=urgent, 4=critical

        Returns:
            Dict with p25, median, p75 in minutes + human recommendation, or None.
        """
        self._try_load()
        if not self.is_loaded:
            return None

        try:
            import numpy as np

            ctx = context if context in C2_CONTEXTS else "other"
            evt = event_type if event_type in C2_EVENT_TYPES else "PEER_OBSERVATION"

            X = np.array(
                [self._build_features(evt, ctx, score, n_obs, urgency_tier)],
                dtype=np.float32,
            )

            p25    = max(1, int(self._p25.predict(X)[0]))
            median = max(p25, int(self._median.predict(X)[0]))
            p75    = max(median, int(self._p75.predict(X)[0]))

            if median <= 3:
                urgency   = "critical"
                advice    = f"Immediate action required — command expected within {p25}-{median} minutes."
            elif median <= 10:
                urgency   = "high"
                advice    = f"Action window: {p25}-{p75} minutes from observation."
            else:
                urgency   = "medium"
                advice    = f"Monitor and respond within {p25}-{p75} minutes."

            return {
                "p25_minutes":   p25,
                "median_minutes": median,
                "p75_minutes":   p75,
                "urgency":       urgency,
                "recommendation": advice,
                "event_type":    event_type,
                "context":       ctx,
            }

        except Exception as e:
            logger.warning("[C2TimingPredictor] Prediction failed: %s", e)
            return None

    def predict_batch(self, observations: List[Dict]) -> List[Optional[Dict]]:
        return [
            self.predict_window(
                event_type=o.get("event_type", "PEER_OBSERVATION"),
                context=o.get("context", "other"),
                score=o.get("score", 50),
                n_obs=o.get("n_obs", 1),
            )
            for o in observations
        ]

    def _reload(self):
        self._load_attempted = False
        self._median = None
        self._p25 = None
        self._p75 = None
        self._meta = None
        self._try_load()

    def get_status(self) -> dict:
        self._try_load()
        return {
            "loaded": self.is_loaded,
            "trained_at": self._meta.get("trained_at") if self._meta else None,
            "cv_mae_minutes": self._meta.get("cv_mae_mean") if self._meta else None,
        }


_predictor: Optional[C2TimingPredictor] = None


def get_timing_predictor() -> C2TimingPredictor:
    global _predictor
    if _predictor is None:
        _predictor = C2TimingPredictor()
    return _predictor
