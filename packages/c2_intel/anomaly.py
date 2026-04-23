"""
C2 Observation Anomaly Detector

Scores how anomalous an observation is relative to that entity's baseline activity.
Anomalous = unusually high event rate = MORE significant, not less.

Usage:
    from c2_intel.anomaly import get_anomaly_detector
    boost = get_anomaly_detector().get_anomaly_boost(event_type, obs_last_5m, ...)
    # boost = 0-15 points added to base score
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
META_PATH  = MODELS_DIR / "anomaly_detector_meta.json"

MAX_BOOST = 15


class C2AnomalyDetector:
    def __init__(self):
        self._models: Dict[str, object] = {}
        self._global_model = None
        self._meta = None
        self._load_attempted = False

    def _try_load(self):
        if self._load_attempted:
            return
        self._load_attempted = True

        if not META_PATH.exists():
            return

        try:
            import joblib

            with open(META_PATH) as f:
                self._meta = json.load(f)

            event_types = self._meta.get("event_types", [])
            loaded = 0
            for ev_type in event_types:
                path = MODELS_DIR / f"anomaly_detector_{ev_type}.joblib"
                if path.exists():
                    self._models[ev_type] = joblib.load(path)
                    loaded += 1

            global_path = MODELS_DIR / "anomaly_detector_global.joblib"
            if global_path.exists():
                self._global_model = joblib.load(global_path)

            if loaded > 0:
                logger.info("[C2AnomalyDetector] Loaded %d per-type models + global", loaded)
        except Exception as e:
            logger.warning("[C2AnomalyDetector] Failed to load: %s", e)

    @property
    def is_loaded(self) -> bool:
        self._try_load()
        return bool(self._models)

    def get_anomaly_boost(
        self,
        event_type: str,
        obs_last_5m: int = 0,
        obs_last_30m: int = 0,
        obs_last_90m: int = 0,
        seconds_since_last: int = 300,
    ) -> int:
        """
        Returns 0-15 point score boost based on anomaly level.

        Higher = more anomalous relative to this entity's baseline.
        0 = normal activity pattern, 15 = extreme event spike.

        Args:
            event_type: C2EventType string (e.g. "COMMS_DEGRADED")
            obs_last_5m: Observations of this type in last 5 minutes
            obs_last_30m: Observations of this type in last 30 minutes
            obs_last_90m: Observations of this type in last 90 minutes
            seconds_since_last: Seconds since last observation of this type
        """
        self._try_load()
        if not self._models:
            return 0

        model = self._models.get(event_type) or self._global_model
        if model is None:
            return 0

        try:
            import numpy as np

            velocity = obs_last_30m / max(obs_last_90m / 3, 0.1)
            X = np.array([[
                float(obs_last_5m),
                float(obs_last_30m),
                float(obs_last_90m),
                float(velocity),
                float(seconds_since_last),
            ]])

            # decision_function: negative = anomalous, positive = normal
            score = float(model.decision_function(X)[0])

            if score >= 0:
                return 0
            boost = int(min(MAX_BOOST, abs(score) * MAX_BOOST * 2))
            return boost

        except Exception as e:
            logger.warning("[C2AnomalyDetector] Scoring failed: %s", e)
            return 0

    def _reload(self):
        self._load_attempted = False
        self._models = {}
        self._global_model = None
        self._meta = None
        self._try_load()

    def get_status(self) -> dict:
        self._try_load()
        return {
            "loaded": self.is_loaded,
            "models_loaded": len(self._models),
            "global_model": self._global_model is not None,
            "trained_at": self._meta.get("trained_at") if self._meta else None,
        }


_detector: Optional[C2AnomalyDetector] = None


def get_anomaly_detector() -> C2AnomalyDetector:
    global _detector
    if _detector is None:
        _detector = C2AnomalyDetector()
    return _detector
