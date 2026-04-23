"""
C2 Observation Relevance Model

Trained gradient-boosted classifier that predicts observation relevance score.
Drop-in replacement for hardcoded base scores in the priority matrix.

Instead of static {COMMS_DEGRADED: 40, THREAT_IDENTIFIED: 50, ...},
this model returns a learned score based on operator engagement history.

Usage:
    from c2_intel.relevance import get_relevance_model

    model = get_relevance_model()
    if model.is_loaded:
        base = model.predict_base_score(event_type, sensor_source, confidence, priority)
    else:
        base = CONDITION_BASE_SCORES.get(event_type, 30)  # fallback
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "c2_relevance_classifier.joblib"
META_PATH  = MODELS_DIR / "c2_relevance_classifier_meta.json"

PRIORITY_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

# Score range: 10-55 base score (same as priority matrix fallback range)
SCORE_MIN = 10
SCORE_MAX = 55


class C2RelevanceModel:
    """
    Wraps the trained gradient-boosted relevance classifier for C2 observations.

    Lazy-loads from disk on first use. Falls back gracefully if not trained yet.
    """

    def __init__(self):
        self._clf = None
        self._meta = None
        self._loaded = False
        self._load_attempted = False

    def _try_load(self):
        if self._load_attempted:
            return
        self._load_attempted = True

        if not MODEL_PATH.exists():
            logger.debug("[C2RelevanceModel] No trained model at %s — using base scores", MODEL_PATH)
            return

        try:
            import joblib
            self._clf = joblib.load(MODEL_PATH)
            if META_PATH.exists():
                with open(META_PATH) as f:
                    self._meta = json.load(f)
            self._loaded = True
            auc = self._meta.get("cv_roc_auc_mean") if self._meta else None
            auc_str = f" (CV ROC-AUC: {auc:.3f})" if auc else ""
            logger.info("[C2RelevanceModel] Loaded trained model%s", auc_str)
        except Exception as e:
            logger.warning("[C2RelevanceModel] Failed to load: %s — falling back to base scores", e)

    @property
    def is_loaded(self) -> bool:
        self._try_load()
        return self._loaded

    @property
    def trained_at(self) -> Optional[str]:
        self._try_load()
        return self._meta.get("trained_at") if self._meta else None

    @property
    def cv_roc_auc(self) -> Optional[float]:
        self._try_load()
        return self._meta.get("cv_roc_auc_mean") if self._meta else None

    def predict_base_score(
        self,
        event_type: str,
        sensor_source: str,
        confidence: float = 0.5,
        priority: str = "MEDIUM",
        is_dismissed: bool = False,
    ) -> Optional[int]:
        """
        Predict base relevance score (10-55) for a C2 observation.

        Args:
            event_type: C2EventType string (e.g. "COMMS_DEGRADED")
            sensor_source: SensorSource string (e.g. "MAVLINK")
            confidence: Sensor confidence 0.0-1.0
            priority: ObservationPriority string
            is_dismissed: Whether operator has previously dismissed this type

        Returns:
            Score int, or None if model not loaded (caller uses base scores).
        """
        self._try_load()
        if not self._loaded or self._clf is None or self._meta is None:
            return None

        try:
            import numpy as np

            event_types    = self._meta.get("event_types", [])
            sensor_sources = self._meta.get("sensor_sources", [])
            source_idx     = {s: i for i, s in enumerate(sensor_sources)}

            type_vec = [0.0] * len(event_types)
            if event_type in event_types:
                type_vec[event_types.index(event_type)] = 1.0

            src_enc      = float(source_idx.get(sensor_source, len(sensor_sources)))
            priority_enc = float(PRIORITY_MAP.get(str(priority).upper(), 1))
            conf         = max(0.0, min(1.0, float(confidence)))
            dismissed    = 1.0 if is_dismissed else 0.0

            features = np.array(
                [type_vec + [src_enc, priority_enc, conf, dismissed]],
                dtype=np.float32,
            )

            prob  = float(self._clf.predict_proba(features)[0][1])
            score = int(SCORE_MIN + prob * (SCORE_MAX - SCORE_MIN))
            return score

        except Exception as e:
            logger.warning("[C2RelevanceModel] Prediction failed: %s", e)
            return None

    def _reload(self):
        self._load_attempted = False
        self._loaded = False
        self._clf = None
        self._meta = None
        self._try_load()

    def get_status(self) -> dict:
        self._try_load()
        return {
            "loaded": self._loaded,
            "model_path": str(MODEL_PATH),
            "model_exists": MODEL_PATH.exists(),
            "trained_at": self.trained_at,
            "cv_roc_auc": self.cv_roc_auc,
        }


_model: Optional[C2RelevanceModel] = None


def get_relevance_model() -> C2RelevanceModel:
    global _model
    if _model is None:
        _model = C2RelevanceModel()
    return _model
