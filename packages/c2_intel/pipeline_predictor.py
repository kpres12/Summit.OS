"""
C2 Pipeline Anomaly Predictor

Multiclass classifier: normal vs. operational anomaly vs. leak suspected.

Usage:
    from c2_intel.pipeline_predictor import get_pipeline_predictor

    predictor = get_pipeline_predictor()
    result = predictor.predict(pressure_delta_pct=-12.0, flow_delta_pct=18.0,
                               acoustic_db=85.0)
    # result = {"anomaly_class": 2, "label": "leak_suspected", "probability": 0.91, "method": "ml"}
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "pipeline_anomaly_classifier.joblib"
META_PATH  = MODELS_DIR / "pipeline_anomaly_classifier_meta.json"

CLASSES = ["normal", "anomaly_operational", "leak_suspected"]


class PipelinePredictor:
    def __init__(self):
        self._model = None
        self._meta: Optional[dict] = None
        self._load_attempted = False

    def _try_load(self):
        if self._load_attempted:
            return
        self._load_attempted = True
        if not MODEL_PATH.exists():
            return
        try:
            import joblib
            self._model = joblib.load(MODEL_PATH)
            if META_PATH.exists():
                self._meta = json.loads(META_PATH.read_text())
            f1 = (self._meta or {}).get("metrics", {}).get("f1_macro_cv", "?")
            logger.info("[PipelinePredictor] Loaded (CV F1-macro: %s)", f1)
        except Exception as e:
            logger.warning("[PipelinePredictor] Failed to load: %s", e)

    @property
    def is_loaded(self) -> bool:
        self._try_load()
        return self._model is not None

    def predict(
        self,
        pressure_bar: float = 60.0,
        pressure_delta_pct: float = 0.0,
        flow_rate_m3h: float = 250.0,
        flow_delta_pct: float = 0.0,
        temp_delta_c: float = 0.0,
        acoustic_db: float = 55.0,
        cp_mv: float = -850.0,
        wall_loss_pct: float = 2.0,
        time_since_pig_days: float = 90.0,
        segment_age_years: float = 10.0,
    ) -> dict:
        self._try_load()

        features = [[
            float(pressure_bar),
            float(pressure_delta_pct),
            float(flow_rate_m3h),
            float(flow_delta_pct),
            float(temp_delta_c),
            float(acoustic_db),
            float(cp_mv),
            float(wall_loss_pct),
            float(time_since_pig_days),
            float(segment_age_years),
        ]]

        if self._model is not None:
            try:
                pred = int(self._model.predict(features)[0])
                proba = self._model.predict_proba(features)[0]
                return {
                    "anomaly_class": pred,
                    "label": CLASSES[pred],
                    "probability": round(float(proba[pred]), 3),
                    "method": "ml",
                }
            except Exception as e:
                logger.warning("[PipelinePredictor] Prediction failed: %s", e)

        # Heuristic fallback: pressure drop is the primary leak signal
        if pressure_delta_pct < -5.0:
            pred = 2  # leak_suspected
        elif abs(pressure_delta_pct) > 5.0 or acoustic_db > 70:
            pred = 1  # anomaly_operational
        else:
            pred = 0  # normal
        return {
            "anomaly_class": pred,
            "label": CLASSES[pred],
            "probability": None,
            "method": "heuristic",
        }

    def get_status(self) -> dict:
        self._try_load()
        return {
            "loaded": self.is_loaded,
            "trained_at": (self._meta or {}).get("trained_at"),
            "f1_macro_cv": (self._meta or {}).get("metrics", {}).get("f1_macro_cv"),
        }


_predictor: Optional[PipelinePredictor] = None


def get_pipeline_predictor() -> PipelinePredictor:
    global _predictor
    if _predictor is None:
        _predictor = PipelinePredictor()
    return _predictor
