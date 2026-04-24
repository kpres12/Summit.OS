"""
C2 Slope Stability Predictor

Multiclass classifier for mining/geohazard slope stability assessment:
  0 = stable
  1 = watch    (monitor closely)
  2 = critical (evacuate/halt operations)

Usage:
    from c2_intel.slope_predictor import get_slope_predictor

    predictor = get_slope_predictor()
    result = predictor.predict(displacement_mm_day=25.0, displacement_accel=1.2,
                               pore_pressure_kpa=200.0, saturation_pct=85.0)
    # result = {"stability_class": 2, "label": "critical", "probability": 0.96, "method": "ml"}
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "slope_stability_classifier.joblib"
META_PATH  = MODELS_DIR / "slope_stability_classifier_meta.json"

CLASSES = ["stable", "watch", "critical"]


class SlopePredictor:
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
            logger.info("[SlopePredictor] Loaded (CV F1-macro: %s)", f1)
        except Exception as e:
            logger.warning("[SlopePredictor] Failed to load: %s", e)

    @property
    def is_loaded(self) -> bool:
        self._try_load()
        return self._model is not None

    def predict(
        self,
        displacement_mm_day: float = 0.5,
        displacement_accel: float = 0.0,
        crack_width_mm: float = 1.0,
        pore_pressure_kpa: float = 30.0,
        rainfall_mm_24h: float = 5.0,
        slope_angle_deg: float = 35.0,
        material_type: int = 0,
        saturation_pct: float = 25.0,
        vibration_ppv: float = 20.0,
        historical_movement_mm: float = 10.0,
    ) -> dict:
        self._try_load()

        features = [[
            float(displacement_mm_day),
            float(displacement_accel),
            float(crack_width_mm),
            float(pore_pressure_kpa),
            float(rainfall_mm_24h),
            float(slope_angle_deg),
            float(material_type),
            float(saturation_pct),
            float(vibration_ppv),
            float(historical_movement_mm),
        ]]

        if self._model is not None:
            try:
                pred = int(self._model.predict(features)[0])
                proba = self._model.predict_proba(features)[0]
                return {
                    "stability_class": pred,
                    "label": CLASSES[pred],
                    "probability": round(float(proba[pred]), 3),
                    "method": "ml",
                }
            except Exception as e:
                logger.warning("[SlopePredictor] Prediction failed: %s", e)

        # Heuristic fallback: displacement rate is the primary geotechnical indicator
        if displacement_mm_day > 10.0 or displacement_accel > 0.5:
            pred = 2  # critical
        elif displacement_mm_day > 1.0 or pore_pressure_kpa > 100:
            pred = 1  # watch
        else:
            pred = 0  # stable
        return {
            "stability_class": pred,
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


_predictor: Optional[SlopePredictor] = None


def get_slope_predictor() -> SlopePredictor:
    global _predictor
    if _predictor is None:
        _predictor = SlopePredictor()
    return _predictor
