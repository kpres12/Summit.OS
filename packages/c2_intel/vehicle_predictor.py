"""
C2 Vehicle Classification Predictor

Classifies ground vehicles from aerial/sensor data into four classes:
  0 = civilian_passenger
  1 = civilian_commercial
  2 = emergency_services
  3 = military_logistics

Usage:
    from c2_intel.vehicle_predictor import get_vehicle_predictor

    predictor = get_vehicle_predictor()
    result = predictor.predict(speed_mps=25.0, size_class=2, convoy_member=1)
    # result = {"vehicle_class": 3, "label": "military_logistics", "probability": 0.87, "method": "ml"}
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "vehicle_classifier.joblib"
META_PATH  = MODELS_DIR / "vehicle_classifier_meta.json"

CLASSES = ["civilian_passenger", "civilian_commercial", "emergency_services", "military_logistics"]


class VehiclePredictor:
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
            logger.info("[VehiclePredictor] Loaded (CV F1-macro: %s)", f1)
        except Exception as e:
            logger.warning("[VehiclePredictor] Failed to load: %s", e)

    @property
    def is_loaded(self) -> bool:
        self._try_load()
        return self._model is not None

    def predict(
        self,
        speed_mps: float = 15.0,
        heading_change_rate: float = 1.0,
        formation_spacing_m: float = 200.0,
        time_of_day_h: float = 12.0,
        area_type: int = 0,
        convoy_member: int = 0,
        stop_frequency: float = 2.0,
        route_deviation: float = 20.0,
        size_class: int = 1,
        thermal_signature: float = 0.2,
        acoustic_level: float = 0.3,
        payload_indicator: int = 0,
    ) -> dict:
        self._try_load()

        features = [[
            float(speed_mps),
            float(heading_change_rate),
            float(formation_spacing_m),
            float(time_of_day_h),
            float(area_type),
            float(convoy_member),
            float(stop_frequency),
            float(route_deviation),
            float(size_class),
            float(thermal_signature),
            float(acoustic_level),
            float(payload_indicator),
        ]]

        if self._model is not None:
            try:
                pred = int(self._model.predict(features)[0])
                proba = self._model.predict_proba(features)[0]
                return {
                    "vehicle_class": pred,
                    "label": CLASSES[pred],
                    "probability": round(float(proba[pred]), 3),
                    "method": "ml",
                }
            except Exception as e:
                logger.warning("[VehiclePredictor] Prediction failed: %s", e)

        # Heuristic fallback: speed + size + convoy membership
        if convoy_member == 1 and size_class >= 2:
            pred = 3  # military_logistics
        elif acoustic_level > 0.5:
            pred = 2  # emergency_services
        elif size_class >= 1 and payload_indicator == 1:
            pred = 1  # civilian_commercial
        else:
            pred = 0  # civilian_passenger
        return {"vehicle_class": pred, "label": CLASSES[pred], "probability": None, "method": "heuristic"}

    def get_status(self) -> dict:
        self._try_load()
        return {
            "loaded": self.is_loaded,
            "trained_at": (self._meta or {}).get("trained_at"),
            "f1_macro_cv": (self._meta or {}).get("metrics", {}).get("f1_macro_cv"),
        }


_predictor: Optional[VehiclePredictor] = None


def get_vehicle_predictor() -> VehiclePredictor:
    global _predictor
    if _predictor is None:
        _predictor = VehiclePredictor()
    return _predictor
