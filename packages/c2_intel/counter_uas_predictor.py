"""
C2 Counter-UAS Predictor

Binary classifier: is this UAS authorized or rogue/threat?

Usage:
    from c2_intel.counter_uas_predictor import get_counter_uas_predictor

    predictor = get_counter_uas_predictor()
    result = predictor.predict(geofence_compliant=0, operator_id_confirmed=0,
                               flight_pattern=2, distance_to_asset_m=150)
    # result = {"is_rogue": True, "probability": 0.94, "method": "ml"}
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "counter_uas_classifier.joblib"
META_PATH  = MODELS_DIR / "counter_uas_classifier_meta.json"

CLASSES = ["authorized", "rogue_threat"]


class CounterUASPredictor:
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
            f1 = (self._meta or {}).get("metrics", {}).get("f1_cv", "?")
            logger.info("[CounterUASPredictor] Loaded (CV F1: %s)", f1)
        except Exception as e:
            logger.warning("[CounterUASPredictor] Failed to load: %s", e)

    @property
    def is_loaded(self) -> bool:
        self._try_load()
        return self._model is not None

    def predict(
        self,
        altitude_m: float = 100.0,
        speed_mps: float = 10.0,
        rcs_dbsm: float = -20.0,
        rf_power_dbm: float = -75.0,
        flight_pattern: int = 0,
        distance_to_asset_m: float = 1000.0,
        time_active_min: float = 15.0,
        operator_id_confirmed: int = 1,
        geofence_compliant: int = 1,
        payload_detected: int = 0,
    ) -> dict:
        self._try_load()

        features = [[
            float(altitude_m),
            float(speed_mps),
            float(rcs_dbsm),
            float(rf_power_dbm),
            float(flight_pattern),
            float(distance_to_asset_m),
            float(time_active_min),
            float(operator_id_confirmed),
            float(geofence_compliant),
            float(payload_detected),
        ]]

        if self._model is not None:
            try:
                pred = int(self._model.predict(features)[0])
                proba = self._model.predict_proba(features)[0]
                return {
                    "is_rogue": bool(pred),
                    "label": CLASSES[pred],
                    "probability": round(float(proba[1]), 3),
                    "method": "ml",
                }
            except Exception as e:
                logger.warning("[CounterUASPredictor] Prediction failed: %s", e)

        # Heuristic fallback: geofence compliance + operator ID are primary indicators
        is_rogue = (geofence_compliant == 0) or (operator_id_confirmed == 0)
        return {
            "is_rogue": is_rogue,
            "label": CLASSES[1] if is_rogue else CLASSES[0],
            "probability": None,
            "method": "heuristic",
        }

    def get_status(self) -> dict:
        self._try_load()
        return {
            "loaded": self.is_loaded,
            "trained_at": (self._meta or {}).get("trained_at"),
            "f1_cv": (self._meta or {}).get("metrics", {}).get("f1_cv"),
        }


_predictor: Optional[CounterUASPredictor] = None


def get_counter_uas_predictor() -> CounterUASPredictor:
    global _predictor
    if _predictor is None:
        _predictor = CounterUASPredictor()
    return _predictor
