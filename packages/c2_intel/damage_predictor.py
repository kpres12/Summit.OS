"""
C2 Building Damage Predictor

Classifies building damage severity (0-3: no-damage → destroyed) from
tabular features extracted from imagery metadata and sensor data.

Usage:
    from c2_intel.damage_predictor import get_damage_predictor

    predictor = get_damage_predictor()
    if predictor.is_loaded:
        result = predictor.predict(disaster_type="earthquake", sar_incoherence=0.8)
        # result = {"damage_class": 3, "label": "destroyed", "confidence": 0.91, ...}
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "damage_classifier.joblib"
META_PATH  = MODELS_DIR / "damage_classifier_meta.json"

DISASTER_TYPES = [
    "hurricane", "wildfire", "flooding", "earthquake",
    "tsunami", "tornado", "volcano", "other",
]

DAMAGE_LABELS = ["no-damage", "minor-damage", "major-damage", "destroyed"]


class DamagePredictor:
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
            logger.info("[DamagePredictor] Loaded (CV F1-macro: %s)", f1)
        except Exception as e:
            logger.warning("[DamagePredictor] Failed to load: %s", e)

    @property
    def is_loaded(self) -> bool:
        self._try_load()
        return self._model is not None

    def predict(
        self,
        disaster_type: str = "other",
        zone_area_m2: float = 5000.0,
        structure_density: float = 500.0,
        thermal_anomaly: bool = False,
        sar_incoherence: float = 0.5,
        time_since_event_h: float = 12.0,
        pre_event_pop_density: float = 2000.0,
    ) -> dict:
        self._try_load()

        dtype = disaster_type.lower()
        dtype_oh = [1.0 if dt in dtype else 0.0 for dt in DISASTER_TYPES]

        features = dtype_oh + [
            min(zone_area_m2 / 100_000, 1.0),
            min(structure_density / 10_000, 1.0),
            float(thermal_anomaly),
            float(sar_incoherence),
            min(time_since_event_h / 72.0, 1.0),
            min(pre_event_pop_density / 20_000, 1.0),
        ]

        if self._model is not None:
            try:
                import numpy as np
                X = np.array([features], dtype=np.float32)
                pred = int(self._model.predict(X)[0])
                proba = self._model.predict_proba(X)[0]
                return {
                    "damage_class": pred,
                    "label": DAMAGE_LABELS[pred],
                    "confidence": round(float(proba[pred]), 3),
                    "method": "ml",
                }
            except Exception as e:
                logger.warning("[DamagePredictor] Prediction failed: %s", e)

        # Fallback: heuristic
        score = sar_incoherence * 3.0 + float(thermal_anomaly) * 0.5
        pred = min(3, int(score))
        return {"damage_class": pred, "label": DAMAGE_LABELS[pred], "confidence": None, "method": "heuristic"}

    def get_status(self) -> dict:
        self._try_load()
        return {
            "loaded": self.is_loaded,
            "trained_at": (self._meta or {}).get("trained_at"),
            "f1_macro_cv": (self._meta or {}).get("metrics", {}).get("f1_macro_cv"),
        }


_predictor: Optional[DamagePredictor] = None


def get_damage_predictor() -> DamagePredictor:
    global _predictor
    if _predictor is None:
        _predictor = DamagePredictor()
    return _predictor
