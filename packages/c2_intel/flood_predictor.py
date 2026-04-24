"""
C2 Flood Extent Predictor

Binary classifier: is this area flooded? Uses SAR backscatter + optical indices
+ terrain features extracted from satellite data.

Usage:
    from c2_intel.flood_predictor import get_flood_predictor

    predictor = get_flood_predictor()
    result = predictor.predict(sar_vv_mean=-18.0, ndwi=0.4, slope_deg=1.5)
    # result = {"is_flooded": True, "probability": 0.93, "method": "ml"}
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "flood_classifier.joblib"
META_PATH  = MODELS_DIR / "flood_classifier_meta.json"


class FloodPredictor:
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
            logger.info("[FloodPredictor] Loaded (CV F1: %s)", f1)
        except Exception as e:
            logger.warning("[FloodPredictor] Failed to load: %s", e)

    @property
    def is_loaded(self) -> bool:
        self._try_load()
        return self._model is not None

    def predict(
        self,
        sar_vv_mean: float = -15.0,
        sar_vh_mean: float = -20.0,
        sar_vv_std: float = 2.5,
        ndwi: float = 0.0,
        ndvi: float = 0.3,
        blue_mean: float = 0.1,
        slope_deg: float = 5.0,
        dem_m: float = 50.0,
    ) -> dict:
        self._try_load()

        features = [
            sar_vv_mean / -30.0,
            sar_vh_mean / -30.0,
            min(sar_vv_std / 5.0, 1.0),
            (ndwi + 1) / 2.0,
            (ndvi + 1) / 2.0,
            float(blue_mean),
            min(slope_deg / 30.0, 1.0),
            min(dem_m / 500.0, 1.0),
        ]

        if self._model is not None:
            try:
                import numpy as np
                X = [[float(f) for f in features]]
                pred = int(self._model.predict(X)[0])
                proba = self._model.predict_proba(X)[0]
                return {
                    "is_flooded": bool(pred),
                    "probability": round(float(proba[1]), 3),
                    "method": "ml",
                }
            except Exception as e:
                logger.warning("[FloodPredictor] Prediction failed: %s", e)

        # Fallback: NDWI + SAR heuristic
        flooded = ndwi > 0.2 or sar_vv_mean < -18
        return {"is_flooded": flooded, "probability": None, "method": "heuristic"}

    def get_status(self) -> dict:
        self._try_load()
        return {
            "loaded": self.is_loaded,
            "trained_at": (self._meta or {}).get("trained_at"),
            "f1_cv": (self._meta or {}).get("metrics", {}).get("f1_cv"),
        }


_predictor: Optional[FloodPredictor] = None


def get_flood_predictor() -> FloodPredictor:
    global _predictor
    if _predictor is None:
        _predictor = FloodPredictor()
    return _predictor
