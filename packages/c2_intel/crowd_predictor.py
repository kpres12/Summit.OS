"""
C2 Crowd Density Estimator

Estimates true crowd count from YOLO detection outputs + sensor metadata,
correcting for altitude-induced occlusion and detection rate.

Usage:
    from c2_intel.crowd_predictor import get_crowd_predictor

    predictor = get_crowd_predictor()
    if predictor.is_loaded:
        result = predictor.predict(person_detections=45, altitude_m=80)
        # result = {"estimated_count": 67, "confidence_interval": [52, 84], ...}
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "crowd_estimator.joblib"
META_PATH  = MODELS_DIR / "crowd_estimator_meta.json"

SCENARIO_TYPES = ["disaster", "event", "street", "evacuation", "sar"]


class CrowdPredictor:
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
            mae = (self._meta or {}).get("metrics", {}).get("mae_cv_persons", "?")
            logger.info("[CrowdPredictor] Loaded (CV MAE: %s persons)", mae)
        except Exception as e:
            logger.warning("[CrowdPredictor] Failed to load: %s", e)

    @property
    def is_loaded(self) -> bool:
        self._try_load()
        return self._model is not None

    def predict(
        self,
        person_detections: int = 0,
        frame_coverage: float = 0.3,
        detection_density: float = 0.01,
        mean_bbox_area: float = 2000.0,
        altitude_m: float = 100.0,
        fov_deg: float = 45.0,
        overlap_ratio: float = 0.2,
        time_of_day: float = 0.5,
        thermal_count: Optional[int] = None,
        scenario: str = "disaster",
    ) -> dict:
        self._try_load()

        detections = float(person_detections)
        thermal = float(thermal_count if thermal_count is not None else detections)
        thermal_norm = min(thermal / max(detections + 1, 1), 2.0)
        scen_idx = SCENARIO_TYPES.index(scenario) / 4.0 if scenario in SCENARIO_TYPES else 0.0

        features = [
            detections / 1000.0,
            min(frame_coverage, 1.0),
            min(detection_density, 1.0),
            min(mean_bbox_area / 10000.0, 1.0),
            min(altitude_m / 500.0, 1.0),
            min(fov_deg / 120.0, 1.0),
            min(overlap_ratio, 1.0),
            min(time_of_day, 1.0),
            thermal_norm,
            scen_idx,
        ]

        if self._model is not None:
            try:
                import numpy as np
                X = np.array([features], dtype=np.float32)
                est = max(0.0, float(self._model.predict(X)[0]))
                mae = (self._meta or {}).get("metrics", {}).get("mae_cv_persons", est * 0.2)
                margin = float(mae) * 1.5
                return {
                    "estimated_count": round(est),
                    "confidence_interval": [max(0, round(est - margin)), round(est + margin)],
                    "raw_detections": person_detections,
                    "correction_factor": round(est / max(detections, 1), 2),
                    "method": "ml",
                }
            except Exception as e:
                logger.warning("[CrowdPredictor] Prediction failed: %s", e)

        # Fallback: altitude-based occlusion correction
        alt_factor = 1.0 + min(altitude_m / 500.0, 0.8)
        est = round(detections * alt_factor)
        return {
            "estimated_count": est,
            "confidence_interval": [round(est * 0.7), round(est * 1.4)],
            "raw_detections": person_detections,
            "correction_factor": round(alt_factor, 2),
            "method": "heuristic",
        }

    def get_status(self) -> dict:
        self._try_load()
        return {
            "loaded": self.is_loaded,
            "trained_at": (self._meta or {}).get("trained_at"),
            "mae_cv_persons": (self._meta or {}).get("metrics", {}).get("mae_cv_persons"),
        }


_predictor: Optional[CrowdPredictor] = None


def get_crowd_predictor() -> CrowdPredictor:
    global _predictor
    if _predictor is None:
        _predictor = CrowdPredictor()
    return _predictor
