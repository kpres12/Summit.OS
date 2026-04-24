"""
C2 Structural Corrosion / Defect Predictor

Multi-label classification of infrastructure defects from sensor features.
Returns which defect types are likely present and an overall severity score.

Usage:
    from c2_intel.corrosion_predictor import get_corrosion_predictor

    predictor = get_corrosion_predictor()
    if predictor.is_loaded:
        result = predictor.predict(rgb_rust_ratio=0.4, context_type="pipeline")
        # result = {"defects": ["corrosion", "crack"], "severity": 0.33, ...}
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "corrosion_classifier.joblib"
META_PATH  = MODELS_DIR / "corrosion_classifier_meta.json"

DEFECT_CLASSES = ["crack", "spalling", "efflorescence", "exposed_rebar", "corrosion", "delamination"]
CONTEXT_TYPES  = ["bridge", "pipeline", "tank", "steel_beam", "concrete_deck"]


class CorrosionPredictor:
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
            f1 = (self._meta or {}).get("metrics", {}).get("corrosion_f1_cv", "?")
            logger.info("[CorrosionPredictor] Loaded (corrosion F1: %s)", f1)
        except Exception as e:
            logger.warning("[CorrosionPredictor] Failed to load: %s", e)

    @property
    def is_loaded(self) -> bool:
        self._try_load()
        return self._model is not None

    def predict(
        self,
        thermal_delta_c: float = 0.0,
        surface_roughness: float = 0.2,
        rgb_rust_ratio: float = 0.1,
        rgb_crack_score: float = 0.1,
        context_type: str = "concrete_deck",
        age_years: float = 10.0,
        last_inspection_months: float = 12.0,
        environment_marine: bool = False,
    ) -> dict:
        self._try_load()

        ctx = context_type.lower()
        ctx_oh = [1.0 if c in ctx else 0.0 for c in CONTEXT_TYPES]

        features = [
            min(thermal_delta_c / 50.0, 1.0),
            min(surface_roughness, 1.0),
            min(rgb_rust_ratio, 1.0),
            min(rgb_crack_score, 1.0),
        ] + ctx_oh + [
            min(age_years / 100.0, 1.0),
            min(last_inspection_months / 120.0, 1.0),
            float(environment_marine),
        ]

        if self._model is not None:
            try:
                import numpy as np
                X = np.array([features], dtype=np.float32)
                y_pred = self._model.predict(X)[0]
                detected = [DEFECT_CLASSES[i] for i, v in enumerate(y_pred) if v == 1]
                severity = len(detected) / len(DEFECT_CLASSES)
                return {
                    "defects": detected,
                    "defect_vector": [int(v) for v in y_pred],
                    "severity": round(severity, 3),
                    "method": "ml",
                }
            except Exception as e:
                logger.warning("[CorrosionPredictor] Prediction failed: %s", e)

        # Fallback: threshold-based heuristic
        detected = []
        if rgb_rust_ratio > 0.2:
            detected.append("corrosion")
        if rgb_crack_score > 0.25:
            detected.append("crack")
        if thermal_delta_c > 10:
            detected.append("delamination")
        severity = len(detected) / len(DEFECT_CLASSES)
        return {"defects": detected, "defect_vector": None, "severity": round(severity, 3), "method": "heuristic"}

    def get_status(self) -> dict:
        self._try_load()
        return {
            "loaded": self.is_loaded,
            "trained_at": (self._meta or {}).get("trained_at"),
            "corrosion_f1_cv": (self._meta or {}).get("metrics", {}).get("corrosion_f1_cv"),
        }


_predictor: Optional[CorrosionPredictor] = None


def get_corrosion_predictor() -> CorrosionPredictor:
    global _predictor
    if _predictor is None:
        _predictor = CorrosionPredictor()
    return _predictor
