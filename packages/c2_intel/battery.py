"""
C2 Battery Degradation Predictor

Predicts minutes remaining until battery reaches critical (15%) and empty (0%)
thresholds using a gradient-boosted model trained on NASA Li-ion discharge data.

Integrates with the c2_intel pipeline to improve BATTERY_CRITICAL/BATTERY_LOW
observation scoring and provide operators with time-to-action estimates.

Usage:
    from c2_intel.battery import get_battery_predictor

    predictor = get_battery_predictor()
    if predictor.is_loaded:
        result = predictor.predict(soc_pct=45.0, discharge_rate_c=1.0)
        # result = {"minutes_to_critical": 18.2, "minutes_to_empty": 27.0, ...}
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODELS_DIR   = Path(__file__).parent / "models"
META_PATH    = MODELS_DIR / "battery_predictor_meta.json"
MODEL_CRIT   = MODELS_DIR / "battery_predictor_critical.joblib"
MODEL_EMPTY  = MODELS_DIR / "battery_predictor_empty.joblib"

# Fallback: simple linear model when ML model isn't trained yet
_FALLBACK_C_RATE = 1.0   # 1C = 60 min full discharge
_CRITICAL_PCT    = 15.0


class BatteryPredictor:
    """
    Predicts time-to-critical and time-to-empty for Li-ion batteries.

    Lazy-loads trained models on first use. Falls back to a physics-based
    linear approximation if models aren't available.
    """

    def __init__(self):
        self._crit_model = None
        self._empty_model = None
        self._meta: Optional[dict] = None
        self._load_attempted = False

    def _try_load(self):
        if self._load_attempted:
            return
        self._load_attempted = True

        if not MODEL_CRIT.exists():
            return

        try:
            import joblib
            self._crit_model  = joblib.load(MODEL_CRIT)
            self._empty_model = joblib.load(MODEL_EMPTY) if MODEL_EMPTY.exists() else None
            if META_PATH.exists():
                with open(META_PATH) as f:
                    self._meta = json.load(f)
            crit_mae = (self._meta or {}).get("metrics", {}).get("minutes_to_critical_mae_min")
            mae_str = f" (MAE: {crit_mae:.1f}min)" if crit_mae else ""
            logger.info("[BatteryPredictor] Loaded battery models%s", mae_str)
        except Exception as e:
            logger.warning("[BatteryPredictor] Failed to load: %s", e)

    @property
    def is_loaded(self) -> bool:
        self._try_load()
        return self._crit_model is not None

    def predict(
        self,
        soc_pct: float,
        discharge_rate_c: float = 1.0,
        temp_celsius: float = 25.0,
        capacity_ratio: float = 1.0,
    ) -> dict:
        """
        Predict battery time-to-critical and time-to-empty.

        Args:
            soc_pct:          Current state of charge (0-100)
            discharge_rate_c: C-rate (1.0 = 60 min full discharge, 2.0 = 30 min)
            temp_celsius:     Ambient temperature (affects discharge rate)
            capacity_ratio:   Relative capacity vs nominal (1.0 = fresh, <1.0 = degraded)

        Returns:
            Dict with minutes_to_critical, minutes_to_empty, method, and urgency.
        """
        self._try_load()

        soc_pct          = max(0.0, min(100.0, float(soc_pct)))
        discharge_rate_c = max(0.1, float(discharge_rate_c))
        temp_celsius     = float(temp_celsius)
        capacity_ratio   = max(0.1, min(1.5, float(capacity_ratio)))

        if self._crit_model is not None:
            return self._predict_ml(soc_pct, discharge_rate_c, temp_celsius, capacity_ratio)
        return self._predict_fallback(soc_pct, discharge_rate_c, temp_celsius, capacity_ratio)

    def _predict_ml(
        self,
        soc_pct: float,
        discharge_rate_c: float,
        temp_celsius: float,
        capacity_ratio: float,
    ) -> dict:
        try:
            import numpy as np
            X = np.array([[soc_pct, discharge_rate_c, temp_celsius, capacity_ratio]],
                         dtype=np.float32)
            t_crit  = max(0.0, float(self._crit_model.predict(X)[0]))
            t_empty = max(t_crit, float(
                self._empty_model.predict(X)[0]
            ) if self._empty_model else t_crit * 1.4)
            return self._format_result(t_crit, t_empty, method="ml")
        except Exception as e:
            logger.warning("[BatteryPredictor] ML prediction failed: %s — falling back", e)
            return self._predict_fallback(soc_pct, discharge_rate_c, temp_celsius, capacity_ratio)

    def _predict_fallback(
        self,
        soc_pct: float,
        discharge_rate_c: float,
        temp_celsius: float,
        capacity_ratio: float,
    ) -> dict:
        # Physics-based approximation
        base_full_min = 60.0 / max(0.1, discharge_rate_c) * capacity_ratio
        temp_factor   = 1.0 - max(0.0, (20.0 - temp_celsius)) * 0.008
        effective_min = base_full_min * temp_factor

        t_empty = effective_min * (soc_pct / 100.0)
        t_crit  = effective_min * max(0.0, soc_pct - _CRITICAL_PCT) / 100.0
        return self._format_result(t_crit, t_empty, method="physics")

    @staticmethod
    def _format_result(t_crit: float, t_empty: float, method: str) -> dict:
        if t_crit <= 5:
            urgency = "critical"
        elif t_crit <= 15:
            urgency = "high"
        elif t_crit <= 30:
            urgency = "medium"
        else:
            urgency = "low"

        return {
            "minutes_to_critical": round(t_crit, 1),
            "minutes_to_empty":    round(t_empty, 1),
            "urgency":             urgency,
            "method":              method,
        }

    def get_status(self) -> dict:
        self._try_load()
        return {
            "loaded":      self.is_loaded,
            "trained_at":  (self._meta or {}).get("trained_at"),
            "model_mae_min": (self._meta or {}).get("metrics", {}).get(
                "minutes_to_critical_mae_min"
            ),
        }


_predictor: Optional[BatteryPredictor] = None


def get_battery_predictor() -> BatteryPredictor:
    global _predictor
    if _predictor is None:
        _predictor = BatteryPredictor()
    return _predictor
