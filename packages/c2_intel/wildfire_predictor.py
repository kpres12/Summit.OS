"""
C2 Wildfire Spread Predictor (LSTM)

Given a 12-step time series of fire observations (FRP, wind, humidity, NDVI),
predicts fire radiative power at the next timestep.

Usage:
    from c2_intel.wildfire_predictor import get_wildfire_predictor

    predictor = get_wildfire_predictor()
    result = predictor.predict(sequence)
    # result = {"frp_predicted_mw": 842.0, "trend": "growing", "method": "lstm"}
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "wildfire_lstm.pt"
META_PATH  = MODELS_DIR / "wildfire_lstm_meta.json"

SEQ_LEN    = 12
N_FEATURES = 9
HIDDEN_DIM = 128
N_LAYERS   = 2
FRP_SCALE  = 5000.0  # normalization factor


class WildfirePredictor:
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
            import torch
            import torch.nn as nn

            class _LSTM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(N_FEATURES, HIDDEN_DIM, N_LAYERS,
                                        batch_first=True, dropout=0.2)
                    self.head = nn.Sequential(
                        nn.Linear(HIDDEN_DIM, 64), nn.ReLU(),
                        nn.Dropout(0.2), nn.Linear(64, 1),
                    )

                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.head(out[:, -1, :]).squeeze(1)

            checkpoint = torch.load(MODEL_PATH, map_location="cpu")
            model = _LSTM()
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            self._model = model
            if META_PATH.exists():
                self._meta = json.loads(META_PATH.read_text())
            mae = (self._meta or {}).get("metrics", {}).get("mae_val_normalized", "?")
            logger.info("[WildfirePredictor] LSTM loaded (val MAE: %s norm FRP)", mae)
        except Exception as e:
            logger.warning("[WildfirePredictor] Failed to load: %s", e)

    @property
    def is_loaded(self) -> bool:
        self._try_load()
        return self._model is not None

    def predict(self, sequence: list[dict]) -> dict:
        """
        Args:
            sequence: List of dicts with keys: frp_mw, wind_speed_ms, wind_dir_deg,
                      rh_pct, temp_c, ndvi, slope_deg, active_pixels.
                      Should be exactly seq_len (12) steps; will be zero-padded if shorter.
        """
        self._try_load()

        seq_len = (self._meta or {}).get("seq_len", SEQ_LEN)
        features = _encode_sequence(sequence, seq_len)

        if self._model is not None:
            try:
                import torch
                X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    pred_norm = float(self._model(X)[0])
                pred_frp = max(0.0, pred_norm * FRP_SCALE)
                last_frp = float(sequence[-1].get("frp_mw", 100)) if sequence else 100.0
                trend = "growing" if pred_frp > last_frp * 1.05 else (
                    "decaying" if pred_frp < last_frp * 0.95 else "stable")
                return {
                    "frp_predicted_mw": round(pred_frp, 1),
                    "trend": trend,
                    "last_frp_mw": round(last_frp, 1),
                    "method": "lstm",
                }
            except Exception as e:
                logger.warning("[WildfirePredictor] LSTM inference failed: %s", e)

        # Fallback: simple exponential based on last two readings
        if len(sequence) >= 2:
            f1 = float(sequence[-2].get("frp_mw", 100))
            f2 = float(sequence[-1].get("frp_mw", 100))
            ratio = f2 / max(f1, 1)
            pred_frp = f2 * ratio
        elif sequence:
            pred_frp = float(sequence[-1].get("frp_mw", 100))
        else:
            pred_frp = 100.0
        return {"frp_predicted_mw": round(pred_frp, 1), "trend": "unknown",
                "last_frp_mw": None, "method": "extrapolation"}

    def get_status(self) -> dict:
        self._try_load()
        return {
            "loaded": self.is_loaded,
            "trained_at": (self._meta or {}).get("trained_at"),
            "mae_normalized": (self._meta or {}).get("metrics", {}).get("mae_val_normalized"),
        }


def _encode_sequence(sequence: list[dict], seq_len: int) -> list[list[float]]:
    rows = []
    for step in sequence[-seq_len:]:
        frp = float(step.get("frp_mw", 100))
        ws  = float(step.get("wind_speed_ms", 5))
        wd  = math.radians(float(step.get("wind_dir_deg", 0)))
        rh  = float(step.get("rh_pct", 40))
        tc  = float(step.get("temp_c", 25))
        ndvi = float(step.get("ndvi", 0.5))
        slope = float(step.get("slope_deg", 5))
        ap  = float(step.get("active_pixels", 10))
        rows.append([
            frp / FRP_SCALE,
            ap / 200.0,
            ws / 25.0,
            math.sin(wd),
            math.cos(wd),
            rh / 100.0,
            (tc + 10) / 65.0,
            ndvi,
            min(slope / 45.0, 1.0),
        ])
    # Zero-pad if shorter than seq_len
    while len(rows) < seq_len:
        rows.insert(0, [0.0] * N_FEATURES)
    return rows


_predictor: Optional[WildfirePredictor] = None


def get_wildfire_predictor() -> WildfirePredictor:
    global _predictor
    if _predictor is None:
        _predictor = WildfirePredictor()
    return _predictor
