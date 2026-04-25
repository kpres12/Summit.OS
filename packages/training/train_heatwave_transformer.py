"""
Heat-Wave Forecast Transformer

Trains a small Transformer encoder on daily weather sequences to predict
the probability of an "extreme heat day" in each of the next 1, 3, and 7
days. Extreme heat = daily temperature_2m_max above the 95th percentile
of the location's full-year distribution (region-specific definition,
matches NWS HeatRisk methodology in spirit).

Per-day features:
  temperature_2m_max, temperature_2m_min, apparent_temperature_max,
  relative_humidity_2m_min, wind_speed_10m_max,
  vapour_pressure_deficit_max, heat_index_estimate

Training data: ~30 US heat-prone city centroids × 730 days (2 years) of
Open-Meteo historical archive. Each city contributes sliding 14-day
windows. Output: 3 binary labels (heat in next 1d, 3d, 7d).

Civilian use:
  CDC / public-health heat-vulnerability alerting,
  power-utility load forecasting,
  outdoor-worker safety planning.

Output: packages/c2_intel/models/heatwave_transformer.pt
        packages/c2_intel/models/heatwave_transformer_meta.json

Usage:
    python train_heatwave_transformer.py [--start 2023-01-01 --end 2024-12-31]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "c2_intel" / "models"
MODEL_PATH = MODELS_DIR / "heatwave_transformer.pt"
META_PATH  = MODELS_DIR / "heatwave_transformer_meta.json"
CACHE_DIR  = Path(__file__).parent / "data" / "heatwave"

OPENMETEO_HIST = "https://archive-api.open-meteo.com/v1/archive"

SEQ_LEN     = 14         # 14 days of context
N_FEATURES  = 7
D_MODEL     = 64
N_HEADS     = 4
N_LAYERS    = 2
N_HORIZONS  = 3          # 1d, 3d, 7d horizons

# Heat-prone or heat-vulnerable US cities (lat, lon)
CITIES: dict[str, tuple[float, float]] = {
    "Phoenix":       (33.448, -112.074),
    "Las Vegas":     (36.169, -115.140),
    "Tucson":        (32.222, -110.974),
    "El Paso":       (31.762, -106.485),
    "Houston":       (29.760, -95.370),
    "Dallas":        (32.776, -96.796),
    "Austin":        (30.267, -97.743),
    "San Antonio":   (29.424, -98.494),
    "New Orleans":   (29.951, -90.071),
    "Atlanta":       (33.749, -84.388),
    "Miami":         (25.762, -80.192),
    "Tampa":         (27.951, -82.458),
    "Charleston":    (32.776, -79.931),
    "Memphis":       (35.149, -90.049),
    "Birmingham":    (33.519, -86.810),
    "Oklahoma City": (35.469, -97.516),
    "Kansas City":   (39.099, -94.578),
    "St Louis":      (38.627, -90.199),
    "Chicago":       (41.878, -87.629),
    "Cincinnati":    (39.103, -84.512),
    "Sacramento":    (38.581, -121.494),
    "Fresno":        (36.737, -119.787),
    "Bakersfield":   (35.373, -119.018),
    "Boise":         (43.615, -116.202),
    "Salt Lake":     (40.760, -111.891),
    "Denver":        (39.739, -104.990),
    "Albuquerque":   (35.084, -106.651),
    "Nashville":     (36.162, -86.781),
    "Louisville":    (38.252, -85.758),
    "Washington":    (38.907, -77.037),
}


def _fetch_city_weather(city: str, lat: float, lon: float,
                        start: str, end: str,
                        max_retries: int = 3) -> dict | None:
    """Fetch daily weather for a city centroid, with backoff on 429."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"wx_{city.replace(' ', '_')}_{start}_{end}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass

    backoff = 5.0
    for attempt in range(max_retries):
        try:
            r = requests.get(OPENMETEO_HIST, params={
                "latitude": lat, "longitude": lon,
                "start_date": start, "end_date": end,
                "daily": ",".join([
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "apparent_temperature_max",
                    "relative_humidity_2m_min",
                    "wind_speed_10m_max",
                    "vapour_pressure_deficit_max",
                    "et0_fao_evapotranspiration",
                ]),
                "timezone": "UTC",
            }, timeout=120)
            if r.status_code == 429:
                logger.warning("[Heat] %s 429 — backing off %.0fs", city, backoff)
                time.sleep(backoff)
                backoff *= 2
                continue
            r.raise_for_status()
            data = r.json().get("daily") or {}
            if not data.get("time"):
                return None
            cache.write_text(json.dumps(data))
            return data
        except Exception as e:
            logger.warning("[Heat] %s fetch failed (try %d): %s", city, attempt + 1, e)
            time.sleep(backoff)
            backoff *= 2
    return None


def _heat_index_estimate(t_max: float, rh_min: float) -> float:
    """Rothfusz heat index approximation (°C input, °C-equivalent output)."""
    if t_max is None or rh_min is None or t_max < 26.7:
        return t_max if t_max is not None else 0.0
    # Convert to Fahrenheit for the canonical Rothfusz formula
    t_f = t_max * 9 / 5 + 32
    rh = rh_min
    hi_f = (
        -42.379 + 2.04901523 * t_f + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 6.83783e-3 * t_f * t_f
        - 5.481717e-2 * rh * rh
        + 1.22874e-3 * t_f * t_f * rh
        + 8.5282e-4 * t_f * rh * rh
        - 1.99e-6 * t_f * t_f * rh * rh
    )
    return (hi_f - 32) * 5 / 9


def _build_dataset(start: str, end: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (N, SEQ_LEN, N_FEATURES) sequences and (N, N_HORIZONS) targets."""
    feat_names = [
        "t_max_norm", "t_min_norm", "apparent_t_max_norm",
        "rh_min_norm", "wind_max_norm", "vpd_max_norm",
        "heat_index_norm",
    ]

    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    n_cities = 0

    for city, (lat, lon) in CITIES.items():
        wx = _fetch_city_weather(city, lat, lon, start, end)
        time.sleep(0.4)  # be polite
        if not wx or not wx.get("time"):
            continue

        t_max = np.array(wx.get("temperature_2m_max") or [], dtype=np.float32)
        t_min = np.array(wx.get("temperature_2m_min") or [], dtype=np.float32)
        at_max = np.array(wx.get("apparent_temperature_max") or [], dtype=np.float32)
        rh_min = np.array(wx.get("relative_humidity_2m_min") or [], dtype=np.float32)
        wind_max = np.array(wx.get("wind_speed_10m_max") or [], dtype=np.float32)
        vpd_max = np.array(wx.get("vapour_pressure_deficit_max") or [], dtype=np.float32)
        et0 = np.array(wx.get("et0_fao_evapotranspiration") or [], dtype=np.float32)

        N = len(t_max)
        if N < SEQ_LEN + 8:
            continue

        # Replace NaN with reasonable defaults
        t_max = np.nan_to_num(t_max, nan=25.0)
        t_min = np.nan_to_num(t_min, nan=15.0)
        at_max = np.nan_to_num(at_max, nan=t_max)
        rh_min = np.nan_to_num(rh_min, nan=50.0)
        wind_max = np.nan_to_num(wind_max, nan=10.0)
        vpd_max = np.nan_to_num(vpd_max, nan=1.5)
        et0 = np.nan_to_num(et0, nan=4.0)

        # City-specific extreme-heat threshold = 95th percentile of t_max
        city_thresh = float(np.percentile(t_max, 95))

        # Build heat-index per day
        hi = np.array([_heat_index_estimate(float(t_max[i]), float(rh_min[i]))
                       for i in range(N)], dtype=np.float32)

        # Normalize features
        feats = np.stack([
            t_max / 50.0,
            t_min / 50.0,
            at_max / 55.0,
            rh_min / 100.0,
            wind_max / 25.0,
            vpd_max / 5.0,
            hi / 50.0,
        ], axis=1).astype(np.float32)

        # Sliding window: window [i-SEQ_LEN+1 .. i] predicts horizons after i
        for i in range(SEQ_LEN - 1, N - 7):
            window = feats[i - SEQ_LEN + 1:i + 1]
            # Targets: any extreme-heat day in next 1d, 3d, 7d
            future_t = t_max[i + 1:i + 8]
            target = np.array([
                1.0 if (future_t[:1] >= city_thresh).any() else 0.0,  # 1d
                1.0 if (future_t[:3] >= city_thresh).any() else 0.0,  # 3d
                1.0 if (future_t[:7] >= city_thresh).any() else 0.0,  # 7d
            ], dtype=np.float32)

            X_list.append(window)
            Y_list.append(target)

        n_cities += 1
        logger.info("[Heat] %s -> %d windows (thresh=%.1fC)", city,
                    N - SEQ_LEN - 6, city_thresh)

    if not X_list:
        return np.zeros((0, SEQ_LEN, N_FEATURES), dtype=np.float32), np.zeros((0, N_HORIZONS), dtype=np.float32), feat_names

    X = np.stack(X_list)
    Y = np.stack(Y_list)
    pos_rates = Y.mean(axis=0)
    logger.info("[Heat] Built %d sequences from %d cities. Pos rates per horizon: 1d=%.3f 3d=%.3f 7d=%.3f",
                len(X), n_cities, pos_rates[0], pos_rates[1], pos_rates[2])
    return X, Y, feat_names


def train(start: str, end: str, epochs: int = 30, batch_size: int = 64,
          device_str: str = "auto") -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if device_str == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("[Heat] Device: %s", device)

    X, Y, feat_names = _build_dataset(start, end)
    if len(X) < 200:
        raise RuntimeError(f"Only {len(X)} sequences — too few.")

    rng = np.random.default_rng(7)
    order = rng.permutation(len(X))
    X, Y = X[order], Y[order]
    split = int(len(X) * 0.8)
    X_train, X_val = torch.tensor(X[:split]), torch.tensor(X[split:])
    Y_train, Y_val = torch.tensor(Y[:split]), torch.tensor(Y[split:])

    train_loader = DataLoader(TensorDataset(X_train, Y_train),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(X_val, Y_val),
                              batch_size=batch_size, shuffle=False, num_workers=0)

    class _Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(N_FEATURES, D_MODEL)
            self.pos_embed = nn.Parameter(torch.zeros(1, SEQ_LEN, D_MODEL))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=128,
                dropout=0.1, batch_first=True, activation="gelu"
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
            self.head = nn.Sequential(
                nn.LayerNorm(D_MODEL),
                nn.Linear(D_MODEL, 32),
                nn.GELU(),
                nn.Linear(32, N_HORIZONS),
            )

        def forward(self, x):
            # x: [B, SEQ_LEN, N_FEATURES]
            h = self.input_proj(x) + self.pos_embed
            h = self.encoder(h)
            # Use the last timestep as summary
            return self.head(h[:, -1, :])

    model = _Transformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
        if labels.sum() == 0 or labels.sum() == len(labels):
            return 0.5
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)
        pos = ranks[labels > 0.5]
        n_pos = len(pos)
        n_neg = len(labels) - n_pos
        return float((pos.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    best_auc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            optimizer.zero_grad()
            criterion(model(X_b), Y_b).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        scores_h = [[] for _ in range(N_HORIZONS)]
        labels_h = [[] for _ in range(N_HORIZONS)]
        with torch.no_grad():
            for X_b, Y_b in val_loader:
                logits = torch.sigmoid(model(X_b.to(device))).cpu().numpy()
                for h in range(N_HORIZONS):
                    scores_h[h].extend(logits[:, h].tolist())
                    labels_h[h].extend(Y_b[:, h].tolist())
        aucs = [_auc(np.array(s), np.array(l)) for s, l in zip(scores_h, labels_h)]
        avg_auc = float(np.mean(aucs))
        logger.info("[Heat] Epoch %d/%d  auc 1d=%.3f 3d=%.3f 7d=%.3f (avg=%.3f)",
                    epoch + 1, epochs, aucs[0], aucs[1], aucs[2], avg_auc)
        if avg_auc > best_auc:
            best_auc = avg_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "seq_len": SEQ_LEN,
        "n_features": N_FEATURES,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "n_horizons": N_HORIZONS,
    }, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "Transformer encoder",
        "task": "Multi-horizon extreme-heat probability (1d/3d/7d)",
        "seq_len_days": SEQ_LEN,
        "n_features": N_FEATURES,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "horizons_days": [1, 3, 7],
        "n_epochs": epochs,
        "n_samples": int(len(X)),
        "n_cities": int(len(set(c for c in CITIES))),
        "data_source": "openmeteo_archive_real",
        "period": f"{start} -> {end}",
        "features": feat_names,
        "extreme_heat_definition": "city-specific p95 of daily t_max over training period",
        "metrics": {
            "auc_val_1d": round(aucs[0], 4),
            "auc_val_3d": round(aucs[1], 4),
            "auc_val_7d": round(aucs[2], 4),
            "auc_val_avg_best": round(best_auc, 4),
        },
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[Heat] Saved -> %s (avg AUC %.3f)", MODEL_PATH, best_auc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--start",  default="2023-01-01")
    p.add_argument("--end",    default="2024-12-31")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    train(start=args.start, end=args.end, epochs=args.epochs, device_str=args.device)
