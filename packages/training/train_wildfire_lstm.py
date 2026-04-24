"""
Wildfire Spread LSTM Trainer

Trains a sequence-to-one LSTM to predict fire intensity at T+1h from a
12-step (12-hour) time series of satellite fire observations.

Input sequence features (per timestep):
  - frp_mw: fire radiative power (MW) — from FIRMS VIIRS/MODIS
  - active_pixels: count of active fire pixels in AOI
  - wind_speed_ms: surface wind speed
  - wind_dir_sin, wind_dir_cos: wind direction (circular encoding)
  - rh_pct: relative humidity (0-100)
  - temp_c: ambient temperature
  - ndvi: vegetation index (-1 to 1, proxy for fuel load)
  - slope_deg: terrain slope (fire climbs slopes faster)

Output: frp_next_mw — predicted fire radiative power at next timestep

Output: packages/c2_intel/models/wildfire_lstm.pt
        packages/c2_intel/models/wildfire_lstm_meta.json

Usage:
    python train_wildfire_lstm.py [--epochs 30] [--seq-len 12] [--device auto]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "c2_intel" / "models"
MODEL_PATH = MODELS_DIR / "wildfire_lstm.pt"
META_PATH  = MODELS_DIR / "wildfire_lstm_meta.json"

SEQ_LEN    = 12
N_FEATURES = 9
HIDDEN_DIM = 128
N_LAYERS   = 2


def _synthetic_sequences(n: int = 5000, seq_len: int = SEQ_LEN):
    """
    Synthetic wildfire time series. Fires grow exponentially under high wind/low RH,
    decay under rain conditions, and accelerate on slopes. Physics-grounded noise.
    """
    rng = np.random.default_rng(42)
    X_all, y_all = [], []

    for _ in range(n):
        # Sample a fire scenario
        base_frp   = rng.uniform(10, 2000)
        wind_speed = rng.uniform(0, 25)
        rh         = rng.uniform(10, 80)
        temp_c     = rng.uniform(15, 45)
        ndvi       = rng.uniform(0.1, 0.9)
        slope      = rng.uniform(0, 35)
        wind_dir   = rng.uniform(0, 2 * math.pi)

        # Growth factor: high wind + low RH + steep slope → fast spread
        growth_rate = (wind_speed / 25.0) * (1.0 - rh / 100.0) * (1 + slope / 70.0) * ndvi
        growth_rate = growth_rate * rng.uniform(0.7, 1.3)  # stochasticity

        seq = []
        frp = base_frp
        for t in range(seq_len + 1):
            noise = rng.normal(1.0, 0.08)
            # Wind perturbation each step
            wd_t = wind_dir + rng.normal(0, 0.2)
            seq.append([
                frp / 5000.0,                       # frp_mw (normalized)
                rng.integers(1, 200) / 200.0,       # active_pixels
                wind_speed / 25.0,                  # wind_speed_ms
                math.sin(wd_t),                     # wind_dir_sin
                math.cos(wd_t),                     # wind_dir_cos
                rh / 100.0,                         # rh_pct
                (temp_c + 10) / 65.0,               # temp_c
                ndvi,                               # ndvi
                slope / 45.0,                       # slope_deg
            ])
            # Evolve fire
            frp = max(0.1, frp * (1 + growth_rate * 0.05) * noise)
            if rh > 70:  # rain suppresses
                frp *= 0.85

        seq_arr = np.array(seq, dtype=np.float32)
        X_all.append(seq_arr[:seq_len])
        y_all.append(seq_arr[seq_len, 0])  # next-step FRP

    return np.stack(X_all), np.array(y_all, dtype=np.float32)


class WildfireLSTM(object):  # defined inside to allow __main__ reimport
    pass


def _real_sequences(seq_len: int = SEQ_LEN, max_clusters: int = 80) -> list[tuple]:
    """
    Build real fire sequences from NASA FIRMS + Open-Meteo weather.

    Groups FIRMS detections by approximate location (0.5° grid cell) and date,
    then fetches weather for each cluster from Open-Meteo.

    Returns list of (feature_array, target_frp_norm) tuples.
    """
    import math
    import csv
    from pathlib import Path
    from datetime import datetime, timedelta
    from datasets.firms import download as firms_download, FIRMS_MODIS_URL
    from datasets.openmeteo import fetch_hourly, weather_at_hour

    firms_path = firms_download()
    if not firms_path.exists():
        return []

    # Parse FIRMS CSV into (lat_grid, lon_grid, date) → [frp, ...]
    clusters: dict = {}
    try:
        with open(firms_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lat   = round(float(row["latitude"]) / 0.5) * 0.5
                    lon   = round(float(row["longitude"]) / 0.5) * 0.5
                    date  = row.get("acq_date", "")
                    frp   = float(row.get("frp", 0) or 0)
                    key   = (lat, lon, date)
                    clusters.setdefault(key, []).append(frp)
                except (KeyError, ValueError):
                    continue
    except Exception:
        return []

    # Sort clusters by date, group by location to build sequences
    by_loc: dict = {}
    for (lat, lon, date), frps in clusters.items():
        loc_key = (lat, lon)
        by_loc.setdefault(loc_key, []).append((date, sum(frps), len(frps)))

    sequences = []
    processed = 0
    for (lat, lon), daily in sorted(by_loc.items())[:max_clusters]:
        daily.sort(key=lambda x: x[0])
        if len(daily) < 2:
            continue

        # Fetch weather for this location over the date range
        dates = [d[0] for d in daily]
        try:
            hourly = fetch_hourly(lat, lon, dates[0], dates[-1])
        except Exception:
            hourly = None

        seq = []
        for i, (date, agg_frp, n_pixels) in enumerate(daily):
            hour_idx = i * 24
            if hourly:
                wx = weather_at_hour(hourly, hour_idx)
            else:
                wx = {"temp_c": 25, "rh_pct": 40, "wind_speed_ms": 5,
                      "wind_dir_deg": 180, "precip_mm": 0}

            import math as _math
            wd_rad = _math.radians(wx["wind_dir_deg"])
            seq.append([
                min(agg_frp / 5000.0, 1.0),
                min(n_pixels / 200.0, 1.0),
                min(wx["wind_speed_ms"] / 25.0, 1.0),
                _math.sin(wd_rad),
                _math.cos(wd_rad),
                wx["rh_pct"] / 100.0,
                (wx["temp_c"] + 10) / 65.0,
                0.5,   # NDVI unknown from FIRMS
                0.1,   # slope unknown
            ])

        # Build overlapping windows of seq_len steps
        for start in range(max(0, len(seq) - seq_len)):
            window = seq[start:start + seq_len]
            if len(window) < seq_len:
                break
            target_frp = daily[start + seq_len][1] if start + seq_len < len(daily) else daily[-1][1]
            sequences.append((window, min(target_frp / 5000.0, 1.0)))

        processed += 1

    logger.info("[WildfireLSTM] Built %d real sequences from %d FIRMS clusters + Open-Meteo",
                len(sequences), processed)
    return sequences


def train(epochs: int = 30, seq_len: int = SEQ_LEN, batch_size: int = 64,
          device_str: str = "auto") -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if device_str == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("[WildfireLSTM] Device: %s", device)

    # Try real FIRMS + Open-Meteo sequences first
    real_seqs = _real_sequences(seq_len=seq_len, max_clusters=80)
    if real_seqs:
        logger.info("[WildfireLSTM] Using %d real sequences from FIRMS+Open-Meteo", len(real_seqs))
        real_X = [s[0] for s in real_seqs]
        real_y = [s[1] for s in real_seqs]
        # Augment with synthetic for volume
        syn_X_arr, syn_y_arr = _synthetic_sequences(n=max(500, 3000 - len(real_seqs)), seq_len=seq_len)
        X = np.vstack([np.array(real_X, dtype=np.float32), syn_X_arr])
        y = np.concatenate([np.array(real_y, dtype=np.float32), syn_y_arr])
        logger.info("[WildfireLSTM] Combined: %d real + %d synthetic = %d total",
                    len(real_seqs), len(syn_X_arr), len(X))
    else:
        logger.info("[WildfireLSTM] No real sequences available — using synthetic only")
        X, y = _synthetic_sequences(n=6000, seq_len=seq_len)

    split = int(len(X) * 0.8)
    X_train, X_val = torch.tensor(X[:split]), torch.tensor(X[split:])
    y_train, y_val = torch.tensor(y[:split]), torch.tensor(y[split:])

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=batch_size, shuffle=False, num_workers=0)

    class _LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(N_FEATURES, HIDDEN_DIM, N_LAYERS,
                                batch_first=True, dropout=0.2)
            self.head = nn.Sequential(
                nn.Linear(HIDDEN_DIM, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(1)

    model = _LSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.HuberLoss(delta=0.1)

    best_mae, best_state = float("inf"), None

    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                preds.append(model(X_b.to(device)).cpu())
                targets.append(y_b)
        preds   = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()
        mae = float(np.mean(np.abs(preds - targets)))
        scheduler.step(mae)
        logger.info("[WildfireLSTM] Epoch %d/%d — val MAE %.4f (norm FRP)", epoch + 1, epochs, mae)
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "seq_len": seq_len,
                "n_features": N_FEATURES, "hidden_dim": HIDDEN_DIM,
                "n_layers": N_LAYERS}, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "LSTM",
        "seq_len": seq_len,
        "n_features": N_FEATURES,
        "hidden_dim": HIDDEN_DIM,
        "n_layers": N_LAYERS,
        "n_epochs": epochs,
        "n_samples": len(X),
        "data_source": "synthetic",
        "features": ["frp_mw", "active_pixels", "wind_speed_ms", "wind_dir_sin",
                     "wind_dir_cos", "rh_pct", "temp_c", "ndvi", "slope_deg"],
        "target": "frp_next_mw (normalized)",
        "metrics": {"mae_val_normalized": round(best_mae, 5)},
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[WildfireLSTM] Model saved → %s (MAE: %.4f)", MODEL_PATH, best_mae)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--seq-len",    type=int, default=SEQ_LEN)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device",     default="auto")
    args = p.parse_args()
    train(epochs=args.epochs, seq_len=args.seq_len, batch_size=args.batch_size,
          device_str=args.device)
