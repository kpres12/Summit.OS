"""
Aftershock LSTM Trainer

Trains a sequence-to-one LSTM to predict the probability of an M3.0+ earthquake
in the next 12 hours within a 0.5deg x 0.5deg cell, given a 7-day history of
seismic activity binned hourly (168 timesteps).

Input sequence features (per 1-hour bin):
  - event_count       : number of events in this hour
  - max_mag           : max magnitude in this hour (0 if none)
  - mean_depth_norm   : mean hypocenter depth / 100km (0 if none)
  - log_energy_sum    : log10(sum of moment-magnitude energy proxy)
  - hours_since_main  : hours since the largest event in the window / 168
  - cumulative_count  : count to date / 50

Output: prob_m3_next_12h (0-1)

Data: USGS ComCat FDSN API, M2.5+ globally over the past 2 years.
License: USGS (public domain, US Government work).

Output: packages/c2_intel/models/aftershock_lstm.pt
        packages/c2_intel/models/aftershock_lstm_meta.json

Usage:
    python train_aftershock_lstm.py [--epochs 30] [--device auto]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "c2_intel" / "models"
MODEL_PATH = MODELS_DIR / "aftershock_lstm.pt"
META_PATH  = MODELS_DIR / "aftershock_lstm_meta.json"
CACHE_DIR  = Path(__file__).parent / "data" / "usgs_aftershock"

SEQ_LEN     = 48           # 2 days x 24 hourly bins — short enough for stable LSTM
N_FEATURES  = 6
HIDDEN_DIM  = 64
N_LAYERS    = 1
HORIZON_H   = 12           # predict next-12h
TRIGGER_MAG = 3.0          # binary target threshold
GRID_DEG    = 1.0          # coarser grid → more events per cell
USGS_URL    = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def _download_catalog(days_back: int = 730, min_mag: float = 2.5) -> list[dict]:
    """Pull USGS M2.5+ catalog. Caches locally as JSONL."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"usgs_m{int(min_mag*10):02d}_d{days_back}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            if data:
                logger.info("[Aftershock] Loaded %d cached events", len(data))
                return data
        except Exception:
            pass

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)

    logger.info("[Aftershock] Fetching USGS M%.1f+ catalog %s -> %s",
                min_mag, start.date(), end.date())

    # USGS limits to 20k events per request — page by month
    all_features: list[dict] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=30), end)
        params = {
            "format": "geojson",
            "minmagnitude": min_mag,
            "starttime": cursor.strftime("%Y-%m-%d"),
            "endtime": chunk_end.strftime("%Y-%m-%d"),
            "orderby": "time-asc",
            "limit": 20000,
        }
        try:
            r = requests.get(USGS_URL, params=params, timeout=60,
                             headers={"User-Agent": "Heli.OS/1.0"})
            r.raise_for_status()
            feats = r.json().get("features", [])
            all_features.extend(feats)
            logger.info("[Aftershock]   %s..%s -> %d events (total %d)",
                        cursor.date(), chunk_end.date(), len(feats), len(all_features))
        except Exception as e:
            logger.warning("[Aftershock] Chunk fetch failed (%s): %s", cursor.date(), e)
        cursor = chunk_end

    events: list[dict] = []
    for feat in all_features:
        props = feat.get("properties") or {}
        geom  = feat.get("geometry") or {}
        coords = geom.get("coordinates") or [0, 0, 10]
        ts_ms = props.get("time")
        mag   = props.get("mag")
        if ts_ms is None or mag is None:
            continue
        events.append({
            "ts": int(ts_ms),                     # ms since epoch
            "mag": float(mag),
            "lon": float(coords[0]),
            "lat": float(coords[1]),
            "depth": float(coords[2]) if len(coords) > 2 else 10.0,
        })

    cache_file.write_text(json.dumps(events))
    logger.info("[Aftershock] Cached %d events to %s", len(events), cache_file)
    return events


def _build_sequences(events: list[dict],
                     seq_len: int = SEQ_LEN,
                     horizon_h: int = HORIZON_H,
                     trigger_mag: float = TRIGGER_MAG,
                     min_events_per_cell: int = 30,
                     max_samples: int = 30000) -> tuple[np.ndarray, np.ndarray]:
    """Bucket events into (cell, hour) bins and slide a (seq_len)-hour window.

    Returns (X[N, seq_len, N_FEATURES], y[N]) where y is 1 iff there is a
    >=trigger_mag event in the next horizon_h hours in the same cell.
    """
    if not events:
        return np.zeros((0, seq_len, N_FEATURES), dtype=np.float32), np.zeros(0, dtype=np.float32)

    # Group events by cell key
    cells: dict[tuple[int, int], list[dict]] = {}
    for e in events:
        key = (int(math.floor(e["lat"] / GRID_DEG)),
               int(math.floor(e["lon"] / GRID_DEG)))
        cells.setdefault(key, []).append(e)

    X_list: list[np.ndarray] = []
    y_list: list[float] = []

    one_hour_ms = 3600 * 1000

    for key, cell_events in cells.items():
        if len(cell_events) < min_events_per_cell:
            continue
        cell_events.sort(key=lambda e: e["ts"])

        t0 = cell_events[0]["ts"]
        t_end = cell_events[-1]["ts"]
        n_hours = int((t_end - t0) // one_hour_ms) + 1
        if n_hours < seq_len + horizon_h + 1:
            continue

        # Allocate per-bin aggregates
        bin_count = np.zeros(n_hours, dtype=np.float32)
        bin_max_mag = np.zeros(n_hours, dtype=np.float32)
        bin_depth_sum = np.zeros(n_hours, dtype=np.float32)
        bin_energy = np.zeros(n_hours, dtype=np.float32)

        for e in cell_events:
            idx = int((e["ts"] - t0) // one_hour_ms)
            if idx < 0 or idx >= n_hours:
                continue
            bin_count[idx] += 1
            if e["mag"] > bin_max_mag[idx]:
                bin_max_mag[idx] = e["mag"]
            bin_depth_sum[idx] += e["depth"]
            # Moment-energy proxy: 10**(1.5 * mag)
            bin_energy[idx] += 10 ** (1.5 * e["mag"])

        # Slide window
        cum = np.cumsum(bin_count)
        for start in range(0, n_hours - seq_len - horizon_h):
            end = start + seq_len
            window_count = bin_count[start:end]
            window_mag = bin_max_mag[start:end]
            window_depth_sum = bin_depth_sum[start:end]
            window_energy = bin_energy[start:end]

            if window_count.sum() < 3:  # too sparse — skip
                continue

            mean_depth = np.where(window_count > 0,
                                  window_depth_sum / np.clip(window_count, 1, None),
                                  0.0) / 100.0
            log_e = np.log10(np.clip(window_energy, 1.0, None)) / 15.0  # ~0..1 scale

            # Hours since largest event in window / seq_len
            largest_idx = int(np.argmax(window_mag))
            hours_since = np.arange(seq_len, dtype=np.float32) - largest_idx
            hours_since = np.clip(hours_since, 0, seq_len) / seq_len

            cum_to_date = cum[start:end] / 50.0

            features = np.stack([
                np.clip(window_count / 5.0, 0, 1),
                np.clip(window_mag / 8.0, 0, 1),
                np.clip(mean_depth, 0, 1),
                np.clip(log_e, 0, 1),
                hours_since,
                np.clip(cum_to_date, 0, 1),
            ], axis=1).astype(np.float32)

            # Target: any >=trigger_mag event in next horizon_h hours
            future = bin_max_mag[end:end + horizon_h]
            target = 1.0 if future.size > 0 and future.max() >= trigger_mag else 0.0

            X_list.append(features)
            y_list.append(target)

            if len(X_list) >= max_samples:
                break

        if len(X_list) >= max_samples:
            break

    if not X_list:
        return np.zeros((0, seq_len, N_FEATURES), dtype=np.float32), np.zeros(0, dtype=np.float32)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    pos_rate = y.mean()
    logger.info("[Aftershock] Built %d sequences, positive rate=%.3f",
                len(X), pos_rate)
    return X, y


def train(epochs: int = 30, batch_size: int = 64, device_str: str = "auto",
          days_back: int = 730, min_mag: float = 2.5,
          max_samples: int = 30000) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if device_str == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("[Aftershock] Device: %s", device)

    events = _download_catalog(days_back=days_back, min_mag=min_mag)
    if not events:
        raise RuntimeError("No USGS events downloaded — cannot train.")

    X, y = _build_sequences(events, max_samples=max_samples)
    if len(X) < 200:
        raise RuntimeError(f"Only {len(X)} sequences — need at least 200.")

    # Class balance: oversample positives if heavily skewed
    pos = np.where(y > 0.5)[0]
    neg = np.where(y < 0.5)[0]
    if len(pos) > 0 and len(neg) > 0 and len(pos) < 0.2 * len(y):
        n_target = min(len(neg), len(pos) * 4)
        pos_idx = np.random.choice(pos, size=n_target, replace=True)
        neg_idx = np.random.choice(neg, size=n_target, replace=False)
        idx = np.concatenate([pos_idx, neg_idx])
        np.random.shuffle(idx)
        X, y = X[idx], y[idx]
        logger.info("[Aftershock] Rebalanced to %d samples (pos rate=%.3f)",
                    len(X), y.mean())

    # Shuffle then split
    rng = np.random.default_rng(7)
    order = rng.permutation(len(X))
    X, y = X[order], y[order]
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
                                batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(HIDDEN_DIM, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            out, (h, _) = self.lstm(x)
            return self.head(h[-1]).squeeze(1)

    model = _LSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # pos_weight pushes the model harder on positives even after rebalancing
    pos_weight = torch.tensor(2.0, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc, best_state = 0.0, None

    def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
        # Quick rank-based AUC; falls back to 0.5 if degenerate
        if labels.sum() == 0 or labels.sum() == len(labels):
            return 0.5
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)
        pos = ranks[labels > 0.5]
        n_pos = len(pos)
        n_neg = len(labels) - n_pos
        return float((pos.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            criterion(logits, y_b).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        all_scores, all_labels, all_loss = [], [], 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                logits = model(X_b.to(device))
                all_loss += criterion(logits, y_b.to(device)).item() * len(y_b)
                all_scores.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(y_b.numpy())
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        auc = _auc(scores, labels)
        loss_val = all_loss / max(len(labels), 1)
        scheduler.step()
        logger.info("[Aftershock] Epoch %d/%d - loss=%.4f auc=%.4f pos@0.5=%.3f",
                    epoch + 1, epochs, loss_val, auc, float((scores >= 0.5).mean()))
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "seq_len": SEQ_LEN,
        "n_features": N_FEATURES,
        "hidden_dim": HIDDEN_DIM,
        "n_layers": N_LAYERS,
    }, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "LSTM (binary classifier)",
        "task": f"Probability of M{TRIGGER_MAG}+ aftershock in next {HORIZON_H}h",
        "seq_len_hours": SEQ_LEN,
        "horizon_hours": HORIZON_H,
        "trigger_magnitude": TRIGGER_MAG,
        "grid_deg": GRID_DEG,
        "n_features": N_FEATURES,
        "hidden_dim": HIDDEN_DIM,
        "n_layers": N_LAYERS,
        "n_epochs": epochs,
        "n_samples": int(len(X)),
        "n_real_events": len(events),
        "data_source": "usgs_comcat_real",
        "features": [
            "event_count_norm", "max_mag_norm", "mean_depth_norm",
            "log_energy_norm", "hours_since_largest_norm", "cum_count_norm",
        ],
        "metrics": {"val_auc": round(best_auc, 4)},
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[Aftershock] Saved -> %s (AUC=%.4f)", MODEL_PATH, best_auc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int, default=30)
    p.add_argument("--batch-size",   type=int, default=64)
    p.add_argument("--device",       default="auto")
    p.add_argument("--days-back",    type=int, default=730)
    p.add_argument("--min-mag",      type=float, default=2.5)
    p.add_argument("--max-samples",  type=int, default=30000)
    args = p.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, device_str=args.device,
          days_back=args.days_back, min_mag=args.min_mag, max_samples=args.max_samples)
