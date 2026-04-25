"""
Aircraft Trajectory Anomaly LSTM (Autoencoder)

Fetches a series of OpenSky Network state-vector snapshots, stitches
per-aircraft trajectories by ICAO24, and trains an LSTM sequence
autoencoder. Anomaly score = reconstruction error.

Civilian use cases:
  - Spoofed/erratic tracks (FAA, ATC)
  - Low-altitude lingering (BLM patrol, fire ops, border)
  - Off-flight-plan deviation (search & rescue assist)
  - UAS detection in unexpected airspace

Real data:
  OpenSky Network /api/states/all — anonymous access, 100 req/day cap.
  Each call returns ~10k current aircraft state vectors globally. We poll
  every POLL_INTERVAL_SEC seconds for SNAPSHOTS samples.

Output: packages/c2_intel/models/aircraft_anomaly_lstm.pt
        packages/c2_intel/models/aircraft_anomaly_lstm_meta.json

Usage:
    python train_aircraft_anomaly_lstm.py [--snapshots 20 --interval 15]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "c2_intel" / "models"
MODEL_PATH = MODELS_DIR / "aircraft_anomaly_lstm.pt"
META_PATH  = MODELS_DIR / "aircraft_anomaly_lstm_meta.json"
CACHE_DIR  = Path(__file__).parent / "data" / "opensky_track"

OPENSKY_URL = "https://opensky-network.org/api/states/all"

SEQ_LEN     = 8           # 8 timesteps per aircraft (≈ 2 min at 15-sec interval)
N_FEATURES  = 8
HIDDEN_DIM  = 64
N_LAYERS    = 1


# State vector index reference (OpenSky API):
#   0 icao24   1 callsign   2 origin_country   3 time_position   4 last_contact
#   5 longitude   6 latitude   7 baro_altitude   8 on_ground   9 velocity
#   10 true_track   11 vertical_rate   12 sensors   13 geo_altitude
#   14 squawk   15 spi   16 position_source

def _fetch_snapshot(retries: int = 3, sleep_on_429: float = 30.0) -> list[list]:
    for attempt in range(retries):
        try:
            r = requests.get(OPENSKY_URL, timeout=30,
                             headers={"User-Agent": "Heli.OS/1.0"})
            if r.status_code == 429:
                logger.warning("[Aircraft] 429 — backing off %.0fs", sleep_on_429)
                time.sleep(sleep_on_429)
                continue
            r.raise_for_status()
            data = r.json()
            return data.get("states") or []
        except Exception as e:
            logger.warning("[Aircraft] OpenSky fetch failed (try %d): %s", attempt + 1, e)
            time.sleep(5)
    return []


def _collect_snapshots(n_snapshots: int = 20, interval_sec: int = 15) -> list[dict]:
    """Poll OpenSky n_snapshots times. Cache the entire run keyed by start time."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"snapshots_{n_snapshots}x{interval_sec}.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            if data and len(data) >= n_snapshots // 2:
                logger.info("[Aircraft] Cached snapshots: %d", len(data))
                return data
        except Exception:
            pass

    snapshots: list[dict] = []
    for i in range(n_snapshots):
        states = _fetch_snapshot()
        if not states:
            logger.warning("[Aircraft] Empty snapshot %d/%d", i + 1, n_snapshots)
        else:
            ts = int(time.time())
            snapshots.append({"ts": ts, "states": states})
            logger.info("[Aircraft] Snapshot %d/%d -> %d aircraft", i + 1, n_snapshots, len(states))
        if i < n_snapshots - 1:
            time.sleep(interval_sec)

    cache.write_text(json.dumps(snapshots))
    return snapshots


def _build_sequences(snapshots: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Group state vectors by icao24 across snapshots, build per-aircraft sequences."""
    by_icao: dict[str, list[tuple[int, list]]] = defaultdict(list)
    for snap in snapshots:
        ts = snap["ts"]
        for s in snap.get("states") or []:
            if not s or len(s) < 12:
                continue
            icao = (s[0] or "").strip().lower()
            if not icao:
                continue
            by_icao[icao].append((ts, s))

    feat_names = [
        "lat_norm", "lon_norm", "alt_norm",
        "velocity_norm", "vertical_rate_norm",
        "track_sin", "track_cos",
        "on_ground",
    ]

    sequences: list[np.ndarray] = []
    icaos: list[str] = []

    for icao, items in by_icao.items():
        items.sort(key=lambda x: x[0])
        # Need at least SEQ_LEN observations
        if len(items) < SEQ_LEN:
            continue

        # Convert to feature rows
        feats: list[np.ndarray] = []
        for _ts, s in items:
            try:
                lon = s[5]; lat = s[6]
                if lon is None or lat is None:
                    feats.append(None)  # type: ignore
                    continue
                baro = s[7] or 0.0
                on_ground = 1.0 if s[8] else 0.0
                velocity = s[9] or 0.0
                track = s[10]
                vrate = s[11] or 0.0
                track_sin = math.sin(math.radians(track)) if track is not None else 0.0
                track_cos = math.cos(math.radians(track)) if track is not None else 0.0
                feats.append(np.array([
                    float(lat) / 90.0,
                    float(lon) / 180.0,
                    min(float(baro) / 13000.0, 1.5),  # FL430 ~ 1.0
                    min(float(velocity) / 300.0, 1.5),
                    np.clip(float(vrate) / 30.0, -1.0, 1.0),
                    track_sin, track_cos, on_ground,
                ], dtype=np.float32))
            except (TypeError, ValueError):
                feats.append(None)  # type: ignore

        # Drop None rows; require continuous run of at least SEQ_LEN
        good_feats = [f for f in feats if f is not None]
        if len(good_feats) < SEQ_LEN:
            continue
        # Use trailing window (most recent SEQ_LEN observations)
        seq = np.stack(good_feats[-SEQ_LEN:])
        sequences.append(seq)
        icaos.append(icao)

    if not sequences:
        return np.zeros((0, SEQ_LEN, N_FEATURES), dtype=np.float32), []
    X = np.stack(sequences)
    logger.info("[Aircraft] Built %d per-aircraft sequences (seq_len=%d, features=%d)",
                len(X), SEQ_LEN, N_FEATURES)
    return X, icaos


def train(snapshots: int = 20, interval_sec: int = 15, epochs: int = 30,
          batch_size: int = 64, device_str: str = "auto") -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if device_str == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("[Aircraft] Device: %s", device)

    snaps = _collect_snapshots(n_snapshots=snapshots, interval_sec=interval_sec)
    if len(snaps) < SEQ_LEN:
        raise RuntimeError(f"Only {len(snaps)} usable snapshots — need at least {SEQ_LEN}.")

    X, icaos = _build_sequences(snaps)
    if len(X) < 100:
        raise RuntimeError(f"Only {len(X)} aircraft sequences — too few.")

    rng = np.random.default_rng(7)
    order = rng.permutation(len(X))
    X = X[order]
    split = int(len(X) * 0.8)
    X_train, X_val = torch.tensor(X[:split]), torch.tensor(X[split:])

    train_loader = DataLoader(TensorDataset(X_train, X_train),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(X_val, X_val),
                              batch_size=batch_size, shuffle=False, num_workers=0)

    class _AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(N_FEATURES, HIDDEN_DIM, N_LAYERS,
                                   batch_first=True)
            self.decoder = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, N_LAYERS,
                                   batch_first=True)
            self.out = nn.Linear(HIDDEN_DIM, N_FEATURES)

        def forward(self, x):
            _, (h, c) = self.encoder(x)
            # decoder input: replicate latent across SEQ_LEN steps
            B = x.size(0)
            latent_seq = h[-1].unsqueeze(1).expand(B, SEQ_LEN, HIDDEN_DIM).contiguous()
            decoded, _ = self.decoder(latent_seq, (h, c))
            return self.out(decoded)

    model = _AE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.MSELoss()

    best_loss, best_state = float("inf"), None

    for epoch in range(epochs):
        model.train()
        for X_b, _ in train_loader:
            X_b = X_b.to(device)
            optimizer.zero_grad()
            criterion(model(X_b), X_b).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            losses = []
            for X_b, _ in val_loader:
                X_b = X_b.to(device)
                losses.append(criterion(model(X_b), X_b).item())
        val_loss = float(np.mean(losses)) if losses else float("inf")
        logger.info("[Aircraft] Epoch %d/%d  val_loss=%.5f", epoch + 1, epochs, val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # Compute reconstruction-error histogram on holdout for anomaly threshold guidance
    model.eval()
    with torch.no_grad():
        recon = model(torch.tensor(X[split:]).to(device)).cpu().numpy()
    errs = ((recon - X[split:]) ** 2).mean(axis=(1, 2))
    err_p50 = float(np.percentile(errs, 50))
    err_p95 = float(np.percentile(errs, 95))
    err_p99 = float(np.percentile(errs, 99))

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
        "model": "LSTM autoencoder",
        "task": "Aircraft trajectory anomaly score (reconstruction error)",
        "seq_len": SEQ_LEN,
        "n_features": N_FEATURES,
        "hidden_dim": HIDDEN_DIM,
        "n_layers": N_LAYERS,
        "n_epochs": epochs,
        "n_aircraft_sequences": int(len(X)),
        "n_snapshots_collected": len(snaps),
        "snapshot_interval_sec": interval_sec,
        "data_source": "opensky_realtime_real",
        "features": [
            "lat_norm", "lon_norm", "alt_norm",
            "velocity_norm", "vertical_rate_norm",
            "track_sin", "track_cos", "on_ground",
        ],
        "metrics": {
            "val_mse_best": round(best_loss, 5),
            "anomaly_thresholds_mse": {
                "p50": round(err_p50, 5),
                "p95": round(err_p95, 5),
                "p99": round(err_p99, 5),
            },
        },
        "usage": ("Score new sequences with model.forward() then compute MSE; "
                  "MSE > p99 ≈ strong anomaly, p95-p99 ≈ moderate."),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[Aircraft] Saved -> %s  (val_mse=%.5f)", MODEL_PATH, best_loss)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--snapshots",    type=int, default=20)
    p.add_argument("--interval",     type=int, default=15)
    p.add_argument("--epochs",       type=int, default=30)
    p.add_argument("--batch-size",   type=int, default=64)
    p.add_argument("--device",       default="auto")
    args = p.parse_args()
    train(snapshots=args.snapshots, interval_sec=args.interval,
          epochs=args.epochs, batch_size=args.batch_size, device_str=args.device)
