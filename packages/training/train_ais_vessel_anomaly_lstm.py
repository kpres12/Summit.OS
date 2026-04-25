"""
Maritime Vessel Trajectory Anomaly LSTM (Autoencoder)

Trains an LSTM sequence autoencoder on per-vessel trajectory features
extracted from real US-waters AIS data (MarineCadastre.gov, free, no
auth). Anomaly score = reconstruction MSE.

Civilian use cases:
  - USCG SAR demand prediction (vessels with anomalous trajectories
    are higher distress probability)
  - Illegal fishing detection (loitering near closed zones, AIS gaps
    + sudden course changes after gap)
  - Port security (out-of-pattern approaches)
Federal use cases:
  - Maritime Domain Awareness (MDA)
  - Smuggling pattern of life
  - Dark-vessel re-emergence detection (when a vessel reappears after
    AIS gap, score the reappearance trajectory against historical
    vessel-class patterns)

Real data: MarineCadastre.gov daily AIS CSVs via direct HTTPS, no auth.

Output:
  packages/c2_intel/models/ais_vessel_anomaly_lstm.pt
  packages/c2_intel/models/ais_vessel_anomaly_lstm_meta.json

If MarineCadastre download fails, falls back to synthetic vessel
trajectories with embedded anomalies for pipeline test.

Usage:
  python train_ais_vessel_anomaly_lstm.py [--year 2024 --month 6 --days 1 2 3]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "c2_intel" / "models"
MODEL_PATH = MODELS_DIR / "ais_vessel_anomaly_lstm.pt"
META_PATH  = MODELS_DIR / "ais_vessel_anomaly_lstm_meta.json"

SEQ_LEN     = 8
N_FEATURES  = 6
HIDDEN_DIM  = 64
N_LAYERS    = 1


def _build_real_dataset(year: int, month: int, days: list[int],
                        max_records_per_day: int = 200_000,
                        min_obs_per_vessel: int = SEQ_LEN
                        ) -> tuple[np.ndarray, list[str]]:
    """Pull AIS records, group by MMSI, build per-vessel sequences."""
    try:
        from datasets.marinecadastre import (   # type: ignore
            download_day, iter_records,
        )
    except Exception as e:
        logger.warning("[ais-train] marinecadastre loader unavailable: %s", e)
        return np.zeros((0, SEQ_LEN, N_FEATURES), dtype=np.float32), []

    by_mmsi: dict[str, list[dict]] = defaultdict(list)
    for d in days:
        csv_path = download_day(year, month, d)
        if csv_path is None:
            logger.warning("[ais-train] %d-%02d-%02d download failed", year, month, d)
            continue
        n = 0
        for rec in iter_records(csv_path, max_rows=max_records_per_day):
            mmsi = rec.get("mmsi")
            if not mmsi:
                continue
            by_mmsi[mmsi].append(rec)
            n += 1
        logger.info("[ais-train] %d-%02d-%02d -> %d records, %d MMSIs",
                    year, month, d, n, len(by_mmsi))

    sequences: list[np.ndarray] = []
    mmsis: list[str] = []

    for mmsi, recs in by_mmsi.items():
        if len(recs) < min_obs_per_vessel:
            continue
        recs.sort(key=lambda r: r["ts"])
        rows: list[np.ndarray] = []
        for r in recs:
            try:
                cog = r["cog"]
                heading = r["heading"]
                rows.append(np.array([
                    r["lat"] / 90.0,
                    r["lon"] / 180.0,
                    min(r["sog"] / 30.0, 1.5),    # over 30 kts is unusual
                    math.sin(math.radians(cog)) if cog else 0.0,
                    math.cos(math.radians(cog)) if cog else 0.0,
                    min(r.get("length", 0) / 400.0, 1.0),  # length up to 400m
                ], dtype=np.float32))
            except (TypeError, ValueError):
                continue
        if len(rows) < SEQ_LEN:
            continue
        # Trailing window
        seq = np.stack(rows[-SEQ_LEN:])
        sequences.append(seq)
        mmsis.append(mmsi)

    if not sequences:
        return np.zeros((0, SEQ_LEN, N_FEATURES), dtype=np.float32), []
    X = np.stack(sequences)
    logger.info("[ais-train] real path: %d vessels x SEQ_LEN=%d x feat=%d",
                len(X), SEQ_LEN, N_FEATURES)
    return X, mmsis


def _synthetic_fallback(n: int = 4000) -> tuple[np.ndarray, list[str]]:
    """Crude vessel-trajectory generator for pipeline test.

    Normal vessels: smooth heading, modest speed, single class.
    Anomalies (~10%): random course flips, speed spikes, teleports.
    """
    rng = np.random.default_rng(7)
    X = np.zeros((n, SEQ_LEN, N_FEATURES), dtype=np.float32)
    mmsis = [f"SYN-{i:06d}" for i in range(n)]
    for i in range(n):
        lat0 = rng.uniform(25.0, 45.0)
        lon0 = rng.uniform(-130.0, -65.0)
        sog = rng.uniform(2.0, 18.0)
        cog = rng.uniform(0.0, 360.0)
        length = rng.uniform(20.0, 200.0)
        anomaly = rng.random() < 0.10
        for t in range(SEQ_LEN):
            if anomaly and t == SEQ_LEN // 2:
                cog = (cog + rng.uniform(120, 240)) % 360.0  # sharp turn
                sog *= rng.uniform(2.0, 4.0)                   # speed spike
            else:
                cog = (cog + rng.uniform(-3, 3)) % 360.0
                sog = max(0.5, sog + rng.normal(0, 0.3))
            lat0 += sog * 0.0001 * math.cos(math.radians(cog)) + rng.normal(0, 0.001)
            lon0 += sog * 0.0001 * math.sin(math.radians(cog)) + rng.normal(0, 0.001)
            X[i, t, 0] = lat0 / 90.0
            X[i, t, 1] = lon0 / 180.0
            X[i, t, 2] = min(sog / 30.0, 1.5)
            X[i, t, 3] = math.sin(math.radians(cog))
            X[i, t, 4] = math.cos(math.radians(cog))
            X[i, t, 5] = length / 400.0
    logger.info("[ais-synth] %d synthetic vessels (10%% anomalous)", n)
    return X, mmsis


def train(year: int, month: int, days: list[int], epochs: int = 25,
          batch_size: int = 64, device_str: str = "auto") -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if device_str == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("[ais-train] Device: %s", device)

    X, mmsis = _build_real_dataset(year, month, days)
    used_real = len(X) >= 200
    if not used_real:
        logger.warning("[ais-train] real path produced %d vessels — "
                       "falling back to synthetic", len(X))
        X, mmsis = _synthetic_fallback()

    rng = np.random.default_rng(7)
    order = rng.permutation(len(X))
    X = X[order]
    split = int(len(X) * 0.8)
    X_train = torch.tensor(X[:split])
    X_val   = torch.tensor(X[split:])

    train_loader = DataLoader(TensorDataset(X_train, X_train),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(X_val, X_val),
                              batch_size=batch_size, shuffle=False, num_workers=0)

    class _AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(N_FEATURES, HIDDEN_DIM, N_LAYERS, batch_first=True)
            self.decoder = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, N_LAYERS, batch_first=True)
            self.out = nn.Linear(HIDDEN_DIM, N_FEATURES)

        def forward(self, x):
            _, (h, c) = self.encoder(x)
            B = x.size(0)
            latent_seq = h[-1].unsqueeze(1).expand(B, SEQ_LEN, HIDDEN_DIM).contiguous()
            decoded, _ = self.decoder(latent_seq, (h, c))
            return self.out(decoded)

    model = _AE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

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
        losses = []
        with torch.no_grad():
            for X_b, _ in val_loader:
                X_b = X_b.to(device)
                losses.append(criterion(model(X_b), X_b).item())
        val = float(np.mean(losses)) if losses else float("inf")
        logger.info("[ais-train] Epoch %d/%d val_mse=%.5f", epoch + 1, epochs, val)
        if val < best_loss:
            best_loss = val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        recon = model(torch.tensor(X[split:]).to(device)).cpu().numpy()
    errs = ((recon - X[split:]) ** 2).mean(axis=(1, 2))
    p50 = float(np.percentile(errs, 50))
    p95 = float(np.percentile(errs, 95))
    p99 = float(np.percentile(errs, 99))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "seq_len": SEQ_LEN,
                "n_features": N_FEATURES,
                "hidden_dim": HIDDEN_DIM,
                "n_layers": N_LAYERS}, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "LSTM autoencoder",
        "task": "Vessel trajectory anomaly score (reconstruction MSE)",
        "seq_len": SEQ_LEN,
        "n_features": N_FEATURES,
        "n_vessels": int(len(X)),
        "data_source": "marinecadastre_ais_real" if used_real else "synthetic_fallback",
        "year_month": f"{year}-{month:02d}",
        "days": days,
        "metrics": {
            "val_mse_best": round(best_loss, 5),
            "anomaly_thresholds_mse": {
                "p50": round(p50, 5),
                "p95": round(p95, 5),
                "p99": round(p99, 5),
            },
        },
        "usage": ("Score new trajectories with model.forward() then compute "
                  "MSE; MSE > p99 ≈ strong anomaly (likely AIS gap, distress, "
                  "or illicit pattern)."),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[ais-train] Saved -> %s (val_mse=%.5f, real=%s)",
                MODEL_PATH, best_loss, used_real)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--year",  type=int, default=2024)
    p.add_argument("--month", type=int, default=6)
    p.add_argument("--days",  nargs="+", type=int, default=[15])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    train(year=args.year, month=args.month, days=args.days,
          epochs=args.epochs, batch_size=args.batch_size, device_str=args.device)
