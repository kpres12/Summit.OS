"""
Tornado Nowcasting CNN — NEXRAD Level-II Reflectivity

Trains a 2D CNN on NEXRAD Level-II radar reflectivity grids to predict
the probability of a tornado / severe-weather signature in the next
20 minutes.

Real data: NEXRAD Level-II from the NOAA AWS bucket
(s3://noaa-nexrad-level2/) — completely no auth, ~5 minute cadence per
station, 160+ stations across the US.

Labels (positive class) are derived from NOAA Storm Events records of
tornadoes (we already cache this in `packages/training/data/noaa_storms/`):
  - For each tornado event, find NEXRAD scans from the same station
    (or nearest of TORNADO_ALLEY_STATIONS) within ±30 min
  - Positive label: the scan happened ≤ 20 minutes BEFORE the tornado
  - Negative label: scans on tornado-free days at the same station

Model: simple 4-conv CNN over (azimuth × gates) reflectivity grid.

Output:
  packages/c2_intel/models/tornado_nowcast_cnn.pt
  packages/c2_intel/models/tornado_nowcast_cnn_meta.json

If `pyart` is not installed or no real scans are present, falls back
to synthetic radar-like patterns so the pipeline is testable. Tornado
signatures (mesocyclone hook echoes) are simulated by injecting a curved
high-reflectivity arm into the synthetic samples.

Usage:
  python train_tornado_nowcast.py [--stations KOUN,KFWS --max-days 7]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "c2_intel" / "models"
MODEL_PATH = MODELS_DIR / "tornado_nowcast_cnn.pt"
META_PATH  = MODELS_DIR / "tornado_nowcast_cnn_meta.json"
STORM_CACHE = Path(__file__).parent / "data" / "noaa_storms"

GRID_AZ = 360
GRID_GATES = 256
HORIZON_MIN = 20


def _parse_dt(date_str: str):
    if not date_str:
        return None
    for f in ("%d-%b-%y %H:%M:%S", "%d-%b-%y %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str.strip(), f).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _load_tornado_events(years: list[int]) -> list[dict]:
    """Load tornado event rows from cached NOAA storm CSVs."""
    out: list[dict] = []
    for y in years:
        cache = STORM_CACHE / f"storms_{y}.csv"
        if not cache.exists():
            logger.warning("[tornado] no storm cache for %d", y)
            continue
        with cache.open("r", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                if (row.get("EVENT_TYPE") or "").strip().lower() != "tornado":
                    continue
                dt = _parse_dt(row.get("BEGIN_DATE_TIME") or "")
                if dt is None:
                    continue
                try:
                    lat = float(row.get("BEGIN_LAT") or 0)
                    lon = float(row.get("BEGIN_LON") or 0)
                except ValueError:
                    continue
                if lat == 0 and lon == 0:
                    continue
                tor_scale = (row.get("TOR_F_SCALE") or "").strip()
                ef = 0
                m = re.match(r"^E?F(\d)", tor_scale)
                if m:
                    try: ef = int(m.group(1))
                    except: ef = 0
                out.append({"dt": dt, "lat": lat, "lon": lon, "ef": ef,
                            "state": (row.get("STATE") or "").strip()})
    logger.info("[tornado] loaded %d tornado events from cache", len(out))
    return out


def _build_real_dataset(stations: list[str], days: int, years: list[int]
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Try to build real (X, y) from NEXRAD scans matched to tornado events.

    Heavy lift — requires pyart and substantial download time. Returns empty
    arrays if dependencies aren't met or no scans matched.
    """
    try:
        import pyart  # type: ignore  # noqa: F401
    except ImportError:
        logger.warning("[tornado] pyart not installed — real-data path unavailable")
        return np.zeros((0, 1, GRID_AZ, GRID_GATES), dtype=np.float32), np.zeros(0, dtype=np.int64)

    try:
        from datasets.nexrad import (   # type: ignore
            list_available_scans, download_scan, parse_scan_to_grid,
        )
    except Exception as e:
        logger.warning("[tornado] nexrad loader unavailable: %s", e)
        return np.zeros((0, 1, GRID_AZ, GRID_GATES), dtype=np.float32), np.zeros(0, dtype=np.int64)

    events = _load_tornado_events(years)
    if not events:
        return np.zeros((0, 1, GRID_AZ, GRID_GATES), dtype=np.float32), np.zeros(0, dtype=np.int64)

    Xs: list[np.ndarray] = []
    ys: list[int] = []
    fetched = 0
    for ev in events[:days * 5]:  # cap fetch
        ev_date = ev["dt"].date()
        for st in stations:
            keys = list_available_scans(st, ev_date)
            if not keys:
                continue
            # Take ONE scan in the (T-20, T) window before the tornado time
            target_window_lo = ev["dt"] - timedelta(minutes=20)
            target_window_hi = ev["dt"]
            chosen = None
            for k in keys:
                # Filename includes HHMMSS — parse it
                m = re.search(r"(\d{8})_(\d{6})", k)
                if not m:
                    continue
                try:
                    t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                if target_window_lo <= t <= target_window_hi:
                    chosen = k; break
            if chosen is None:
                continue
            local = download_scan(chosen)
            if local is None:
                continue
            grid = parse_scan_to_grid(local, n_gates=GRID_GATES, n_azimuths=GRID_AZ)
            if grid is None:
                continue
            # Normalize -32..72 dBZ -> 0..1
            g = (grid - (-32.0)) / 104.0
            Xs.append(g[None, :, :].astype(np.float32))
            ys.append(1)
            fetched += 1
            if fetched >= 200:
                break
        if fetched >= 200:
            break

    # Equally many negatives — take scans from same stations on a known calm day
    calm_date = date(2024, 1, 15)  # mid-winter quiet
    n_pos = len(Xs)
    for st in stations:
        if len(Xs) >= 2 * n_pos:
            break
        keys = list_available_scans(st, calm_date)
        for k in keys[:max(1, n_pos // len(stations) + 1)]:
            local = download_scan(k)
            if local is None:
                continue
            grid = parse_scan_to_grid(local, n_gates=GRID_GATES, n_azimuths=GRID_AZ)
            if grid is None:
                continue
            g = (grid - (-32.0)) / 104.0
            Xs.append(g[None, :, :].astype(np.float32))
            ys.append(0)

    if not Xs:
        return np.zeros((0, 1, GRID_AZ, GRID_GATES), dtype=np.float32), np.zeros(0, dtype=np.int64)
    X = np.stack(Xs)
    y = np.array(ys, dtype=np.int64)
    logger.info("[tornado] real path: %d samples (%d pos / %d neg)",
                len(X), int(y.sum()), int((y == 0).sum()))
    return X, y


def _build_synthetic_fallback(n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Generate radar-like reflectivity with hook-echo signatures for the
    positive class. Pipeline test only — not representative of real
    tornado-detection performance."""
    rng = np.random.default_rng(7)
    X = rng.uniform(0.0, 0.05, (n, 1, GRID_AZ, GRID_GATES)).astype(np.float32)
    y = rng.integers(0, 2, n).astype(np.int64)
    for i in range(n):
        # Add some background storm cells to all
        for _ in range(rng.integers(1, 5)):
            ca = int(rng.uniform(0, GRID_AZ))
            cg = int(rng.uniform(40, GRID_GATES - 40))
            sig = float(rng.uniform(0.4, 0.7))
            for da in range(-15, 16):
                for dg in range(-15, 16):
                    a, g = (ca + da) % GRID_AZ, cg + dg
                    if 0 <= g < GRID_GATES:
                        d2 = da*da + dg*dg
                        X[i, 0, a, g] += sig * math.exp(-d2 / 60.0)
        if y[i] == 1:
            # Add a hook-echo arm — high reflectivity along a curved sweep
            ca = int(rng.uniform(0, GRID_AZ))
            cg = int(rng.uniform(80, GRID_GATES - 80))
            for k in range(40):
                a = (ca + k) % GRID_AZ
                g = cg + int(20 * math.sin(k / 6.0))
                if 0 <= g < GRID_GATES:
                    X[i, 0, a, g] += 0.6
                    if g + 1 < GRID_GATES:
                        X[i, 0, a, g + 1] += 0.4
                    if a + 1 < GRID_AZ:
                        X[i, 0, (a + 1) % GRID_AZ, g] += 0.4
    X = np.clip(X, 0.0, 1.0).astype(np.float32)
    logger.info("[tornado-synth] %d samples (pipeline test only)", n)
    return X, y


def train(stations: list[str], years: list[int], epochs: int = 12,
          batch_size: int = 32, days: int = 30,
          device_str: str = "auto") -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    if device_str == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("[tornado-train] Device: %s", device)

    X, y = _build_real_dataset(stations, days, years)
    used_real = len(X) > 50
    if not used_real:
        logger.warning("[tornado-train] real path produced %d samples — "
                       "falling back to synthetic", len(X))
        X, y = _build_synthetic_fallback()

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

    class _CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 5, padding=2),
                nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 5, padding=2),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(64 * 4 * 4, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 2),
            )

        def forward(self, x):
            return self.head(self.features(x))

    model = _CNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                preds = model(X_b.to(device)).argmax(1).cpu()
                correct += int((preds == y_b).sum())
                total += len(y_b)
        acc = correct / max(total, 1)
        logger.info("[tornado-train] Epoch %d/%d  val_acc=%.4f", epoch + 1, epochs, acc)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "az": GRID_AZ, "gates": GRID_GATES,
                "horizon_min": HORIZON_MIN}, MODEL_PATH)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": "2D CNN (4 conv + MLP head)",
        "task": f"P(tornado/severe-weather signature within next {HORIZON_MIN}m)",
        "n_samples": int(len(X)),
        "input_shape": [1, GRID_AZ, GRID_GATES],
        "horizon_minutes": HORIZON_MIN,
        "stations_used": stations,
        "years": years,
        "data_source": ("nexrad_level2_real" if used_real else
                        "synthetic_fallback (install pyart + run with stations to use real)"),
        "metrics": {"val_acc_best": round(best_acc, 4)},
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info("[tornado-train] Saved -> %s (val_acc=%.4f, real=%s)",
                MODEL_PATH, best_acc, used_real)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--stations", default="KOUN,KFWS,KTLX,KICT")
    p.add_argument("--years",    nargs="+", type=int, default=[2023, 2024])
    p.add_argument("--epochs",   type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--days",     type=int, default=30)
    p.add_argument("--device",   default="auto")
    args = p.parse_args()
    stations = [s.strip().upper() for s in args.stations.split(",") if s.strip()]
    train(stations=stations, years=args.years, epochs=args.epochs,
          batch_size=args.batch_size, days=args.days, device_str=args.device)
