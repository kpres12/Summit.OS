"""
C2 Anomaly Detector Training
==============================
Trains IsolationForest models on NASA SMAP/MSL telemetry data.
Produces per-event-type models + a global fallback model that plug directly
into packages/c2_intel/anomaly.py without any code changes.

Feature vector (5 features — must match C2AnomalyDetector.get_anomaly_boost):
  [obs_last_5m, obs_last_30m, obs_last_90m, velocity, seconds_since_last]

Training strategy:
  - Extract windowed observation-burst features from SMAP/MSL time series
  - Train IsolationForest on NORMAL windows (unsupervised)
  - Per-event-type: train on channel subsets mapped to C2EventType categories
  - Global: train on all channels combined

Usage:
    python train_anomaly.py \\
        --data   /tmp/heli-training-data/smap_msl \\
        --output ../../packages/c2_intel/models
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Map SMAP/MSL channel name prefixes to C2EventType categories
# Spacecraft telemetry categories → drone/robot event analogues
_CHANNEL_TO_EVENT_TYPE: dict[str, str] = {
    "P": "COMMS_DEGRADED",      # P-series: power subsystem → comms link
    "S": "SENSOR_LOSS",         # S-series: sensor subsystem
    "E": "BATTERY_CRITICAL",    # E-series: electrical → battery
    "A": "ASSET_DEGRADED",      # A-series: attitude → asset state
    "D": "LINK_DEGRADED",       # D-series: data handling → link
    "T": "WEATHER_ALERT",       # T-series: thermal → environmental
    "F": "NODE_FAILED",         # F-series: fault management
    "G": "GEOFENCE_BREACH",     # G-series: guidance
    "M": "MISSION_ABORTED",     # M-series: mission planning
    "B": "ENTITY_DETECTED",     # B-series: payload/bus
    "C": "THREAT_IDENTIFIED",   # C-series: command
    "R": "PEER_OBSERVATION",    # R-series: relay
    "U": "AIRSPACE_CONFLICT",   # U-series: utilities
}

# C2EventType values referenced by anomaly.py
_C2_EVENT_TYPES = [
    "COMMS_DEGRADED", "SENSOR_LOSS", "BATTERY_CRITICAL", "ASSET_DEGRADED",
    "LINK_DEGRADED", "WEATHER_ALERT", "NODE_FAILED", "GEOFENCE_BREACH",
    "MISSION_ABORTED", "ENTITY_DETECTED", "THREAT_IDENTIFIED", "PEER_OBSERVATION",
    "AIRSPACE_CONFLICT",
]


def _extract_features(series: np.ndarray, sampling_hz: float = 1.0) -> np.ndarray:
    """
    Extract C2AnomalyDetector-compatible feature windows from a time series.

    Args:
        series: 1-D or 2-D array (timesteps [, channels])
        sampling_hz: samples per second (SMAP ~1/60 Hz ≈ 1 per minute)

    Returns:
        Feature matrix of shape (n_windows, 5)
    """
    if series.ndim > 1:
        series = series[:, 0]  # use first channel

    n = len(series)
    # Convert sample counts to time windows (SMAP: ~1 sample/min)
    # 5min → 5 samples, 30min → 30 samples, 90min → 90 samples
    w5  = max(1, int(5 / (1 / sampling_hz / 60)))
    w30 = max(1, int(30 / (1 / sampling_hz / 60)))
    w90 = max(1, int(90 / (1 / sampling_hz / 60)))

    # Threshold for "significant observation" (anomalous delta)
    std = max(1e-6, float(np.std(series)))
    threshold = 0.5 * std

    features = []
    for i in range(w90, n):
        window_90 = series[i - w90:i]
        window_30 = series[i - min(w30, i):][:w30]
        window_5  = series[i - min(w5, i):][:w5]

        def count_significant(w):
            if len(w) < 2:
                return 0
            deltas = np.abs(np.diff(w))
            return int(np.sum(deltas > threshold))

        cnt_5m  = count_significant(window_5)
        cnt_30m = count_significant(window_30)
        cnt_90m = count_significant(window_90)

        velocity = cnt_30m / max(cnt_90m / 3, 0.1)

        # Time since last significant delta
        deltas_90 = np.abs(np.diff(window_90))
        sig_indices = np.where(deltas_90 > threshold)[0]
        if sig_indices.size > 0:
            secs_since = int((len(window_90) - 1 - sig_indices[-1]) * 60)
        else:
            secs_since = 300  # default: 5 min ago

        features.append([
            float(cnt_5m),
            float(cnt_30m),
            float(cnt_90m),
            float(velocity),
            float(secs_since),
        ])

    return np.array(features, dtype=np.float32)


def _load_channel(path: Path) -> np.ndarray | None:
    try:
        arr = np.load(path)
        if arr.ndim == 1:
            return arr
        return arr[:, 0]  # take first column if multi-channel
    except Exception as e:
        logger.debug("Failed to load %s: %s", path, e)
        return None


def _load_anomaly_labels(data_dir: Path) -> dict[str, list[list[int]]]:
    """Load labeled anomaly windows from labeled_anomalies.csv."""
    import csv
    labels = {}
    csv_path = data_dir / "labeled_anomalies.csv"
    if not csv_path.exists():
        return labels
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            chan = row.get("chan_id", "")
            seqs_str = row.get("anomaly_sequences", "[]")
            try:
                import ast
                seqs = ast.literal_eval(seqs_str)
                labels[chan] = seqs
            except Exception:
                pass
    return labels


def train(data_dir: str, output_dir: str, contamination: float = 0.05) -> None:
    try:
        import joblib
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
    except ImportError:
        raise RuntimeError("scikit-learn and joblib required. Run: pip install scikit-learn joblib")

    data = Path(data_dir)
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    anomaly_labels = _load_anomaly_labels(data)
    train_dir = data / "train"

    if not train_dir.exists():
        raise FileNotFoundError(f"SMAP/MSL train directory not found: {train_dir}")

    # Group channels by C2EventType prefix
    type_features: dict[str, list[np.ndarray]] = {evt: [] for evt in _C2_EVENT_TYPES}
    all_features: list[np.ndarray] = []

    npy_files = list(train_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files in {train_dir}")

    logger.info("Extracting features from %d SMAP/MSL channels ...", len(npy_files))

    for npy_path in npy_files:
        chan = npy_path.stem   # e.g. "P-1"
        series = _load_channel(npy_path)
        if series is None or len(series) < 100:
            continue

        # Mask out labeled anomaly windows (train on normal only)
        normal_series = series.copy()
        for seq in anomaly_labels.get(chan, []):
            if len(seq) == 2:
                s, e = seq
                normal_series[s:e+1] = np.nan

        # Drop NaN windows
        valid_mask = ~np.isnan(normal_series)
        if valid_mask.sum() < 100:
            continue

        # Interpolate gaps
        normal_series = series[valid_mask]  # simplify: just use valid segments

        features = _extract_features(normal_series)
        if len(features) < 10:
            continue

        # Remove any NaN/inf rows
        finite_mask = np.isfinite(features).all(axis=1)
        features = features[finite_mask]
        if len(features) < 5:
            continue

        all_features.append(features)

        # Assign to event type bucket
        prefix = chan.split("-")[0]
        evt = _CHANNEL_TO_EVENT_TYPE.get(prefix, "PEER_OBSERVATION")
        type_features[evt].append(features)

    if not all_features:
        raise RuntimeError("No usable features extracted from SMAP/MSL data")

    all_X = np.vstack(all_features)
    logger.info("Total feature windows: %d", len(all_X))

    trained_types = []

    # Train per-event-type models
    for evt, feat_list in type_features.items():
        if not feat_list:
            continue
        X = np.vstack(feat_list)
        if len(X) < 20:
            continue

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("iso",    IsolationForest(
                n_estimators=200,
                contamination=contamination,
                max_samples="auto",
                random_state=42,
            )),
        ])
        pipe.fit(X)

        model_path = out / f"anomaly_detector_{evt}.joblib"
        joblib.dump(pipe, model_path)
        trained_types.append(evt)
        logger.info("Trained %s → %s (%d windows)", evt, model_path.name, len(X))

    # Train global fallback model
    global_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("iso",    IsolationForest(
            n_estimators=300,
            contamination=contamination,
            max_samples=min(len(all_X), 5000),
            random_state=42,
        )),
    ])
    global_pipe.fit(all_X)
    global_path = out / "anomaly_detector_global.joblib"
    joblib.dump(global_pipe, global_path)
    logger.info("Global model → %s (%d windows)", global_path.name, len(all_X))

    # Write metadata
    meta = {
        "trained_at":   datetime.now(timezone.utc).isoformat(),
        "event_types":  trained_types,
        "n_features":   5,
        "feature_names": ["obs_last_5m", "obs_last_30m", "obs_last_90m",
                           "velocity", "seconds_since_last"],
        "contamination": contamination,
        "training_windows": len(all_X),
        "source": "NASA SMAP/MSL telemetry anomaly dataset",
    }
    meta_path = out / "anomaly_detector_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Anomaly detector training complete. Models at %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Train C2 anomaly detector on SMAP/MSL")
    p.add_argument("--data",          required=True)
    p.add_argument("--output",        required=True)
    p.add_argument("--contamination", type=float, default=0.05,
                   help="IsolationForest contamination fraction (default: 0.05)")
    args = p.parse_args()
    train(args.data, args.output, args.contamination)
