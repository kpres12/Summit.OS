"""
Train the Summit.OS sequence anomaly detector.

Detects anomalous entity behavior in time-series telemetry sequences.
Catches things cameras miss — a vessel that suddenly stops in a shipping lane,
a drone flying erratically, a person who stops moving in a flood zone.

The model operates on a 16-float feature vector extracted from a sliding
window of 10 consecutive telemetry pings for a single entity.

Outputs:
  packages/ml/models/sequence_anomaly.onnx          — runtime model (IsolationForest)
  packages/ml/models/sequence_anomaly_feature_names.json

Feature vector (16 floats, index-stable):
  [0]  mean_speed_mps          — average speed over the window
  [1]  speed_std               — std deviation of speed (high = erratic)
  [2]  max_speed_mps           — peak speed in window
  [3]  heading_change_mean_deg — mean absolute heading change between steps
  [4]  heading_change_std      — std of heading changes (high = erratic)
  [5]  stop_duration_s         — seconds entity has been stationary (speed < 0.5 m/s)
  [6]  position_variance_m     — variance of lat/lon positions converted to metres
  [7]  altitude_change_mean_m  — mean altitude change per step (UAV signal)
  [8]  altitude_variance_m     — variance of altitude (UAV instability)
  [9]  time_gap_mean_s         — mean time between telemetry pings
  [10] time_gap_max_s          — single longest gap (connectivity loss signal)
  [11] entity_type_uav         — 1 if UAV
  [12] entity_type_vessel      — 1 if vessel
  [13] entity_type_person      — 1 if person / survivor
  [14] entity_type_vehicle     — 1 if ground vehicle
  [15] mission_active          — 1 if entity currently has an active mission

Usage:
  python train_sequence_anomaly.py
  python train_sequence_anomaly.py --samples 80000 --output-dir packages/ml/models
"""

import onnx_compat  # noqa: F401 — Python 3.14 compat patch
import argparse
import json
import os
import sys

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

FEATURE_DIM = 16
FEATURE_NAMES = [
    "mean_speed_mps",
    "speed_std",
    "max_speed_mps",
    "heading_change_mean_deg",
    "heading_change_std",
    "stop_duration_s",
    "position_variance_m",
    "altitude_change_mean_m",
    "altitude_variance_m",
    "time_gap_mean_s",
    "time_gap_max_s",
    "entity_type_uav",
    "entity_type_vessel",
    "entity_type_person",
    "entity_type_vehicle",
    "mission_active",
]

# IsolationForest convention: -1 = anomaly, 1 = normal.
# We map that to binary labels for evaluation: 1 = anomaly, 0 = normal.

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _clip(v, lo, hi):
    return float(np.clip(v, lo, hi))


def _make_uav_on_mission(rng, n: int) -> np.ndarray:
    """UAV flying an active mission: steady speed, stable heading, 1-s pings."""
    speed = rng.uniform(8, 15, n)
    return np.column_stack([
        speed,                                      # mean_speed_mps
        rng.uniform(0.2, 1.5, n),                   # speed_std   — tight
        speed + rng.uniform(0.5, 2.5, n),           # max_speed_mps
        rng.uniform(2, 15, n),                      # heading_change_mean_deg
        rng.uniform(1, 8, n),                       # heading_change_std
        np.zeros(n),                                # stop_duration_s
        rng.uniform(0.5, 8, n),                     # position_variance_m
        rng.uniform(0.2, 1.5, n),                   # altitude_change_mean_m
        rng.uniform(0.5, 5, n),                     # altitude_variance_m
        np.ones(n) * 1.0,                           # time_gap_mean_s
        rng.uniform(1.0, 2.5, n),                   # time_gap_max_s
        np.ones(n),                                 # entity_type_uav
        np.zeros(n), np.zeros(n), np.zeros(n),      # vessel / person / vehicle
        np.ones(n),                                 # mission_active
    ])


def _make_uav_hovering(rng, n: int) -> np.ndarray:
    """UAV holding position: near-zero speed, minimal position variance."""
    return np.column_stack([
        rng.uniform(0, 2, n),
        rng.uniform(0.1, 0.8, n),
        rng.uniform(0.5, 2.5, n),
        rng.uniform(5, 25, n),
        rng.uniform(2, 12, n),
        rng.uniform(0, 20, n),
        rng.uniform(0.1, 4, n),
        rng.uniform(0.1, 0.8, n),
        rng.uniform(0.1, 3, n),
        np.ones(n) * 1.0,
        rng.uniform(1.0, 2.0, n),
        np.ones(n),
        np.zeros(n), np.zeros(n), np.zeros(n),
        rng.integers(0, 2, n).astype(float),
    ])


def _make_vessel_underway(rng, n: int) -> np.ndarray:
    """Vessel making way: moderate speed, slow heading changes, regular AIS pings."""
    speed = rng.uniform(3, 8, n)
    return np.column_stack([
        speed,
        rng.uniform(0.2, 1.0, n),
        speed + rng.uniform(0.2, 1.5, n),
        rng.uniform(0.5, 8, n),
        rng.uniform(0.3, 3, n),
        np.zeros(n),
        rng.uniform(5, 40, n),
        np.zeros(n),
        np.zeros(n),
        rng.uniform(2, 6, n),
        rng.uniform(3, 10, n),
        np.zeros(n),
        np.ones(n),
        np.zeros(n), np.zeros(n),
        rng.integers(0, 2, n).astype(float),
    ])


def _make_vessel_anchored(rng, n: int) -> np.ndarray:
    """Vessel at anchor: essentially stationary, position variance from swinging."""
    return np.column_stack([
        rng.uniform(0, 0.5, n),
        rng.uniform(0.05, 0.3, n),
        rng.uniform(0.1, 0.7, n),
        rng.uniform(0, 5, n),
        rng.uniform(0, 3, n),
        rng.uniform(30, 600, n),
        rng.uniform(2, 10, n),
        np.zeros(n),
        np.zeros(n),
        rng.uniform(2, 6, n),
        rng.uniform(3, 12, n),
        np.zeros(n),
        np.ones(n),
        np.zeros(n), np.zeros(n),
        np.zeros(n),
    ])


def _make_ground_vehicle(rng, n: int) -> np.ndarray:
    """Ground vehicle driving: variable speed, larger heading changes at intersections."""
    speed = rng.uniform(5, 20, n)
    return np.column_stack([
        speed,
        rng.uniform(1, 4, n),
        speed + rng.uniform(1, 5, n),
        rng.uniform(5, 40, n),
        rng.uniform(3, 20, n),
        rng.uniform(0, 30, n),
        rng.uniform(5, 80, n),
        np.zeros(n),
        np.zeros(n),
        rng.uniform(1, 5, n),
        rng.uniform(2, 12, n),
        np.zeros(n), np.zeros(n),
        np.zeros(n),
        np.ones(n),
        rng.integers(0, 2, n).astype(float),
    ])


def _make_person_walking(rng, n: int) -> np.ndarray:
    """Person on foot: slow speed, irregular heading changes."""
    speed = rng.uniform(0.8, 2.0, n)
    return np.column_stack([
        speed,
        rng.uniform(0.1, 0.6, n),
        speed + rng.uniform(0.1, 0.5, n),
        rng.uniform(10, 60, n),
        rng.uniform(5, 30, n),
        rng.uniform(0, 15, n),
        rng.uniform(1, 20, n),
        np.zeros(n),
        np.zeros(n),
        rng.uniform(1, 4, n),
        rng.uniform(2, 8, n),
        np.zeros(n), np.zeros(n),
        np.ones(n),
        np.zeros(n),
        rng.integers(0, 2, n).astype(float),
    ])


# --- Anomalous sequences ---------------------------------------------------

def _make_erratic_uav(rng, n: int) -> np.ndarray:
    """UAV with wildly varying speed and heading — likely malfunction or evasion."""
    speed = rng.uniform(2, 20, n)
    return np.column_stack([
        speed,
        rng.uniform(8, 18, n),                      # speed_std >> normal
        speed + rng.uniform(5, 15, n),
        rng.uniform(40, 120, n),                    # heading_change_mean_deg >> normal
        rng.uniform(60, 100, n),                    # heading_change_std >> normal
        np.zeros(n),
        rng.uniform(5, 30, n),
        rng.uniform(2, 10, n),
        rng.uniform(5, 20, n),
        np.ones(n) * 1.0,
        rng.uniform(1.0, 3.0, n),
        np.ones(n),
        np.zeros(n), np.zeros(n), np.zeros(n),
        rng.integers(0, 2, n).astype(float),
    ])


def _make_uav_connectivity_loss(rng, n: int) -> np.ndarray:
    """UAV with telemetry dropout — long gap in ping stream."""
    speed = rng.uniform(5, 12, n)
    return np.column_stack([
        speed,
        rng.uniform(0.5, 2, n),
        speed + rng.uniform(0.5, 2, n),
        rng.uniform(3, 20, n),
        rng.uniform(2, 10, n),
        np.zeros(n),
        rng.uniform(1, 10, n),
        rng.uniform(0.2, 1, n),
        rng.uniform(0.5, 3, n),
        rng.uniform(15, 60, n),                     # time_gap_mean elevated
        rng.uniform(30, 120, n),                    # time_gap_max >> normal
        np.ones(n),
        np.zeros(n), np.zeros(n), np.zeros(n),
        np.ones(n),
    ])


def _make_vessel_stopped_no_mission(rng, n: int) -> np.ndarray:
    """Vessel stationary in shipping lane with no assigned mission — suspicious."""
    return np.column_stack([
        rng.uniform(0, 0.3, n),
        rng.uniform(0.05, 0.2, n),
        rng.uniform(0.1, 0.5, n),
        rng.uniform(0, 5, n),
        rng.uniform(0, 3, n),
        rng.uniform(300, 1800, n),                  # stop_duration >> normal vessel
        rng.uniform(2, 15, n),
        np.zeros(n),
        np.zeros(n),
        rng.uniform(2, 6, n),
        rng.uniform(3, 10, n),
        np.zeros(n),
        np.ones(n),
        np.zeros(n), np.zeros(n),
        np.zeros(n),                                # no mission
    ])


def _make_fast_vessel(rng, n: int) -> np.ndarray:
    """Rapidly accelerating vessel — suspicious fast boat, possible smuggling."""
    speed = rng.uniform(15, 25, n)
    return np.column_stack([
        speed,
        rng.uniform(3, 8, n),
        speed + rng.uniform(3, 10, n),              # max_speed > 15 m/s
        rng.uniform(5, 30, n),
        rng.uniform(3, 15, n),
        np.zeros(n),
        rng.uniform(50, 200, n),
        np.zeros(n),
        np.zeros(n),
        rng.uniform(1, 4, n),
        rng.uniform(2, 8, n),
        np.zeros(n),
        np.ones(n),
        np.zeros(n), np.zeros(n),
        np.zeros(n),
    ])


def _make_person_stationary_flood(rng, n: int) -> np.ndarray:
    """Person stationary in a flood zone with an active mission — likely trapped."""
    return np.column_stack([
        rng.uniform(0, 0.3, n),
        rng.uniform(0.05, 0.2, n),
        rng.uniform(0.1, 0.4, n),
        rng.uniform(2, 20, n),
        rng.uniform(1, 10, n),
        rng.uniform(120, 900, n),                   # stop_duration >> normal person
        rng.uniform(0.5, 5, n),
        np.zeros(n),
        np.zeros(n),
        rng.uniform(1, 4, n),
        rng.uniform(2, 8, n),
        np.zeros(n), np.zeros(n),
        np.ones(n),
        np.zeros(n),
        np.ones(n),                                 # mission_active — SAR in progress
    ])


def _make_gps_spoofing(rng, n: int) -> np.ndarray:
    """GPS spoofing signature: position variance spikes but reported speed is near zero."""
    return np.column_stack([
        rng.uniform(0, 0.5, n),                     # speed ~ 0
        rng.uniform(0.1, 0.5, n),
        rng.uniform(0.2, 1.0, n),
        rng.uniform(5, 30, n),
        rng.uniform(3, 15, n),
        rng.uniform(30, 300, n),
        rng.uniform(500, 5000, n),                  # position_variance wildly high
        np.zeros(n),
        np.zeros(n),
        rng.uniform(1, 5, n),
        rng.uniform(2, 10, n),
        rng.integers(0, 2, n).astype(float),
        rng.integers(0, 2, n).astype(float),
        rng.integers(0, 2, n).astype(float),
        rng.integers(0, 2, n).astype(float),
        rng.integers(0, 2, n).astype(float),
    ])


def generate_data(n_total: int = 80000, seed: int = 42):
    """
    Generate synthetic telemetry window feature vectors.

    Returns:
        X      — float32 array of shape (n_total, FEATURE_DIM)
        labels — int array: 0 = normal, 1 = anomalous  (for evaluation only)
    """
    rng = _rng(seed)

    # Target: ~5% anomalies to match IsolationForest contamination=0.05
    n_anomaly = int(n_total * 0.05)
    n_normal = n_total - n_anomaly

    # --- Normal samples (6 archetypes, roughly equal split) -----------------
    per_normal = n_normal // 6
    remainder = n_normal - per_normal * 6
    normal_counts = [per_normal] * 6
    normal_counts[-1] += remainder

    normal_builders = [
        _make_uav_on_mission,
        _make_uav_hovering,
        _make_vessel_underway,
        _make_vessel_anchored,
        _make_ground_vehicle,
        _make_person_walking,
    ]
    X_normal_parts = [fn(rng, cnt) for fn, cnt in zip(normal_builders, normal_counts)]
    X_normal = np.vstack(X_normal_parts).astype(np.float32)
    y_normal = np.zeros(len(X_normal), dtype=np.int32)

    # --- Anomalous samples (6 archetypes) ------------------------------------
    per_anom = n_anomaly // 6
    remainder_a = n_anomaly - per_anom * 6
    anom_counts = [per_anom] * 6
    anom_counts[-1] += remainder_a

    anom_builders = [
        _make_erratic_uav,
        _make_uav_connectivity_loss,
        _make_vessel_stopped_no_mission,
        _make_fast_vessel,
        _make_person_stationary_flood,
        _make_gps_spoofing,
    ]
    X_anom_parts = [fn(rng, cnt) for fn, cnt in zip(anom_builders, anom_counts)]
    X_anom = np.vstack(X_anom_parts).astype(np.float32)
    y_anom = np.ones(len(X_anom), dtype=np.int32)

    X = np.vstack([X_normal, X_anom])
    y = np.concatenate([y_normal, y_anom])

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx].astype(np.float32), y[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_real_csv(csv_path: str):
    """Load real observations and map to sequence anomaly feature space (16 floats).

    Real CSV contains single-point observations, not telemetry windows.  We map
    each row to the 16-float feature vector using the entity's class to set entity-
    type flags and sensible defaults for the kinematic features.  Rows are treated
    as normal (non-anomalous) background samples since the CSV represents confirmed
    real-world observations rather than flagged anomalies.
    """
    import csv as _csv
    X = []
    try:
        with open(csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    cls = row.get("class", "").strip().lower()
                    conf = float(row["confidence"])

                    # Map class → entity type flags
                    is_uav = any(kw in cls for kw in ["uav", "drone", "aircraft"])
                    is_vessel = any(kw in cls for kw in ["boat", "vessel", "ship", "marine"])
                    is_person = any(kw in cls for kw in ["person", "human", "survivor", "missing"])
                    is_vehicle = any(kw in cls for kw in ["vehicle", "truck", "car"])

                    # If no entity type matches, treat as person (most common in SAR context)
                    if not any([is_uav, is_vessel, is_person, is_vehicle]):
                        is_person = True

                    # Build representative normal kinematics based on entity type
                    if is_uav:
                        feat = [10.0, 0.8, 12.0, 8.0, 4.0, 0.0, 3.0, 0.5, 1.5, 1.0, 2.0,
                                1.0, 0.0, 0.0, 0.0, 1.0]
                    elif is_vessel:
                        feat = [5.0, 0.5, 6.5, 3.0, 1.5, 0.0, 20.0, 0.0, 0.0, 4.0, 7.0,
                                0.0, 1.0, 0.0, 0.0, 0.0]
                    elif is_vehicle:
                        feat = [12.0, 2.0, 17.0, 20.0, 10.0, 10.0, 40.0, 0.0, 0.0, 2.0, 5.0,
                                0.0, 0.0, 0.0, 1.0, 0.0]
                    else:  # person
                        feat = [1.2, 0.3, 1.8, 30.0, 15.0, 5.0, 8.0, 0.0, 0.0, 2.0, 4.0,
                                0.0, 0.0, 1.0, 0.0, 0.0]

                    # Scale mean_speed by confidence as a mild proxy for activity level
                    feat[0] = feat[0] * conf
                    X.append(feat)
                except (ValueError, KeyError):
                    continue
        print(f"  Loaded {len(X)} real samples from {os.path.basename(csv_path)}")
    except FileNotFoundError:
        print(f"  CSV not found: {csv_path}")
    return np.array(X, dtype=np.float32) if X else None


def train(n_samples: int = 80000, output_dir: str = None, real_csv: str = None):
    """Train IsolationForest anomaly detector and export to ONNX."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {n_samples:,} synthetic telemetry window samples...")
    X, y_labels = generate_data(n_total=n_samples)
    n_normal = int((y_labels == 0).sum())
    n_anom = int((y_labels == 1).sum())
    print(f"  Normal: {n_normal:,}  |  Anomalous: {n_anom:,}  "
          f"(contamination rate: {n_anom / len(X):.3f})")

    if real_csv:
        X_real = load_real_csv(real_csv)
        if X_real is not None:
            # Cap real normal samples at 2x the synthetic normal count to avoid skewing
            cap = n_normal * 2
            if len(X_real) > cap:
                import random as _rand
                _rand.seed(42)
                indices = list(range(len(X_real)))
                _rand.shuffle(indices)
                X_real = X_real[indices[:cap]]
            # Real observations are treated as normal (label=0)
            y_real = np.zeros(len(X_real), dtype=np.int32)
            X = np.vstack([X, X_real])
            y_labels = np.concatenate([y_labels, y_real])
            print(f"  Real data blended: {len(X_real)} normal samples")
            print(f"  Combined: {len(X)} total samples (real + synthetic)")

    # IsolationForest is trained on ALL data — no labels required.
    # contamination=0.05 tells it to expect ~5% outliers when scoring.
    print("\nFitting IsolationForest(n_estimators=200, contamination=0.05)...")
    clf = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X)

    # --- Evaluation against synthetic ground-truth labels -------------------
    # IsolationForest predict: -1 = anomaly, 1 = normal
    preds_raw = clf.predict(X)              # -1 or +1
    scores = clf.score_samples(X)          # lower = more anomalous

    preds_binary = (preds_raw == -1).astype(int)  # 1 = predicted anomaly

    # Anomaly score distribution
    normal_scores = scores[y_labels == 0]
    anom_scores = scores[y_labels == 1]
    print("\nAnomaly score distribution (lower = more anomalous):")
    print(f"  Normal    — mean: {normal_scores.mean():.4f}  std: {normal_scores.std():.4f}  "
          f"min: {normal_scores.min():.4f}  max: {normal_scores.max():.4f}")
    print(f"  Anomalous — mean: {anom_scores.mean():.4f}  std: {anom_scores.std():.4f}  "
          f"min: {anom_scores.min():.4f}  max: {anom_scores.max():.4f}")

    # AUC-ROC (negate scores so higher = more anomalous for roc_auc_score)
    auc = roc_auc_score(y_labels, -scores)
    print(f"\nAUC-ROC (normal vs anomalous): {auc:.4f}")

    # Precision / recall at the model's built-in threshold
    from sklearn.metrics import precision_score, recall_score, f1_score
    prec = precision_score(y_labels, preds_binary, zero_division=0)
    rec = recall_score(y_labels, preds_binary, zero_division=0)
    f1 = f1_score(y_labels, preds_binary, zero_division=0)
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")

    # Top features driving anomalies — use mean absolute deviation from the
    # mean as a proxy (features with widest normal/anomalous spread are most
    # discriminative).
    normal_mean = X[y_labels == 0].mean(axis=0)
    anom_mean = X[y_labels == 1].mean(axis=0)
    delta = np.abs(anom_mean - normal_mean)
    ranked = np.argsort(delta)[::-1]
    print("\nTop anomaly-discriminating features (mean shift normal → anomalous):")
    for rank, idx in enumerate(ranked[:8], 1):
        print(f"  {rank}. {FEATURE_NAMES[idx]:<30}  "
              f"normal={normal_mean[idx]:.3f}  anomalous={anom_mean[idx]:.3f}  "
              f"delta={delta[idx]:.3f}")

    # --- ONNX export (matches project skl2onnx pattern) ---------------------
    onnx_path = os.path.join(output_dir, "sequence_anomaly.onnx")
    initial_type = [("float_input", FloatTensorType([None, FEATURE_DIM]))]
    onnx_model = convert_sklearn(
        clf, initial_types=initial_type,
        target_opset={"": 12, "ai.onnx.ml": 3},
    )
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"\nModel saved: {onnx_path}")

    # --- Feature names JSON -------------------------------------------------
    names_path = os.path.join(output_dir, "sequence_anomaly_feature_names.json")
    with open(names_path, "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    print(f"Feature names saved: {names_path}")

    # --- Smoke test via ONNX runtime ----------------------------------------
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        smoke_cases = [
            ("UAV on mission (normal)",
             [12.0, 0.8, 14.2, 8.0, 4.0, 0.0, 3.0, 0.5, 1.5, 1.0, 1.8, 1, 0, 0, 0, 1]),
            ("Erratic UAV (anomaly)",
             [9.0, 14.0, 22.0, 90.0, 85.0, 0.0, 12.0, 6.0, 15.0, 1.0, 2.0, 1, 0, 0, 0, 1]),
            ("Vessel underway (normal)",
             [5.5, 0.5, 6.8, 3.0, 1.5, 0.0, 20.0, 0.0, 0.0, 4.0, 6.0, 0, 1, 0, 0, 0]),
            ("Vessel stopped no mission (anomaly)",
             [0.1, 0.1, 0.2, 1.0, 0.8, 600.0, 5.0, 0.0, 0.0, 4.0, 7.0, 0, 1, 0, 0, 0]),
            ("GPS spoofing (anomaly)",
             [0.2, 0.3, 0.4, 10.0, 8.0, 60.0, 3500.0, 0.0, 0.0, 2.0, 4.0, 0, 1, 0, 0, 0]),
        ]
        print("\nSmoke test (label_map: -1=anomaly, 1=normal):")
        for label, feat in smoke_cases:
            x = np.array([feat], dtype=np.float32)
            result = sess.run(None, {"float_input": x})
            pred_label = int(np.asarray(result[0]).flat[0])
            tag = "ANOMALY" if pred_label == -1 else "normal"
            print(f"  {label:<40} → {tag}")
    except ImportError:
        print("\nNote: onnxruntime not installed — skipping smoke test.")

    return onnx_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Summit.OS sequence anomaly detector and export to ONNX."
    )
    parser.add_argument(
        "--samples", type=int, default=80000,
        help="Total number of synthetic telemetry window samples (default: 80000)"
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="Directory to write .onnx and .json files (default: packages/ml/models/)"
    )
    parser.add_argument(
        "--real-csv", dest="real_csv", default=None,
        help="Path to real observations CSV to blend with synthetic data"
    )
    args = parser.parse_args()
    train(n_samples=args.samples, output_dir=args.output_dir, real_csv=args.real_csv)
