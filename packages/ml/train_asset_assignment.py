"""
Train the Summit.OS asset assignment scorer.

Given a (mission, candidate asset) pair, outputs a score 0-1 indicating how
well-suited the asset is for the mission.  At runtime the mission planner runs
inference for every available asset and dispatches the highest scorer,
replacing the previous "nearest available" heuristic.

Feature vector (15 features, one row per asset-mission pair):
  [0]  mission_type        0-6  (SURVEY/MONITOR/SEARCH/PERIMETER/ORBIT/DELIVER/INSPECT)
  [1]  asset_type          0-4  (QUADCOPTER/FIXED_WING/VTOL/GROUND_UGV/TOWER)
  [2]  battery_pct         0-1  (raw 0-100 normalised)
  [3]  distance_km         0-1  (raw 0-200 normalised)
  [4]  capability_thermal  0/1
  [5]  capability_rgb      0/1
  [6]  capability_lidar    0/1
  [7]  capability_payload  0/1
  [8]  currently_idle      0/1
  [9]  endurance_min       0-1  (raw 0-180 normalised)
  [10] wind_speed_mps      0-1  (raw 0-20 normalised)
  [11] time_of_day_sin     cyclical hour encoding
  [12] time_of_day_cos     cyclical hour encoding
  [13] terrain_difficulty  0=flat, 1=mountainous
  [14] mission_priority    0/0.33/0.66/1.0

Outputs:
  packages/ml/models/asset_assignment.onnx
  packages/ml/models/asset_assignment_feature_names.json

Usage:
  python train_asset_assignment.py
  python train_asset_assignment.py --samples 100000
"""

import onnx_compat  # noqa: F401 — Python 3.14 compat patch
import argparse
import json
import os
import sys

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models")

FEATURE_NAMES = [
    "mission_type",
    "asset_type",
    "battery_pct",
    "distance_km",
    "capability_thermal",
    "capability_rgb",
    "capability_lidar",
    "capability_payload",
    "currently_idle",
    "endurance_min",
    "wind_speed_mps",
    "time_of_day_sin",
    "time_of_day_cos",
    "terrain_difficulty",
    "mission_priority",
]
FEATURE_DIM = len(FEATURE_NAMES)

# Encoded mission types
MISSION_SURVEY    = 0
MISSION_MONITOR   = 1
MISSION_SEARCH    = 2
MISSION_PERIMETER = 3
MISSION_ORBIT     = 4
MISSION_DELIVER   = 5
MISSION_INSPECT   = 6

# Encoded asset types
ASSET_QUADCOPTER  = 0
ASSET_FIXED_WING  = 1
ASSET_VTOL        = 2
ASSET_GROUND_UGV  = 3
ASSET_TOWER       = 4

PRIORITY_MAP = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}  # LOW/MEDIUM/HIGH/CRITICAL


def _score_pair(mission_type, asset_type, battery_pct_norm, distance_km_norm,
                cap_thermal, cap_rgb, cap_lidar, cap_payload, idle,
                endurance_norm, wind_norm, tod_sin, tod_cos,
                terrain, priority, rng):
    """
    Return (features, label) for a single asset-mission pair.
    Label 1 = good assignment, 0 = poor assignment.
    A continuous suitability score is thresholded with controlled noise so the
    model learns distributions rather than hard rules.
    """
    score = 0.5  # baseline

    battery_pct = battery_pct_norm * 100.0
    distance_km = distance_km_norm * 200.0
    wind_mps    = wind_norm * 20.0

    # ---- mission-specific rules ----------------------------------------
    if mission_type == MISSION_DELIVER:
        if cap_payload:
            score += 0.35
        else:
            score -= 0.45  # cannot deliver without payload
        if asset_type == ASSET_GROUND_UGV:
            score -= 0.10  # slow for time-sensitive deliveries
        if asset_type in (ASSET_VTOL, ASSET_FIXED_WING):
            score += 0.10

    elif mission_type == MISSION_SEARCH:
        score += endurance_norm * 0.30   # long endurance critical
        if cap_thermal:
            score += 0.20               # find people in smoke/night
        if cap_rgb:
            score += 0.10
        if asset_type == ASSET_TOWER:
            score -= 0.20               # towers can't search wide area

    elif mission_type == MISSION_ORBIT:
        if asset_type in (ASSET_FIXED_WING, ASSET_VTOL):
            score += 0.30               # efficient for sustained loiter
        elif asset_type == ASSET_QUADCOPTER:
            score -= 0.15               # inefficient hover
        elif asset_type == ASSET_TOWER:
            score -= 0.40               # immobile
        score += endurance_norm * 0.15

    elif mission_type == MISSION_INSPECT:
        if cap_rgb:
            score += 0.30
        else:
            score -= 0.25
        if cap_lidar:
            score += 0.15
        if asset_type == ASSET_TOWER:
            score -= 0.30               # can't get close enough

    elif mission_type == MISSION_SURVEY:
        score += endurance_norm * 0.20
        if cap_rgb:
            score += 0.15
        if cap_lidar:
            score += 0.10
        if asset_type == ASSET_TOWER:
            score -= 0.25

    elif mission_type == MISSION_MONITOR:
        if cap_thermal:
            score += 0.15
        if cap_rgb:
            score += 0.10
        if asset_type == ASSET_TOWER:
            score += 0.15               # fixed monitoring point is fine

    elif mission_type == MISSION_PERIMETER:
        score += endurance_norm * 0.15
        if asset_type in (ASSET_FIXED_WING, ASSET_VTOL):
            score += 0.20

    # ---- universal penalties -------------------------------------------
    # Low battery
    if battery_pct < 20.0:
        score -= 0.40
    elif battery_pct < 35.0:
        score -= 0.15

    # High wind vs fixed-wing
    if wind_mps > 12.0 and asset_type == ASSET_FIXED_WING:
        score -= 0.25

    # Large distance vs quadcopter (short range)
    if distance_km > 40.0 and asset_type == ASSET_QUADCOPTER:
        score -= 0.20 + (distance_km - 40.0) / 200.0 * 0.15

    # Busy asset is worse than idle
    if not idle:
        score -= 0.10

    # Night penalty if no thermal (tod_sin/cos encode hour; sin≈-1 is ~18:00 UTC)
    # Approximate night: cos(hour * 2π/24) roughly negative → late evening/night
    is_night = tod_cos < -0.3
    if is_night and not cap_thermal:
        if mission_type in (MISSION_SEARCH, MISSION_INSPECT):
            score -= 0.25

    # Terrain difficulty
    if terrain > 0.5 and asset_type == ASSET_GROUND_UGV:
        score -= 0.20

    # Priority boost — high-priority missions penalise borderline assets more
    if priority >= 0.66 and battery_pct < 50.0:
        score -= 0.10

    # ---- controlled noise so the model learns distributions ------------
    score += rng.normal(0, 0.12)

    label = int(score >= 0.5)
    features = np.array([
        mission_type,
        asset_type,
        battery_pct_norm,
        distance_km_norm,
        float(cap_thermal),
        float(cap_rgb),
        float(cap_lidar),
        float(cap_payload),
        float(idle),
        endurance_norm,
        wind_norm,
        tod_sin,
        tod_cos,
        float(terrain),
        float(priority),
    ], dtype=np.float32)

    return features, label


def generate_samples(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)

    X, y = [], []

    # Capability sets per asset type (rough real-world defaults)
    asset_cap_profiles = {
        ASSET_QUADCOPTER:  dict(thermal=0.45, rgb=0.90, lidar=0.25, payload=0.30),
        ASSET_FIXED_WING:  dict(thermal=0.55, rgb=0.85, lidar=0.20, payload=0.15),
        ASSET_VTOL:        dict(thermal=0.60, rgb=0.85, lidar=0.35, payload=0.50),
        ASSET_GROUND_UGV:  dict(thermal=0.40, rgb=0.80, lidar=0.60, payload=0.70),
        ASSET_TOWER:       dict(thermal=0.75, rgb=0.90, lidar=0.30, payload=0.00),
    }

    for _ in range(n):
        mission_type   = int(rng.integers(0, 7))
        asset_type     = int(rng.integers(0, 5))
        battery_norm   = float(rng.uniform(0.05, 1.0))
        distance_norm  = float(rng.beta(2, 5))          # skewed toward short distances
        cap_profile    = asset_cap_profiles[asset_type]
        cap_thermal    = int(rng.random() < cap_profile["thermal"])
        cap_rgb        = int(rng.random() < cap_profile["rgb"])
        cap_lidar      = int(rng.random() < cap_profile["lidar"])
        cap_payload    = int(rng.random() < cap_profile["payload"])
        idle           = int(rng.random() < 0.65)
        endurance_norm = float(rng.beta(3, 2))          # most assets have decent endurance
        wind_norm      = float(rng.beta(1.5, 4))        # usually low wind
        hour           = float(rng.uniform(0, 24))
        tod_sin        = float(np.sin(2 * np.pi * hour / 24.0))
        tod_cos        = float(np.cos(2 * np.pi * hour / 24.0))
        terrain        = int(rng.random() < 0.30)       # 30% mountainous
        priority_idx   = int(rng.choice([0, 1, 2, 3], p=[0.25, 0.35, 0.25, 0.15]))
        priority       = PRIORITY_MAP[priority_idx]

        feat, label = _score_pair(
            mission_type, asset_type, battery_norm, distance_norm,
            cap_thermal, cap_rgb, cap_lidar, cap_payload, idle,
            endurance_norm, wind_norm, tod_sin, tod_cos,
            terrain, priority, rng,
        )
        X.append(feat)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def load_real_csv(csv_path: str):
    """Load real observations and map to asset assignment feature space (15 floats)."""
    import csv as _csv
    X, y = [], []

    mission_type_map = {
        "SURVEY": MISSION_SURVEY,
        "MONITOR": MISSION_MONITOR,
        "SEARCH": MISSION_SEARCH,
        "PERIMETER": MISSION_PERIMETER,
        "ORBIT": MISSION_ORBIT,
        "DELIVER": MISSION_DELIVER,
        "INSPECT": MISSION_INSPECT,
    }
    priority_map = {"LOW": 0.0, "MEDIUM": 0.33, "HIGH": 0.66, "CRITICAL": 1.0}
    # Proxy label: CRITICAL/HIGH risk → good-match assignment needed (1); LOW/MEDIUM → 0
    risk_to_label = {"LOW": 0, "MEDIUM": 0, "HIGH": 1, "CRITICAL": 1}

    try:
        with open(csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    mission_str = row.get("mission_type", "SURVEY").strip().upper()
                    mission_type = float(mission_type_map.get(mission_str, MISSION_SURVEY))
                    risk_str = row.get("risk_level", "MEDIUM").strip().upper()
                    if risk_str not in priority_map:
                        continue
                    priority = priority_map[risk_str]
                    conf = float(row["confidence"])
                    feat = np.array([
                        mission_type,         # mission_type
                        float(ASSET_VTOL),    # asset_type (VTOL default)
                        0.75,                 # battery_pct
                        0.15,                 # distance_km (normalized)
                        1.0,                  # capability_thermal (VTOL default)
                        1.0,                  # capability_rgb
                        0.0,                  # capability_lidar
                        0.5,                  # capability_payload (50/50)
                        1.0,                  # currently_idle
                        0.70,                 # endurance_min (normalized)
                        0.20,                 # wind_speed_mps (normalized)
                        float(np.sin(2 * np.pi * 12 / 24.0)),  # time_of_day_sin
                        float(np.cos(2 * np.pi * 12 / 24.0)),  # time_of_day_cos
                        0.20,                 # terrain_difficulty
                        priority,             # mission_priority
                    ], dtype=np.float32)
                    label = risk_to_label[risk_str]
                    X.append(feat)
                    y.append(label)
                except (ValueError, KeyError):
                    continue
        print(f"  Loaded {len(X)} real samples from {os.path.basename(csv_path)}")
    except FileNotFoundError:
        print(f"  CSV not found: {csv_path}")
    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)) if X else (None, None)


def train(n_samples: int = 50_000, real_csv: str = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Generating {n_samples} synthetic asset-mission pair samples...")
    X, y = generate_samples(n_samples)

    if real_csv:
        X_real, y_real = load_real_csv(real_csv)
        if X_real is not None:
            from collections import defaultdict
            import random as _rand
            _rand.seed(42)
            syn_counts = defaultdict(int)
            for lbl in y:
                syn_counts[int(lbl)] += 1
            real_capped_X, real_capped_y = [], []
            real_counts = defaultdict(int)
            combined = list(zip(X_real.tolist(), y_real.tolist()))
            _rand.shuffle(combined)
            for feat, lbl in combined:
                cap = syn_counts[int(lbl)] * 2
                if real_counts[int(lbl)] < cap:
                    real_capped_X.append(feat)
                    real_capped_y.append(lbl)
                    real_counts[int(lbl)] += 1
            if real_capped_X:
                X = np.vstack([X, np.array(real_capped_X, dtype=np.float32)])
                y = np.concatenate([y, np.array(real_capped_y, dtype=np.int64)])
                label_map = {0: "poor_match", 1: "good_match"}
                print(f"  Real data (capped): { {label_map[k]: v for k, v in real_counts.items()} }")
                print(f"  Combined: {len(X)} total samples (real + synthetic)")

    pos = int(y.sum())
    neg = int((y == 0).sum())
    n_total = len(y)
    print(f"  {n_total} samples — positive (good match): {pos}  negative: {neg}")
    print(f"  Positive rate: {pos / n_total:.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # HistGradientBoostingClassifier wrapped in probability calibration.
    # CalibratedClassifierCV with cv=5 ensures the output probabilities are
    # well-calibrated (not just raw sigmoid), which is what the runtime scorer
    # uses to rank candidate assets.
    base_clf = HistGradientBoostingClassifier(
        max_iter=400,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=8,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    CalibratedClassifierCV(base_clf, cv=5, method="isotonic")),
    ])

    print("Training HistGradientBoostingClassifier + isotonic calibration (cv=5)...")
    pipe.fit(X_train, y_train)

    # Evaluation
    y_pred      = pipe.predict(X_test)
    y_proba     = pipe.predict_proba(X_test)[:, 1]
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred,
                                target_names=["poor_match", "good_match"],
                                zero_division=0))

    # Mean probability on correctly vs incorrectly labelled positives
    tp_mean = float(y_proba[(y_test == 1) & (y_pred == 1)].mean()) if any((y_test == 1) & (y_pred == 1)) else float("nan")
    fp_mean = float(y_proba[(y_test == 0) & (y_pred == 1)].mean()) if any((y_test == 0) & (y_pred == 1)) else float("nan")
    print(f"Mean score — true positives: {tp_mean:.3f}  false positives: {fp_mean:.3f}")

    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
    print(f"5-fold CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Export to ONNX
    onnx_path = os.path.join(OUTPUT_DIR, "asset_assignment.onnx")
    initial_type = [("float_input", FloatTensorType([None, FEATURE_DIM]))]
    onnx_model   = convert_sklearn(pipe, initial_types=initial_type, target_opset=12)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"\nModel saved: {onnx_path}")

    # Save feature names
    feat_path = os.path.join(OUTPUT_DIR, "asset_assignment_feature_names.json")
    with open(feat_path, "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    print(f"Feature names saved: {feat_path}")

    # Smoke test
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    smoke_cases = [
        # (description, feature vector)
        ("DELIVER + payload + VTOL + full battery",
         [MISSION_DELIVER, ASSET_VTOL,       0.95, 0.10, 0, 1, 0, 1, 1, 0.80, 0.10, 0.0, 1.0, 0, 1.0]),
        ("DELIVER + no payload + QUADCOPTER",
         [MISSION_DELIVER, ASSET_QUADCOPTER,  0.80, 0.10, 0, 1, 0, 0, 1, 0.50, 0.10, 0.0, 1.0, 0, 1.0]),
        ("SEARCH + thermal + long endurance",
         [MISSION_SEARCH,  ASSET_VTOL,        0.85, 0.20, 1, 1, 0, 0, 1, 0.90, 0.15, 0.0, 1.0, 0, 0.66]),
        ("SEARCH + low battery",
         [MISSION_SEARCH,  ASSET_VTOL,        0.12, 0.20, 1, 1, 0, 0, 1, 0.90, 0.15, 0.0, 1.0, 0, 0.66]),
        ("ORBIT + FIXED_WING",
         [MISSION_ORBIT,   ASSET_FIXED_WING,  0.75, 0.15, 0, 1, 0, 0, 1, 0.85, 0.20, 0.0, 1.0, 0, 0.33]),
        ("ORBIT + TOWER (immobile)",
         [MISSION_ORBIT,   ASSET_TOWER,       1.00, 0.00, 1, 1, 0, 0, 1, 1.00, 0.10, 0.0, 1.0, 0, 0.33]),
        ("INSPECT + RGB camera",
         [MISSION_INSPECT, ASSET_QUADCOPTER,  0.70, 0.05, 0, 1, 1, 0, 1, 0.60, 0.10, 0.0, 1.0, 0, 0.66]),
        ("INSPECT + no camera",
         [MISSION_INSPECT, ASSET_QUADCOPTER,  0.70, 0.05, 0, 0, 0, 0, 1, 0.60, 0.10, 0.0, 1.0, 0, 0.66]),
        ("High wind + FIXED_WING",
         [MISSION_MONITOR, ASSET_FIXED_WING,  0.80, 0.20, 0, 1, 0, 0, 1, 0.75, 0.80, 0.0, 1.0, 0, 0.33]),
        ("Long distance + QUADCOPTER",
         [MISSION_SURVEY,  ASSET_QUADCOPTER,  0.90, 0.85, 0, 1, 0, 0, 1, 0.55, 0.15, 0.0, 1.0, 0, 0.33]),
    ]

    print("\nSmoke test predictions:")
    label_map = {0: "poor_match", 1: "good_match"}
    for desc, feats in smoke_cases:
        x = np.array([feats], dtype=np.float32)
        proba = sess.run(["probabilities"], {input_name: x})[0][0]
        pred  = sess.run([output_name],    {input_name: x})[0][0]
        print(f"  {desc:<45} → {label_map[int(pred)]}  (score={proba[1]:.3f})")

    return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Summit.OS asset assignment scorer")
    parser.add_argument("--samples", type=int, default=50_000,
                        help="Number of synthetic training samples (default: 50000)")
    parser.add_argument(
        "--real-csv", dest="real_csv", default=None,
        help="Path to real observations CSV to blend with synthetic data"
    )
    args = parser.parse_args()
    train(n_samples=args.samples, real_csv=args.real_csv)
