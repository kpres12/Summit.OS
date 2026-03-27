"""
Train the Summit.OS mission outcome predictor.

Given pre-dispatch conditions, predicts the probability that a mission will
succeed.  The operator console uses the output score to show a confidence
indicator before the DISPATCH button is pressed.

Feature vector (15 features):
  [0]  mission_type        0-6  (SURVEY/MONITOR/SEARCH/PERIMETER/ORBIT/DELIVER/INSPECT)
  [1]  asset_type          0-4  (QUADCOPTER/FIXED_WING/VTOL/GROUND_UGV/TOWER)
  [2]  battery_at_start    0-1
  [3]  distance_km_norm    0-1  (raw 0-200 normalised)
  [4]  observation_conf    0-1  (confidence of the triggering detection)
  [5]  risk_level          0/0.33/0.66/1.0  (LOW/MEDIUM/HIGH/CRITICAL)
  [6]  weather_wind_norm   0-1
  [7]  weather_visibility  0-1  (1=perfect, 0=zero)
  [8]  terrain_difficulty  0-1
  [9]  time_of_day_sin     cyclical hour encoding
  [10] time_of_day_cos     cyclical hour encoding
  [11] operator_approved   0/1
  [12] auto_generated      0/1
  [13] num_assets_avail    0-1  (raw 0-20 normalised)
  [14] has_backup_asset    0/1

Outputs:
  packages/ml/models/outcome_predictor.onnx
  packages/ml/models/outcome_predictor_feature_names.json

Usage:
  python train_outcome_predictor.py
  python train_outcome_predictor.py --samples 80000
"""

import onnx_compat  # noqa: F401 — Python 3.14 compat patch
import argparse
import json
import os
import sys

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, brier_score_loss
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models")

FEATURE_NAMES = [
    "mission_type",
    "asset_type",
    "battery_at_start",
    "distance_km_norm",
    "observation_conf",
    "risk_level",
    "weather_wind_norm",
    "weather_visibility",
    "terrain_difficulty",
    "time_of_day_sin",
    "time_of_day_cos",
    "operator_approved",
    "auto_generated",
    "num_assets_avail",
    "has_backup_asset",
]
FEATURE_DIM = len(FEATURE_NAMES)

# Mission types
MISSION_SURVEY = 0
MISSION_MONITOR = 1
MISSION_SEARCH = 2
MISSION_PERIMETER = 3
MISSION_ORBIT = 4
MISSION_DELIVER = 5
MISSION_INSPECT = 6

# Asset types
ASSET_QUADCOPTER = 0
ASSET_FIXED_WING = 1
ASSET_VTOL = 2
ASSET_GROUND_UGV = 3
ASSET_TOWER = 4

RISK_MAP = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}


def _success_probability(
    mission_type,
    asset_type,
    battery,
    distance_norm,
    obs_conf,
    risk,
    wind_norm,
    visibility,
    terrain,
    tod_sin,
    tod_cos,
    op_approved,
    auto_gen,
    num_assets_avail,
    has_backup,
    rng,
):
    """
    Compute a continuous success probability then threshold with noise.
    Returns (features_array, label).
    """
    p = 0.65  # baseline success rate

    # ---- battery -----------------------------------------------------------
    if battery < 0.15:
        p -= 0.45  # very likely to fail — battery RTL mid-mission
    elif battery < 0.30:
        p -= 0.20
    elif battery > 0.80:
        p += 0.05

    # ---- visibility × mission type ----------------------------------------
    if visibility < 0.3:
        if mission_type == MISSION_SEARCH:
            p -= 0.40  # can't find target in near-zero visibility
        elif mission_type == MISSION_INSPECT:
            p -= 0.30
        else:
            p -= 0.15
    elif visibility < 0.6:
        p -= 0.10

    # ---- wind × asset type ------------------------------------------------
    if wind_norm > 0.70:  # > 14 m/s
        if asset_type == ASSET_QUADCOPTER:
            p -= 0.45
        elif asset_type == ASSET_FIXED_WING:
            p -= 0.20
        elif asset_type == ASSET_VTOL:
            p -= 0.15
    elif wind_norm > 0.50:
        if asset_type == ASSET_QUADCOPTER:
            p -= 0.20

    # ---- distance × asset endurance (encoded via asset_type proxy) ---------
    if distance_norm > 0.60 and asset_type == ASSET_QUADCOPTER:
        p -= 0.25  # likely to run out of battery before returning

    # ---- mission-specific -------------------------------------------------
    if mission_type == MISSION_DELIVER:
        # Assume we only dispatch deliver missions with payload-capable assets —
        # score reflects runtime planning quality
        if asset_type in (ASSET_VTOL, ASSET_FIXED_WING):
            p += 0.20
        elif asset_type == ASSET_GROUND_UGV:
            p -= 0.10  # slow, terrain-sensitive

    elif mission_type == MISSION_ORBIT:
        if asset_type in (ASSET_FIXED_WING, ASSET_VTOL):
            p += 0.20  # aerodynamically efficient sustained loiter

    elif mission_type == MISSION_SEARCH:
        if wind_norm > 0.4 and visibility < 0.5:
            p -= 0.15  # compound weather penalty

    # ---- operator approval ------------------------------------------------
    if op_approved:
        p += 0.08  # human review reduces dispatch errors
    if auto_gen and not op_approved:
        p -= 0.05  # auto-dispatched, no review

    # ---- backup asset availability ----------------------------------------
    if has_backup:
        p += 0.10  # backup can take over if primary fails
    if num_assets_avail > 0.5:
        p += 0.05  # general fleet health

    # ---- night + no thermal (approximate with cos encoding) ---------------
    is_night = tod_cos < -0.3
    if is_night and mission_type in (MISSION_SEARCH, MISSION_INSPECT):
        p -= 0.20

    # ---- terrain ----------------------------------------------------------
    if terrain > 0.5:
        if asset_type == ASSET_GROUND_UGV:
            p -= 0.25
        elif asset_type in (ASSET_FIXED_WING,):
            p -= 0.10  # limited loiter altitude near terrain

    # ---- risk level (high risk missions in bad conditions fail more) -------
    if risk >= 0.66:
        if wind_norm > 0.50 or visibility < 0.40:
            p -= 0.10

    # ---- observation confidence (weak prior on mission rationale) ----------
    p += (obs_conf - 0.5) * 0.10

    # ---- noise for distribution learning ----------------------------------
    p += rng.normal(0, 0.10)

    label = int(p >= 0.5)
    features = np.array(
        [
            float(mission_type),
            float(asset_type),
            battery,
            distance_norm,
            obs_conf,
            risk,
            wind_norm,
            visibility,
            terrain,
            tod_sin,
            tod_cos,
            float(op_approved),
            float(auto_gen),
            num_assets_avail,
            float(has_backup),
        ],
        dtype=np.float32,
    )
    return features, label


def generate_samples(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    X, y = [], []

    for _ in range(n):
        mission_type = int(rng.integers(0, 7))
        asset_type = int(rng.integers(0, 5))
        battery = float(
            rng.beta(4, 2)
        )  # skewed high — operators charge before dispatch
        distance_norm = float(rng.beta(2, 4))
        obs_conf = float(rng.beta(5, 2))  # skewed high — low-conf obs usually filtered
        risk_idx = int(rng.choice([0, 1, 2, 3], p=[0.20, 0.40, 0.28, 0.12]))
        risk = RISK_MAP[risk_idx]
        wind_norm = float(rng.beta(1.5, 5))
        visibility = float(rng.beta(5, 1.5))  # usually good
        terrain = float(rng.beta(1, 3))  # mostly flat
        hour = float(rng.uniform(0, 24))
        tod_sin = float(np.sin(2 * np.pi * hour / 24.0))
        tod_cos = float(np.cos(2 * np.pi * hour / 24.0))
        op_approved = int(rng.random() < 0.72)  # most missions are operator-approved
        auto_gen = int(rng.random() < 0.30)
        num_assets_norm = float(rng.beta(3, 2))
        has_backup = int(rng.random() < 0.45)

        feat, label = _success_probability(
            mission_type,
            asset_type,
            battery,
            distance_norm,
            obs_conf,
            risk,
            wind_norm,
            visibility,
            terrain,
            tod_sin,
            tod_cos,
            op_approved,
            auto_gen,
            num_assets_norm,
            has_backup,
            rng,
        )
        X.append(feat)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def print_calibration_stats(y_true, y_proba, n_bins: int = 10):
    """Print calibration curve summary and Brier score."""
    frac_pos, mean_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )
    brier = brier_score_loss(y_true, y_proba)

    print("\nCalibration curve (fraction_positive vs mean_predicted_prob):")
    print(f"  {'Bin center':>12}  {'Mean pred':>10}  {'Frac positive':>14}")
    for mp, fp in zip(mean_pred, frac_pos):
        bar = "#" * int(fp * 20)
        print(f"  {mp:12.3f}  {mp:10.3f}  {fp:14.3f}  {bar}")
    print(f"\n  Brier score (lower=better, perfect=0): {brier:.4f}")
    print(
        f"  Perfect calibration Brier benchmark:   {float(np.mean(y_true) * (1 - np.mean(y_true))):.4f}"
    )


def load_real_csv(csv_path: str):
    """Load real observations and map to outcome predictor feature space (15 floats)."""
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
    risk_to_encoded = {"LOW": 0.0, "MEDIUM": 0.33, "HIGH": 0.66, "CRITICAL": 1.0}
    # Proxy success label: HIGH/CRITICAL risk missions in real data are harder → success=0
    risk_to_label = {"LOW": 1, "MEDIUM": 1, "HIGH": 0, "CRITICAL": 0}

    try:
        with open(csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    conf = float(row["confidence"])
                    mission_str = row.get("mission_type", "SURVEY").strip().upper()
                    mission_type = mission_type_map.get(mission_str, MISSION_SURVEY)
                    risk_str = row.get("risk_level", "MEDIUM").strip().upper()
                    if risk_str not in risk_to_encoded:
                        continue
                    risk_enc = risk_to_encoded[risk_str]
                    # Use neutral defaults for fields not in CSV
                    feat = np.array(
                        [
                            float(mission_type),  # mission_type
                            float(
                                ASSET_VTOL
                            ),  # asset_type (VTOL as default capable asset)
                            0.75,  # battery_at_start (assume charged)
                            0.15,  # distance_km_norm (typical short mission)
                            conf,  # observation_conf
                            risk_enc,  # risk_level
                            0.20,  # weather_wind_norm (moderate default)
                            0.80,  # weather_visibility (good default)
                            0.20,  # terrain_difficulty (mostly flat)
                            float(
                                np.sin(2 * np.pi * 12 / 24.0)
                            ),  # time_of_day_sin (midday)
                            float(np.cos(2 * np.pi * 12 / 24.0)),  # time_of_day_cos
                            1.0,  # operator_approved
                            0.0,  # auto_generated
                            0.60,  # num_assets_avail
                            1.0,  # has_backup_asset
                        ],
                        dtype=np.float32,
                    )
                    label = risk_to_label[risk_str]
                    X.append(feat)
                    y.append(label)
                except (ValueError, KeyError):
                    continue
        print(f"  Loaded {len(X)} real samples from {os.path.basename(csv_path)}")
    except FileNotFoundError:
        print(f"  CSV not found: {csv_path}")
    return (
        (np.array(X, dtype=np.float32), np.array(y, dtype=np.int64))
        if X
        else (None, None)
    )


def train(n_samples: int = 40_000, real_csv: str = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Generating {n_samples} synthetic mission outcome samples...")
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
                label_map = {0: "failure", 1: "success"}
                print(
                    f"  Real data (capped): { {label_map[k]: v for k, v in real_counts.items()} }"
                )
                print(f"  Combined: {len(X)} total samples (real + synthetic)")

    pos = int(y.sum())
    neg = int((y == 0).sum())
    n_total = len(y)
    print(f"  {n_total} samples — success: {pos}  failure: {neg}")
    print(f"  Base success rate: {pos / n_total:.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # HistGradientBoostingClassifier + isotonic calibration.
    # Isotonic regression calibration is preferred over sigmoid for larger
    # training sets (cv=5 gives ~32k calibration samples per fold).
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

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(base_clf, cv=5, method="isotonic")),
        ]
    )

    print("Training HistGradientBoostingClassifier + isotonic calibration (cv=5)...")
    pipe.fit(X_train, y_train)

    # Evaluation
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    print("\nClassification report (test set):")
    print(
        classification_report(
            y_test, y_pred, target_names=["failure", "success"], zero_division=0
        )
    )

    # Feature importances — average across calibrated estimators
    importances = np.zeros(FEATURE_DIM)
    clf_cv = pipe.named_steps["clf"]
    for estimator in clf_cv.calibrated_classifiers_:
        base = estimator.estimator
        if hasattr(base, "feature_importances_"):
            importances += base.feature_importances_
    if importances.any():
        importances /= len(clf_cv.calibrated_classifiers_)
        ranked = sorted(
            zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True
        )
        print("\nFeature importances (averaged across calibration folds):")
        for name, imp in ranked:
            bar = "#" * int(imp * 60)
            print(f"  {name:<24} {imp:.4f}  {bar}")

    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
    print(f"\n5-fold CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    print_calibration_stats(y_test, y_proba)

    # Export to ONNX
    onnx_path = os.path.join(OUTPUT_DIR, "outcome_predictor.onnx")
    initial_type = [("float_input", FloatTensorType([None, FEATURE_DIM]))]
    onnx_model = convert_sklearn(pipe, initial_types=initial_type, target_opset=12)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"\nModel saved: {onnx_path}")

    # Save feature names
    feat_path = os.path.join(OUTPUT_DIR, "outcome_predictor_feature_names.json")
    with open(feat_path, "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    print(f"Feature names saved: {feat_path}")

    # Smoke test
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    smoke_cases = [
        # (description, feature vector)
        (
            "DELIVER + VTOL + full battery + op approved",
            [
                MISSION_DELIVER,
                ASSET_VTOL,
                0.95,
                0.10,
                0.90,
                0.33,
                0.10,
                0.95,
                0.0,
                0.0,
                1.0,
                1,
                0,
                0.70,
                1,
            ],
        ),
        (
            "SEARCH + low battery + poor visibility",
            [
                MISSION_SEARCH,
                ASSET_VTOL,
                0.12,
                0.20,
                0.80,
                0.66,
                0.20,
                0.20,
                0.0,
                0.0,
                1.0,
                1,
                0,
                0.60,
                0,
            ],
        ),
        (
            "ORBIT + FIXED_WING + good weather",
            [
                MISSION_ORBIT,
                ASSET_FIXED_WING,
                0.80,
                0.20,
                0.88,
                0.33,
                0.15,
                0.90,
                0.0,
                0.0,
                1.0,
                1,
                0,
                0.50,
                1,
            ],
        ),
        (
            "QUADCOPTER + high wind",
            [
                MISSION_MONITOR,
                ASSET_QUADCOPTER,
                0.75,
                0.15,
                0.85,
                0.33,
                0.80,
                0.80,
                0.0,
                0.0,
                1.0,
                1,
                0,
                0.50,
                0,
            ],
        ),
        (
            "SEARCH + night + no op approval",
            [
                MISSION_SEARCH,
                ASSET_VTOL,
                0.70,
                0.25,
                0.75,
                0.66,
                0.20,
                0.70,
                0.1,
                -0.7,
                -0.7,
                0,
                1,
                0.30,
                0,
            ],
        ),
        (
            "INSPECT + good conditions + backup",
            [
                MISSION_INSPECT,
                ASSET_QUADCOPTER,
                0.85,
                0.08,
                0.92,
                0.33,
                0.10,
                0.95,
                0.0,
                0.5,
                0.9,
                1,
                0,
                0.80,
                1,
            ],
        ),
        (
            "DELIVER + GROUND_UGV + mountainous",
            [
                MISSION_DELIVER,
                ASSET_GROUND_UGV,
                0.90,
                0.30,
                0.80,
                0.66,
                0.10,
                0.85,
                0.9,
                0.0,
                1.0,
                1,
                0,
                0.40,
                0,
            ],
        ),
        (
            "SURVEY + normal conditions",
            [
                MISSION_SURVEY,
                ASSET_VTOL,
                0.75,
                0.20,
                0.85,
                0.33,
                0.20,
                0.85,
                0.2,
                0.3,
                0.9,
                1,
                0,
                0.60,
                1,
            ],
        ),
        (
            "Critical risk + poor visibility",
            [
                MISSION_SEARCH,
                ASSET_FIXED_WING,
                0.70,
                0.30,
                0.90,
                1.00,
                0.60,
                0.25,
                0.3,
                0.0,
                1.0,
                1,
                0,
                0.50,
                1,
            ],
        ),
        (
            "Auto-generated, no backup, very low battery",
            [
                MISSION_PERIMETER,
                ASSET_QUADCOPTER,
                0.10,
                0.40,
                0.70,
                0.33,
                0.25,
                0.80,
                0.0,
                0.0,
                1.0,
                0,
                1,
                0.20,
                0,
            ],
        ),
    ]

    print("\nSmoke test predictions:")
    for desc, feats in smoke_cases:
        x = np.array([feats], dtype=np.float32)
        output_names = [o.name for o in sess.get_outputs()]
        results = sess.run(output_names, {input_name: x})
        pred = int(np.asarray(results[0]).flat[0])
        label = "success" if pred == 1 else "failure"
        # probability output is second output if available, else just show pred
        if len(results) > 1:
            proba_raw = results[1]
            try:
                conf = float(np.asarray(proba_raw).flat[1])
                print(f"  {desc:<52} → {label}  (confidence={conf:.3f})")
            except Exception:
                print(f"  {desc:<52} → {label}")
        else:
            print(f"  {desc:<52} → {label}")

    return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Summit.OS mission outcome predictor"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=40_000,
        help="Number of synthetic training samples (default: 40000)",
    )
    parser.add_argument(
        "--real-csv",
        dest="real_csv",
        default=None,
        help="Path to real observations CSV to blend with synthetic data",
    )
    args = parser.parse_args()
    train(n_samples=args.samples, real_csv=args.real_csv)
