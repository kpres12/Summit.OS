"""
Train the Summit.OS escalation predictor.

Predicts whether an alert will escalate — i.e., the operator will NOT
acknowledge it within the configured timeout window.  KOFA uses the output
probability to pre-escalate critical events rather than waiting for the timer
to expire, cutting median escalation latency on night-shift incidents.

Feature vector (21 floats):
  [0-14]  Standard feature vector from features.py
            (confidence, has_location, 13 binary domain-keyword groups)
  [15]    severity_encoded          — CRITICAL=1.0  HIGH=0.75  MEDIUM=0.5  LOW=0.25
  [16]    hour_of_day_sin           — sin(2π * hour / 24)
  [17]    hour_of_day_cos           — cos(2π * hour / 24)
  [18]    day_of_week_sin           — sin(2π * dow / 7)
  [19]    day_of_week_cos           — cos(2π * dow / 7)
  [20]    active_mission_count_norm — active missions / 10.0  (workload proxy)

Labels:
  1 = will escalate   (operator does not acknowledge in time)
  0 = acknowledged    (operator responds within timeout)

Outputs:
  packages/ml/models/escalation_predictor.onnx
  packages/ml/models/escalation_predictor_feature_names.json

Usage:
  python train_escalation_predictor.py
  python train_escalation_predictor.py --samples 80000 --output-dir /tmp/models
"""

import onnx_compat  # noqa: F401 — Python 3.14 compat patch
import argparse
import json
import os
import sys

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_DIM, FEATURE_NAMES  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

SEVERITY_MAP = {
    "CRITICAL": 1.00,
    "HIGH": 0.75,
    "MEDIUM": 0.50,
    "LOW": 0.25,
}
SEVERITY_LEVELS = list(SEVERITY_MAP.keys())

EXTENDED_FEATURE_NAMES = FEATURE_NAMES + [
    "severity_encoded",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "active_mission_count_norm",
]

EXTENDED_FEATURE_DIM = len(EXTENDED_FEATURE_NAMES)  # 21


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def generate_escalation_samples(n: int, seed: int = 42) -> tuple:
    """
    Generate n synthetic (X, y) samples for the escalation predictor.

    Returns
    -------
    X : np.ndarray of shape (n, 21), dtype float32
    y : np.ndarray of shape (n,), dtype int64  (1=escalates, 0=acknowledged)
    """
    rng = np.random.default_rng(seed)

    # Domain-keyword feature indices (2-14) for fire/smoke detection
    cls_to_idx = {
        "fire_smoke": 2,
        "person": 3,
        "flood_water": 4,
        "structural": 5,
        "vehicle": 6,
        "hazmat": 7,
        "wildlife": 8,
        "infrastructure": 9,
        "agricultural": 10,
        "medical": 11,
        "security": 12,
        "search_target": 13,
        "logistics": 14,
    }
    cls_names = ["none"] + list(cls_to_idx.keys())

    X = []
    y = []

    for _ in range(n):
        cls = rng.choice(cls_names)
        conf = float(rng.beta(4, 2))
        has_location = float(rng.random() > 0.25)
        hour = int(rng.integers(0, 24))
        dow = int(rng.integers(0, 7))  # 0=Mon … 6=Sun
        severity = rng.choice(SEVERITY_LEVELS, p=[0.10, 0.25, 0.45, 0.20])
        sev_enc = SEVERITY_MAP[severity]
        active_missions_norm = float(rng.beta(1.5, 4))  # mostly low workload

        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        dow_sin = np.sin(2 * np.pi * dow / 7.0)
        dow_cos = np.cos(2 * np.pi * dow / 7.0)

        is_nighttime = hour >= 22 or hour <= 6
        is_weekend = dow >= 5
        high_workload = active_missions_norm > 0.5
        is_fire_smoke = cls == "fire_smoke"

        # ------------------------------------------------------------------
        # Decision rules — determine p_escalate
        # ------------------------------------------------------------------

        # Base escalation rate by severity
        base_esc = {
            "CRITICAL": 0.45,
            "HIGH": 0.35,
            "MEDIUM": 0.25,
            "LOW": 0.10,
        }[severity]

        p_esc = base_esc

        # Night + high workload → much harder to acknowledge in time
        if is_nighttime and high_workload:
            p_esc = min(1.0, p_esc + 0.30)

        # CRITICAL at night → 70% escalation
        if severity == "CRITICAL" and is_nighttime:
            p_esc = max(p_esc, 0.70)

        # Daytime CRITICAL → operators are alert and respond fast
        if severity == "CRITICAL" and not is_nighttime:
            p_esc = min(p_esc, 0.35)

        # Weekend → higher base escalation (skeleton crew)
        if is_weekend:
            p_esc = min(1.0, p_esc + 0.12)

        # High workload alone raises escalation
        if high_workload and not is_nighttime:
            p_esc = min(1.0, p_esc + 0.10)

        # Fire/smoke: operators are specifically trained to respond fast
        if is_fire_smoke:
            p_esc = max(0.0, p_esc - 0.15)

        # LOW severity → operators usually handle quickly regardless
        if severity == "LOW":
            p_esc = min(p_esc, 0.18)

        # No location + low confidence → ambiguous alert, may be ignored
        if not has_location and conf < 0.45 and severity in ("MEDIUM", "LOW"):
            p_esc = min(1.0, p_esc + 0.10)

        # Add calibration noise
        noise = rng.normal(0, 0.10)
        p_esc = float(np.clip(p_esc + noise, 0.0, 1.0))

        label = int(rng.random() < p_esc)

        # Build feature row
        keyword_vec = [0.0] * FEATURE_DIM
        keyword_vec[0] = conf
        keyword_vec[1] = has_location
        if cls in cls_to_idx:
            keyword_vec[cls_to_idx[cls]] = 1.0

        row = keyword_vec + [
            sev_enc,
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,
            active_missions_norm,
        ]
        X.append(row)
        y.append(label)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int64)
    return X_arr, y_arr


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def load_real_csv(csv_path: str):
    """Load real observations and map to escalation predictor feature space (21 floats)."""
    import csv as _csv
    from features import extract as _extract

    X, y = [], []
    # Map risk_level → severity encoding (proxy for escalation label)
    risk_to_sev = {"LOW": 0.25, "MEDIUM": 0.50, "HIGH": 0.75, "CRITICAL": 1.00}
    # HIGH/CRITICAL observations proxy as likely-to-escalate; LOW/MEDIUM as acknowledged
    risk_to_label = {"LOW": 0, "MEDIUM": 0, "HIGH": 1, "CRITICAL": 1}
    try:
        with open(csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    conf = float(row["confidence"])
                    lat = float(row["lat"])
                    lon = float(row["lon"])
                    risk = row.get("risk_level", "MEDIUM").strip().upper()
                    if risk not in risk_to_sev:
                        continue
                    obs = {
                        "class": row["class"],
                        "confidence": conf,
                        "lat": lat,
                        "lon": lon,
                    }
                    base_vec = _extract(obs)  # 15 floats
                    sev_enc = risk_to_sev[risk]
                    # Use midday weekday as default temporal context (unknown)
                    hour, dow = 12, 2
                    hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
                    hour_cos = float(np.cos(2 * np.pi * hour / 24.0))
                    dow_sin = float(np.sin(2 * np.pi * dow / 7.0))
                    dow_cos = float(np.cos(2 * np.pi * dow / 7.0))
                    active_missions_norm = 0.3  # moderate workload default
                    feat = base_vec + [
                        sev_enc,
                        hour_sin,
                        hour_cos,
                        dow_sin,
                        dow_cos,
                        active_missions_norm,
                    ]
                    label = risk_to_label[risk]
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


def train(
    n_samples: int = 50000, output_dir: str = MODELS_DIR, real_csv: str = None
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {n_samples:,} synthetic samples...")
    X, y = generate_escalation_samples(n_samples)

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
                label_map = {0: "acknowledged", 1: "escalated"}
                print(
                    f"  Real data (capped): { {label_map[k]: v for k, v in real_counts.items()} }"
                )
                print(f"  Combined: {len(X)} total samples (real + synthetic)")

    esc_rate = y.mean()
    print(
        f"  Class balance: {esc_rate:.1%} escalate  |  {1 - esc_rate:.1%} acknowledged"
    )
    print(f"  Feature dim: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_clf = HistGradientBoostingClassifier(
        max_iter=400,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=10,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )

    calibrated = CalibratedClassifierCV(base_clf, cv=5, method="isotonic")

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", calibrated),
        ]
    )

    print("Training CalibratedClassifierCV(HistGradientBoostingClassifier)...")
    pipe.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print("\n--- Test Set Metrics ---")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")

    print("\n--- Classification Report ---")
    print(
        classification_report(
            y_test, y_pred, target_names=["acknowledged", "escalated"], zero_division=0
        )
    )

    # ------------------------------------------------------------------
    # Escalation rate by hour-of-day — illustrates the nighttime pattern
    # ------------------------------------------------------------------
    print("--- Predicted Escalation Rate by Hour of Day ---")
    rng_hour = np.random.default_rng(seed=77)
    hour_stats = []
    for h in range(24):
        n_h = 300
        # Use MEDIUM severity, moderate workload, weekday — isolate hour effect
        row = np.zeros((n_h, EXTENDED_FEATURE_DIM), dtype=np.float32)
        row[:, 0] = rng_hour.beta(4, 2, n_h).astype(np.float32)  # conf
        row[:, 1] = (rng_hour.random(n_h) > 0.25).astype(np.float32)  # has_loc
        row[:, 2] = (rng_hour.random(n_h) > 0.5).astype(
            np.float32
        )  # fire_smoke keyword
        row[:, 15] = 0.50  # MEDIUM severity
        row[:, 16] = np.sin(2 * np.pi * h / 24.0)
        row[:, 17] = np.cos(2 * np.pi * h / 24.0)
        row[:, 18] = np.sin(2 * np.pi * 2 / 7.0)  # Wednesday
        row[:, 19] = np.cos(2 * np.pi * 2 / 7.0)
        row[:, 20] = 0.30  # moderate workload
        preds = pipe.predict(row)
        rate = preds.mean()
        hour_stats.append((h, rate))

    for h, rate in hour_stats:
        bar = "#" * int(rate * 40)
        marker = " <-- NIGHT" if (h >= 22 or h <= 6) else ""
        print(f"  {h:02d}:00  {rate:.1%}  {bar}{marker}")

    # ------------------------------------------------------------------
    # Export to ONNX
    # ------------------------------------------------------------------
    onnx_path = os.path.join(output_dir, "escalation_predictor.onnx")
    initial_type = [("float_input", FloatTensorType([None, EXTENDED_FEATURE_DIM]))]
    onnx_model = convert_sklearn(pipe, initial_types=initial_type, target_opset=12)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"\nModel saved:    {onnx_path}")

    # Save feature names
    feat_names_path = os.path.join(
        output_dir, "escalation_predictor_feature_names.json"
    )
    with open(feat_names_path, "w") as f:
        json.dump(EXTENDED_FEATURE_NAMES, f, indent=2)
    print(f"Features saved: {feat_names_path}")

    # ------------------------------------------------------------------
    # Quick smoke test via onnxruntime
    # ------------------------------------------------------------------
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        smoke_cases = [
            # (description, conf, has_loc, cls_idx, severity, hour, dow, workload)
            ("CRITICAL fire  night  high workload", 0.92, 1, 2, 1.00, 23, 2, 0.8),
            ("CRITICAL fire  day    low workload", 0.92, 1, 2, 1.00, 10, 2, 0.1),
            ("LOW alert  day  no-loc", 0.60, 0, -1, 0.25, 11, 1, 0.2),
            ("HIGH structural night weekend", 0.75, 1, 5, 0.75, 2, 6, 0.6),
            ("MEDIUM hazmat night weekday", 0.78, 1, 7, 0.50, 3, 3, 0.4),
            ("HIGH person day low workload", 0.85, 1, 3, 0.75, 14, 0, 0.2),
        ]
        print("\nSmoke test:")
        for desc, conf, has_loc, cls_idx, sev, hour, dow, workload in smoke_cases:
            row = np.zeros((1, EXTENDED_FEATURE_DIM), dtype=np.float32)
            row[0, 0] = conf
            row[0, 1] = has_loc
            if cls_idx >= 0:
                row[0, cls_idx] = 1.0
            row[0, 15] = sev
            row[0, 16] = np.sin(2 * np.pi * hour / 24.0)
            row[0, 17] = np.cos(2 * np.pi * hour / 24.0)
            row[0, 18] = np.sin(2 * np.pi * dow / 7.0)
            row[0, 19] = np.cos(2 * np.pi * dow / 7.0)
            row[0, 20] = workload
            pred = sess.run(None, {"float_input": row})[0][0]
            label = "ESCALATE" if pred == 1 else "ACK_OK"
            print(f"  {desc:<45} → {label}")
    except ImportError:
        print("onnxruntime not installed — skipping smoke test")

    return onnx_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Summit.OS escalation predictor model."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50000,
        help="Number of synthetic training samples (default: 50000)",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=MODELS_DIR,
        help="Directory to write .onnx and .json files (default: packages/ml/models/)",
    )
    parser.add_argument(
        "--real-csv",
        dest="real_csv",
        default=None,
        help="Path to real observations CSV to blend with synthetic data",
    )
    args = parser.parse_args()
    train(n_samples=args.samples, output_dir=args.output_dir, real_csv=args.real_csv)
