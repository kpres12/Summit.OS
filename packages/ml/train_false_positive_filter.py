"""
Train the Summit.OS false-positive filter.

Binary classifier that filters out noisy/spurious sensor detections before
they reach the KOFA dispatch engine.  Reducing alert fatigue is the primary
goal: operators should only see detections that warrant action.

Feature vector (18 floats):
  [0-14]  Standard feature vector from features.py
            (confidence, has_location, 13 binary domain-keyword groups)
  [15]    detection_frequency   — how many times this class fired in the last
                                  5 min at this location (0.0 rare → 1.0 repeated)
  [16]    hour_of_day_sin       — sin(2π * hour / 24), cyclic time encoding
  [17]    hour_of_day_cos       — cos(2π * hour / 24)

Labels:
  1 = real detection (should be acted on)
  0 = false positive (noise / artifact)

Outputs:
  packages/ml/models/false_positive_filter.onnx
  packages/ml/models/false_positive_filter_feature_names.json

Usage:
  python train_false_positive_filter.py
  python train_false_positive_filter.py --samples 80000 --output-dir /tmp/models
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

EXTENDED_FEATURE_NAMES = FEATURE_NAMES + [
    "detection_frequency",
    "hour_of_day_sin",
    "hour_of_day_cos",
]

EXTENDED_FEATURE_DIM = len(EXTENDED_FEATURE_NAMES)  # 18

# Domain-keyword feature indices (indices 2-14 in the base vector)
_FIRE_SMOKE_IDX = 2
_STRUCTURAL_IDX = 5

# Base false-positive rates by class (used to shape synthetic data)
_CLASS_FP_RATES = {
    "fire_smoke": 0.15,  # well-calibrated sensors, low FP rate
    "person": 0.25,
    "flood_water": 0.20,
    "structural": 0.40,  # highest FP rate — debris vs. real collapse ambiguous
    "vehicle": 0.30,
    "hazmat": 0.18,
    "wildlife": 0.35,
    "infrastructure": 0.28,
    "agricultural": 0.32,
    "medical": 0.20,
    "security": 0.25,
    "search_target": 0.18,
    "logistics": 0.30,
    "none": 0.55,  # no keyword match → high FP
}

_CLASS_NAMES = list(_CLASS_FP_RATES.keys())


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def _sample_class(rng: np.random.Generator) -> tuple:
    """Return (class_name, base_fp_rate, keyword_feature_vec_15).

    The keyword_feature_vec_15 is the 15-float base feature vector with
    confidence=0 and has_location=0 (those are filled in later).
    """
    cls = rng.choice(_CLASS_NAMES)
    fp_rate = _CLASS_FP_RATES[cls]
    base_vec = [0.0] * FEATURE_DIM
    # Map class name → feature index (indices 2-14 in order of _CLASS_NAMES minus "none")
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
    if cls in cls_to_idx:
        base_vec[cls_to_idx[cls]] = 1.0
    return cls, fp_rate, base_vec


def generate_false_positive_samples(n: int, seed: int = 42) -> tuple:
    """
    Generate n synthetic (X, y) samples for the false-positive filter.

    Returns
    -------
    X : np.ndarray of shape (n, 18), dtype float32
    y : np.ndarray of shape (n,), dtype int64  (1=real, 0=false_positive)
    """
    rng = np.random.default_rng(seed)
    X = []
    y = []

    for _ in range(n):
        cls, base_fp_rate, keyword_vec = _sample_class(rng)

        conf = float(rng.beta(4, 2))  # skewed toward higher confidence
        has_location = float(rng.random() > 0.25)  # 75% of detections have location
        hour = float(rng.integers(0, 24))
        detection_frequency = float(
            rng.beta(1.5, 3)
        )  # skewed low (most are single-shot)

        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)

        keyword_vec[0] = conf
        keyword_vec[1] = has_location

        has_domain_keyword = any(keyword_vec[2:])
        is_nighttime = hour >= 22 or hour <= 5

        # ------------------------------------------------------------------
        # Decision rules — determine is_real probability
        # ------------------------------------------------------------------
        p_real = 1.0 - base_fp_rate  # class-specific prior

        # Strong positive signals → real
        if conf > 0.85:
            p_real = max(p_real, 0.92)

        if conf > 0.70 and has_location and has_domain_keyword:
            p_real = max(p_real, 0.88)

        if detection_frequency > 0.5:
            p_real = min(1.0, p_real + 0.15)  # repeated detections corroborate

        # Nighttime fire/smoke with moderate confidence → real
        if cls == "fire_smoke" and is_nighttime and conf > 0.60:
            p_real = max(p_real, 0.82)

        # Strong negative signals → false positive
        if conf < 0.35:
            p_real = min(p_real, 0.25)

        if 0.35 <= conf <= 0.55 and not has_location and not has_domain_keyword:
            p_real = min(p_real, 0.20)

        if (
            detection_frequency < 0.05
            and conf < 0.50
            and cls in ("none", "structural", "wildlife")
        ):
            p_real = min(p_real, 0.18)  # single isolated ambiguous detection

        # Add calibration noise
        noise = rng.normal(0, 0.08)
        p_real = float(np.clip(p_real + noise, 0.0, 1.0))

        label = int(rng.random() < p_real)

        row = keyword_vec + [detection_frequency, hour_sin, hour_cos]
        X.append(row)
        y.append(label)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int64)
    return X_arr, y_arr


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def load_real_csv(csv_path: str):
    """Load real observations from download_real_data.py CSV and map to FP filter features."""
    import csv as _csv

    X, y = [], []
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
    # Map class string → feature index via features.extract keyword groups
    from features import extract as _extract

    try:
        with open(csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    conf = float(row["confidence"])
                    lat = float(row["lat"])
                    lon = float(row["lon"])
                    obs = {
                        "class": row["class"],
                        "confidence": conf,
                        "lat": lat,
                        "lon": lon,
                    }
                    base_vec = _extract(obs)  # 15 floats
                    # Extend to 18 floats: detection_frequency=0.5 (unknown), hour=12 (midday)
                    hour = 12.0
                    hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
                    hour_cos = float(np.cos(2 * np.pi * hour / 24.0))
                    feat = base_vec + [0.5, hour_sin, hour_cos]
                    # Use risk_level as a proxy for real/false-positive label:
                    # LOW risk → likely false positive (0); anything else → real (1)
                    risk = row.get("risk_level", "MEDIUM").strip().upper()
                    label = 0 if risk == "LOW" else 1
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
    n_samples: int = 60000, output_dir: str = MODELS_DIR, real_csv: str = None
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {n_samples:,} synthetic samples...")
    X, y = generate_false_positive_samples(n_samples)

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
                label_map = {0: "false_positive", 1: "real"}
                print(
                    f"  Real data (capped): { {label_map[k]: v for k, v in real_counts.items()} }"
                )
                print(f"  Combined: {len(X)} total samples (real + synthetic)")

    fp_rate = 1.0 - y.mean()
    print(f"  Class balance: {y.mean():.1%} real  |  {fp_rate:.1%} false positive")
    print(f"  Feature dim: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_clf = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=6,
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
            y_test, y_pred, target_names=["false_positive", "real"], zero_division=0
        )
    )

    # ------------------------------------------------------------------
    # FP rate by class (use test portion of the synthetic data)
    # Re-generate a small labelled set with class annotations for analysis
    # ------------------------------------------------------------------
    print("--- False Positive Rate by Class (estimated from held-out data) ---")
    rng_analysis = np.random.default_rng(seed=99)
    analysis_rows = []
    for cls in _CLASS_NAMES:
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
        # Build 500 samples of a single class
        n_cls = 500
        confs = rng_analysis.beta(4, 2, n_cls).astype(np.float32)
        has_locs = (rng_analysis.random(n_cls) > 0.25).astype(np.float32)
        hours = rng_analysis.integers(0, 24, n_cls).astype(np.float32)
        freqs = rng_analysis.beta(1.5, 3, n_cls).astype(np.float32)
        h_sin = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
        h_cos = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)

        X_cls = np.zeros((n_cls, EXTENDED_FEATURE_DIM), dtype=np.float32)
        X_cls[:, 0] = confs
        X_cls[:, 1] = has_locs
        if cls in cls_to_idx:
            X_cls[:, cls_to_idx[cls]] = 1.0
        X_cls[:, 15] = freqs
        X_cls[:, 16] = h_sin
        X_cls[:, 17] = h_cos

        preds = pipe.predict(X_cls)
        fp_est = 1.0 - preds.mean()
        analysis_rows.append((cls, fp_est))

    analysis_rows.sort(key=lambda t: t[1], reverse=True)
    for cls, fp_est in analysis_rows:
        bar = "#" * int(fp_est * 30)
        print(f"  {cls:<20} FP rate {fp_est:.1%}  {bar}")

    # ------------------------------------------------------------------
    # Export to ONNX
    # ------------------------------------------------------------------
    onnx_path = os.path.join(output_dir, "false_positive_filter.onnx")
    initial_type = [("float_input", FloatTensorType([None, EXTENDED_FEATURE_DIM]))]
    onnx_model = convert_sklearn(pipe, initial_types=initial_type, target_opset=12)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"\nModel saved:    {onnx_path}")

    # Save feature names
    feat_names_path = os.path.join(
        output_dir, "false_positive_filter_feature_names.json"
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
            # (description, conf, has_loc, cls_idx, freq, hour)
            ("fire conf=0.92 night loc", 0.92, 1, 2, 0.6, 23),
            ("smoke conf=0.40 no-loc", 0.40, 0, 2, 0.0, 9),
            ("structural conf=0.31", 0.31, 0, 5, 0.0, 14),
            ("person conf=0.88 loc", 0.88, 1, 3, 0.4, 10),
            ("unknown conf=0.28", 0.28, 0, -1, 0.0, 15),
            ("vehicle conf=0.72 repeated", 0.72, 1, 6, 0.8, 12),
        ]
        print("\nSmoke test:")
        for desc, conf, has_loc, cls_idx, freq, hour in smoke_cases:
            row = np.zeros((1, EXTENDED_FEATURE_DIM), dtype=np.float32)
            row[0, 0] = conf
            row[0, 1] = has_loc
            if cls_idx >= 0:
                row[0, cls_idx] = 1.0
            row[0, 15] = freq
            row[0, 16] = np.sin(2 * np.pi * hour / 24.0)
            row[0, 17] = np.cos(2 * np.pi * hour / 24.0)
            pred = sess.run(None, {"float_input": row})[0][0]
            label = "REAL" if pred == 1 else "FALSE_POS"
            print(f"  {desc:<40} → {label}")
    except ImportError:
        print("onnxruntime not installed — skipping smoke test")

    return onnx_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Summit.OS false-positive filter model."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=60000,
        help="Number of synthetic training samples (default: 60000)",
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
