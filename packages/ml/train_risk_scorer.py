"""
Train the Summit.OS risk level scorer.

Outputs:
  packages/ml/models/risk_scorer.onnx
  packages/ml/models/risk_scorer_labels.json   — {index: risk_level}

Replaces the hardcoded _calculate_risk_level() rule in intelligence/main.py
with a trained model that accounts for observation class, confidence,
and presence of location data simultaneously.
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import onnx_compat  # noqa: F401 — must be before skl2onnx

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from generate_data import generate_risk_samples, RISK_LABELS
from features import FEATURE_DIM

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_real_csv(csv_path: str):
    """Load risk-labeled observations from download_real_data.py CSV output."""
    label_inv = {v: k for k, v in RISK_LABELS.items()}
    X, y = [], []
    from features import extract
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = label_inv.get(row.get("risk_level", ""))
                if label is None:
                    continue
                try:
                    obs = {
                        "class": row["class"],
                        "confidence": float(row["confidence"]),
                        "lat": float(row["lat"]),
                        "lon": float(row["lon"]),
                    }
                    X.append(extract(obs))
                    y.append(label)
                except (ValueError, KeyError):
                    continue
        print(f"  Loaded {len(X)} real risk samples from {os.path.basename(csv_path)}")
    except FileNotFoundError:
        print(f"  CSV not found: {csv_path}")
    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)) if X else (None, None)


def train(n_synthetic: int = 8000, real_csv: str = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating synthetic risk training data...")
    X, y = generate_risk_samples(n_synthetic)
    print(f"  {len(X)} samples across {len(set(y.tolist()))} risk levels")

    if real_csv:
        X_real, y_real = load_real_csv(real_csv)
        if X_real is not None:
            from collections import defaultdict
            import random as _rand
            _rand.seed(42)
            syn_counts = defaultdict(int)
            for lbl in y:
                syn_counts[lbl] += 1
            real_capped_X, real_capped_y = [], []
            real_counts = defaultdict(int)
            combined = list(zip(X_real.tolist(), y_real.tolist()))
            _rand.shuffle(combined)
            for feat, lbl in combined:
                cap = syn_counts[lbl] * 2
                if real_counts[lbl] < cap:
                    real_capped_X.append(feat)
                    real_capped_y.append(lbl)
                    real_counts[lbl] += 1
            if real_capped_X:
                X = np.vstack([X, np.array(real_capped_X, dtype=np.float32)])
                y = np.concatenate([y, np.array(real_capped_y, dtype=np.int64)])
                print(f"  Real data (capped): { {RISK_LABELS[k]: v for k, v in real_counts.items()} }")
                print(f"  Combined: {len(X)} total samples (real + synthetic)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=400,
            max_depth=5,
            learning_rate=0.05,
            min_samples_leaf=8,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])

    print("Training HistGradientBoostingClassifier (balanced)...")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    target_names = [RISK_LABELS[i] for i in sorted(RISK_LABELS)]
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Export to ONNX
    onnx_path = os.path.join(OUTPUT_DIR, "risk_scorer.onnx")
    initial_type = [("float_input", FloatTensorType([None, FEATURE_DIM]))]
    onnx_model = convert_sklearn(pipe, initial_types=initial_type, target_opset=12)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"\nModel saved: {onnx_path}")

    labels_path = os.path.join(OUTPUT_DIR, "risk_scorer_labels.json")
    with open(labels_path, "w") as f:
        json.dump({str(k): v for k, v in RISK_LABELS.items()}, f, indent=2)
    print(f"Labels saved: {labels_path}")

    # Smoke test
    import onnxruntime as ort
    from features import extract
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    test_obs = [
        {"class": "active fire front", "confidence": 0.95, "lat": 34.0, "lon": -118.0},
        {"class": "crop survey", "confidence": 0.90, "lat": 38.0, "lon": -122.0},
        {"class": "suspicious activity", "confidence": 0.60, "lat": 40.0, "lon": -74.0},
        {"class": "mass casualty", "confidence": 0.88, "lat": 33.0, "lon": -117.0},
        {"class": "unknown blob", "confidence": 0.30, "lat": 35.0, "lon": -105.0},
        {"class": "capsized boat", "confidence": 0.85, "lat": 25.0, "lon": -80.0},
    ]
    print("\nSmoke test predictions:")
    for obs in test_obs:
        feat = np.array([extract(obs)], dtype=np.float32)
        pred = sess.run(None, {"float_input": feat})[0][0]
        print(f"  {obs['class']:<35} conf={obs['confidence']:.2f} → {RISK_LABELS[int(pred)]}")

    return onnx_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-csv", dest="real_csv", default=None)
    parser.add_argument("--samples", type=int, default=8000)
    args = parser.parse_args()
    train(n_synthetic=args.samples, real_csv=args.real_csv)
