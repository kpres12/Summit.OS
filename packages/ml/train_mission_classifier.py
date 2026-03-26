"""
Train the Summit.OS mission type classifier.

Outputs:
  packages/ml/models/mission_classifier.onnx   — runtime model
  packages/ml/models/mission_classifier_labels.json — {index: mission_type}

Usage:
  python train_mission_classifier.py            # synthetic data only
  python train_mission_classifier.py --real-data postgres://...  # blend real operator decisions

Real-data schema expected (tasking DB):
  SELECT m.mission_type,
         o.class    AS obs_class,
         o.confidence,
         o.lat,
         o.lon
  FROM missions m
  JOIN advisories a ON a.advisory_id = m.source_advisory_id
  JOIN observations o ON o.advisory_id = a.advisory_id
  WHERE m.operator_approved = TRUE;
"""

import onnx_compat  # noqa: F401 — Python 3.14 compat patch
import argparse
import csv
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

sys.path.insert(0, os.path.dirname(__file__))
from generate_data import generate_mission_samples, MISSION_LABELS
from features import FEATURE_DIM, FEATURE_NAMES

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_real_csv(csv_path: str):
    """Load labeled observations from a CSV file (output of download_real_data.py)."""
    label_inv = {v: k for k, v in MISSION_LABELS.items()}
    X, y = [], []
    from features import extract
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = label_inv.get(row.get("mission_type", ""))
                if label is None:
                    continue
                try:
                    obs = {
                        "class":      row["class"],
                        "confidence": float(row["confidence"]),
                        "lat":        float(row["lat"]),
                        "lon":        float(row["lon"]),
                    }
                    X.append(extract(obs))
                    y.append(label)
                except (ValueError, KeyError):
                    continue
        print(f"  Loaded {len(X)} real observations from {os.path.basename(csv_path)}")
        from collections import Counter
        dist = Counter(MISSION_LABELS[i] for i in y)
        print(f"  Real class distribution: {dict(dist)}")
    except FileNotFoundError:
        print(f"  CSV not found: {csv_path}")
    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)) if X else (None, None)


def load_real_data(pg_url: str):
    """Load operator-approved mission decisions from the tasking DB."""
    try:
        import psycopg2
        conn = psycopg2.connect(pg_url)
        cur = conn.cursor()
        cur.execute("""
            SELECT m.mission_type, o.class, o.confidence, o.lat, o.lon
            FROM missions m
            JOIN advisories a ON a.advisory_id = m.source_advisory_id
            WHERE m.operator_approved = TRUE
            LIMIT 50000
        """)
        rows = cur.fetchall()
        conn.close()
        label_inv = {v: k for k, v in MISSION_LABELS.items()}
        X, y = [], []
        from features import extract
        for mission_type_str, cls, conf, lat, lon in rows:
            label = label_inv.get(mission_type_str)
            if label is None:
                continue
            obs = {"class": cls, "confidence": float(conf or 0),
                   "lat": lat, "lon": lon}
            X.append(extract(obs))
            y.append(label)
        print(f"  Loaded {len(X)} real operator samples from DB")
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
    except Exception as e:
        print(f"  Real data load failed ({e}) — synthetic only")
        return None, None


def train(pg_url: str = None, real_csv: str = None, n_synthetic: int = 8000):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating synthetic training data...")
    X_syn, y_syn = generate_mission_samples(n_synthetic)
    print(f"  {len(X_syn)} synthetic samples, {len(set(y_syn.tolist()))} classes")
    print(f"  Class distribution: { {MISSION_LABELS[i]: int((y_syn==i).sum()) for i in sorted(MISSION_LABELS)} }")

    X, y = X_syn, y_syn

    if real_csv:
        X_real, y_real = load_real_csv(real_csv)
        if X_real is not None:
            # Cap each real class at 2× its synthetic count to prevent dominance
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
                print(f"  Real data (capped): { {MISSION_LABELS[k]: v for k, v in real_counts.items()} }")
                print(f"  Combined: {len(X)} total samples (real + synthetic)")

    if pg_url:
        X_real, y_real = load_real_data(pg_url)
        if X_real is not None:
            X_real_rep = np.tile(X_real, (3, 1))
            y_real_rep = np.tile(y_real, 3)
            X = np.vstack([X, X_real_rep])
            y = np.concatenate([y, y_real_rep])
            print(f"  Combined: {len(X)} total samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # HistGradientBoosting: faster than GBT, native class balancing,
    # handles sparse binary features better, supports early stopping.
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=400,
            max_depth=6,
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

    # Evaluation
    y_pred = pipe.predict(X_test)
    target_names = [MISSION_LABELS[i] for i in sorted(MISSION_LABELS)]
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Export to ONNX
    onnx_path = os.path.join(OUTPUT_DIR, "mission_classifier.onnx")
    initial_type = [("float_input", FloatTensorType([None, FEATURE_DIM]))]
    onnx_model = convert_sklearn(pipe, initial_types=initial_type, target_opset=12)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"\nModel saved: {onnx_path}")

    # Save label map
    labels_path = os.path.join(OUTPUT_DIR, "mission_classifier_labels.json")
    with open(labels_path, "w") as f:
        json.dump({str(k): v for k, v in MISSION_LABELS.items()}, f, indent=2)
    print(f"Labels saved: {labels_path}")

    # Quick smoke test
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    test_obs = [
        {"class": "smoke plume", "confidence": 0.92, "lat": 34.05, "lon": -118.24},
        {"class": "missing hiker", "confidence": 0.80, "lat": 37.5, "lon": -119.5},
        {"class": "chemical spill", "confidence": 0.75, "lat": 33.0, "lon": -117.0},
        {"class": "pipeline leak", "confidence": 0.88, "lat": 35.0, "lon": -100.0},
        {"class": "unauthorized uav", "confidence": 0.91, "lat": 40.0, "lon": -74.0},
        {"class": "capsized boat", "confidence": 0.85, "lat": 25.0, "lon": -80.0},
        {"class": "crop blight", "confidence": 0.78, "lat": 38.0, "lon": -122.0},
        {"class": "delivery waypoint", "confidence": 0.95, "lat": 37.8, "lon": -122.4},
    ]
    from features import extract as feat_extract
    print("\nSmoke test predictions:")
    for obs in test_obs:
        feat = np.array([feat_extract(obs)], dtype=np.float32)
        pred = sess.run(None, {"float_input": feat})[0][0]
        print(f"  {obs['class']:<30} → {MISSION_LABELS[int(pred)]}")

    return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data", dest="pg_url", default=None,
                        help="PostgreSQL URL to blend real operator decisions")
    parser.add_argument("--real-csv", dest="real_csv", default=None,
                        help="CSV file from download_real_data.py")
    parser.add_argument("--samples", type=int, default=8000)
    args = parser.parse_args()
    train(pg_url=args.pg_url, real_csv=args.real_csv, n_synthetic=args.samples)
