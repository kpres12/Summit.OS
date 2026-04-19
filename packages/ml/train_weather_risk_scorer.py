"""
Train the Heli.OS weather-adjusted risk scorer.

Extends the base risk scorer (which uses class + confidence only) by incorporating
real-time weather conditions into risk severity scoring.  A fire detection with
40 mph winds and 5% humidity is far more critical than the same detection in calm,
rainy conditions.

Feature vector (21 floats):
  [0-14]  Standard features from features.py (index-stable)
  [15]    wind_speed_mps_norm   — wind speed 0-30 m/s, normalized /30
  [16]    wind_gust_mps_norm    — peak gust / 40
  [17]    humidity_pct_inv      — (100 - humidity%) / 100  (dry=high=dangerous for fire)
  [18]    temp_c_norm           — (temp_c - (-20)) / 70   (range -20°C to 50°C)
  [19]    precip_mm_norm        — min(precip_mm / 50, 1.0)
  [20]    visibility_km_norm    — min(visibility_km / 20, 1.0)

Labels:
  0 = LOW, 1 = MEDIUM, 2 = HIGH, 3 = CRITICAL

Outputs:
  packages/ml/models/weather_risk_scorer.onnx
  packages/ml/models/weather_risk_scorer_feature_names.json

Usage:
  python train_weather_risk_scorer.py
  python train_weather_risk_scorer.py --samples 80000 --output-dir ./models
"""

import onnx_compat  # noqa: F401 — Python 3.14 compat patch
import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from features import FEATURE_DIM, FEATURE_NAMES, extract

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "models"

# ── Constants ─────────────────────────────────────────────────────────────────

RISK_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}

WEATHER_FEATURE_DIM = 21  # 15 base + 6 weather

WEATHER_FEATURE_NAMES = FEATURE_NAMES + [
    "wind_speed_mps_norm",  # [15]
    "wind_gust_mps_norm",  # [16]
    "humidity_pct_inv",  # [17]  (100 - humidity%) / 100
    "temp_c_norm",  # [18]  (temp_c + 20) / 70
    "precip_mm_norm",  # [19]  min(precip_mm / 50, 1.0)
    "visibility_km_norm",  # [20]  min(visibility_km / 20, 1.0)
]

# ── Scenario catalogue ────────────────────────────────────────────────────────
# (class_string, base_risk_label)
# base_risk_label: 0=LOW 1=MEDIUM 2=HIGH 3=CRITICAL
# Weather modifiers are applied on top of this baseline.

_SCENARIOS = [
    # Fire / smoke
    ("active fire front", 3),
    ("wildfire", 3),
    ("smoke plume", 2),
    ("smoke column", 2),
    ("hotspot", 2),
    ("ember shower", 3),
    ("burning structure", 3),
    ("burning vehicle", 2),
    # Flood / water
    ("flash flood", 3),
    ("flooding", 2),
    ("storm surge", 3),
    ("levee breach", 3),
    ("tsunami wave", 3),
    ("rising water", 2),
    ("submerged vehicle", 1),
    # SAR / people
    ("missing person", 2),
    ("stranded hiker", 2),
    ("distress signal", 3),
    ("man overboard", 3),
    ("injured survivor", 2),
    ("casualty", 2),
    # Hazmat
    ("chemical spill", 3),
    ("toxic gas leak", 3),
    ("radiation leak", 3),
    ("industrial explosion", 3),
    ("ammonia release", 2),
    ("hazmat plume", 2),
    # Infrastructure damage
    ("bridge collapse", 3),
    ("power line down", 2),
    ("dam damage", 3),
    ("pipeline rupture", 3),
    ("tower collapse", 2),
    # Agricultural
    ("crop drought stress", 1),
    ("locust swarm", 2),
    ("field fire", 2),
    ("irrigation failure", 1),
    # Medical
    ("mass casualty", 3),
    ("field hospital overload", 2),
    ("outbreak", 2),
    ("heat stroke cluster", 2),
    ("hypothermia victim", 2),
    # Security
    ("perimeter intrusion", 2),
    ("unauthorized vessel", 1),
    ("suspicious activity", 1),
    # Logistics / low risk baseline
    ("delivery waypoint", 0),
    ("supply drop", 0),
    ("crop survey", 0),
    ("wildlife survey", 0),
    ("routine inspection", 0),
]


# ── Synthetic weather generation ──────────────────────────────────────────────


def _sample_weather(rng: np.random.Generator) -> dict:
    """Sample random weather conditions."""
    wind_speed_mps = rng.uniform(0.0, 30.0)
    wind_gust_mps = wind_speed_mps + rng.uniform(0.0, 15.0)  # gust >= wind
    humidity_pct = rng.uniform(2.0, 100.0)
    temp_c = rng.uniform(-20.0, 50.0)
    precip_mm = (
        rng.choice([0.0], p=[1.0]) if rng.random() < 0.45 else rng.exponential(8.0)
    )
    visibility_km = rng.uniform(0.1, 40.0)
    return {
        "wind_speed_mps": float(wind_speed_mps),
        "wind_gust_mps": float(min(wind_gust_mps, 60.0)),
        "humidity_pct": float(humidity_pct),
        "temp_c": float(temp_c),
        "precip_mm": float(precip_mm),
        "visibility_km": float(visibility_km),
    }


def _normalize_weather(w: dict) -> list:
    """Convert raw weather values to normalized [0,1] feature vector."""
    return [
        min(w["wind_speed_mps"] / 30.0, 1.0),  # [15]
        min(w["wind_gust_mps"] / 40.0, 1.0),  # [16]
        (100.0 - w["humidity_pct"]) / 100.0,  # [17]
        (w["temp_c"] - (-20.0)) / 70.0,  # [18]
        min(w["precip_mm"] / 50.0, 1.0),  # [19]
        min(w["visibility_km"] / 20.0, 1.0),  # [20]
    ]


# ── Weather amplification rules ───────────────────────────────────────────────


def _apply_weather_modifier(
    base_risk: int, class_str: str, w: dict, rng: np.random.Generator
) -> int:
    """
    Apply domain-expert weather amplification rules to a base risk level.

    Rules are evaluated in priority order; the first matching rule that changes
    the label wins.  Gaussian noise N(0, 0.1) on a 0-3 scale is applied before
    rounding to add realistic label fuzziness.
    """
    cls = class_str.lower()
    wind = w["wind_speed_mps"]
    gust = w["wind_gust_mps"]
    humid = w["humidity_pct"]
    temp = w["temp_c"]
    precip = w["precip_mm"]
    vis = w["visibility_km"]

    # Normalized equivalents for threshold checks
    wind_n = wind / 30.0
    gust_n = gust / 40.0
    humid_n = (100.0 - humid) / 100.0  # inverted: high = dry
    precip_n = min(precip / 50.0, 1.0)
    vis_n = min(vis / 20.0, 1.0)
    temp_n = (temp - (-20.0)) / 70.0

    delta = 0.0  # modifier in risk-label units

    is_fire = any(
        kw in cls
        for kw in ["fire", "smoke", "ember", "hotspot", "burning", "wildfire", "blaze"]
    )
    is_flood = any(
        kw in cls
        for kw in ["flood", "surge", "inundation", "tsunami", "rising water", "levee"]
    )
    is_sar = any(
        kw in cls
        for kw in [
            "missing",
            "stranded",
            "distress",
            "casualty",
            "survivor",
            "overboard",
        ]
    )
    is_hazmat = any(
        kw in cls
        for kw in [
            "hazmat",
            "chemical",
            "spill",
            "toxic",
            "gas",
            "radiation",
            "plume",
            "ammonia",
        ]
    )
    is_infra = any(
        kw in cls
        for kw in ["bridge", "dam", "power line", "tower", "pipeline", "levee"]
    )
    is_agri = any(
        kw in cls for kw in ["crop", "drought", "field", "farm", "irrigation"]
    )

    # ── Fire / smoke rules ──────────────────────────────────────────────────
    if is_fire:
        # Dry + windy → maximum escalation
        if wind_n > 0.5 and humid_n > 0.7:
            delta += 1.5  # → CRITICAL
        elif wind_n > 0.3 and humid_n > 0.5:
            delta += 0.8  # → HIGH/CRITICAL
        # Raining + calm → downgrade
        if precip_n > 0.3 and wind_n < 0.2:
            delta -= 1.5  # → MEDIUM/LOW
        elif precip_n > 0.1:
            delta -= 0.5

    # ── Flood rules ─────────────────────────────────────────────────────────
    if is_flood:
        # Heavy rain + strong gust = active storm surge
        if precip_n > 0.5 and gust_n > 0.5:
            delta += 1.5  # → CRITICAL
        elif precip_n > 0.3:
            delta += 0.5
        # No rain, stale flood detection → downgrade
        if precip_n == 0.0 and base_risk < 3:
            delta -= 0.8

    # ── SAR rules ───────────────────────────────────────────────────────────
    if is_sar:
        # Low visibility makes SAR harder
        if vis_n < 0.3:
            delta += 0.8  # fog/smoke — harder to find
        elif vis_n < 0.5:
            delta += 0.4
        # Cold weather → hypothermia risk
        if temp < 5.0:
            delta += 0.6
        elif temp < 0.0:
            delta += 1.0

    # ── Hazmat rules ────────────────────────────────────────────────────────
    if is_hazmat:
        # Wind spreads plume
        if wind_n > 0.6:
            delta += 1.0  # plume spreading rapidly
        elif wind_n > 0.4:
            delta += 0.5
        # Calm: plume stays local, slightly less critical
        elif wind_n < 0.1:
            delta -= 0.3

    # ── Infrastructure rules ─────────────────────────────────────────────────
    if is_infra:
        # Active storm / high gusts
        if gust_n > 0.7:
            delta += 0.8
        elif gust_n > 0.5:
            delta += 0.4

    # ── Agricultural / drought rules ─────────────────────────────────────────
    if is_agri and "drought" in cls:
        if humid_n > 0.8 and temp_n > 0.7:
            delta += 1.5  # extreme fire risk conditions
        elif humid_n > 0.6:
            delta += 0.6

    # Add noise to represent label uncertainty
    delta += rng.normal(0.0, 0.1)

    # Apply modifier, clamp to [0, 3]
    adjusted = float(base_risk) + delta
    return int(np.clip(round(adjusted), 0, 3))


# ── Data generation ───────────────────────────────────────────────────────────


def generate_weather_risk_samples(n_samples: int, rng_seed: int = 42) -> tuple:
    """
    Generate n_samples (X, y) pairs with 21-float feature vectors.

    Returns:
        X: np.ndarray (n_samples, 21) float32
        y: np.ndarray (n_samples,)   int64  — risk label 0-3
    """
    rng = np.random.default_rng(rng_seed)
    X_list, y_list = [], []

    for i in range(n_samples):
        # Pick a scenario
        cls_str, base_risk = _SCENARIOS[i % len(_SCENARIOS)]

        # Randomize confidence slightly
        confidence = float(rng.uniform(0.45, 1.0))

        # Build base observation (lat/lon ~50% present)
        has_loc = rng.random() > 0.5
        obs = {
            "class": cls_str,
            "confidence": confidence,
            "lat": float(rng.uniform(-60.0, 80.0)) if has_loc else None,
            "lon": float(rng.uniform(-180.0, 180.0)) if has_loc else None,
        }

        # Extract 15-float base feature vector
        base_feat = extract(obs)

        # Sample weather conditions
        weather = _sample_weather(rng)
        weather_feats = _normalize_weather(weather)

        # Combine into 21-float vector
        feat_21 = base_feat + weather_feats
        X_list.append(feat_21)

        # Derive risk label with weather amplification
        label = _apply_weather_modifier(base_risk, cls_str, weather, rng)
        y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


# ── Feature importance (permutation-based for HGBT) ──────────────────────────


def _weather_feature_importances(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Print permutation importance for weather features (indices 15-20)."""
    from sklearn.metrics import accuracy_score

    base_acc = accuracy_score(y_test, model.predict(X_test))
    print("\nWeather feature importances (permutation, drop in accuracy):")
    print(f"  {'Feature':<30}  Importance")
    print(f"  {'-'*30}  {'-'*10}")

    for idx in range(15, WEATHER_FEATURE_DIM):
        X_perm = X_test.copy()
        rng = np.random.default_rng(idx)
        X_perm[:, idx] = rng.permutation(X_perm[:, idx])
        perm_acc = accuracy_score(y_test, model.predict(X_perm))
        importance = base_acc - perm_acc
        name = WEATHER_FEATURE_NAMES[idx]
        bar = "#" * max(0, int(importance * 200))
        print(f"  {name:<30}  {importance:+.4f}  {bar}")


# ── Weather comparison examples ───────────────────────────────────────────────


def _print_weather_examples(model) -> None:
    """
    Print 5 example predictions showing how weather changes the risk score.
    Each example: same detection class, two weather conditions (calm vs. severe).
    """
    from features import extract as feat_extract

    examples = [
        {
            "desc": "Wildfire smoke — calm rainy vs. dry windy",
            "class": "smoke plume",
            "conf": 0.88,
            "calm": {
                "wind_speed_mps": 2.0,
                "wind_gust_mps": 3.0,
                "humidity_pct": 85.0,
                "temp_c": 18.0,
                "precip_mm": 12.0,
                "visibility_km": 15.0,
            },
            "severe": {
                "wind_speed_mps": 22.0,
                "wind_gust_mps": 35.0,
                "humidity_pct": 6.0,
                "temp_c": 38.0,
                "precip_mm": 0.0,
                "visibility_km": 4.0,
            },
        },
        {
            "desc": "Flash flood — no rain vs. active storm",
            "class": "flash flood",
            "conf": 0.75,
            "calm": {
                "wind_speed_mps": 3.0,
                "wind_gust_mps": 5.0,
                "humidity_pct": 60.0,
                "temp_c": 20.0,
                "precip_mm": 0.0,
                "visibility_km": 18.0,
            },
            "severe": {
                "wind_speed_mps": 18.0,
                "wind_gust_mps": 28.0,
                "humidity_pct": 95.0,
                "temp_c": 15.0,
                "precip_mm": 35.0,
                "visibility_km": 3.0,
            },
        },
        {
            "desc": "Missing person SAR — clear day vs. fog + cold",
            "class": "missing person",
            "conf": 0.80,
            "calm": {
                "wind_speed_mps": 4.0,
                "wind_gust_mps": 6.0,
                "humidity_pct": 55.0,
                "temp_c": 22.0,
                "precip_mm": 0.0,
                "visibility_km": 20.0,
            },
            "severe": {
                "wind_speed_mps": 8.0,
                "wind_gust_mps": 12.0,
                "humidity_pct": 92.0,
                "temp_c": 1.0,
                "precip_mm": 4.0,
                "visibility_km": 0.5,
            },
        },
        {
            "desc": "Chemical spill — calm vs. high wind (plume spread)",
            "class": "chemical spill",
            "conf": 0.90,
            "calm": {
                "wind_speed_mps": 0.5,
                "wind_gust_mps": 1.0,
                "humidity_pct": 60.0,
                "temp_c": 20.0,
                "precip_mm": 0.0,
                "visibility_km": 15.0,
            },
            "severe": {
                "wind_speed_mps": 20.0,
                "wind_gust_mps": 30.0,
                "humidity_pct": 45.0,
                "temp_c": 25.0,
                "precip_mm": 0.0,
                "visibility_km": 8.0,
            },
        },
        {
            "desc": "Agricultural drought — humid cool vs. extreme heat/dry",
            "class": "crop drought stress",
            "conf": 0.70,
            "calm": {
                "wind_speed_mps": 3.0,
                "wind_gust_mps": 5.0,
                "humidity_pct": 70.0,
                "temp_c": 15.0,
                "precip_mm": 5.0,
                "visibility_km": 20.0,
            },
            "severe": {
                "wind_speed_mps": 10.0,
                "wind_gust_mps": 18.0,
                "humidity_pct": 8.0,
                "temp_c": 44.0,
                "precip_mm": 0.0,
                "visibility_km": 10.0,
            },
        },
    ]

    print("\n" + "=" * 65)
    print("Weather amplification examples (same detection, two conditions):")
    print("=" * 65)

    for ex in examples:
        obs = {
            "class": ex["class"],
            "confidence": ex["conf"],
            "lat": 35.0,
            "lon": -118.0,
        }
        base_feat = feat_extract(obs)

        row_calm = np.array(
            [base_feat + _normalize_weather(ex["calm"])], dtype=np.float32
        )
        row_severe = np.array(
            [base_feat + _normalize_weather(ex["severe"])], dtype=np.float32
        )

        pred_calm = model.predict(row_calm)[0]
        pred_severe = model.predict(row_severe)[0]

        label_calm = RISK_LABELS[int(pred_calm)]
        label_severe = RISK_LABELS[int(pred_severe)]

        changed = (
            " <-- ESCALATED"
            if pred_severe > pred_calm
            else (" <-- DOWNGRADED" if pred_severe < pred_calm else "")
        )

        print(f"\n  {ex['desc']}")
        print(f"    Calm conditions :  {label_calm:<8}")
        print(f"    Severe conditions: {label_severe:<8}{changed}")


# ── Training ──────────────────────────────────────────────────────────────────


def load_real_csv(csv_path: str):
    """Load real observations and map to weather risk scorer feature space (21 floats)."""
    import csv as _csv

    label_inv = {v: k for k, v in RISK_LABELS.items()}
    X, y = [], []
    # Default neutral weather (temperate, calm) for rows without weather data
    default_weather = {
        "wind_speed_mps": 5.0,
        "wind_gust_mps": 8.0,
        "humidity_pct": 55.0,
        "temp_c": 18.0,
        "precip_mm": 0.0,
        "visibility_km": 15.0,
    }
    try:
        with open(csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    risk_str = row.get("risk_level", "").strip().upper()
                    label = label_inv.get(risk_str)
                    if label is None:
                        continue
                    conf = float(row["confidence"])
                    lat_str = row.get("lat", "")
                    lon_str = row.get("lon", "")
                    obs = {
                        "class": row.get("class", ""),
                        "confidence": conf,
                        "lat": float(lat_str) if lat_str else None,
                        "lon": float(lon_str) if lon_str else None,
                    }
                    base_feat = extract(obs)
                    weather_feats = _normalize_weather(default_weather)
                    feat = base_feat + weather_feats
                    X.append(feat)
                    y.append(label)
                except (ValueError, KeyError):
                    continue
        print(
            f"  Loaded {len(X)} real weather-risk samples from {os.path.basename(csv_path)}"
        )
    except FileNotFoundError:
        print(f"  CSV not found: {csv_path}")
    return (
        (np.array(X, dtype=np.float32), np.array(y, dtype=np.int64))
        if X
        else (None, None)
    )


def train(
    n_samples: int = 80000, output_dir: Path = DEFAULT_OUTPUT_DIR, real_csv: str = None
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Heli.OS — Weather Risk Scorer Training")
    print("=" * 60)
    print(f"  Samples:     {n_samples:,}")
    print(f"  Features:    {WEATHER_FEATURE_DIM} (15 base + 6 weather)")
    print(f"  Output dir:  {output_dir}")
    print()

    # ── Generate data ──────────────────────────────────────────────────────
    print("Generating synthetic training data with weather amplification ...")
    X, y = generate_weather_risk_samples(n_samples)

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
                print(
                    f"  Real data (capped): { {RISK_LABELS[k]: v for k, v in real_counts.items()} }"
                )
                print(f"  Combined: {len(X)} total samples (real + synthetic)")

    dist = Counter(RISK_LABELS[i] for i in y.tolist())
    print(f"  {len(X):,} samples  |  class distribution: {dict(dist)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print()

    # ── Model ──────────────────────────────────────────────────────────────
    # CalibratedClassifierCV wraps HGBT for well-calibrated probability outputs.
    # Isotonic regression handles the non-linear probability calibration needed
    # for weather-skewed class distributions.
    print("Training CalibratedClassifierCV(HistGradientBoosting) ...")
    print("  (HGBT max_iter=500, max_depth=7, lr=0.04 | isotonic cv=5)")

    base_clf = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=7,
        learning_rate=0.04,
        min_samples_leaf=10,
        l2_regularization=0.05,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=42,
    )
    model = CalibratedClassifierCV(
        estimator=base_clf,
        cv=5,
        method="isotonic",
    )
    model.fit(X_train, y_train)
    print("  Training complete.")
    print()

    # ── Evaluation ─────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    target_names = [RISK_LABELS[i] for i in sorted(RISK_LABELS)]

    print("Classification report (test set):")
    print(
        classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        )
    )

    # ── Weather feature importances ─────────────────────────────────────────
    _weather_feature_importances(model, X_test, y_test)

    # ── Weather comparison examples ─────────────────────────────────────────
    _print_weather_examples(model)

    # ── Export to ONNX ──────────────────────────────────────────────────────
    print()
    print("Exporting to ONNX (opset 17) ...")
    onnx_path = output_dir / "weather_risk_scorer.onnx"
    initial_type = [("float_input", FloatTensorType([None, WEATHER_FEATURE_DIM]))]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=12,
    )
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    size_kb = onnx_path.stat().st_size / 1024
    print(f"  Model saved: {onnx_path}  ({size_kb:.1f} KB)")

    # ── Save feature names JSON ─────────────────────────────────────────────
    meta = {
        "feature_names": WEATHER_FEATURE_NAMES,
        "feature_dim": WEATHER_FEATURE_DIM,
        "base_feature_dim": FEATURE_DIM,
        "weather_feature_dim": 6,
        "weather_feature_indices": {
            "wind_speed_mps_norm": 15,
            "wind_gust_mps_norm": 16,
            "humidity_pct_inv": 17,
            "temp_c_norm": 18,
            "precip_mm_norm": 19,
            "visibility_km_norm": 20,
        },
        "normalization": {
            "wind_speed_mps_norm": "raw_value / 30.0  (clip to 1.0)",
            "wind_gust_mps_norm": "raw_value / 40.0  (clip to 1.0)",
            "humidity_pct_inv": "(100 - humidity_pct) / 100",
            "temp_c_norm": "(temp_c - (-20)) / 70  (range -20C to 50C)",
            "precip_mm_norm": "min(precip_mm / 50, 1.0)",
            "visibility_km_norm": "min(visibility_km / 20, 1.0)",
        },
        "risk_labels": {str(k): v for k, v in RISK_LABELS.items()},
        "onnx_input_name": "float_input",
        "model_class": "CalibratedClassifierCV(HistGradientBoostingClassifier)",
    }
    meta_path = output_dir / "weather_risk_scorer_feature_names.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved: {meta_path}")

    # ── Smoke test via onnxruntime ──────────────────────────────────────────
    print()
    print("Smoke test via onnxruntime ...")
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        test_cases = [
            (
                "active fire front",
                0.95,
                {
                    "wind_speed_mps": 25.0,
                    "wind_gust_mps": 35.0,
                    "humidity_pct": 4.0,
                    "temp_c": 40.0,
                    "precip_mm": 0.0,
                    "visibility_km": 5.0,
                },
            ),
            (
                "active fire front",
                0.95,
                {
                    "wind_speed_mps": 2.0,
                    "wind_gust_mps": 3.0,
                    "humidity_pct": 88.0,
                    "temp_c": 15.0,
                    "precip_mm": 20.0,
                    "visibility_km": 12.0,
                },
            ),
            (
                "missing person",
                0.82,
                {
                    "wind_speed_mps": 5.0,
                    "wind_gust_mps": 8.0,
                    "humidity_pct": 90.0,
                    "temp_c": 0.5,
                    "precip_mm": 2.0,
                    "visibility_km": 0.4,
                },
            ),
            (
                "chemical spill",
                0.91,
                {
                    "wind_speed_mps": 18.0,
                    "wind_gust_mps": 24.0,
                    "humidity_pct": 50.0,
                    "temp_c": 22.0,
                    "precip_mm": 0.0,
                    "visibility_km": 8.0,
                },
            ),
            (
                "delivery waypoint",
                0.97,
                {
                    "wind_speed_mps": 3.0,
                    "wind_gust_mps": 5.0,
                    "humidity_pct": 60.0,
                    "temp_c": 18.0,
                    "precip_mm": 1.0,
                    "visibility_km": 20.0,
                },
            ),
        ]
        for cls_str, conf, w in test_cases:
            obs = {"class": cls_str, "confidence": conf, "lat": 35.0, "lon": -118.0}
            feat = extract(obs) + _normalize_weather(w)
            row = np.array([feat], dtype=np.float32)
            pred_idx = sess.run(None, {"float_input": row})[0][0]
            label = RISK_LABELS[int(pred_idx)]
            wind_str = f"wind={w['wind_speed_mps']:.0f}m/s hum={w['humidity_pct']:.0f}%"
            print(f"  {cls_str:<25} conf={conf:.2f}  [{wind_str:<25}] → {label}")

    except ImportError:
        print("  onnxruntime not installed — skipping smoke test.")

    print()
    print("Done.")
    print(f"  ONNX model:  {onnx_path}")
    print(f"  Metadata:    {meta_path}")
    return str(onnx_path)


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train Heli.OS weather-adjusted risk scorer"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=80000,
        help="Number of synthetic training samples (default: 80000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for .onnx + .json (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--real-csv",
        dest="real_csv",
        default=None,
        help="Path to real observations CSV to blend with synthetic data",
    )
    args = parser.parse_args()
    train(n_samples=args.samples, output_dir=args.output_dir, real_csv=args.real_csv)


if __name__ == "__main__":
    main()
