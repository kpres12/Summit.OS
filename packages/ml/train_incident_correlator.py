"""
Train the Summit.OS incident correlator.

Pairwise binary classifier — given two detections, predicts whether they
represent the same incident (1) or separate incidents (0). Prevents the
KOFA tasking engine from dispatching multiple missions for the same event
(e.g. five UAVs for one wildfire reported by different sensors).

Outputs:
  packages/ml/models/incident_correlator.onnx         — runtime model
  packages/ml/models/incident_correlator_feature_names.json

Feature vector (20 floats, index-stable):
  [0]  spatial_distance_km       — great-circle distance between detections
  [1]  temporal_distance_s       — seconds between detections
  [2]  confidence_a              — confidence of detection A
  [3]  confidence_b              — confidence of detection B
  [4]  confidence_diff           — abs(conf_a - conf_b)
  [5]  same_domain               — 1 if both detections share the same hazard domain
  [6]  same_class                — 1 if exact class label match
  [7]  complementary_class       — 1 if known co-occurrence pair (fire+smoke, flood+surge)
  [8]  feat_dot_product          — dot product of 15-float feature vectors (cosine proxy)
  [9]  feat_l2_distance          — L2 distance between the two feature vectors
  [10] both_have_location        — 1 if both detections carry lat/lon
  [11] bearing_deg_normalized    — bearing A→B divided by 360
  [12] wind_alignment            — proxy for spatial offset aligned with drift direction
  [13] sensor_same_type          — 1 if same sensor modality (camera+camera, AIS+AIS)
  [14] rapid_succession          — 1 if temporal_distance < 30 s
  [15] close_proximity           — 1 if spatial_distance < 0.5 km
  [16] medium_proximity          — 1 if 0.5 ≤ spatial_distance < 2 km
  [17] far_apart                 — 1 if spatial_distance > 10 km
  [18] long_time_gap             — 1 if temporal_distance > 600 s (10 min)
  [19] escalating_confidence     — 1 if detection B has higher confidence than A

Label: 1 = same incident, 0 = different incidents

Usage:
  python train_incident_correlator.py
  python train_incident_correlator.py --samples 70000 --output-dir packages/ml/models
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

FEATURE_DIM = 20
FEATURE_NAMES = [
    "spatial_distance_km",
    "temporal_distance_s",
    "confidence_a",
    "confidence_b",
    "confidence_diff",
    "same_domain",
    "same_class",
    "complementary_class",
    "feat_dot_product",
    "feat_l2_distance",
    "both_have_location",
    "bearing_deg_normalized",
    "wind_alignment",
    "sensor_same_type",
    "rapid_succession",
    "close_proximity",
    "medium_proximity",
    "far_apart",
    "long_time_gap",
    "escalating_confidence",
]

# ---------------------------------------------------------------------------
# Known co-occurrence (complementary) class pairs
# ---------------------------------------------------------------------------

_COMPLEMENTARY_PAIRS = [
    ("fire", "smoke"),
    ("smoke", "fire"),
    ("flood", "surge"),
    ("surge", "flood"),
    ("flood", "submerged"),
    ("submerged", "flood"),
    ("flood", "inundation"),
    ("inundation", "flood"),
    ("fire", "ember"),
    ("ember", "fire"),
    ("earthquake", "collapse"),
    ("collapse", "earthquake"),
    ("chemical", "plume"),
    ("plume", "chemical"),
    ("spill", "plume"),
    ("plume", "spill"),
]

_COMPLEMENTARY_SET = frozenset(_COMPLEMENTARY_PAIRS)

# Domain groupings (must match features.py categories)
_DOMAIN_MAP = {
    "fire": "fire_smoke",
    "smoke": "fire_smoke",
    "wildfire": "fire_smoke",
    "ember": "fire_smoke",
    "hotspot": "fire_smoke",
    "flood": "flood_water",
    "inundation": "flood_water",
    "surge": "flood_water",
    "submerged": "flood_water",
    "tsunami": "flood_water",
    "person": "person",
    "survivor": "person",
    "missing": "person",
    "stranded": "person",
    "vehicle": "vehicle",
    "drone": "vehicle",
    "uav": "vehicle",
    "vessel": "vehicle",
    "boat": "vehicle",
    "ship": "vehicle",
    "chemical": "hazmat",
    "spill": "hazmat",
    "plume": "hazmat",
    "toxic": "hazmat",
    "collapse": "structural",
    "earthquake": "structural",
    "landslide": "structural",
}

# Domain lists used during synthesis
_DOMAINS = {
    "fire_smoke": ["fire", "smoke", "wildfire", "ember", "hotspot"],
    "flood_water": ["flood", "inundation", "surge", "submerged", "tsunami"],
    "person": ["person", "survivor", "missing", "stranded"],
    "vehicle": ["vehicle", "drone", "uav", "vessel", "boat"],
    "hazmat": ["chemical", "spill", "plume", "toxic"],
    "structural": ["collapse", "earthquake", "landslide"],
}
_DOMAIN_NAMES = list(_DOMAINS.keys())


def _get_domain(cls: str) -> str:
    return _DOMAIN_MAP.get(cls, "unknown")


def _random_class(rng, domain: str) -> str:
    choices = _DOMAINS[domain]
    return choices[rng.integers(0, len(choices))]


def _build_feature_vector(
    dist_km: float,
    time_s: float,
    conf_a: float,
    conf_b: float,
    cls_a: str,
    cls_b: str,
    feat_a: np.ndarray,
    feat_b: np.ndarray,
    has_loc: bool,
    bearing_norm: float,
    wind_align: float,
    same_sensor: bool,
    noise_rng,
) -> np.ndarray:
    """Assemble the 20-float feature vector from raw detection pair attributes."""
    dom_a = _get_domain(cls_a)
    dom_b = _get_domain(cls_b)

    same_domain = float(dom_a == dom_b and dom_a != "unknown")
    same_class = float(cls_a == cls_b)
    complementary = float((cls_a, cls_b) in _COMPLEMENTARY_SET)
    dot_prod = float(np.dot(feat_a, feat_b))
    l2_dist = float(np.linalg.norm(feat_a - feat_b))
    conf_diff = abs(conf_a - conf_b)
    rapid = float(time_s < 30)
    close = float(dist_km < 0.5)
    medium = float(0.5 <= dist_km < 2.0)
    far = float(dist_km > 10.0)
    long_gap = float(time_s > 600)
    escalating = float(conf_b > conf_a)

    vec = np.array(
        [
            dist_km,
            time_s,
            conf_a,
            conf_b,
            conf_diff,
            same_domain,
            same_class,
            complementary,
            dot_prod,
            l2_dist,
            float(has_loc),
            bearing_norm,
            wind_align,
            float(same_sensor),
            rapid,
            close,
            medium,
            far,
            long_gap,
            escalating,
        ],
        dtype=np.float32,
    )

    # Add small Gaussian noise to continuous features only
    noise = noise_rng.normal(0, 0.05, FEATURE_DIM).astype(np.float32)
    noise[5:8] = 0.0  # binary class-match flags — no noise
    noise[10] = 0.0  # both_have_location
    noise[13] = 0.0  # sensor_same_type
    noise[14:] = 0.0  # binary derived flags
    vec = vec + noise

    # Clip continuous features to valid ranges
    vec[0] = max(0.0, vec[0])  # distance >= 0
    vec[1] = max(0.0, vec[1])  # time >= 0
    vec[2] = float(np.clip(vec[2], 0.0, 1.0))
    vec[3] = float(np.clip(vec[3], 0.0, 1.0))
    vec[4] = float(np.clip(vec[4], 0.0, 1.0))

    return vec


def _make_feature_vec_for_class(rng, cls: str) -> np.ndarray:
    """
    Build a synthetic 15-float detection feature vector (matches features.py FEATURE_DIM=15).
    Encodes confidence + has_location + 13 binary domain flags.
    """
    domain = _get_domain(cls)
    conf = float(rng.uniform(0.5, 0.99))
    vec = np.zeros(15, dtype=np.float32)
    vec[0] = conf
    vec[1] = 1.0  # has_location

    domain_to_idx = {
        "fire_smoke": 2,
        "person": 3,
        "flood_water": 4,
        "structural": 5,
        "vehicle": 6,
        "hazmat": 7,
    }
    idx = domain_to_idx.get(domain)
    if idx is not None:
        vec[idx] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def generate_data(n_total: int = 70000, seed: int = 42):
    """
    Generate synthetic detection-pair feature vectors with binary labels.

    Label distribution: ~40% same-incident (1), ~60% different (0).

    Returns:
        X      — float32 array of shape (n_total, FEATURE_DIM)
        y      — int32 array: 1 = same incident, 0 = different incidents
    """
    rng = np.random.default_rng(seed)

    n_same = int(n_total * 0.40)
    n_diff = n_total - n_same

    samples_X = []
    samples_y = []

    # -----------------------------------------------------------------------
    # SAME-INCIDENT pairs (label = 1)
    # -----------------------------------------------------------------------

    # Archetype A: Close spatial + recent temporal + same domain
    # Very high correlation — nearly always same incident.
    n_a = n_same // 5
    for _ in range(n_a):
        domain = _DOMAIN_NAMES[rng.integers(0, len(_DOMAIN_NAMES))]
        cls_a = _random_class(rng, domain)
        cls_b = _random_class(rng, domain)
        dist = float(rng.uniform(0.0, 1.0))
        t = float(rng.uniform(0, 300))
        conf_a = float(rng.uniform(0.6, 0.99))
        conf_b = float(rng.uniform(0.55, 0.99))
        fa = _make_feature_vec_for_class(rng, cls_a)
        fb = _make_feature_vec_for_class(rng, cls_b)
        vec = _build_feature_vector(
            dist,
            t,
            conf_a,
            conf_b,
            cls_a,
            cls_b,
            fa,
            fb,
            True,
            float(rng.uniform(0, 1)),
            float(rng.uniform(0, 1)),
            bool(rng.integers(0, 2)),
            rng,
        )
        samples_X.append(vec)
        samples_y.append(1)

    # Archetype B: Same class + close temporal
    n_b = n_same // 5
    for _ in range(n_b):
        domain = _DOMAIN_NAMES[rng.integers(0, len(_DOMAIN_NAMES))]
        cls_a = _random_class(rng, domain)
        cls_b = cls_a  # exact class match
        dist = float(rng.uniform(0.0, 3.0))
        t = float(rng.uniform(0, 120))
        conf_a = float(rng.uniform(0.5, 0.95))
        conf_b = float(rng.uniform(0.5, 0.99))
        fa = _make_feature_vec_for_class(rng, cls_a)
        fb = fa.copy()
        fb[0] = conf_b
        vec = _build_feature_vector(
            dist,
            t,
            conf_a,
            conf_b,
            cls_a,
            cls_b,
            fa,
            fb,
            True,
            float(rng.uniform(0, 1)),
            float(rng.uniform(0, 1)),
            True,
            rng,
        )
        samples_X.append(vec)
        samples_y.append(1)

    # Archetype C: fire + smoke within 2 km, < 5 min
    n_c = n_same // 5
    for _ in range(n_c):
        cls_a = "fire"
        cls_b = "smoke"
        dist = float(rng.uniform(0.0, 2.0))
        t = float(rng.uniform(0, 300))
        conf_a = float(rng.uniform(0.65, 0.99))
        conf_b = float(rng.uniform(0.55, 0.95))
        fa = _make_feature_vec_for_class(rng, cls_a)
        fb = _make_feature_vec_for_class(rng, cls_b)
        vec = _build_feature_vector(
            dist,
            t,
            conf_a,
            conf_b,
            cls_a,
            cls_b,
            fa,
            fb,
            True,
            float(rng.uniform(0, 1)),
            float(rng.uniform(0, 1)),
            bool(rng.integers(0, 2)),
            rng,
        )
        samples_X.append(vec)
        samples_y.append(1)

    # Archetype D: flood + surge/submerged within 5 km, < 30 min
    n_d = n_same // 5
    flood_classes = ["flood", "surge", "submerged", "inundation"]
    for _ in range(n_d):
        cls_a = "flood"
        cls_b = flood_classes[rng.integers(1, len(flood_classes))]
        dist = float(rng.uniform(0.0, 5.0))
        t = float(rng.uniform(0, 1800))
        conf_a = float(rng.uniform(0.6, 0.99))
        conf_b = float(rng.uniform(0.55, 0.99))
        fa = _make_feature_vec_for_class(rng, cls_a)
        fb = _make_feature_vec_for_class(rng, cls_b)
        vec = _build_feature_vector(
            dist,
            t,
            conf_a,
            conf_b,
            cls_a,
            cls_b,
            fa,
            fb,
            True,
            float(rng.uniform(0, 1)),
            float(rng.uniform(0, 1)),
            bool(rng.integers(0, 2)),
            rng,
        )
        samples_X.append(vec)
        samples_y.append(1)

    # Archetype E: rapid succession, same domain, escalating confidence
    n_e = n_same - (n_a + n_b + n_c + n_d)
    for _ in range(n_e):
        domain = _DOMAIN_NAMES[rng.integers(0, len(_DOMAIN_NAMES))]
        cls_a = _random_class(rng, domain)
        cls_b = _random_class(rng, domain)
        dist = float(rng.uniform(0.0, 1.5))
        t = float(rng.uniform(0, 30))  # rapid succession
        conf_a = float(rng.uniform(0.4, 0.75))
        conf_b = float(rng.uniform(conf_a, 0.99))  # escalating
        fa = _make_feature_vec_for_class(rng, cls_a)
        fb = _make_feature_vec_for_class(rng, cls_b)
        vec = _build_feature_vector(
            dist,
            t,
            conf_a,
            conf_b,
            cls_a,
            cls_b,
            fa,
            fb,
            True,
            float(rng.uniform(0, 1)),
            float(rng.uniform(0, 1)),
            bool(rng.integers(0, 2)),
            rng,
        )
        samples_X.append(vec)
        samples_y.append(1)

    # -----------------------------------------------------------------------
    # DIFFERENT-INCIDENT pairs (label = 0)
    # -----------------------------------------------------------------------

    # Archetype F: far apart geographically
    n_f = n_diff // 3
    for _ in range(n_f):
        dom_a = _DOMAIN_NAMES[rng.integers(0, len(_DOMAIN_NAMES))]
        dom_b = _DOMAIN_NAMES[rng.integers(0, len(_DOMAIN_NAMES))]
        cls_a = _random_class(rng, dom_a)
        cls_b = _random_class(rng, dom_b)
        dist = float(rng.uniform(20.0, 500.0))
        t = float(rng.uniform(0, 7200))
        conf_a = float(rng.uniform(0.4, 0.99))
        conf_b = float(rng.uniform(0.4, 0.99))
        fa = _make_feature_vec_for_class(rng, cls_a)
        fb = _make_feature_vec_for_class(rng, cls_b)
        vec = _build_feature_vector(
            dist,
            t,
            conf_a,
            conf_b,
            cls_a,
            cls_b,
            fa,
            fb,
            bool(rng.integers(0, 2)),
            float(rng.uniform(0, 1)),
            float(rng.uniform(0, 1)),
            bool(rng.integers(0, 2)),
            rng,
        )
        samples_X.append(vec)
        samples_y.append(0)

    # Archetype G: different domains (fire vs flood, etc.)
    n_g = n_diff // 3
    domain_pairs = [
        ("fire_smoke", "flood_water"),
        ("fire_smoke", "person"),
        ("hazmat", "vehicle"),
        ("structural", "flood_water"),
        ("person", "hazmat"),
        ("vehicle", "structural"),
    ]
    for _ in range(n_g):
        pair = domain_pairs[rng.integers(0, len(domain_pairs))]
        cls_a = _random_class(rng, pair[0])
        cls_b = _random_class(rng, pair[1])
        dist = float(rng.uniform(0.5, 100.0))
        t = float(rng.uniform(30, 14400))
        conf_a = float(rng.uniform(0.4, 0.99))
        conf_b = float(rng.uniform(0.4, 0.99))
        fa = _make_feature_vec_for_class(rng, cls_a)
        fb = _make_feature_vec_for_class(rng, cls_b)
        vec = _build_feature_vector(
            dist,
            t,
            conf_a,
            conf_b,
            cls_a,
            cls_b,
            fa,
            fb,
            bool(rng.integers(0, 2)),
            float(rng.uniform(0, 1)),
            float(rng.uniform(0, 1)),
            bool(rng.integers(0, 2)),
            rng,
        )
        samples_X.append(vec)
        samples_y.append(0)

    # Archetype H: large time gap (> 1 hour) — likely distinct incidents
    n_h = n_diff - (n_f + n_g)
    for _ in range(n_h):
        dom_a = _DOMAIN_NAMES[rng.integers(0, len(_DOMAIN_NAMES))]
        dom_b = _DOMAIN_NAMES[rng.integers(0, len(_DOMAIN_NAMES))]
        cls_a = _random_class(rng, dom_a)
        cls_b = _random_class(rng, dom_b)
        dist = float(rng.uniform(0.0, 50.0))
        t = float(rng.uniform(3600, 86400))  # 1 h → 24 h gap
        conf_a = float(rng.uniform(0.4, 0.99))
        conf_b = float(rng.uniform(0.4, 0.99))
        fa = _make_feature_vec_for_class(rng, cls_a)
        fb = _make_feature_vec_for_class(rng, cls_b)
        vec = _build_feature_vector(
            dist,
            t,
            conf_a,
            conf_b,
            cls_a,
            cls_b,
            fa,
            fb,
            bool(rng.integers(0, 2)),
            float(rng.uniform(0, 1)),
            float(rng.uniform(0, 1)),
            bool(rng.integers(0, 2)),
            rng,
        )
        samples_X.append(vec)
        samples_y.append(0)

    X = np.array(samples_X, dtype=np.float32)
    y = np.array(samples_y, dtype=np.int32)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def load_real_csv(csv_path: str):
    """Load real observations and form detection pairs for incident correlator.

    Because the CSV contains single observations (not pre-paired), we form pairs
    by pairing consecutive rows that share the same class (same incident) and
    pairing rows from different classes (different incidents), using spatial
    distance from lat/lon and a fixed synthetic time gap.
    """
    import csv as _csv
    import math

    def _haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    rows = []
    try:
        with open(csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    conf = float(row["confidence"])
                    lat = float(row["lat"])
                    lon = float(row["lon"])
                    cls = row["class"].strip().lower()
                    rows.append(
                        {"class": cls, "confidence": conf, "lat": lat, "lon": lon}
                    )
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        print(f"  CSV not found: {csv_path}")
        return None, None

    # Build a fast feature vec per row
    def _row_feat(r):
        v = np.zeros(15, dtype=np.float32)
        v[0] = r["confidence"]
        v[1] = 1.0  # has location
        domain = _get_domain(r["class"].split()[0] if r["class"] else "")
        domain_to_idx = {
            "fire_smoke": 2,
            "person": 3,
            "flood_water": 4,
            "structural": 5,
            "vehicle": 6,
            "hazmat": 7,
        }
        idx = domain_to_idx.get(domain)
        if idx is not None:
            v[idx] = 1.0
        return v

    X, y = [], []
    import random as _rand

    _rand.seed(42)

    # Group by class for same-incident pairs
    from collections import defaultdict

    by_class = defaultdict(list)
    for r in rows:
        by_class[r["class"]].append(r)

    # Same-incident pairs: consecutive rows of same class within 5 km
    for cls, group in by_class.items():
        for i in range(len(group) - 1):
            a, b = group[i], group[i + 1]
            try:
                dist = _haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
                if dist > 50.0:
                    continue  # skip implausible same-incident pairs
                time_s = float(_rand.uniform(0, 600))
                fa = _row_feat(a)
                fb = _row_feat(b)
                vec = _build_feature_vector(
                    dist,
                    time_s,
                    a["confidence"],
                    b["confidence"],
                    a["class"].split()[0],
                    b["class"].split()[0],
                    fa,
                    fb,
                    True,
                    float(_rand.uniform(0, 1)),
                    float(_rand.uniform(0, 1)),
                    True,
                    np.random.default_rng(42),
                )
                X.append(vec)
                y.append(1)
            except Exception:
                continue

    # Different-incident pairs: random rows of different classes
    row_list = rows[: min(len(rows), 20000)]
    _rand.shuffle(row_list)
    n_diff = min(len(X), len(row_list) // 2)
    for i in range(n_diff):
        a = row_list[i]
        b = row_list[-(i + 1)]
        if a["class"] == b["class"]:
            continue
        try:
            dist = _haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
            time_s = float(_rand.uniform(600, 86400))
            fa = _row_feat(a)
            fb = _row_feat(b)
            vec = _build_feature_vector(
                dist,
                time_s,
                a["confidence"],
                b["confidence"],
                a["class"].split()[0],
                b["class"].split()[0],
                fa,
                fb,
                True,
                float(_rand.uniform(0, 1)),
                float(_rand.uniform(0, 1)),
                False,
                np.random.default_rng(43),
            )
            X.append(vec)
            y.append(0)
        except Exception:
            continue

    print(
        f"  Loaded {len(X)} real detection-pair samples from {os.path.basename(csv_path)}"
    )
    return (
        (np.array(X, dtype=np.float32), np.array(y, dtype=np.int32))
        if X
        else (None, None)
    )


def train(n_samples: int = 70000, output_dir: str = None, real_csv: str = None):
    """Train CalibratedClassifierCV(HistGradientBoostingClassifier) and export to ONNX."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating {n_samples:,} synthetic detection-pair samples...")
    X, y = generate_data(n_total=n_samples)
    n_same = int((y == 1).sum())
    n_diff = int((y == 0).sum())
    print(f"  Same incident (1): {n_same:,}  ({n_same/len(y)*100:.1f}%)")
    print(f"  Different    (0): {n_diff:,}  ({n_diff/len(y)*100:.1f}%)")

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
                y = np.concatenate([y, np.array(real_capped_y, dtype=np.int32)])
                label_map = {0: "different", 1: "same"}
                print(
                    f"  Real data (capped): { {label_map[k]: v for k, v in real_counts.items()} }"
                )
                print(f"  Combined: {len(X)} total samples (real + synthetic)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # CalibratedClassifierCV wraps HGBC for calibrated probability outputs.
    # Isotonic regression calibration with 5-fold CV corrects for class
    # imbalance and gives reliable P(same_incident) scores at inference time.
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
    clf = CalibratedClassifierCV(base_clf, cv=5, method="isotonic")

    print("\nFitting CalibratedClassifierCV(HistGradientBoostingClassifier, cv=5)...")
    clf.fit(X_train, y_train)

    # --- Evaluation ----------------------------------------------------------
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print(
        f"\nPrecision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  AUC-ROC: {auc:.4f}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["different_incident", "same_incident"],
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"              pred_diff  pred_same")
    print(f"  actual_diff  {cm[0, 0]:>8d}  {cm[0, 1]:>8d}")
    print(f"  actual_same  {cm[1, 0]:>8d}  {cm[1, 1]:>8d}")

    # Feature importance — use mean feature values for same vs different
    # as a discriminative proxy (underlying HGBC feature_importances_ is
    # not directly accessible through CalibratedClassifierCV wrapper)
    same_mean = X_test[y_test == 1].mean(axis=0)
    diff_mean = X_test[y_test == 0].mean(axis=0)
    delta = np.abs(same_mean - diff_mean)
    ranked = np.argsort(delta)[::-1]
    print("\nMost discriminating features (mean shift different → same):")
    for rank, idx in enumerate(ranked[:10], 1):
        print(
            f"  {rank:>2}. {FEATURE_NAMES[idx]:<28}  "
            f"same={same_mean[idx]:.3f}  diff={diff_mean[idx]:.3f}  "
            f"delta={delta[idx]:.3f}"
        )

    # --- ONNX export (matches project skl2onnx pattern) ---------------------
    onnx_path = os.path.join(output_dir, "incident_correlator.onnx")
    initial_type = [("float_input", FloatTensorType([None, FEATURE_DIM]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type, target_opset=12)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"\nModel saved: {onnx_path}")

    # --- Feature names JSON -------------------------------------------------
    names_path = os.path.join(output_dir, "incident_correlator_feature_names.json")
    with open(names_path, "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    print(f"Feature names saved: {names_path}")

    # --- Smoke test via ONNX runtime ----------------------------------------
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        # fmt: (description, feature_vector)
        smoke_cases = [
            (
                "fire + smoke, 0.3 km, 45 s  (SAME)",
                # dist  time   ca    cb   diff  s_dom s_cls compl  dot   l2
                [
                    0.3,
                    45.0,
                    0.88,
                    0.76,
                    0.12,
                    1.0,
                    0.0,
                    1.0,
                    6.5,
                    1.2,
                    # hasloc bear  wind  sens  rapid close med   far   lgap  esc
                    1.0,
                    0.23,
                    0.61,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ),
            (
                "flood + surge, 1.2 km, 8 min (SAME)",
                [
                    1.2,
                    480.0,
                    0.82,
                    0.91,
                    0.09,
                    1.0,
                    0.0,
                    1.0,
                    5.8,
                    1.4,
                    1.0,
                    0.44,
                    0.55,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            ),
            (
                "fire vs flood, 2 km, 1 min   (DIFF)",
                [
                    2.0,
                    60.0,
                    0.90,
                    0.85,
                    0.05,
                    0.0,
                    0.0,
                    0.0,
                    1.2,
                    3.8,
                    1.0,
                    0.12,
                    0.30,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ),
            (
                "same class, 150 km apart      (DIFF)",
                [
                    150.0,
                    300.0,
                    0.75,
                    0.80,
                    0.05,
                    1.0,
                    1.0,
                    0.0,
                    7.1,
                    0.3,
                    1.0,
                    0.77,
                    0.42,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                ],
            ),
            (
                "same class, 0.1 km, 2 h ago  (DIFF)",
                [
                    0.1,
                    7200.0,
                    0.88,
                    0.71,
                    0.17,
                    1.0,
                    1.0,
                    0.0,
                    6.9,
                    0.4,
                    1.0,
                    0.05,
                    0.50,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
            ),
        ]

        print("\nSmoke test:")
        for desc, feat in smoke_cases:
            x = np.array([feat], dtype=np.float32)
            result = sess.run(None, {"float_input": x})
            pred_label = int(result[0][0])
            prob = result[1][0]  # probability dict or array
            if isinstance(prob, dict):
                p_same = prob.get(1, 0.0)
            else:
                p_same = float(prob[1]) if len(prob) > 1 else float(prob[0])
            tag = "SAME INCIDENT" if pred_label == 1 else "different"
            print(f"  {desc:<42}  → {tag:<15} P(same)={p_same:.3f}")
    except ImportError:
        print("\nNote: onnxruntime not installed — skipping smoke test.")

    return onnx_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Summit.OS incident correlator and export to ONNX."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=70000,
        help="Total number of synthetic detection-pair samples (default: 70000)",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
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
