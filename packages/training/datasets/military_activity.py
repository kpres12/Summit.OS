"""
Military Activity Pattern Dataset Generator

Synthetic dataset for pattern-of-life anomaly detection in military/sensitive contexts.
No real data source — all samples are procedurally generated from realistic
temporal, spatial, and behavioural distributions.

Features (10):
  - entity_count              : number of entities in cluster
  - speed_variance            : variance in entity speeds (m/s)²
  - dwell_time_min            : time spent in area (minutes)
  - time_of_day_h             : hour of observation (0-23)
  - formation_type            : 0=dispersed, 1=column, 2=wedge, 3=coil
  - route_deviation_m         : deviation from known/expected route (m)
  - vehicle_class             : 0=light, 1=medium, 2=heavy, 3=mixed
  - area_sensitivity          : 0=low, 1=medium, 2=high
  - comms_activity            : normalised RF/comms activity (0-1)
  - activity_cluster_radius_m : spatial radius of activity cluster (m)

Labels:
  0 = routine        (known pattern, expected timing/location)
  1 = anomalous      (unusual but not necessarily threatening)
  2 = high_threat_indicator
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import NamedTuple

import numpy as np


FEATURES = [
    "entity_count",
    "speed_variance",
    "dwell_time_min",
    "time_of_day_h",
    "formation_type",
    "route_deviation_m",
    "vehicle_class",
    "area_sensitivity",
    "comms_activity",
    "activity_cluster_radius_m",
]

LABELS = ["routine", "anomalous", "high_threat_indicator"]


class MilitaryActivitySample(NamedTuple):
    features: list[float]
    label: int


def _routine_sample(rng: random.Random) -> list[float]:
    """Known pattern: expected timing, formation, low deviation, low comms."""
    return [
        rng.uniform(2, 12),          # entity_count: small unit
        rng.uniform(0.0, 2.0),       # speed_variance: consistent
        rng.uniform(5, 60),          # dwell_time_min: brief
        rng.choices(                 # time_of_day_h: daytime ops
            list(range(7, 19)),
            weights=[1] * 12,
        )[0],
        rng.choices([0, 1], weights=[0.5, 0.5])[0],   # formation: dispersed/column
        rng.uniform(0, 200),         # route_deviation_m: on route
        rng.choices([0, 1], weights=[0.6, 0.4])[0],   # vehicle_class: light/medium
        rng.choices([0, 1], weights=[0.7, 0.3])[0],   # area_sensitivity: low/medium
        rng.uniform(0.1, 0.4),       # comms_activity: low
        rng.uniform(100, 800),       # activity_cluster_radius_m: normal spread
    ]


def _anomalous_sample(rng: random.Random) -> list[float]:
    """Unusual but not immediately threatening: off-route, wrong time, high variance."""
    return [
        rng.uniform(5, 25),          # entity_count: medium group
        rng.uniform(1.5, 6.0),       # speed_variance: inconsistent
        rng.uniform(30, 180),        # dwell_time_min: prolonged dwell
        rng.choices(                 # time_of_day_h: twilight / early morning
            list(range(0, 7)) + list(range(19, 24)),
            weights=[1] * 12,
        )[0],
        rng.choices([0, 2, 3], weights=[0.4, 0.3, 0.3])[0],  # formation varies
        rng.uniform(200, 1000),      # route_deviation_m: off route
        rng.choices([1, 2], weights=[0.5, 0.5])[0],   # vehicle_class: medium/heavy
        rng.choices([1, 2], weights=[0.5, 0.5])[0],   # area_sensitivity: medium/high
        rng.uniform(0.3, 0.7),       # comms_activity: elevated
        rng.uniform(200, 1500),      # activity_cluster_radius_m
    ]


def _high_threat_sample(rng: random.Random) -> list[float]:
    """Threat indicators: high entity count, coil formation, near sensitive areas,
    high comms, dark-hours ops, large deviation, heavy vehicles."""
    return [
        rng.uniform(15, 50),         # entity_count: large force
        rng.uniform(0.5, 3.0),       # speed_variance: coordinated (low-ish variance)
        rng.uniform(60, 360),        # dwell_time_min: staging / long dwell
        rng.choices(                 # time_of_day_h: night ops
            list(range(0, 5)) + list(range(20, 24)),
            weights=[1] * 9,
        )[0],
        rng.choices([2, 3], weights=[0.5, 0.5])[0],   # formation: wedge/coil
        rng.uniform(500, 3000),      # route_deviation_m: avoiding known routes
        rng.choices([2, 3], weights=[0.5, 0.5])[0],   # vehicle_class: heavy/mixed
        2,                           # area_sensitivity: high
        rng.uniform(0.6, 1.0),       # comms_activity: very high
        rng.uniform(500, 3000),      # activity_cluster_radius_m: large assembly
    ]


def generate(n: int = 600, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate n synthetic military activity samples.

    Returns:
        X: ndarray of shape (n, 10) — feature matrix
        y: ndarray of shape (n,)   — integer labels (0/1/2)
    """
    rng = random.Random(seed)

    per_class = n // 3
    remainder = n - per_class * 3

    samples: list[MilitaryActivitySample] = []

    for _ in range(per_class):
        samples.append(MilitaryActivitySample(_routine_sample(rng), 0))
    for _ in range(per_class):
        samples.append(MilitaryActivitySample(_anomalous_sample(rng), 1))
    for _ in range(per_class + remainder):
        samples.append(MilitaryActivitySample(_high_threat_sample(rng), 2))

    rng.shuffle(samples)

    X = np.array([s.features for s in samples], dtype=np.float32)
    y = np.array([s.label for s in samples], dtype=np.int32)
    return X, y


def load_as_training_samples(data_dir: Path | None = None) -> list[dict]:
    """
    Compatibility shim — returns list of dicts matching the standard training
    sample format used by other dataset loaders in this package.

    ``data_dir`` is accepted but ignored (data is fully synthetic).
    """
    X, y = generate()
    samples = []
    for row, label in zip(X.tolist(), y.tolist()):
        d = {feat: val for feat, val in zip(FEATURES, row)}
        d["label"] = int(label)
        samples.append(d)
    return samples


if __name__ == "__main__":
    X, y = generate(n=600)
    class_counts = {0: int((y == 0).sum()), 1: int((y == 1).sum()), 2: int((y == 2).sum())}
    print(f"Generated {len(X)} samples: {class_counts}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Sample row: {X[0].tolist()}")
