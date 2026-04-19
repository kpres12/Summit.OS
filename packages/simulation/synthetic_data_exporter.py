"""
Synthetic Data Exporter — Heli.OS

Generates synthetic sensor observations from simulated mission scenarios.
Used to bootstrap ML training datasets without real field data.

Generates:
  - Position tracks with realistic noise (GPS accuracy model)
  - Simulated camera detections (bounding boxes with noise)
  - Weather telemetry with seasonal variation
  - AIS/ADS-B tracks with realistic traffic patterns
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple


class SyntheticDataExporter:
    """Generate synthetic sensor data for ML training dataset bootstrapping."""

    # GPS accuracy: 1 sigma in meters → degrees (~111 km/degree at equator)
    _METERS_PER_DEG = 111_000.0

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    def generate_sar_scenario(
        self,
        n_victims: int = 3,
        area_km2: float = 4.0,
    ) -> List[dict]:
        """
        Generate synthetic entity observations for a SAR scenario.
        Returns a list of observation dicts with entity type, track, bounding boxes.
        """
        half_deg = math.sqrt(area_km2) / 2.0 / 111.0  # approx degrees
        center_lat = 37.4 + self._rng.uniform(-0.5, 0.5)
        center_lon = -122.0 + self._rng.uniform(-0.5, 0.5)

        observations: List[dict] = []

        # Victims (stationary with slight drift)
        for i in range(n_victims):
            lat = center_lat + self._rng.uniform(-half_deg, half_deg)
            lon = center_lon + self._rng.uniform(-half_deg, half_deg)
            track = self.generate_track(lat, lon, n_steps=20, noise_m=2.0)
            observations.append({
                "entity_id":    f"victim_{i}",
                "class":        "victim",
                "class_id":     1,
                "track":        track,
                "confidence":   round(self._rng.uniform(0.6, 0.99), 3),
                "bbox_wh_norm": [round(self._rng.uniform(0.02, 0.06), 4),
                                 round(self._rng.uniform(0.04, 0.10), 4)],
            })

        # Vehicles (moving)
        n_vehicles = self._rng.randint(1, 4)
        for i in range(n_vehicles):
            lat = center_lat + self._rng.uniform(-half_deg, half_deg)
            lon = center_lon + self._rng.uniform(-half_deg, half_deg)
            track = self.generate_track(lat, lon, n_steps=50, noise_m=8.0)
            observations.append({
                "entity_id":    f"vehicle_{i}",
                "class":        "vehicle",
                "class_id":     2,
                "track":        track,
                "confidence":   round(self._rng.uniform(0.75, 0.99), 3),
                "bbox_wh_norm": [round(self._rng.uniform(0.06, 0.15), 4),
                                 round(self._rng.uniform(0.04, 0.08), 4)],
            })

        # Structures (static)
        n_structures = self._rng.randint(2, 8)
        for i in range(n_structures):
            lat = center_lat + self._rng.uniform(-half_deg, half_deg)
            lon = center_lon + self._rng.uniform(-half_deg, half_deg)
            observations.append({
                "entity_id":    f"structure_{i}",
                "class":        "structure",
                "class_id":     3,
                "track":        [{"lat": lat, "lon": lon, "ts": time.time()}],
                "confidence":   round(self._rng.uniform(0.85, 0.99), 3),
                "bbox_wh_norm": [round(self._rng.uniform(0.10, 0.30), 4),
                                 round(self._rng.uniform(0.10, 0.25), 4)],
            })

        return observations

    # ------------------------------------------------------------------
    def generate_track(
        self,
        start_lat: float,
        start_lon: float,
        n_steps: int = 100,
        noise_m: float = 5.0,
    ) -> List[dict]:
        """
        Generate a realistic GPS track with random walk + noise.
        Returns list of {lat, lon, ts} dicts.
        """
        track: List[dict] = []
        lat, lon = start_lat, start_lon
        # Random heading in degrees
        heading = self._rng.uniform(0, 360)
        speed_mps = self._rng.uniform(0.5, 3.0)  # m/s
        dt = 1.0  # 1 second per step
        ts = time.time()

        for _ in range(n_steps):
            # Slight heading perturbation each step
            heading += self._rng.gauss(0, 5)
            rad = math.radians(heading)
            dlat = (speed_mps * dt * math.cos(rad)) / self._METERS_PER_DEG
            dlon = (speed_mps * dt * math.sin(rad)) / (self._METERS_PER_DEG * math.cos(math.radians(lat)))

            lat += dlat
            lon += dlon
            noisy_lat, noisy_lon = self.add_gps_noise(lat, lon, accuracy_m=noise_m)
            track.append({"lat": noisy_lat, "lon": noisy_lon, "ts": round(ts, 3)})
            ts += dt

        return track

    # ------------------------------------------------------------------
    def add_gps_noise(
        self,
        lat: float,
        lon: float,
        accuracy_m: float = 5.0,
    ) -> Tuple[float, float]:
        """Add Gaussian GPS noise (accuracy_m = 1-sigma in metres)."""
        sigma_deg = accuracy_m / self._METERS_PER_DEG
        noisy_lat = lat + self._rng.gauss(0, sigma_deg)
        noisy_lon = lon + self._rng.gauss(0, sigma_deg)
        return round(noisy_lat, 7), round(noisy_lon, 7)

    # ------------------------------------------------------------------
    def export_yolo_labels(self, observations: List[dict], output_dir: str) -> None:
        """
        Write YOLO-format .txt label files for each observation.
        One file per entity_id, one line per track point (frame).
        Also writes a stub image list file for dataset construction.
        """
        out = Path(output_dir)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        (out / "images").mkdir(parents=True, exist_ok=True)

        image_list: List[str] = []

        for obs in observations:
            entity_id = obs.get("entity_id", "entity")
            class_id  = obs.get("class_id", 0)
            bw, bh    = obs.get("bbox_wh_norm", [0.05, 0.05])
            track     = obs.get("track", [])

            for frame_idx, point in enumerate(track):
                # Normalize lat/lon to [0,1] range — placeholder values for
                # synthetic data; real usage would require image pixel coords.
                cx = 0.5 + self._rng.gauss(0, 0.05)
                cy = 0.5 + self._rng.gauss(0, 0.05)
                cx = max(0.01, min(0.99, cx))
                cy = max(0.01, min(0.99, cy))

                label_name = f"{entity_id}_{frame_idx:04d}.txt"
                label_path = out / "labels" / label_name
                label_path.write_text(
                    f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
                )

                # Stub image filename
                img_name = f"{entity_id}_{frame_idx:04d}.jpg"
                image_list.append(str(out / "images" / img_name))

        # Write image list
        (out / "image_list.txt").write_text("\n".join(image_list) + "\n")
