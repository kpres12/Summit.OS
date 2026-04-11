"""
Deconfliction Volume — Summit.OS

Represents an asset's occupied airspace as a cylinder:
  - center: lat/lon
  - radius_m: horizontal buffer (default 50m)
  - alt_floor_m / alt_ceil_m: vertical extent
  - ts: timestamp (volumes become stale after TTL)

Also represents planned flight paths as sequences of cylinder volumes
(one per waypoint segment).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class CylinderVolume:
    """3D airspace volume modelled as a vertical cylinder."""

    entity_id: str
    lat: float
    lon: float
    radius_m: float = 50.0
    alt_floor_m: float = 0.0
    alt_ceil_m: float = 120.0
    ts: float = field(default_factory=time.time)
    ttl_s: float = 30.0
    priority: int = 1

    def is_stale(self, now: float = None) -> bool:
        """Return True if this volume has expired."""
        return (now or time.time()) - self.ts > self.ttl_s


# ── Overlap predicates ────────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def horizontal_overlap(a: CylinderVolume, b: CylinderVolume) -> bool:
    """True if the horizontal circles of two cylinders intersect."""
    dist_m = _haversine_m(a.lat, a.lon, b.lat, b.lon)
    return dist_m < (a.radius_m + b.radius_m)


def vertical_overlap(a: CylinderVolume, b: CylinderVolume) -> bool:
    """True if the altitude bands of two cylinders overlap."""
    return not (a.alt_ceil_m < b.alt_floor_m or b.alt_ceil_m < a.alt_floor_m)


def overlaps(a: CylinderVolume, b: CylinderVolume) -> bool:
    """True if two cylinder volumes overlap in both horizontal and vertical extent."""
    return horizontal_overlap(a, b) and vertical_overlap(a, b)


# ── Path to volume sequence ────────────────────────────────────────────────────

def _interpolate_waypoints(
    wp_a: dict, wp_b: dict, step_m: float = 50.0
) -> List[dict]:
    """Interpolate intermediate points between two waypoints at ~step_m intervals."""
    lat1 = float(wp_a.get("lat", 0))
    lon1 = float(wp_a.get("lon", 0))
    lat2 = float(wp_b.get("lat", 0))
    lon2 = float(wp_b.get("lon", 0))
    alt1 = float(wp_a.get("alt_m", wp_a.get("altitude_m", 50.0)))
    alt2 = float(wp_b.get("alt_m", wp_b.get("altitude_m", 50.0)))

    dist_m = _haversine_m(lat1, lon1, lat2, lon2)
    if dist_m <= step_m:
        return []

    n_steps = max(1, int(dist_m / step_m))
    points = []
    for i in range(1, n_steps):
        t = i / n_steps
        interp_lat = lat1 + t * (lat2 - lat1)
        interp_lon = lon1 + t * (lon2 - lon1)
        interp_alt = alt1 + t * (alt2 - alt1)
        points.append({"lat": interp_lat, "lon": interp_lon, "alt_m": interp_alt})
    return points


def path_to_volumes(
    waypoints: List[dict],
    entity_id: str,
    radius_m: float = 50.0,
) -> List[CylinderVolume]:
    """Convert a list of waypoints into a sequence of CylinderVolume objects.

    Creates one volume per waypoint plus interpolated volumes between consecutive
    waypoints so that the full flight path is covered.
    """
    if not waypoints:
        return []

    now = time.time()
    volumes: List[CylinderVolume] = []

    for i, wp in enumerate(waypoints):
        lat = float(wp.get("lat", 0))
        lon = float(wp.get("lon", 0))
        alt_m = float(wp.get("alt_m", wp.get("altitude_m", 50.0)))
        half_buffer = radius_m
        volumes.append(
            CylinderVolume(
                entity_id=entity_id,
                lat=lat,
                lon=lon,
                radius_m=radius_m,
                alt_floor_m=max(0.0, alt_m - half_buffer),
                alt_ceil_m=alt_m + half_buffer,
                ts=now,
            )
        )
        # Interpolate between this waypoint and the next
        if i + 1 < len(waypoints):
            interp_points = _interpolate_waypoints(wp, waypoints[i + 1])
            for pt in interp_points:
                alt_i = float(pt.get("alt_m", alt_m))
                volumes.append(
                    CylinderVolume(
                        entity_id=entity_id,
                        lat=float(pt["lat"]),
                        lon=float(pt["lon"]),
                        radius_m=radius_m,
                        alt_floor_m=max(0.0, alt_i - half_buffer),
                        alt_ceil_m=alt_i + half_buffer,
                        ts=now,
                    )
                )

    return volumes
