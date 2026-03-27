"""
KOFA Swarm Planner — multi-asset mission sector decomposition.

When multiple UAVs are available, KOFA calls SwarmPlanner to divide a single
mission area into N non-overlapping sectors and return one MissionPlan per
drone. Each plan is dispatched independently; Tasking assigns them to
separate assets.

Sector patterns by mission type:
  SEARCH    → lawnmower grid  (strips, each drone flies back-and-forth rows)
  SURVEY    → radial sectors  (pie slices from center outward)
  PERIMETER → perimeter arc   (each drone patrols a 360/N degree arc)
  MONITOR   → orbital ring    (drones equally spaced, orbit at fixed radius)
  default   → radial sectors

All math is in geodetic coordinates (WGS-84).
No external dependencies beyond the standard library.

Usage:
    from swarm_planner import SwarmPlanner
    from mission_planner import MissionPlan

    plans = SwarmPlanner().expand(base_plan, n_assets=4)
    # → list of 4 MissionPlan objects with sector waypoints attached
"""

import math
import os
import uuid
from dataclasses import dataclass, field
from typing import List, Tuple

# ── constants ──────────────────────────────────────────────────────────────────
EARTH_R_M = 6_371_000.0  # mean radius in metres
DEG_PER_M = 1.0 / 111_320.0  # approximate degrees latitude per metre at equator

# Min/max swarm size
MIN_SWARM = 2
MAX_SWARM = int(os.getenv("KOFA_MAX_SWARM_SIZE", "12"))

# Default coverage radius when the base plan has no explicit radius
DEFAULT_RADIUS_M = {
    "SEARCH": 800.0,
    "SURVEY": 600.0,
    "PERIMETER": 400.0,
    "MONITOR": 200.0,
    "ORBIT": 150.0,
}


# ── waypoint ──────────────────────────────────────────────────────────────────


@dataclass
class Waypoint:
    lat: float
    lon: float
    alt_m: float
    action: str = "WAYPOINT"  # WAYPOINT | LOITER | PHOTO | RTB


# ── sector ────────────────────────────────────────────────────────────────────


@dataclass
class Sector:
    """One drone's assigned area + ordered list of waypoints to fly."""

    sector_id: str
    drone_index: int
    waypoints: List[Waypoint]
    coverage_deg_start: float = 0.0  # for radial/perimeter sectors
    coverage_deg_end: float = 360.0
    swarm_id: str = ""


# ── geo helpers ────────────────────────────────────────────────────────────────


def _offset(
    lat: float, lon: float, bearing_deg: float, dist_m: float
) -> Tuple[float, float]:
    """Return (lat, lon) offset from a point by dist_m in bearing_deg (true north = 0)."""
    bearing = math.radians(bearing_deg)
    dlat = (dist_m * math.cos(bearing)) / EARTH_R_M
    dlon = (dist_m * math.sin(bearing)) / (EARTH_R_M * math.cos(math.radians(lat)))
    return lat + math.degrees(dlat), lon + math.degrees(dlon)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return EARTH_R_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── sector decomposition ──────────────────────────────────────────────────────


def _lawnmower_grid(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    n: int,
    alt_m: float,
    row_spacing_m: float = 50.0,
) -> List[Sector]:
    """
    Divide a circular search area into N vertical strips.
    Each drone flies a back-and-forth lawnmower pattern within its strip.

    Strip width = 2 * radius_m / n
    Rows spaced row_spacing_m apart (default 50 m for 60% camera overlap at 80 m AGL).
    """
    swarm_id = str(uuid.uuid4())[:8]
    strip_w = (2 * radius_m) / n
    sectors = []

    for i in range(n):
        # Strip x bounds relative to center (west→east)
        x_start = -radius_m + i * strip_w
        x_center = x_start + strip_w / 2

        # Convert x to lon offset
        lon_start = center_lon + (
            x_start / (EARTH_R_M * math.cos(math.radians(center_lat)))
        ) * (180 / math.pi)
        lon_center = center_lon + (
            x_center / (EARTH_R_M * math.cos(math.radians(center_lat)))
        ) * (180 / math.pi)

        # Rows within the strip (north→south, capped by the circle)
        rows = []
        y = radius_m
        row_idx = 0
        while y >= -radius_m:
            # Clip strip to circle: x² + y² ≤ R²
            max_x_at_y = math.sqrt(max(radius_m**2 - y**2, 0))
            if abs(x_start) <= max_x_at_y or abs(x_start + strip_w) <= max_x_at_y:
                row_lat = center_lat + (y / EARTH_R_M) * (180 / math.pi)
                # Alternate direction for lawnmower
                if row_idx % 2 == 0:
                    lon_a, lon_b = lon_start, lon_start + (
                        strip_w / (EARTH_R_M * math.cos(math.radians(center_lat)))
                    ) * (180 / math.pi)
                else:
                    lon_a, lon_b = (
                        lon_start
                        + (strip_w / (EARTH_R_M * math.cos(math.radians(center_lat))))
                        * (180 / math.pi),
                        lon_start,
                    )
                rows.append(Waypoint(lat=row_lat, lon=lon_a, alt_m=alt_m))
                rows.append(Waypoint(lat=row_lat, lon=lon_b, alt_m=alt_m))
            y -= row_spacing_m
            row_idx += 1

        if not rows:
            # Empty strip (outside circle) — put one waypoint at center of strip
            rows = [
                Waypoint(lat=center_lat, lon=lon_center, alt_m=alt_m, action="LOITER")
            ]

        sectors.append(
            Sector(
                sector_id=f"s{i+1}",
                drone_index=i,
                waypoints=rows,
                swarm_id=swarm_id,
            )
        )

    return sectors


def _radial_sectors(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    n: int,
    alt_m: float,
    spokes: int = 3,
) -> List[Sector]:
    """
    Divide a circular survey area into N pie slices.
    Each drone flies from center outward along evenly-spaced radial spokes
    within its angular sector.
    """
    swarm_id = str(uuid.uuid4())[:8]
    sector_deg = 360.0 / n
    sectors = []

    for i in range(n):
        start_deg = i * sector_deg
        mid_deg = start_deg + sector_deg / 2

        waypoints: List[Waypoint] = []

        # Start at center
        waypoints.append(Waypoint(lat=center_lat, lon=center_lon, alt_m=alt_m))

        # Fly along evenly-spaced spokes within the sector
        for s in range(spokes):
            angle = start_deg + (sector_deg / (spokes + 1)) * (s + 1)
            # Intermediate points along spoke
            for step in [0.4, 0.7, 1.0]:
                wp_lat, wp_lon = _offset(center_lat, center_lon, angle, radius_m * step)
                waypoints.append(Waypoint(lat=wp_lat, lon=wp_lon, alt_m=alt_m))
            # Back to center for next spoke
            waypoints.append(Waypoint(lat=center_lat, lon=center_lon, alt_m=alt_m))

        # End loiter at midpoint of sector outer edge
        end_lat, end_lon = _offset(center_lat, center_lon, mid_deg, radius_m * 0.8)
        waypoints.append(
            Waypoint(lat=end_lat, lon=end_lon, alt_m=alt_m, action="LOITER")
        )

        sectors.append(
            Sector(
                sector_id=f"s{i+1}",
                drone_index=i,
                waypoints=waypoints,
                coverage_deg_start=start_deg,
                coverage_deg_end=start_deg + sector_deg,
                swarm_id=swarm_id,
            )
        )

    return sectors


def _perimeter_arcs(
    center_lat: float,
    center_lon: float,
    radius_m: float,
    n: int,
    alt_m: float,
    patrol_points: int = 5,
) -> List[Sector]:
    """
    Divide the perimeter into N arcs.
    Each drone patrols back-and-forth along its arc at the given radius.
    Used for PERIMETER and MONITOR missions.
    """
    swarm_id = str(uuid.uuid4())[:8]
    arc_deg = 360.0 / n
    sectors = []

    for i in range(n):
        start_deg = i * arc_deg
        end_deg = start_deg + arc_deg

        waypoints: List[Waypoint] = []
        for p in range(patrol_points + 1):
            angle = start_deg + (arc_deg / patrol_points) * p
            wp_lat, wp_lon = _offset(center_lat, center_lon, angle, radius_m)
            action = "LOITER" if p == patrol_points // 2 else "WAYPOINT"
            waypoints.append(
                Waypoint(lat=wp_lat, lon=wp_lon, alt_m=alt_m, action=action)
            )

        # Return to start of arc for continuous patrol
        start_lat, start_lon = _offset(center_lat, center_lon, start_deg, radius_m)
        waypoints.append(Waypoint(lat=start_lat, lon=start_lon, alt_m=alt_m))

        sectors.append(
            Sector(
                sector_id=f"s{i+1}",
                drone_index=i,
                waypoints=waypoints,
                coverage_deg_start=start_deg,
                coverage_deg_end=end_deg,
                swarm_id=swarm_id,
            )
        )

    return sectors


# ── SwarmPlanner ───────────────────────────────────────────────────────────────


class SwarmPlanner:
    """
    Expands a single MissionPlan into N sector-specific MissionPlans for swarm dispatch.

    Selection logic:
      SEARCH    → lawnmower grid  (maximises coverage probability)
      SURVEY    → radial sectors  (each drone owns a pie slice)
      PERIMETER → perimeter arcs  (continuous boundary patrol)
      MONITOR   → perimeter arcs  (orbit at tighter radius)
      others    → radial sectors  (sensible default)
    """

    def expand(self, base_plan, n_assets: int, radius_m: float = 0.0) -> List:
        """
        Return a list of MissionPlan objects, one per sector.
        base_plan: MissionPlan from mission_planner.py
        n_assets:  number of available drones (clamped to MIN_SWARM..MAX_SWARM)
        radius_m:  coverage radius; 0 = use DEFAULT_RADIUS_M for mission type
        """
        n = max(MIN_SWARM, min(int(n_assets), MAX_SWARM))
        r = radius_m or DEFAULT_RADIUS_M.get(base_plan.mission_type, 500.0)

        if base_plan.mission_type == "SEARCH":
            sectors = _lawnmower_grid(
                base_plan.lat, base_plan.lon, r, n, base_plan.alt_m
            )
        elif base_plan.mission_type in ("PERIMETER", "MONITOR", "ORBIT"):
            sectors = _perimeter_arcs(
                base_plan.lat, base_plan.lon, r, n, base_plan.alt_m
            )
        else:
            sectors = _radial_sectors(
                base_plan.lat, base_plan.lon, r, n, base_plan.alt_m
            )

        # Import here to avoid circular import (mission_planner imports nothing from here)
        from mission_planner import MissionPlan

        plans = []
        for sec in sectors:
            if not sec.waypoints:
                continue
            primary = sec.waypoints[0]
            sector_plan = MissionPlan(
                mission_type=base_plan.mission_type,
                lat=primary.lat,
                lon=primary.lon,
                alt_m=base_plan.alt_m,
                priority=base_plan.priority,
                asset_class=base_plan.asset_class,
                loiter=base_plan.loiter,
                rationale=f"{base_plan.rationale} [swarm {sec.swarm_id} sector {sec.sector_id}/{n}]",
                raw_observation={
                    **base_plan.raw_observation,
                    "_swarm_id": sec.swarm_id,
                    "_sector_id": sec.sector_id,
                    "_drone_index": sec.drone_index,
                    "_n_sectors": n,
                    # Full waypoint list for tasking service
                    "_waypoints": [
                        {
                            "lat": wp.lat,
                            "lon": wp.lon,
                            "alt_m": wp.alt_m,
                            "action": wp.action,
                        }
                        for wp in sec.waypoints
                    ],
                },
            )
            plans.append(sector_plan)

        return plans

    def should_swarm(self, mission_type: str, n_available: int) -> bool:
        """Return True if this mission type benefits from multi-drone coverage."""
        _SWARMABLE = {"SEARCH", "SURVEY", "PERIMETER", "MONITOR"}
        return mission_type in _SWARMABLE and n_available >= MIN_SWARM


# Module-level singleton
_swarm_planner: SwarmPlanner | None = None


def get_swarm_planner() -> SwarmPlanner:
    global _swarm_planner
    if _swarm_planner is None:
        _swarm_planner = SwarmPlanner()
    return _swarm_planner
