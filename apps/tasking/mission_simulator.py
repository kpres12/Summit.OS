"""
apps/tasking/mission_simulator.py — Pre-execution mission dry run.

Simulates a planned mission forward in time using physics models
(battery drain, speed, distance) to predict outcomes before any
real hardware is touched.

CANVAS TA1: "virtualizing the C2 layer — simulate execution of dynamic
workflows under operationally relevant conditions before pushing changes."

The simulator:
  - Models per-asset flight/movement time based on waypoint distances
  - Tracks battery drain (configurable rate, default 1% per 60s flight)
  - Detects: battery depletion mid-mission, time-to-complete per asset,
    asset conflicts (two assets assigned same waypoint), and coverage gaps
  - Returns a SimulationReport with per-asset predictions and an overall
    mission success probability

Usage:
    sim = MissionSimulator()
    report = sim.simulate(
        assignments_map=assignments_map,
        assets=available_assets,
        area=req.area,
    )
    # report.success_probability: 0.0–1.0
    # report.asset_reports: per-asset outcome prediction
    # report.warnings: list of issues found
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("tasking.simulator")

# Physics constants (tunable via constructor)
DEFAULT_BATTERY_DRAIN_PCT_PER_MIN = 1.0   # 100% battery = 100 min flight
DEFAULT_GROUND_DRAIN_PCT_PER_MIN  = 0.4   # ground robots drain slower
MIN_SAFE_BATTERY_PCT              = 15.0  # reserve — land if below this


@dataclass
class AssetSimState:
    asset_id:       str
    start_battery:  float           # % at mission start
    battery_now:    float           # % after simulated mission
    domain:         str             # aerial | ground | camera | ...
    position:       Dict[str, float]  # final simulated position
    elapsed_secs:   float           # total mission time
    distance_m:     float           # total distance covered
    waypoints_done: int
    waypoints_total: int
    depleted:       bool            # ran out of battery mid-mission
    depletion_at_wp: Optional[int]  # which waypoint battery ran out
    status:         str             # "OK" | "BATTERY_CRITICAL" | "DEPLETED" | "UNREACHABLE"


@dataclass
class SimulationReport:
    mission_viable:     bool                    # True if mission can proceed as planned
    success_probability: float                  # 0.0–1.0
    estimated_duration_secs: float             # time until last asset completes
    asset_reports:      List[AssetSimState]
    warnings:           List[str]
    recommendations:    List[str]
    coverage_pct:       Optional[float] = None  # estimated area coverage %


class MissionSimulator:
    """
    Stateless pre-execution mission simulator.

    Runs entirely in memory — no DB, no MQTT, no hardware.
    Safe to call before committing a mission.
    """

    def __init__(
        self,
        battery_drain_aerial_pct_per_min: float = DEFAULT_BATTERY_DRAIN_PCT_PER_MIN,
        battery_drain_ground_pct_per_min: float = DEFAULT_GROUND_DRAIN_PCT_PER_MIN,
        min_safe_battery_pct:             float = MIN_SAFE_BATTERY_PCT,
    ) -> None:
        self._drain_aerial = battery_drain_aerial_pct_per_min
        self._drain_ground = battery_drain_ground_pct_per_min
        self._min_safe     = min_safe_battery_pct

    def simulate(
        self,
        assignments_map: Dict[str, Any],
        assets: List[Dict[str, Any]],
        area: Optional[Dict[str, Any]] = None,
    ) -> SimulationReport:
        """
        Simulate the planned mission and return a prediction report.

        Args:
            assignments_map: asset_id → plan dict (from _plan_assignments or role decomposer)
            assets:          raw asset dicts from DB (for battery/type info)
            area:            mission area dict (for coverage calculation)
        """
        asset_lookup: Dict[str, Dict] = {a["asset_id"]: a for a in assets}
        warnings: List[str] = []
        recommendations: List[str] = []
        asset_reports: List[AssetSimState] = []

        for asset_id, plan in assignments_map.items():
            asset = asset_lookup.get(asset_id, {})
            report = self._simulate_asset(asset_id, asset, plan)
            asset_reports.append(report)

            if report.depleted:
                warnings.append(
                    f"{asset_id}: battery depleted at waypoint "
                    f"{report.depletion_at_wp}/{report.waypoints_total} "
                    f"(starts at {report.start_battery:.0f}%)"
                )
                recommendations.append(
                    f"Reduce waypoints for {asset_id} or increase starting battery above "
                    f"{self._estimate_required_battery(plan):.0f}%"
                )
            elif report.status == "BATTERY_CRITICAL":
                warnings.append(
                    f"{asset_id}: will land with only {report.battery_now:.1f}% battery "
                    f"(below {self._min_safe}% reserve)"
                )

        # Detect duplicate waypoint assignments (two assets at same point)
        wp_owners: Dict[Tuple, str] = {}
        for asset_id, plan in assignments_map.items():
            for wp in plan.get("waypoints", []):
                key = (round(wp.get("lat", 0), 4), round(wp.get("lon", 0), 4))
                if key in wp_owners and wp_owners[key] != asset_id:
                    warnings.append(
                        f"Potential conflict: {asset_id} and {wp_owners[key]} "
                        f"assigned to same waypoint ({key[0]}, {key[1]})"
                    )
                else:
                    wp_owners[key] = asset_id

        # Overall metrics
        depleted_count  = sum(1 for r in asset_reports if r.depleted)
        critical_count  = sum(1 for r in asset_reports if r.status == "BATTERY_CRITICAL")
        total           = len(asset_reports)
        max_duration    = max((r.elapsed_secs for r in asset_reports), default=0.0)

        # Success probability: 1.0 minus penalty for each problem
        p = 1.0
        if total > 0:
            p -= (depleted_count / total) * 0.6   # depletion = severe
            p -= (critical_count / total) * 0.2   # critical = moderate
            p -= (len(warnings) / max(1, total * 3)) * 0.1  # misc warnings
            p = max(0.0, min(1.0, p))

        viable = depleted_count == 0 and p >= 0.5

        if not viable:
            recommendations.append(
                "Mission has critical issues. Review battery levels and reduce waypoint count."
            )
        elif warnings:
            recommendations.append(
                "Mission viable but has warnings. Consider adjusting asset assignments."
            )
        else:
            recommendations.append("Mission looks good. All assets have sufficient battery.")

        # Coverage estimate
        coverage = self._estimate_coverage(assignments_map, area) if area else None

        logger.info(
            "Simulation complete: viable=%s p=%.2f duration=%.0fs assets=%d warnings=%d",
            viable, p, max_duration, total, len(warnings),
        )

        return SimulationReport(
            mission_viable          = viable,
            success_probability     = round(p, 3),
            estimated_duration_secs = round(max_duration, 1),
            asset_reports           = asset_reports,
            warnings                = warnings,
            recommendations         = recommendations,
            coverage_pct            = coverage,
        )

    # ── Per-asset simulation ──────────────────────────────────────────────────

    def _simulate_asset(
        self,
        asset_id: str,
        asset: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> AssetSimState:
        domain       = plan.get("domain", "aerial")
        waypoints    = plan.get("waypoints", [])
        speed_mps    = float(plan.get("speed", 5.0))
        start_batt   = float(asset.get("battery") or 100.0)
        battery      = start_batt
        drain_rate   = self._drain_aerial if domain == "aerial" else self._drain_ground

        elapsed_secs  = float(plan.get("start_delay_sec", 0.0))
        distance_m    = 0.0
        depleted      = False
        depletion_wp  = None
        start_pos     = {"lat": 0.0, "lon": 0.0}

        # Infer start position from first waypoint or asset's last known position
        if waypoints:
            start_pos = {"lat": waypoints[0].get("lat", 0), "lon": waypoints[0].get("lon", 0)}

        prev_lat = start_pos["lat"]
        prev_lon = start_pos["lon"]
        wp_done  = 0

        for wp_idx, wp in enumerate(waypoints):
            lat = float(wp.get("lat", prev_lat))
            lon = float(wp.get("lon", prev_lon))
            spd = float(wp.get("speed", speed_mps)) or speed_mps

            seg_m   = self._haversine_m(prev_lat, prev_lon, lat, lon)
            seg_s   = seg_m / max(0.1, spd)
            drain   = (seg_s / 60.0) * drain_rate

            if battery - drain < 0:
                # Would run out before reaching this waypoint
                depleted     = True
                depletion_wp = wp_idx
                # Partial drain up to 0
                elapsed_secs += (battery / drain_rate) * 60.0
                distance_m   += (battery / drain_rate) * 60.0 * spd
                battery       = 0.0
                break

            battery      -= drain
            elapsed_secs += seg_s
            distance_m   += seg_m
            prev_lat, prev_lon = lat, lon
            wp_done += 1

        if domain == "aerial" and not depleted:
            # Add RTB time: fly back to start
            rtb_m = self._haversine_m(prev_lat, prev_lon, start_pos["lat"], start_pos["lon"])
            rtb_s = rtb_m / max(0.1, speed_mps)
            rtb_drain = (rtb_s / 60.0) * drain_rate
            if battery - rtb_drain < 0 and not depleted:
                depleted     = True
                depletion_wp = len(waypoints)  # fails on RTB
                elapsed_secs += (battery / drain_rate) * 60.0
                battery       = 0.0
            else:
                battery      -= rtb_drain
                elapsed_secs += rtb_s
                distance_m   += rtb_m

        if depleted:
            status = "DEPLETED"
        elif battery < self._min_safe:
            status = "BATTERY_CRITICAL"
        else:
            status = "OK"

        return AssetSimState(
            asset_id        = asset_id,
            start_battery   = start_batt,
            battery_now     = round(battery, 1),
            domain          = domain,
            position        = {"lat": prev_lat, "lon": prev_lon},
            elapsed_secs    = round(elapsed_secs, 1),
            distance_m      = round(distance_m, 1),
            waypoints_done  = wp_done,
            waypoints_total = len(waypoints),
            depleted        = depleted,
            depletion_at_wp = depletion_wp,
            status          = status,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6_371_000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi  = math.radians(lat2 - lat1)
        dlam  = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return R * 2 * math.asin(min(1.0, math.sqrt(a)))

    def _estimate_required_battery(self, plan: Dict[str, Any]) -> float:
        """Estimate minimum starting battery needed to complete the plan."""
        waypoints  = plan.get("waypoints", [])
        speed_mps  = float(plan.get("speed", 5.0))
        drain_rate = self._drain_aerial
        total_dist = 0.0
        prev_lat = waypoints[0].get("lat", 0) if waypoints else 0
        prev_lon = waypoints[0].get("lon", 0) if waypoints else 0

        for wp in waypoints:
            lat, lon = float(wp.get("lat", prev_lat)), float(wp.get("lon", prev_lon))
            total_dist += self._haversine_m(prev_lat, prev_lon, lat, lon)
            prev_lat, prev_lon = lat, lon

        total_secs = total_dist / max(0.1, speed_mps)
        required   = (total_secs / 60.0) * drain_rate + self._min_safe
        return min(100.0, required)

    def _estimate_coverage(
        self,
        assignments_map: Dict[str, Any],
        area: Dict[str, Any],
    ) -> Optional[float]:
        """
        Rough estimate of area coverage based on total distance flown
        vs. area size. Not geometrically exact — used for reporting only.
        """
        try:
            radius_m   = float(area.get("radius_m", 500.0))
            area_m2    = math.pi * radius_m ** 2
            swath_m    = 40.0   # assume 40m sensor swath width
            total_dist = sum(
                sum(
                    self._haversine_m(
                        wp.get("lat", 0), wp.get("lon", 0),
                        waypoints[i + 1].get("lat", 0), waypoints[i + 1].get("lon", 0),
                    )
                    for i, wp in enumerate(waypoints[:-1])
                )
                for plan in assignments_map.values()
                for waypoints in [plan.get("waypoints", [])]
                if len(waypoints) >= 2
            )
            covered_m2 = total_dist * swath_m
            return round(min(100.0, (covered_m2 / max(1.0, area_m2)) * 100.0), 1)
        except Exception:
            return None
