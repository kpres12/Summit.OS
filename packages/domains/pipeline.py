"""
Pipeline Monitoring Domain — Situation Assessment and Mission Planning

Covers: oil, gas, water, and slurry pipelines — right-of-way patrol,
leak detection, third-party intrusion, geohazard crossings, cathodic
protection, pressure anomalies, excavation encroachment, corrosion.

Alert types:
  LEAK_DETECTED              — pressure drop or sensor trigger
  THIRD_PARTY_INTRUSION      — unauthorized entity within pipeline RoW
  GEOHAZARD_CROSSING         — active landslide / subsidence at pipeline
  CATHODIC_PROTECTION_FAILURE — CP current below threshold
  PRESSURE_ANOMALY           — pressure deviating > 2 sigma from baseline
  EXCAVATION_NEAR_PIPELINE   — digging activity within 50m
  CORROSION_WALL_LOSS        — ILI result above wall-loss threshold
  VALVE_FAILURE              — isolation valve offline on critical segment

Mission types:
  LEAK_SURVEY       — rapid transit, optical + thermal + methane sensor
  INTRUSION_MONITOR — orbit intrusion point, relay to control room
  GEOHAZARD_SURVEY  — photogrammetry of slope, displacement measurement
  ROW_PATROL        — systematic patrol of right-of-way corridor

Thresholds:
  Pressure drop: >5% in 10 min = warning, >15% = critical
  RoW buffer:    10m each side of centreline
  CP threshold:  <-850 mV CSE
  Wall loss:     >20% = alert
"""

from __future__ import annotations

import math
from typing import Any, Optional

# ── Thresholds ────────────────────────────────────────────────────────────────

_PRESSURE_DROP_WARN_PCT  = 5.0
_PRESSURE_DROP_CRIT_PCT  = 15.0
_ROW_BUFFER_M            = 10       # right-of-way half-width
_EXCAVATION_BUFFER_M     = 50       # exclusion zone for digging
_CP_THRESHOLD_MV         = -850     # mV vs CSE — below this = failure
_WALL_LOSS_WARN_PCT      = 20
_SIGMA_THRESHOLD         = 2.0      # standard deviations for pressure anomaly


def assess_pipeline_situation(entities: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Assess pipeline integrity and right-of-way situation.

    Args:
        entities: Pipeline segments, sensors, valves, terrain features,
                  third-party entities (people, vehicles, machinery).

    Returns:
        {
            "alerts": [...],
            "leak_risk_segments": int,
            "intrusion_count": int,
            "critical_valves_offline": int,
            "mission_recommended": bool,
        }
    """
    alerts: list[dict] = []
    leak_risk_segments     = 0
    intrusion_count        = 0
    critical_valves_offline = 0

    pipeline_segments = [e for e in entities if e.get("type", "").upper() in
                         ("PIPELINE_SEGMENT", "PIPELINE", "PIPE")]
    valves            = [e for e in entities if e.get("type", "").upper() in
                         ("VALVE", "ISOLATION_VALVE", "BLOCK_VALVE")]
    sensors           = [e for e in entities if e.get("type", "").upper() in
                         ("PRESSURE_SENSOR", "FLOW_SENSOR", "LEAK_DETECTOR",
                          "GAS_DETECTOR", "CP_MONITOR")]
    geohazard_zones   = [e for e in entities if e.get("metadata", {}).get("active_geohazard")]
    excavators        = [e for e in entities if e.get("type", "").upper() in
                         ("EXCAVATOR", "BACKHOE", "DIGGER", "TRENCHER")]

    # ── Leak detection (sensor-based) ─────────────────────────────────────────
    for sensor in sensors:
        meta = sensor.get("metadata", {})

        # Direct leak sensor trigger
        if meta.get("leak_sensor"):
            leak_risk_segments += 1
            alerts.append({
                "entity_id": sensor.get("entity_id"),
                "alert_type": "LEAK_DETECTED",
                "description": "Leak sensor triggered — pipeline breach suspected",
                "severity": "critical",
                "sensor_id": sensor.get("entity_id"),
            })

        # Pressure drop calculation
        baseline_bar  = float(meta.get("baseline_pressure_bar", 0))
        current_bar   = float(meta.get("pressure_bar", 0))
        if baseline_bar > 0:
            drop_pct = (baseline_bar - current_bar) / baseline_bar * 100
            if drop_pct > _PRESSURE_DROP_WARN_PCT:
                leak_risk_segments += 1
                sev = "critical" if drop_pct > _PRESSURE_DROP_CRIT_PCT else "high"
                alerts.append({
                    "entity_id": sensor.get("entity_id"),
                    "alert_type": "LEAK_DETECTED",
                    "description": f"Pressure drop {drop_pct:.1f}% over 10 min",
                    "severity": sev,
                    "pressure_drop_pct": drop_pct,
                    "baseline_bar": baseline_bar,
                    "current_bar": current_bar,
                })

        # Pressure anomaly (statistical deviation)
        stddev_bar = float(meta.get("pressure_stddev_bar", 0))
        if baseline_bar > 0 and stddev_bar > 0:
            deviation = abs(current_bar - baseline_bar) / stddev_bar
            if deviation > _SIGMA_THRESHOLD:
                alerts.append({
                    "entity_id": sensor.get("entity_id"),
                    "alert_type": "PRESSURE_ANOMALY",
                    "description": f"Pressure {deviation:.1f}σ from baseline",
                    "severity": "high" if deviation > 3 else "medium",
                    "sigma": deviation,
                    "current_bar": current_bar,
                    "baseline_bar": baseline_bar,
                })

        # Cathodic protection failure
        cp_mv = meta.get("cp_mv")
        if cp_mv is not None and float(cp_mv) > _CP_THRESHOLD_MV:
            # More positive than threshold means under-protection
            alerts.append({
                "entity_id": sensor.get("entity_id"),
                "alert_type": "CATHODIC_PROTECTION_FAILURE",
                "description": f"CP reading {cp_mv} mV — inadequate cathodic protection",
                "severity": "high",
                "cp_mv": float(cp_mv),
                "threshold_mv": _CP_THRESHOLD_MV,
            })

    # ── Inline inspection (ILI) corrosion results ─────────────────────────────
    for seg in pipeline_segments:
        meta       = seg.get("metadata", {})
        wall_loss  = float(meta.get("wall_loss_pct", 0))
        if wall_loss > _WALL_LOSS_WARN_PCT:
            alerts.append({
                "entity_id": seg.get("entity_id"),
                "alert_type": "CORROSION_WALL_LOSS",
                "description": f"ILI wall loss {wall_loss:.1f}% — integrity risk",
                "severity": "critical" if wall_loss > 40 else "high",
                "wall_loss_pct": wall_loss,
            })

    # ── Valve failure ─────────────────────────────────────────────────────────
    for valve in valves:
        meta = valve.get("metadata", {})
        if not meta.get("operational", True) and meta.get("critical_isolation"):
            critical_valves_offline += 1
            alerts.append({
                "entity_id": valve.get("entity_id"),
                "alert_type": "VALVE_FAILURE",
                "description": "Critical isolation valve offline — cannot isolate segment",
                "severity": "critical",
                "valve_id": valve.get("entity_id"),
            })

    # ── Third-party intrusion in RoW ──────────────────────────────────────────
    all_centrelines = pipeline_segments  # segments act as RoW centreline references
    third_party = [
        e for e in entities
        if e.get("type", "").upper() in ("PERSON", "VEHICLE", "TRUCK")
        and not e.get("metadata", {}).get("authorized_row_access")
    ]
    for tp in third_party:
        for seg in all_centrelines:
            dist = _distance_m(tp, seg)
            if dist is not None and dist < _ROW_BUFFER_M:
                intrusion_count += 1
                alerts.append({
                    "entity_id": tp.get("entity_id"),
                    "alert_type": "THIRD_PARTY_INTRUSION",
                    "description": f"Unauthorized entity {dist:.0f}m from pipeline centreline (RoW = {_ROW_BUFFER_M}m)",
                    "severity": "high",
                    "distance_m": dist,
                    "segment_id": seg.get("entity_id"),
                })
                break  # one alert per entity

    # ── Excavation near pipeline ──────────────────────────────────────────────
    for exc in excavators:
        for seg in pipeline_segments:
            dist = _distance_m(exc, seg)
            if dist is not None and dist < _EXCAVATION_BUFFER_M:
                alerts.append({
                    "entity_id": exc.get("entity_id"),
                    "alert_type": "EXCAVATION_NEAR_PIPELINE",
                    "description": f"Digging activity {dist:.0f}m from pipeline — damage risk",
                    "severity": "critical" if dist < _ROW_BUFFER_M else "high",
                    "distance_m": dist,
                    "segment_id": seg.get("entity_id"),
                })
                break

    # ── Geohazard crossing ────────────────────────────────────────────────────
    for geo in geohazard_zones:
        for seg in pipeline_segments:
            dist = _distance_m(geo, seg)
            if dist is not None and dist < 100:
                hazard_type = geo.get("metadata", {}).get("hazard_type", "unknown")
                alerts.append({
                    "entity_id": seg.get("entity_id"),
                    "alert_type": "GEOHAZARD_CROSSING",
                    "description": f"Active {hazard_type} geohazard at pipeline crossing",
                    "severity": "high",
                    "hazard_id": geo.get("entity_id"),
                    "hazard_type": hazard_type,
                })

    mission_recommended = bool(alerts)

    return {
        "alerts":                   alerts,
        "leak_risk_segments":       leak_risk_segments,
        "intrusion_count":          intrusion_count,
        "critical_valves_offline":  critical_valves_offline,
        "mission_recommended":      mission_recommended,
    }


def plan_pipeline_mission(
    situation: dict[str, Any],
    area: Optional[dict] = None,
) -> dict[str, Any]:
    """Generate a UAV mission plan from a pipeline situation assessment."""
    alert_types = {a["alert_type"] for a in situation.get("alerts", [])}

    if "LEAK_DETECTED" in alert_types:
        return {
            "mission_type": "LEAK_SURVEY",
            "pattern": "LINEAR_TRANSIT",
            "altitude_m": 30,
            "speed_mps": 10,
            "objectives": [
                "Rapid transit along suspect pipeline segment",
                "Optical, thermal, and methane sensor pass",
                "Locate and confirm breach point",
                "Relay precise coordinates to control room for isolation",
            ],
            "sensor_config": {"optical": True, "thermal": True, "methane": True},
            "domain": "pipeline",
            "area": area,
        }

    if "EXCAVATION_NEAR_PIPELINE" in alert_types or "THIRD_PARTY_INTRUSION" in alert_types:
        return {
            "mission_type": "INTRUSION_MONITOR",
            "pattern": "ORBIT",
            "altitude_m": 50,
            "objectives": [
                "Orbit intrusion or excavation location",
                "Capture and relay imagery to control room and permit office",
                "Monitor for pipeline contact or damage",
                "Maintain overwatch until activity ceases or permit confirmed",
            ],
            "sensor_config": {"optical": True, "thermal": False},
            "domain": "pipeline",
            "area": area,
        }

    if "GEOHAZARD_CROSSING" in alert_types:
        return {
            "mission_type": "GEOHAZARD_SURVEY",
            "pattern": "GRID",
            "altitude_m": 60,
            "objectives": [
                "Photogrammetry of slope or subsidence area at pipeline crossing",
                "Generate DEM and measure displacement vectors",
                "Compare to baseline for movement quantification",
                "Relay to integrity management team",
            ],
            "sensor_config": {"optical": True, "photogrammetry": True, "lidar": True},
            "domain": "pipeline",
            "area": area,
        }

    # Default: routine right-of-way patrol
    return {
        "mission_type": "ROW_PATROL",
        "pattern": "LINEAR_TRANSIT",
        "altitude_m": 40,
        "objectives": [
            "Systematic patrol of right-of-way corridor",
            "Inspect for encroachment, vegetation growth, erosion",
            "Verify marker posts and valve sites",
            "Relay anomalies to pipeline control room",
        ],
        "sensor_config": {"optical": True, "thermal": True},
        "domain": "pipeline",
        "area": area,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _distance_m(a: dict, b: dict) -> Optional[float]:
    pa = a.get("position") or {}
    pb = b.get("position") or {}
    if not (pa.get("lat") and pb.get("lat")):
        return None
    lat1, lon1 = math.radians(pa["lat"]), math.radians(pa["lon"])
    lat2, lon2 = math.radians(pb["lat"]), math.radians(pb["lon"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a_ = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * math.asin(math.sqrt(a_)) * 6_371_000
