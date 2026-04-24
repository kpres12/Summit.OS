"""
Mining Domain — Situation Assessment and Mission Planning

Covers: open-pit mines, underground mines, quarries, tailings storage
facilities (TSF), haul roads, blast zones, gas monitoring.

Alert types:
  SLOPE_FAILURE_RISK      — pit wall displacement exceeding safe rate
  SUBSIDENCE_DETECTED     — ground deformation in production zone
  TAILINGS_BREACH_RISK    — TSF near capacity or seepage detected
  BLAST_ZONE_INTRUSION    — entity inside blast radius during scheduled blast
  UNDERGROUND_EMERGENCY   — underground entity in emergency state
  TOXIC_GAS_DETECTED      — CO or H2S above threshold
  EQUIPMENT_BREAKDOWN     — critical-path heavy equipment offline
  HAUL_ROAD_BLOCKED       — haul road obstruction

Mission types:
  SLOPE_MONITOR       — persistent orbit of pit wall, photogrammetry
  TAILINGS_SURVEY     — perimeter survey of TSF, thermal + optical
  BLAST_CLEARANCE     — sweep blast zone for personnel before firing
  UNDERGROUND_RESCUE  — surface relay, locate beacon, guide rescue team
  GAS_PLUME_TRACK     — upwind standoff, map gas dispersal

Thresholds:
  Displacement: >50 mm/day = warning, >100 mm/day = critical
  CO:  >25 ppm = warning, >200 ppm = critical
  H2S: >10 ppm = warning, >50 ppm = critical
  TSF capacity: >90% = warning
"""

from __future__ import annotations

import math
from typing import Any, Optional

# ── Thresholds ────────────────────────────────────────────────────────────────

_DISP_WARNING_MM_DAY  = 50
_DISP_CRITICAL_MM_DAY = 100
_CO_WARNING_PPM       = 25
_CO_CRITICAL_PPM      = 200
_H2S_WARNING_PPM      = 10
_H2S_CRITICAL_PPM     = 50
_TSF_CAPACITY_WARN    = 0.90   # fraction


def assess_mining_situation(entities: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Assess hazard and operational situation across a mine site.

    Args:
        entities: World model entities (pit walls, sensors, equipment, personnel, roads)

    Returns:
        {
            "alerts": [...],
            "active_emergencies": int,
            "gas_hazards": int,
            "slope_warnings": int,
            "mission_recommended": bool,
        }
    """
    alerts: list[dict] = []
    active_emergencies = 0
    gas_hazards        = 0
    slope_warnings     = 0

    for e in entities:
        etype = e.get("type", "").upper()
        meta  = e.get("metadata", e.get("properties", {}))

        # ── Slope / pit wall stability ────────────────────────────────────────
        if etype in ("PIT_WALL", "SLOPE_MONITOR", "GEOSENSOR", "PRISM"):
            disp = float(meta.get("displacement_mm_day", 0))
            accel = meta.get("accelerating", False)
            if disp > _DISP_WARNING_MM_DAY or accel:
                sev = "critical" if disp > _DISP_CRITICAL_MM_DAY or accel else "high"
                slope_warnings += 1
                alerts.append({
                    "entity_id": e.get("entity_id"),
                    "alert_type": "SLOPE_FAILURE_RISK",
                    "description": f"Pit wall displacement {disp:.0f} mm/day"
                                   + (" — accelerating" if accel else ""),
                    "severity": sev,
                    "displacement_mm_day": disp,
                    "accelerating": accel,
                })

        # ── Ground subsidence ─────────────────────────────────────────────────
        elif etype in ("SUBSIDENCE_SENSOR", "INSAR_POINT", "SETTLEMENT_GAUGE"):
            deform_mm = float(meta.get("deformation_mm", 0))
            zone      = meta.get("zone", "unknown")
            if deform_mm > 20:
                alerts.append({
                    "entity_id": e.get("entity_id"),
                    "alert_type": "SUBSIDENCE_DETECTED",
                    "description": f"Ground deformation {deform_mm:.0f}mm in {zone} zone",
                    "severity": "high" if deform_mm > 50 else "medium",
                    "deformation_mm": deform_mm,
                    "zone": zone,
                })

        # ── Tailings Storage Facility ─────────────────────────────────────────
        elif etype in ("TSF", "TAILINGS_POND", "TAILINGS_DAM"):
            level_frac = float(meta.get("capacity_fraction", 0))
            seepage    = meta.get("seepage_detected", False)
            if level_frac > _TSF_CAPACITY_WARN or seepage:
                sev = "critical" if seepage or level_frac > 0.97 else "high"
                alerts.append({
                    "entity_id": e.get("entity_id"),
                    "alert_type": "TAILINGS_BREACH_RISK",
                    "description": f"TSF at {level_frac*100:.0f}% capacity"
                                   + (" — seepage detected" if seepage else ""),
                    "severity": sev,
                    "capacity_fraction": level_frac,
                    "seepage": seepage,
                })

        # ── Blast zone intrusion ──────────────────────────────────────────────
        elif etype in ("BLAST_ZONE", "EXCLUSION_ZONE"):
            if meta.get("blast_scheduled"):
                blast_r = float(meta.get("blast_radius_m", 500))
                # Check all other entities for proximity
                for other in entities:
                    if other.get("entity_id") == e.get("entity_id"):
                        continue
                    dist = _distance_m(other, e)
                    if dist is not None and dist < blast_r:
                        alerts.append({
                            "entity_id": other.get("entity_id"),
                            "alert_type": "BLAST_ZONE_INTRUSION",
                            "description": f"Entity within blast radius ({dist:.0f}m < {blast_r:.0f}m) — blast scheduled",
                            "severity": "critical",
                            "distance_m": dist,
                            "blast_radius_m": blast_r,
                            "zone_id": e.get("entity_id"),
                        })

        # ── Underground emergency ─────────────────────────────────────────────
        elif meta.get("underground"):
            if meta.get("emergency"):
                active_emergencies += 1
                alerts.append({
                    "entity_id": e.get("entity_id"),
                    "alert_type": "UNDERGROUND_EMERGENCY",
                    "description": "Underground entity in emergency state — rescue required",
                    "severity": "critical",
                    "level": meta.get("level", "unknown"),
                })

        # ── Gas sensors ───────────────────────────────────────────────────────
        if etype in ("GAS_SENSOR", "AIR_QUALITY_MONITOR") or meta.get("gas_ppm") is not None or meta.get("h2s_ppm") is not None:
            co_ppm  = float(meta.get("gas_ppm", 0))
            h2s_ppm = float(meta.get("h2s_ppm", 0))
            if co_ppm > _CO_WARNING_PPM:
                gas_hazards += 1
                sev = "critical" if co_ppm > _CO_CRITICAL_PPM else "high"
                alerts.append({
                    "entity_id": e.get("entity_id"),
                    "alert_type": "TOXIC_GAS_DETECTED",
                    "description": f"CO reading {co_ppm:.0f} ppm",
                    "severity": sev,
                    "gas_type": "CO",
                    "ppm": co_ppm,
                })
            if h2s_ppm > _H2S_WARNING_PPM:
                gas_hazards += 1
                sev = "critical" if h2s_ppm > _H2S_CRITICAL_PPM else "high"
                alerts.append({
                    "entity_id": e.get("entity_id"),
                    "alert_type": "TOXIC_GAS_DETECTED",
                    "description": f"H2S reading {h2s_ppm:.1f} ppm",
                    "severity": sev,
                    "gas_type": "H2S",
                    "ppm": h2s_ppm,
                })

        # ── Heavy equipment breakdown ─────────────────────────────────────────
        if etype in ("HAUL_TRUCK", "EXCAVATOR", "SHOVEL", "DOZER", "DRILL") and not meta.get("operational", True):
            if meta.get("critical_path"):
                alerts.append({
                    "entity_id": e.get("entity_id"),
                    "alert_type": "EQUIPMENT_BREAKDOWN",
                    "description": f"Critical-path {etype} offline — production impact",
                    "severity": "high",
                    "equipment_type": etype,
                })

        # ── Haul road blockage ────────────────────────────────────────────────
        if etype in ("HAUL_ROAD", "ACCESS_ROAD") and meta.get("blocked"):
            alerts.append({
                "entity_id": e.get("entity_id"),
                "alert_type": "HAUL_ROAD_BLOCKED",
                "description": "Haul road blocked — logistics disruption",
                "severity": "medium",
                "road_id": e.get("entity_id"),
            })

    mission_recommended = (
        active_emergencies > 0
        or gas_hazards > 0
        or slope_warnings > 0
        or any(a["alert_type"] == "BLAST_ZONE_INTRUSION" for a in alerts)
    )

    return {
        "alerts":              alerts,
        "active_emergencies":  active_emergencies,
        "gas_hazards":         gas_hazards,
        "slope_warnings":      slope_warnings,
        "mission_recommended": mission_recommended,
    }


def plan_mining_mission(
    situation: dict[str, Any],
    area: Optional[dict] = None,
) -> dict[str, Any]:
    """Generate a UAV mission plan from a mine site situation assessment."""
    alert_types = {a["alert_type"] for a in situation.get("alerts", [])}

    if "UNDERGROUND_EMERGENCY" in alert_types:
        return {
            "mission_type": "UNDERGROUND_RESCUE",
            "pattern": "DIRECT",
            "altitude_m": 50,
            "objectives": [
                "Establish surface-to-underground comms relay",
                "Locate personal emergency device (PED) beacon",
                "Guide rescue team to entry point",
                "Relay status to mine control room",
            ],
            "sensor_config": {"optical": True, "thermal": True, "comms_relay": True},
            "domain": "mining",
            "area": area,
        }

    if "BLAST_ZONE_INTRUSION" in alert_types:
        return {
            "mission_type": "BLAST_CLEARANCE",
            "pattern": "EXPANDING_SQUARE",
            "altitude_m": 40,
            "objectives": [
                "Sweep blast exclusion zone for personnel and equipment",
                "Confirm zone clear before firing authorization",
                "Relay clear/not-clear status to blast controller",
            ],
            "sensor_config": {"optical": True, "thermal": True},
            "domain": "mining",
            "area": area,
        }

    if "TOXIC_GAS_DETECTED" in alert_types:
        return {
            "mission_type": "GAS_PLUME_TRACK",
            "pattern": "UPWIND_TRANSECT",
            "altitude_m": 30,
            "objectives": [
                "Approach from upwind — maintain safe standoff",
                "Map gas plume dispersal extent",
                "Relay concentration readings to safety officer",
                "Track wind-driven migration",
            ],
            "sensor_config": {"optical": True, "gas_sensor": True},
            "domain": "mining",
            "area": area,
        }

    if "TAILINGS_BREACH_RISK" in alert_types:
        return {
            "mission_type": "TAILINGS_SURVEY",
            "pattern": "PERIMETER",
            "altitude_m": 60,
            "objectives": [
                "Perimeter survey of tailings storage facility",
                "Thermal scan for seepage hot spots",
                "Optical inspection of dam face and spillway",
                "Measure freeboard and surface extent",
            ],
            "sensor_config": {"optical": True, "thermal": True},
            "domain": "mining",
            "area": area,
        }

    if "SLOPE_FAILURE_RISK" in alert_types or "SUBSIDENCE_DETECTED" in alert_types:
        return {
            "mission_type": "SLOPE_MONITOR",
            "pattern": "PERSISTENT_ORBIT",
            "altitude_m": 80,
            "objectives": [
                "Persistent orbit of pit wall or subsidence zone",
                "Photogrammetry pass — generate 3D deformation model",
                "Compare with baseline DEM",
                "Relay change vectors to geotechnical team",
            ],
            "sensor_config": {"optical": True, "lidar": True, "photogrammetry": True},
            "domain": "mining",
            "area": area,
        }

    # Default: general site survey
    return {
        "mission_type": "SLOPE_MONITOR",
        "pattern": "LAWNMOWER",
        "altitude_m": 100,
        "objectives": [
            "Routine photogrammetry of pit walls and TSF",
            "Check haul roads and active working areas",
        ],
        "sensor_config": {"optical": True},
        "domain": "mining",
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
