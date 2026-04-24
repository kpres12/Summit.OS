"""
Urban Search and Rescue (USAR) Domain — Situation Assessment and Mission Planning

Covers: building collapse, structural failure, USAR operations, void space
identification, survivor location, secondary collapse risk, hazmat in
structures, access route clearance.

Alert types:
  BUILDING_COLLAPSE_DETECTED  — structure confidence < 0.3 or collapsed flag
  VOID_SPACE_IDENTIFIED       — thermal anomaly suggesting survivable void
  SURVIVOR_SIGNAL_DETECTED    — motion or acoustic hit in rubble
  SECONDARY_COLLAPSE_RISK     — structural instability index > 0.7
  HAZMAT_IN_STRUCTURE         — collapsed structure has hazmat_present flag
  ACCESS_ROUTE_BLOCKED        — debris field blocking rescue vehicle path
  DUST_CLOUD_ACTIVE           — recent collapse, visibility < 20m
  FIRE_IN_RUBBLE              — thermal hot spot in collapse zone

Mission types:
  COLLAPSE_ASSESSMENT  — systematic grid at 30m, photogrammetry + thermal
  VOID_DETECTION       — slow pass at 15m, downward-looking thermal
  SURVIVOR_RELAY       — loiter over survivor signal, relay coordinates
  ROUTE_SURVEY         — map access corridors through debris field
  HAZMAT_STANDOFF      — upwind 100m standoff, map hazmat extent

Building type vulnerability: URM (unreinforced masonry) > Wood frame > RC
"""

from __future__ import annotations

import math
from typing import Any, Optional

# ── Thresholds ────────────────────────────────────────────────────────────────

_COLLAPSE_CONF_THRESH  = 0.3    # building confidence below this = collapse alert
_INSTABILITY_THRESH    = 0.7    # structural instability index above this = secondary risk
_DUST_VISIBILITY_M     = 20     # dust cloud visibility threshold
_THERMAL_VOID_DELTA_C  = 3.0    # thermal anomaly delta vs surroundings (°C) for void


def assess_urban_sar_situation(entities: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Assess a USAR incident scene from world model entities.

    Args:
        entities: Structures, survivor signals, sensors, access routes,
                  environment sensors, hazmat sources.

    Returns:
        {
            "alerts": [...],
            "collapse_score_max": float,
            "survivor_signals": int,
            "hazmat_structures": int,
            "mission_recommended": bool,
        }
    """
    alerts: list[dict] = []
    survivor_signals   = 0
    hazmat_structures  = 0
    collapse_score_max = 0.0

    structures = [e for e in entities if e.get("type", "").upper() in
                  ("BUILDING", "STRUCTURE", "COLLAPSED_STRUCTURE")]
    routes     = [e for e in entities if e.get("type", "").upper() in
                  ("ROAD", "ACCESS_ROUTE", "CORRIDOR")]

    # ── Building collapse assessment ──────────────────────────────────────────
    for s in structures:
        meta       = s.get("metadata", {})
        confidence = float(meta.get("confidence", 1.0))
        collapsed  = meta.get("collapsed", False)
        score      = _collapse_score(s)
        collapse_score_max = max(collapse_score_max, score)

        if collapsed or confidence < _COLLAPSE_CONF_THRESH:
            sev = "critical" if score > 0.7 or collapsed else "high"
            alerts.append({
                "entity_id": s.get("entity_id"),
                "alert_type": "BUILDING_COLLAPSE_DETECTED",
                "description": f"Structure collapsed or confidence {confidence:.2f} — collapse score {score:.2f}",
                "severity": sev,
                "confidence": confidence,
                "collapse_score": score,
                "building_type": meta.get("building_type", "unknown"),
            })

            # Hazmat in collapsed structure
            if meta.get("hazmat_present"):
                hazmat_structures += 1
                alerts.append({
                    "entity_id": s.get("entity_id"),
                    "alert_type": "HAZMAT_IN_STRUCTURE",
                    "description": "Collapsed structure contains hazmat — standoff required",
                    "severity": "critical",
                    "hazmat_type": meta.get("hazmat_type", "unknown"),
                })

        # Secondary collapse risk
        instability = float(meta.get("instability_index", 0))
        if instability > _INSTABILITY_THRESH:
            alerts.append({
                "entity_id": s.get("entity_id"),
                "alert_type": "SECONDARY_COLLAPSE_RISK",
                "description": f"Structural instability index {instability:.2f} — secondary collapse risk",
                "severity": "high",
                "instability_index": instability,
            })

    # ── Void space identification (thermal) ───────────────────────────────────
    for e in entities:
        meta = e.get("metadata", {})
        thermal_delta = float(meta.get("thermal_delta_c", 0))
        if meta.get("void_indicator") or thermal_delta > _THERMAL_VOID_DELTA_C:
            alerts.append({
                "entity_id": e.get("entity_id"),
                "alert_type": "VOID_SPACE_IDENTIFIED",
                "description": f"Thermal anomaly (+{thermal_delta:.1f}°C) — possible survivable void",
                "severity": "high",
                "thermal_delta_c": thermal_delta,
            })

    # ── Survivor signals ──────────────────────────────────────────────────────
    for e in entities:
        meta = e.get("metadata", {})
        if meta.get("motion_ping") or meta.get("acoustic_hit") or meta.get("survivor_signal"):
            survivor_signals += 1
            alerts.append({
                "entity_id": e.get("entity_id"),
                "alert_type": "SURVIVOR_SIGNAL_DETECTED",
                "description": "Motion/acoustic survivor signal in rubble",
                "severity": "critical",
                "signal_type": (
                    "motion" if meta.get("motion_ping")
                    else "acoustic" if meta.get("acoustic_hit")
                    else "generic"
                ),
            })

    # ── Access route blockage ─────────────────────────────────────────────────
    for route in routes:
        meta = route.get("metadata", {})
        if meta.get("blocked") or meta.get("debris_field"):
            alerts.append({
                "entity_id": route.get("entity_id"),
                "alert_type": "ACCESS_ROUTE_BLOCKED",
                "description": "Debris field blocking rescue vehicle access",
                "severity": "high",
                "route_id": route.get("entity_id"),
            })

    # ── Environmental sensors ─────────────────────────────────────────────────
    for e in entities:
        etype = e.get("type", "").upper()
        meta  = e.get("metadata", {})

        # Dust cloud
        visibility_m = float(meta.get("visibility_m", 9999))
        if meta.get("dust_cloud_active") or visibility_m < _DUST_VISIBILITY_M:
            alerts.append({
                "entity_id": e.get("entity_id"),
                "alert_type": "DUST_CLOUD_ACTIVE",
                "description": f"Dust cloud active — visibility {visibility_m:.0f}m",
                "severity": "medium",
                "visibility_m": visibility_m,
            })

        # Fire in rubble (thermal hotspot in collapse zone)
        if etype in ("THERMAL_SENSOR", "HOTSPOT") or meta.get("hotspot_detected"):
            temp_c = float(meta.get("temp_c", 0))
            if temp_c > 200 or meta.get("hotspot_detected"):
                alerts.append({
                    "entity_id": e.get("entity_id"),
                    "alert_type": "FIRE_IN_RUBBLE",
                    "description": f"Thermal hotspot {temp_c:.0f}°C in collapse zone — fire in rubble",
                    "severity": "critical",
                    "temp_c": temp_c,
                })

    mission_recommended = bool(alerts)

    return {
        "alerts":              alerts,
        "collapse_score_max":  collapse_score_max,
        "survivor_signals":    survivor_signals,
        "hazmat_structures":   hazmat_structures,
        "mission_recommended": mission_recommended,
    }


def plan_urban_sar_mission(
    situation: dict[str, Any],
    area: Optional[dict] = None,
) -> dict[str, Any]:
    """Generate a UAV mission plan from a USAR situation assessment."""
    alert_types = {a["alert_type"] for a in situation.get("alerts", [])}

    if "SURVIVOR_SIGNAL_DETECTED" in alert_types:
        return {
            "mission_type": "SURVIVOR_RELAY",
            "pattern": "LOITER",
            "altitude_m": 20,
            "objectives": [
                "Loiter directly over survivor signal location",
                "Relay precise coordinates to ground USAR team",
                "Provide downward-facing thermal/optical feed",
                "Maintain position until team arrival",
            ],
            "sensor_config": {"optical": True, "thermal": True},
            "domain": "urban_sar",
            "area": area,
        }

    if "HAZMAT_IN_STRUCTURE" in alert_types:
        return {
            "mission_type": "HAZMAT_STANDOFF",
            "pattern": "UPWIND_ORBIT",
            "altitude_m": 60,
            "standoff_m": 100,
            "objectives": [
                "Maintain 100m upwind standoff from hazmat structure",
                "Map hazmat extent and potential plume direction",
                "Relay to USAR incident commander for exclusion zone",
            ],
            "sensor_config": {"optical": True, "thermal": True},
            "domain": "urban_sar",
            "area": area,
        }

    if "FIRE_IN_RUBBLE" in alert_types or "SECONDARY_COLLAPSE_RISK" in alert_types:
        return {
            "mission_type": "COLLAPSE_ASSESSMENT",
            "pattern": "GRID",
            "altitude_m": 30,
            "objectives": [
                "Systematic grid survey at 30m altitude",
                "Photogrammetry pass — identify collapse extent",
                "Thermal pass — locate hot spots and void signatures",
                "Generate 3D model of collapse zone for incident commander",
            ],
            "sensor_config": {"optical": True, "thermal": True, "photogrammetry": True},
            "domain": "urban_sar",
            "area": area,
        }

    if "VOID_SPACE_IDENTIFIED" in alert_types:
        return {
            "mission_type": "VOID_DETECTION",
            "pattern": "SLOW_PASS",
            "altitude_m": 15,
            "objectives": [
                "Slow low-altitude pass over identified void area",
                "Downward-looking thermal for survivable void confirmation",
                "Acoustic relay if equipped",
                "Mark void coordinates for ground team",
            ],
            "sensor_config": {"thermal": True, "optical": True, "acoustic": True},
            "domain": "urban_sar",
            "area": area,
        }

    if "ACCESS_ROUTE_BLOCKED" in alert_types:
        return {
            "mission_type": "ROUTE_SURVEY",
            "pattern": "LINEAR",
            "altitude_m": 25,
            "objectives": [
                "Survey access corridors through debris field",
                "Map passable routes for rescue vehicles",
                "Identify debris requiring clearance",
                "Relay route status to incident command",
            ],
            "sensor_config": {"optical": True},
            "domain": "urban_sar",
            "area": area,
        }

    # Default: initial collapse assessment
    return {
        "mission_type": "COLLAPSE_ASSESSMENT",
        "pattern": "GRID",
        "altitude_m": 30,
        "objectives": [
            "Initial systematic assessment of collapse zone",
            "Photogrammetry and thermal survey",
            "Establish USAR operational picture",
        ],
        "sensor_config": {"optical": True, "thermal": True},
        "domain": "urban_sar",
        "area": area,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _collapse_score(entity: dict) -> float:
    """
    Score building collapse vulnerability 0–1.
    Considers building_type, floors_pancaked, and metadata confidence.
    URM is most vulnerable, then wood frame, then RC.
    """
    meta  = entity.get("metadata", {})
    btype = str(meta.get("building_type", "")).upper()

    # Base vulnerability by structural type
    type_score = {"URM": 0.9, "UNREINFORCED_MASONRY": 0.9,
                  "WOOD": 0.6, "WOOD_FRAME": 0.6,
                  "RC": 0.3, "REINFORCED_CONCRETE": 0.3,
                  "STEEL": 0.25}.get(btype, 0.5)

    floors_pancaked = int(meta.get("floors_pancaked", 0))
    total_floors    = max(int(meta.get("floors", 1)), 1)
    floor_factor    = min(floors_pancaked / total_floors, 1.0)

    collapsed   = 1.0 if meta.get("collapsed") else 0.0
    confidence  = float(meta.get("confidence", 1.0))
    conf_factor = 1.0 - confidence   # low confidence → higher score

    score = max(collapsed, (type_score * 0.4) + (floor_factor * 0.4) + (conf_factor * 0.2))
    return min(score, 1.0)


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
