"""
Port and Maritime Logistics Domain — Situation Assessment and Mission Planning

Covers: container terminals, bulk terminals, cruise ports, port approach
channels, berths, cranes, dangerous goods storage, fuel handling, AIS
vessel tracking.

Alert types:
  VESSEL_COLLISION_RISK     — converging vessels with CPA < 100m within 10 min
  BERTH_CAPACITY_EXCEEDED   — vessel tonnage above rated berth capacity
  DANGEROUS_CARGO_PROXIMITY — DG class 1/2/3 cargo near ignition source
  CRANE_OVERLOAD            — lift weight exceeds safe working load
  UNAUTHORIZED_VESSEL       — vessel without valid AIS in restricted zone
  PORT_SECURITY_BREACH      — person in sterile/restricted area
  FUEL_SPILL_DETECTED       — hydrocarbon sheen or sensor threshold exceeded
  CONTAINER_OVERHEAT        — reefer container temperature break

Mission types:
  VESSEL_INTERCEPT    — intercept course, ID vessel, relay to port authority
  SPILL_SURVEY        — downwind/downstream spill extent mapping
  SECURITY_PATROL     — perimeter orbit of terminal
  CRANE_OVERWATCH     — hover near crane during critical lifts
"""

from __future__ import annotations

import math
from typing import Any, Optional

# ── Thresholds ────────────────────────────────────────────────────────────────

_CPA_WARN_M            = 100    # closest point of approach warning threshold
_CPA_TIME_MIN          = 10     # CPA must occur within this many minutes
_DG_PROXIMITY_M        = 50     # dangerous goods to ignition source buffer
_HYDROCARBON_PPM_WARN  = 5      # surface hydrocarbon sensor threshold
_REEFER_TEMP_MAX_C     = 8.0    # cold-chain break threshold
_DG_FIRE_CLASSES       = {"1", "2", "3"}   # explosive, gas, flammable liquid


def assess_ports_situation(entities: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Assess operational and safety situation across a port or terminal.

    Args:
        entities: World model entities (vessels, cranes, containers, sensors,
                  persons, cargo lots)

    Returns:
        {
            "alerts": [...],
            "security_incidents": int,
            "navigation_hazards": int,
            "spill_detected": bool,
            "mission_recommended": bool,
        }
    """
    alerts: list[dict] = []
    security_incidents = 0
    navigation_hazards = 0
    spill_detected     = False

    vessels    = [e for e in entities if e.get("type", "").upper() in
                  ("VESSEL", "SHIP", "FERRY", "TUG", "BARGE", "TANKER")]
    cranes     = [e for e in entities if e.get("type", "").upper() in
                  ("CRANE", "STS_CRANE", "RTG_CRANE", "MOBILE_CRANE")]
    persons    = [e for e in entities if e.get("type", "").upper() in
                  ("PERSON", "PEDESTRIAN", "WORKER")]
    cargo      = [e for e in entities if e.get("type", "").upper() in
                  ("CONTAINER", "CARGO_LOT", "DANGEROUS_GOODS")]
    berths     = [e for e in entities if e.get("type", "").upper() == "BERTH"]
    ignition   = [e for e in entities if e.get("metadata", {}).get("ignition_source")]

    # ── Vessel collision risk ─────────────────────────────────────────────────
    for i, v1 in enumerate(vessels):
        for v2 in vessels[i + 1:]:
            cpa = _estimate_cpa_m(v1, v2)
            if cpa is not None and cpa < _CPA_WARN_M:
                navigation_hazards += 1
                sev = "critical" if cpa < 30 else "high"
                alerts.append({
                    "entity_id": v1.get("entity_id"),
                    "alert_type": "VESSEL_COLLISION_RISK",
                    "description": f"CPA {cpa:.0f}m between vessels within {_CPA_TIME_MIN} min",
                    "severity": sev,
                    "vessel_a": v1.get("entity_id"),
                    "vessel_b": v2.get("entity_id"),
                    "cpa_m": cpa,
                })

    # ── Berth capacity ────────────────────────────────────────────────────────
    for berth in berths:
        meta       = berth.get("metadata", {})
        swl_tonnes = float(meta.get("max_tonnage", 0))
        if swl_tonnes <= 0:
            continue
        for vessel in vessels:
            dist = _distance_m(vessel, berth)
            if dist is not None and dist < 50:   # vessel at berth
                gt = float(vessel.get("metadata", {}).get("gross_tonnage", 0))
                if gt > swl_tonnes:
                    alerts.append({
                        "entity_id": vessel.get("entity_id"),
                        "alert_type": "BERTH_CAPACITY_EXCEEDED",
                        "description": f"Vessel {gt:.0f}T exceeds berth rating {swl_tonnes:.0f}T",
                        "severity": "high",
                        "vessel_tonnage": gt,
                        "berth_max_tonnage": swl_tonnes,
                        "berth_id": berth.get("entity_id"),
                    })

    # ── Dangerous cargo proximity to ignition ─────────────────────────────────
    dg_cargo = [
        c for c in cargo
        if str(c.get("metadata", {}).get("dg_class", "")) in _DG_FIRE_CLASSES
    ]
    for dg in dg_cargo:
        for ign in ignition:
            dist = _distance_m(dg, ign)
            if dist is not None and dist < _DG_PROXIMITY_M:
                alerts.append({
                    "entity_id": dg.get("entity_id"),
                    "alert_type": "DANGEROUS_CARGO_PROXIMITY",
                    "description": f"DG class {dg.get('metadata', {}).get('dg_class')} cargo "
                                   f"{dist:.0f}m from ignition source",
                    "severity": "critical",
                    "distance_m": dist,
                    "dg_class": dg.get("metadata", {}).get("dg_class"),
                    "ignition_source_id": ign.get("entity_id"),
                })

    # ── Crane overload ────────────────────────────────────────────────────────
    for crane in cranes:
        meta      = crane.get("metadata", {})
        swl_t     = float(meta.get("swl_tonnes", 0))
        lift_t    = float(meta.get("lift_weight_tonnes", 0))
        if swl_t > 0 and lift_t > swl_t:
            alerts.append({
                "entity_id": crane.get("entity_id"),
                "alert_type": "CRANE_OVERLOAD",
                "description": f"Lift {lift_t:.1f}T exceeds SWL {swl_t:.1f}T",
                "severity": "critical",
                "lift_weight_tonnes": lift_t,
                "swl_tonnes": swl_t,
            })

    # ── Unauthorized vessel ───────────────────────────────────────────────────
    restricted_zones = [e for e in entities if e.get("metadata", {}).get("restricted_zone")]
    for vessel in vessels:
        if vessel.get("metadata", {}).get("ais_active", True):
            continue   # has valid AIS — authorised
        for zone in restricted_zones:
            dist = _distance_m(vessel, zone)
            radius = float(zone.get("metadata", {}).get("zone_radius_m", 200))
            if dist is not None and dist < radius:
                navigation_hazards += 1
                alerts.append({
                    "entity_id": vessel.get("entity_id"),
                    "alert_type": "UNAUTHORIZED_VESSEL",
                    "description": "Vessel without AIS/transponder in restricted zone",
                    "severity": "high",
                    "zone_id": zone.get("entity_id"),
                    "distance_m": dist,
                })

    # ── Port security breach ──────────────────────────────────────────────────
    sterile_zones = [e for e in entities if e.get("metadata", {}).get("sterile_zone")]
    for person in persons:
        if person.get("metadata", {}).get("authorized"):
            continue
        for zone in sterile_zones:
            dist = _distance_m(person, zone)
            radius = float(zone.get("metadata", {}).get("zone_radius_m", 100))
            if dist is not None and dist < radius:
                security_incidents += 1
                alerts.append({
                    "entity_id": person.get("entity_id"),
                    "alert_type": "PORT_SECURITY_BREACH",
                    "description": "Unauthorized person detected in sterile/restricted area",
                    "severity": "high",
                    "zone_id": zone.get("entity_id"),
                })

    # ── Fuel spill / hydrocarbon detection ───────────────────────────────────
    for e in entities:
        meta = e.get("metadata", {})
        hc_ppm = float(meta.get("hydrocarbon_ppm", 0))
        optical_sheen = meta.get("optical_sheen_detected", False)
        if hc_ppm > _HYDROCARBON_PPM_WARN or optical_sheen:
            spill_detected = True
            alerts.append({
                "entity_id": e.get("entity_id"),
                "alert_type": "FUEL_SPILL_DETECTED",
                "description": "Hydrocarbon sheen or sensor threshold exceeded — spill suspected",
                "severity": "high",
                "hydrocarbon_ppm": hc_ppm,
                "optical_sheen": optical_sheen,
            })

    # ── Reefer container cold-chain break ─────────────────────────────────────
    for c in cargo:
        meta   = c.get("metadata", {})
        temp_c = float(meta.get("temp_c", -999))
        if temp_c != -999 and temp_c > _REEFER_TEMP_MAX_C:
            alerts.append({
                "entity_id": c.get("entity_id"),
                "alert_type": "CONTAINER_OVERHEAT",
                "description": f"Reefer container temp {temp_c:.1f}°C exceeds {_REEFER_TEMP_MAX_C}°C limit",
                "severity": "medium",
                "temp_c": temp_c,
                "limit_c": _REEFER_TEMP_MAX_C,
            })

    mission_recommended = bool(alerts)

    return {
        "alerts":              alerts,
        "security_incidents":  security_incidents,
        "navigation_hazards":  navigation_hazards,
        "spill_detected":      spill_detected,
        "mission_recommended": mission_recommended,
    }


def plan_ports_mission(
    situation: dict[str, Any],
    area: Optional[dict] = None,
) -> dict[str, Any]:
    """Generate a UAV mission plan from a port situation assessment."""
    alert_types = {a["alert_type"] for a in situation.get("alerts", [])}

    if "VESSEL_COLLISION_RISK" in alert_types or "UNAUTHORIZED_VESSEL" in alert_types:
        return {
            "mission_type": "VESSEL_INTERCEPT",
            "pattern": "INTERCEPT_COURSE",
            "altitude_m": 60,
            "objectives": [
                "Intercept and visually ID vessel",
                "Capture registration markings / hull ID",
                "Relay position and description to port authority / VTS",
                "Maintain track until authority response",
            ],
            "sensor_config": {"optical": True, "thermal": False},
            "domain": "ports",
            "area": area,
        }

    if "FUEL_SPILL_DETECTED" in alert_types:
        return {
            "mission_type": "SPILL_SURVEY",
            "pattern": "DOWNWIND_TRANSECT",
            "altitude_m": 40,
            "objectives": [
                "Map spill extent from downwind/downstream position",
                "Optical and thermal survey of affected water surface",
                "Estimate spill area and drift direction",
                "Relay to port environmental response team",
            ],
            "sensor_config": {"optical": True, "thermal": True},
            "domain": "ports",
            "area": area,
        }

    if "PORT_SECURITY_BREACH" in alert_types:
        return {
            "mission_type": "SECURITY_PATROL",
            "pattern": "PERIMETER",
            "altitude_m": 50,
            "objectives": [
                "Orbit terminal perimeter — optical surveillance",
                "Locate and track unauthorized person",
                "Relay description and position to security response",
                "Maintain overwatch until security clearance",
            ],
            "sensor_config": {"optical": True, "thermal": True},
            "domain": "ports",
            "area": area,
        }

    if "CRANE_OVERLOAD" in alert_types:
        return {
            "mission_type": "CRANE_OVERWATCH",
            "pattern": "HOVER",
            "altitude_m": 30,
            "objectives": [
                "Position for visual overwatch of overloaded crane",
                "Monitor lift operation in real time",
                "Relay structural observations to crane operator and safety officer",
            ],
            "sensor_config": {"optical": True, "thermal": False},
            "domain": "ports",
            "area": area,
        }

    # Default: general security patrol
    return {
        "mission_type": "SECURITY_PATROL",
        "pattern": "PERIMETER",
        "altitude_m": 60,
        "objectives": [
            "Routine perimeter patrol of terminal",
            "Monitor vessel traffic and berth occupancy",
        ],
        "sensor_config": {"optical": True},
        "domain": "ports",
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


def _estimate_cpa_m(v1: dict, v2: dict) -> Optional[float]:
    """
    Estimate closest point of approach (CPA) in metres using linear dead
    reckoning over the next _CPA_TIME_MIN minutes.  Returns None if
    position or velocity data are missing.
    """
    p1  = v1.get("position") or {}
    p2  = v2.get("position") or {}
    m1  = v1.get("metadata", {})
    m2  = v2.get("metadata", {})

    if not (p1.get("lat") and p2.get("lat")):
        return None

    # Degrees-per-minute displacement for each vessel
    spd1_kts = float(m1.get("speed_kts", 0))
    hdg1_deg = float(m1.get("heading_deg", 0))
    spd2_kts = float(m2.get("speed_kts", 0))
    hdg2_deg = float(m2.get("heading_deg", 0))

    # Convert knots to degrees/min (1 knot ≈ 1/60 NM/min ≈ 1/60 * 1852/111_320 deg/min)
    _kts_to_deg_min = 1.852 / (111_320 * 60)

    def _project(lat, lon, spd_kts, hdg_deg, t_min):
        hdg_r  = math.radians(hdg_deg)
        dd     = spd_kts * _kts_to_deg_min * t_min
        return lat + dd * math.cos(hdg_r), lon + dd * math.sin(hdg_r) / max(math.cos(math.radians(lat)), 1e-9)

    min_dist = None
    for t in range(1, _CPA_TIME_MIN + 1):
        lat1f, lon1f = _project(p1["lat"], p1["lon"], spd1_kts, hdg1_deg, t)
        lat2f, lon2f = _project(p2["lat"], p2["lon"], spd2_kts, hdg2_deg, t)
        d = _distance_m({"position": {"lat": lat1f, "lon": lon1f}},
                        {"position": {"lat": lat2f, "lon": lon2f}})
        if d is not None and (min_dist is None or d < min_dist):
            min_dist = d
    return min_dist
