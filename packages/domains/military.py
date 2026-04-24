"""
Military Domain Logic

Covers: Humanitarian Assistance / Disaster Response (HADR), Agile Combat
Employment (ACE), force protection perimeter, CASEVAC escort coordination,
base/FOB perimeter monitoring, logistics convoy tracking, counter-UAS,
Battle Damage Assessment, CBRN proximity, pattern of life, personnel
recovery (CSAR), communications relay, logistics/resupply, and IED/mine
area marking.

IMPORTANT: Heli.OS is a coordination and situational awareness platform.
This module handles tracking, tasking, and reporting ONLY — absolutely no
weapons targeting, fire control, targeting solutions, or lethal engagement
logic of any kind. Military use = coordination, ISR, HADR, CASEVAC, and
force protection tracking only.

Primary standards: CoT/ATAK (MIL-STD-2525B type codes), SALUTE reports,
9-line MEDEVAC, SITREP.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

# ── Threat assessment thresholds ──────────────────────────────────────────────

_PERIMETER_BREACH_M   = 500     # unknown entity within 500m of FOB = alert
_CASEVAC_MAX_WAIT_MIN = 60      # CASEVAC response window (Golden Hour)
_CONVOY_GAP_M         = 500     # inter-vehicle gap that triggers lost contact
_COUNTER_UAS_RANGE_M  = 2_000   # rogue UAS detection radius around protected asset
_CBRN_STANDOFF_M      = 500     # proximity threshold for CBRN hazard alert
_COMMS_BLACKOUT_MIN   = 30      # minutes without contact before alert
_POL_CONCENTRATION_R  = 200     # radius (m) for unusual concentration check
_POL_CONCENTRATION_N  = 10      # entity count threshold for unusual concentration


def assess_military_situation(entities: list[dict]) -> dict:
    """
    Assess the tactical situation for military/government operations.

    Args:
        entities: World model entities (ATAK tracks, mavlink drones, AIS, etc.)

    Returns:
        {
            "threat_level": "routine" | "elevated" | "high" | "imminent",
            "alerts": [...],
            "friendly_count": int,
            "unknown_count": int,
            "casevac_pending": bool,
            "recommendations": [str],
        }
    """
    alerts = []
    friendly  = [e for e in entities if _is_friendly(e)]
    unknown   = [e for e in entities if _is_unknown(e)]
    fob_locs  = [e for e in entities if e.get("metadata", {}).get("is_fob")]
    convoys   = [e for e in entities if e.get("metadata", {}).get("convoy_id")]

    # ── Perimeter breach ──────────────────────────────────────────────────────
    for unk in unknown:
        for fob in fob_locs:
            dist_m = _distance_m(unk, fob)
            if dist_m is not None and dist_m < _PERIMETER_BREACH_M:
                alerts.append({
                    "entity_id": unk.get("entity_id"),
                    "alert_type": "PERIMETER_BREACH",
                    "description": f"Unknown entity {dist_m:.0f}m from FOB",
                    "severity": "critical",
                    "distance_m": dist_m,
                })

    # ── CASEVAC pending ───────────────────────────────────────────────────────
    casevac_pending = any(
        e.get("metadata", {}).get("casevac_requested") for e in entities
    )
    if casevac_pending:
        alerts.append({
            "entity_id": "casevac",
            "alert_type": "CASEVAC_REQUESTED",
            "description": "CASEVAC/MEDEVAC request active — escort required",
            "severity": "critical",
        })

    # ── Convoy gap (lost vehicle) ─────────────────────────────────────────────
    convoy_ids = {e.get("metadata", {}).get("convoy_id") for e in convoys}
    for cid in convoy_ids:
        members = [e for e in convoys if e.get("metadata", {}).get("convoy_id") == cid]
        if _has_gap(members):
            alerts.append({
                "entity_id": f"convoy-{cid}",
                "alert_type": "CONVOY_GAP",
                "description": f"Convoy {cid} has inter-vehicle gap >{_CONVOY_GAP_M}m",
                "severity": "high",
            })

    # ── ACE dispersal monitoring ──────────────────────────────────────────────
    ace_assets = [e for e in friendly if e.get("metadata", {}).get("ace_role")]
    ace_alert  = _check_ace_clustering(ace_assets)
    if ace_alert:
        alerts.append(ace_alert)

    # ── Counter-UAS ───────────────────────────────────────────────────────────
    protected_assets = [e for e in entities if e.get("metadata", {}).get("protected_asset")]
    small_uas = [
        e for e in unknown
        if e.get("metadata", {}).get("uas_class") in ("small", "micro", "nano")
        or e.get("metadata", {}).get("mass_kg", 99) < 1
    ]
    if len(small_uas) >= 3:
        alerts.append({
            "entity_id": "counter-uas-swarm",
            "alert_type": "COUNTER_UAS_SWARM",
            "description": f"{len(small_uas)} unknown small UAS detected simultaneously — swarm indicator",
            "severity": "critical",
            "uas_count": len(small_uas),
        })
    else:
        for uas in small_uas:
            for asset in protected_assets:
                dist_m = _distance_m(uas, asset)
                if dist_m is not None and dist_m < _COUNTER_UAS_RANGE_M:
                    alerts.append({
                        "entity_id": uas.get("entity_id"),
                        "alert_type": "COUNTER_UAS_DETECTED",
                        "description": f"Rogue small UAS {dist_m:.0f}m from protected asset",
                        "severity": "high",
                        "distance_m": dist_m,
                        "asset_id": asset.get("entity_id"),
                    })

    # ── Battle Damage Assessment ──────────────────────────────────────────────
    for e in entities:
        meta = e.get("metadata", {})
        if meta.get("post_strike"):
            alerts.append({
                "entity_id": e.get("entity_id"),
                "alert_type": "BDA_REQUIRED",
                "description": "Post-strike entity flagged — BDA survey required",
                "severity": "high",
            })
        conf = float(meta.get("building_confidence", 1.0))
        if e.get("type", "").upper() in ("BUILDING", "STRUCTURE") and conf < 0.2:
            alerts.append({
                "entity_id": e.get("entity_id"),
                "alert_type": "STRUCTURE_DESTROYED",
                "description": f"Building confidence {conf:.2f} — probable destruction",
                "severity": "high",
                "confidence": conf,
            })

    # ── CBRN proximity ────────────────────────────────────────────────────────
    cbrn_sources = [e for e in entities if e.get("metadata", {}).get("cbrn_type")]
    for src in cbrn_sources:
        cbrn_type = src.get("metadata", {}).get("cbrn_type", "unknown")
        wind_dir  = float(src.get("metadata", {}).get("wind_dir_deg", 0))
        for other in entities:
            if other.get("entity_id") == src.get("entity_id"):
                continue
            dist_m = _distance_m(other, src)
            if dist_m is not None and dist_m < _CBRN_STANDOFF_M:
                alerts.append({
                    "entity_id": other.get("entity_id"),
                    "alert_type": "CBRN_HAZARD_PROXIMITY",
                    "description": f"Entity within {dist_m:.0f}m of {cbrn_type} CBRN source",
                    "severity": "critical",
                    "distance_m": dist_m,
                    "cbrn_type": cbrn_type,
                })
        # Plume track alert — flag source itself for monitoring
        alerts.append({
            "entity_id": src.get("entity_id"),
            "alert_type": "CBRN_PLUME_TRACK",
            "description": f"CBRN plume tracking required — wind {wind_dir:.0f}°, standoff 300m upwind",
            "severity": "high",
            "wind_dir_deg": wind_dir,
        })

    # ── Pattern of Life / Anomaly ─────────────────────────────────────────────
    current_hour = datetime.now(timezone.utc).hour
    for e in entities:
        meta = e.get("metadata", {})
        baseline_speed = float(meta.get("baseline_speed_mps", 0))
        current_speed  = float(meta.get("speed_mps", 0))
        baseline_dwell = float(meta.get("baseline_dwell_min", 0))
        current_dwell  = float(meta.get("dwell_min", 0))

        if baseline_speed > 0 and current_speed > 3 * baseline_speed:
            alerts.append({
                "entity_id": e.get("entity_id"),
                "alert_type": "PATTERN_OF_LIFE_ANOMALY",
                "description": f"Speed {current_speed:.1f} m/s is {current_speed/baseline_speed:.1f}x above baseline",
                "severity": "medium",
                "current_speed_mps": current_speed,
                "baseline_speed_mps": baseline_speed,
            })
        elif baseline_dwell > 0 and current_dwell > 2 * baseline_dwell:
            alerts.append({
                "entity_id": e.get("entity_id"),
                "alert_type": "PATTERN_OF_LIFE_ANOMALY",
                "description": f"Dwell {current_dwell:.0f} min is {current_dwell/baseline_dwell:.1f}x above baseline",
                "severity": "medium",
                "current_dwell_min": current_dwell,
                "baseline_dwell_min": baseline_dwell,
            })

    # Unusual concentration during off-hours (0000–0500 local UTC)
    if 0 <= current_hour <= 5:
        for anchor in entities:
            nearby = [
                e for e in entities
                if e.get("entity_id") != anchor.get("entity_id")
                and _distance_m(e, anchor) is not None
                and _distance_m(e, anchor) < _POL_CONCENTRATION_R
            ]
            if len(nearby) > _POL_CONCENTRATION_N:
                alerts.append({
                    "entity_id": anchor.get("entity_id"),
                    "alert_type": "UNUSUAL_CONCENTRATION",
                    "description": f"{len(nearby)} entities within {_POL_CONCENTRATION_R}m during off-hours",
                    "severity": "high",
                    "entity_count": len(nearby),
                    "hour_utc": current_hour,
                })
                break  # one alert per assessment pass

    # ── Personnel Recovery (CSAR) ─────────────────────────────────────────────
    for e in entities:
        meta = e.get("metadata", {})
        if meta.get("emergency_beacon"):
            alerts.append({
                "entity_id": e.get("entity_id"),
                "alert_type": "PR_BEACON_ACTIVE",
                "description": "Emergency beacon active — initiate personnel recovery",
                "severity": "critical",
            })
        elif meta.get("isolated") and meta.get("pr_signal"):
            alerts.append({
                "entity_id": e.get("entity_id"),
                "alert_type": "ISOLATED_PERSONNEL",
                "description": "Isolated personnel with PR signal — CSAR required",
                "severity": "critical",
            })

    # ── Communications relay ──────────────────────────────────────────────────
    for e in friendly:
        last_contact = float(e.get("metadata", {}).get("last_contact_min", 0))
        if last_contact > _COMMS_BLACKOUT_MIN:
            alerts.append({
                "entity_id": e.get("entity_id"),
                "alert_type": "COMMS_BLACKOUT",
                "description": f"Friendly element dark for {last_contact:.0f} min — relay required",
                "severity": "high",
                "last_contact_min": last_contact,
            })

    # ── Logistics / Resupply ──────────────────────────────────────────────────
    for e in friendly:
        meta = e.get("metadata", {})
        fuel_pct = float(meta.get("fuel_pct", 100))
        if fuel_pct < 10 or meta.get("ammo_critical"):
            alerts.append({
                "entity_id": e.get("entity_id"),
                "alert_type": "RESUPPLY_CRITICAL",
                "description": f"Critical resupply needed — fuel {fuel_pct:.0f}%"
                               + (" ammo critical" if meta.get("ammo_critical") else ""),
                "severity": "high",
                "fuel_pct": fuel_pct,
            })

    # ── Mine / IED area marking ───────────────────────────────────────────────
    for e in entities:
        meta = e.get("metadata", {})
        if meta.get("ied_indicator") or meta.get("ground_sign_activity"):
            alerts.append({
                "entity_id": e.get("entity_id"),
                "alert_type": "MINEFIELD_SUSPECTED",
                "description": "IED indicator or ground sign activity — standoff survey required",
                "severity": "critical",
            })

    threat = _score_threat(alerts)
    recommendations = _military_recommendations(threat, casevac_pending, len(unknown))

    return {
        "threat_level":    threat,
        "alerts":          alerts,
        "friendly_count":  len(friendly),
        "unknown_count":   len(unknown),
        "casevac_pending": casevac_pending,
        "recommendations": recommendations,
    }


def plan_military_mission(
    situation: dict,
    area: Optional[dict] = None,
) -> dict:
    """
    Generate a military mission plan from a situation assessment.

    Mission types are limited to coordination, ISR, HADR, and logistics.
    NO weapons targeting, fire control, or lethal engagement logic.
    """
    threat      = situation.get("threat_level", "routine")
    alert_types = {a["alert_type"] for a in situation.get("alerts", [])}

    # Priority order: life safety first, then threat, then logistics

    if "PR_BEACON_ACTIVE" in alert_types or "ISOLATED_PERSONNEL" in alert_types:
        return {
            "mission_type": "CSAR_ESCORT",
            "pattern": "DIRECT",
            "altitude_m": 50,
            "objectives": [
                "Authenticate IFF with isolated personnel",
                "Relay precise grid to CSAR element",
                "Low-altitude escort to extraction point",
            ],
            "domain": "military",
            "threat_level": threat,
            "area": area,
        }

    if situation.get("casevac_pending") or "CASEVAC_REQUESTED" in alert_types:
        return {
            "mission_type": "CASEVAC_ESCORT",
            "pattern": "DIRECT",
            "altitude_m": 100,
            "objectives": [
                "Escort CASEVAC asset to collection point",
                "Clear route of obstacles",
                "Relay 9-line to receiving facility",
            ],
            "domain": "military",
            "threat_level": threat,
            "area": area,
        }

    if "MINEFIELD_SUSPECTED" in alert_types:
        return {
            "mission_type": "IED_SURVEY",
            "pattern": "ORBIT",
            "altitude_m": 60,
            "objectives": [
                "Standoff 200m oblique imagery of suspected area",
                "Mark no-go zones with GPS polygons",
                "Relay grid to route clearance element",
            ],
            "sensor_config": {"optical": True, "thermal": True},
            "domain": "military",
            "threat_level": threat,
            "area": area,
        }

    if "CBRN_HAZARD_PROXIMITY" in alert_types or "CBRN_PLUME_TRACK" in alert_types:
        return {
            "mission_type": "CBRN_RECON",
            "pattern": "UPWIND_APPROACH",
            "altitude_m": 80,
            "standoff_m": 300,
            "objectives": [
                "Approach from upwind, maintain 300m standoff",
                "Sensor pass: CBRN detector + thermal + optical",
                "Map plume extent and wind dispersion",
                "Relay readings to CBRN response team",
            ],
            "sensor_config": {"thermal": True, "optical": True, "cbrn": True},
            "domain": "military",
            "threat_level": threat,
            "area": area,
        }

    if "COUNTER_UAS_SWARM" in alert_types or "COUNTER_UAS_DETECTED" in alert_types:
        return {
            "mission_type": "COUNTER_UAS_TRACK",
            "pattern": "ORBIT",
            "altitude_m": 120,
            "objectives": [
                "Orbit and visually/electronically ID rogue UAS",
                "Track and relay telemetry to ground-based C-UAS system",
                "Do NOT engage — coordination and tracking only",
            ],
            "sensor_config": {"optical": True, "thermal": True, "rf_detect": True},
            "domain": "military",
            "threat_level": threat,
            "area": area,
        }

    if "PERIMETER_BREACH" in alert_types:
        return {
            "mission_type": "FORCE_PROTECT",
            "pattern": "ORBIT",
            "altitude_m": 80,
            "objectives": [
                "Identify and track unknown entity",
                "Relay contact report to TOC",
                "Maintain overwatch until QRF arrival",
            ],
            "domain": "military",
            "threat_level": threat,
            "area": area,
        }

    if "BDA_REQUIRED" in alert_types or "STRUCTURE_DESTROYED" in alert_types:
        return {
            "mission_type": "BDA_SURVEY",
            "pattern": "GRID",
            "altitude_m": 100,
            "objectives": [
                "Systematic grid of strike area",
                "Photo and IR capture at each waypoint",
                "Generate BDA report for command assessment",
            ],
            "sensor_config": {"optical": True, "thermal": True},
            "domain": "military",
            "threat_level": threat,
            "area": area,
        }

    if "COMMS_BLACKOUT" in alert_types:
        return {
            "mission_type": "COMMS_RELAY",
            "pattern": "LOITER",
            "altitude_m": 500,
            "objectives": [
                "Establish relay position between blacked-out element and TOC",
                "Loiter at altitude until comms restored",
                "Relay authentication and status messages",
            ],
            "domain": "military",
            "threat_level": threat,
            "area": area,
        }

    if "RESUPPLY_CRITICAL" in alert_types:
        return {
            "mission_type": "RESUPPLY_ESCORT",
            "pattern": "DIRECT",
            "altitude_m": 120,
            "objectives": [
                "Escort resupply element to forward logistics point",
                "Clear route ahead of convoy",
                "Confirm delivery and return status",
            ],
            "domain": "military",
            "threat_level": threat,
            "area": area,
        }

    if "UNUSUAL_CONCENTRATION" in alert_types or "PATTERN_OF_LIFE_ANOMALY" in alert_types:
        return {
            "mission_type": "PATTERN_OF_LIFE_ISR",
            "pattern": "PERSISTENT_ORBIT",
            "altitude_m": 300,
            "dwell_hr": 24,
            "objectives": [
                "Establish persistent orbit over anomaly location",
                "24-hour dwell — collect pattern of life baseline",
                "Relay change-detection events to intelligence cell",
            ],
            "sensor_config": {"optical": True, "thermal": True, "eo_ir": True},
            "domain": "military",
            "threat_level": threat,
            "area": area,
        }

    if "CONVOY_GAP" in alert_types:
        return {
            "mission_type": "RECON",
            "pattern": "LAWNMOWER",
            "altitude_m": 150,
            "objectives": [
                "Locate separated convoy vehicle",
                "Confirm grid and status",
                "Guide recovery element",
            ],
            "domain": "military",
            "threat_level": threat,
            "area": area,
        }

    # Default: HADR or general ISR
    return {
        "mission_type": "HADR",
        "pattern": "LAWNMOWER",
        "altitude_m": 250,
        "objectives": [
            "Survey affected area",
            "Identify survivors and infrastructure damage",
            "Generate SITREP for command",
        ],
        "domain": "military",
        "threat_level": threat,
        "area": area,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _is_friendly(entity: dict) -> bool:
    cot_type = entity.get("metadata", {}).get("cot_type", "")
    return cot_type.startswith("a-f-") or entity.get("classification") == "FRIENDLY"


def _is_unknown(entity: dict) -> bool:
    cot_type = entity.get("metadata", {}).get("cot_type", "")
    return cot_type.startswith("a-u-") or entity.get("classification") == "UNKNOWN"


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


def _has_gap(members: list[dict]) -> bool:
    if len(members) < 2:
        return False
    positions = [e.get("position") for e in members if e.get("position")]
    if len(positions) < 2:
        return False
    for i in range(len(positions) - 1):
        a = {"position": positions[i]}
        b = {"position": positions[i + 1]}
        d = _distance_m(a, b)
        if d is not None and d > _CONVOY_GAP_M:
            return True
    return False


def _check_ace_clustering(assets: list[dict]) -> Optional[dict]:
    if len(assets) < 2:
        return None
    positions = [e.get("position") for e in assets if e.get("position")]
    if len(positions) < 2:
        return None
    # If all ACE assets are within 200m of each other, flag clustering risk
    for i, pa in enumerate(positions):
        for pb in positions[i + 1:]:
            a_e = {"position": pa}
            b_e = {"position": pb}
            d = _distance_m(a_e, b_e)
            if d is None or d > 200:
                return None
    return {
        "entity_id": "ace-cluster",
        "alert_type": "ACE_CLUSTERING",
        "description": "Forward ACE assets clustered — vulnerability to area effects",
        "severity": "medium",
    }


def _score_threat(alerts: list[dict]) -> str:
    critical = sum(1 for a in alerts if a.get("severity") == "critical")
    high     = sum(1 for a in alerts if a.get("severity") == "high")
    if critical:
        return "imminent"
    if high:
        return "high"
    if alerts:
        return "elevated"
    return "routine"


def _military_recommendations(
    threat: str, casevac: bool, unknown_count: int
) -> list[str]:
    recs = []
    if casevac:
        recs.append("Activate CASEVAC escort — relay 9-line to receiving facility")
    if threat == "imminent":
        recs.append("Notify TOC immediately — initiate QRF standby")
        recs.append("Increase ISR coverage of threat axis")
    if unknown_count > 0:
        recs.append(f"Generate SALUTE report for {unknown_count} unknown contact(s)")
    if threat in ("high", "imminent"):
        recs.append("Disperse ACE assets to reduce vulnerability")
        recs.append("Verify CBRN and IED threat assessment before route clearance")
    return recs
