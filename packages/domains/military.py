"""
Military Domain Logic

Covers: Humanitarian Assistance / Disaster Response (HADR), Agile Combat
Employment (ACE), force protection perimeter, CASEVAC escort coordination,
base/FOB perimeter monitoring, logistics convoy tracking.

IMPORTANT: Heli.OS is a coordination and situational awareness platform.
This module handles tracking, tasking, and reporting only — no weapons
targeting, fire control, or lethal engagement logic.

Primary standards: CoT/ATAK (MIL-STD-2525B type codes), SALUTE reports,
9-line MEDEVAC, SITREP.
"""

from __future__ import annotations

from typing import Optional

# ── Threat assessment thresholds ──────────────────────────────────────────────

_PERIMETER_BREACH_M = 500        # unknown entity within 500m of FOB = alert
_CASEVAC_MAX_WAIT_MIN = 60       # CASEVAC response window (Golden Hour)
_CONVOY_GAP_M = 500              # inter-vehicle gap that triggers lost contact


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
    friendly = [e for e in entities if _is_friendly(e)]
    unknown = [e for e in entities if _is_unknown(e)]
    fob_locations = [e for e in entities if e.get("metadata", {}).get("is_fob")]
    convoys = [e for e in entities if e.get("metadata", {}).get("convoy_id")]

    # Perimeter breach check
    for unk in unknown:
        for fob in fob_locations:
            dist_m = _distance_m(unk, fob)
            if dist_m is not None and dist_m < _PERIMETER_BREACH_M:
                alerts.append({
                    "entity_id": unk.get("entity_id"),
                    "alert_type": "PERIMETER_BREACH",
                    "description": f"Unknown entity {dist_m:.0f}m from FOB",
                    "severity": "critical",
                    "distance_m": dist_m,
                })

    # CASEVAC pending check
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

    # Convoy gap (lost vehicle)
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

    # ACE dispersal monitoring — check if forward assets are spread for survivability
    ace_assets = [e for e in friendly if e.get("metadata", {}).get("ace_role")]
    ace_alert = _check_ace_clustering(ace_assets)
    if ace_alert:
        alerts.append(ace_alert)

    threat = _score_threat(alerts)
    recommendations = _military_recommendations(threat, casevac_pending, len(unknown))

    return {
        "threat_level":   threat,
        "alerts":         alerts,
        "friendly_count": len(friendly),
        "unknown_count":  len(unknown),
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
    """
    threat = situation.get("threat_level", "routine")
    alert_types = {a["alert_type"] for a in situation.get("alerts", [])}

    if situation.get("casevac_pending"):
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

import math


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
    high = sum(1 for a in alerts if a.get("severity") == "high")
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
    return recs
