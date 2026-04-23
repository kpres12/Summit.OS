"""
Wildlife Domain Logic

Covers: anti-poaching patrol, species population tracking, habitat
health monitoring, human-wildlife conflict detection, migration corridor
surveillance, ranger coordination.

Primary sources: thermal (nighttime detection), RGB (daytime ID),
acoustic sensors (lorawan), GPS collars (via LoRaWAN or satellite).
"""

from __future__ import annotations

from typing import Optional

# ── Thresholds ────────────────────────────────────────────────────────────────

_POACHING_CLUSTER_COUNT = 2      # humans + vehicle at night in reserve = alert
_SNARE_CONFIDENCE = 0.65         # ML confidence for snare/trap detection
_COLLAR_HEARTBEAT_HOURS = 6      # GPS collar silence that flags mortality
_CONFLICT_BUFFER_M = 200.0       # predator proximity to human settlement


def assess_wildlife_situation(entities: list[dict]) -> dict:
    """
    Assess conservation threats and wildlife status.

    Returns:
        {
            "risk_level": str,
            "alerts": list[dict],
            "species_contacts": int,
            "human_contacts": int,
            "recommendations": list[str],
        }
    """
    alerts = []
    humans = [e for e in entities if e.get("classification") == "PERSON"]
    vehicles = [e for e in entities if e.get("classification") == "VEHICLE"]
    animals = [e for e in entities if e.get("metadata", {}).get("asset_class") == "WILDLIFE"]
    collars = [e for e in entities if e.get("metadata", {}).get("collar_id")]
    structures = [e for e in entities if e.get("metadata", {}).get("asset_class") in (
        "SNARE", "TRAP", "WIRE",
    )]
    settlements = [e for e in entities if e.get("metadata", {}).get("asset_class") == "SETTLEMENT"]

    # Anti-poaching: humans + vehicles in restricted zone at night
    night_mode = any(e.get("metadata", {}).get("night_mode") for e in entities)
    in_reserve = [h for h in humans if h.get("metadata", {}).get("in_reserve")]
    reserve_vehicles = [v for v in vehicles if v.get("metadata", {}).get("in_reserve")]

    if night_mode and len(in_reserve) + len(reserve_vehicles) >= _POACHING_CLUSTER_COUNT:
        alerts.append({
            "entity_id": "poaching-cluster",
            "alert_type": "POACHING_ACTIVITY",
            "description": (
                f"{len(in_reserve)} person(s) + {len(reserve_vehicles)} vehicle(s) "
                "in reserve after dark"
            ),
            "severity": "critical",
        })

    # Snare/trap detection (ML)
    for struct in structures:
        conf = struct.get("metadata", {}).get("detection_confidence", 0)
        if conf >= _SNARE_CONFIDENCE:
            alerts.append({
                "entity_id": struct.get("entity_id"),
                "alert_type": "SNARE_DETECTED",
                "description": f"Snare/trap detected (confidence {conf:.0%})",
                "severity": "high",
            })

    # GPS collar mortality check (heartbeat gap)
    for collar in collars:
        gap_h = collar.get("metadata", {}).get("heartbeat_gap_hours", 0)
        if gap_h >= _COLLAR_HEARTBEAT_HOURS:
            alerts.append({
                "entity_id": collar.get("entity_id"),
                "alert_type": "COLLAR_SILENCE",
                "description": f"GPS collar silent {gap_h:.0f}h — possible mortality/capture",
                "severity": "high",
            })

    # Human-wildlife conflict (predator near settlement)
    predators = [a for a in animals if a.get("metadata", {}).get("predator")]
    for pred in predators:
        for settlement in settlements:
            d = _distance_m(pred, settlement)
            if d is not None and d < _CONFLICT_BUFFER_M:
                alerts.append({
                    "entity_id": pred.get("entity_id"),
                    "alert_type": "HUMAN_WILDLIFE_CONFLICT",
                    "description": f"Predator {d:.0f}m from settlement",
                    "severity": "critical",
                })
                break

    risk = _score_risk(alerts)
    return {
        "risk_level":      risk,
        "alerts":          alerts,
        "species_contacts": len(animals),
        "human_contacts":  len(humans),
        "recommendations": _wildlife_recommendations(alerts),
    }


def plan_wildlife_mission(
    situation: dict,
    area: Optional[dict] = None,
) -> dict:
    """Generate a wildlife patrol or monitoring mission."""
    alert_types = {a["alert_type"] for a in situation.get("alerts", [])}
    risk = situation.get("risk_level", "low")

    if "POACHING_ACTIVITY" in alert_types:
        return {
            "mission_type": "ANTI_POACH",
            "pattern":      "ORBIT",
            "altitude_m":   80,
            "objectives":   [
                "Track and document poaching activity",
                "Relay coordinates to ranger unit",
                "Maintain overwatch — do not engage",
            ],
            "domain":       "wildlife",
            "risk_level":   risk,
            "area":         area,
        }

    if "HUMAN_WILDLIFE_CONFLICT" in alert_types:
        return {
            "mission_type": "MONITOR",
            "pattern":      "ORBIT",
            "altitude_m":   60,
            "objectives":   [
                "Track predator movement",
                "Alert wildlife management team",
                "Monitor until predator clears settlement buffer",
            ],
            "domain":       "wildlife",
            "risk_level":   risk,
            "area":         area,
        }

    return {
        "mission_type": "ANTI_POACH",
        "pattern":      "EXPANDING_SQUARE",
        "altitude_m":   120,
        "objectives":   [
            "Systematic reserve patrol",
            "Detect snares and unauthorized presence",
            "Species count and GPS logging",
        ],
        "domain":       "wildlife",
        "risk_level":   risk,
        "area":         area,
    }


import math


def _distance_m(a: dict, b: dict) -> Optional[float]:
    pa = a.get("position") or {}
    pb = b.get("position") or {}
    if not (pa.get("lat") and pb.get("lat")):
        return None
    lat1, lon1 = math.radians(pa["lat"]), math.radians(pa["lon"])
    lat2, lon2 = math.radians(pb["lat"]), math.radians(pb["lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a_ = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * math.asin(math.sqrt(a_)) * 6_371_000


def _score_risk(alerts: list[dict]) -> str:
    if any(a["severity"] == "critical" for a in alerts):
        return "critical"
    if any(a["severity"] == "high" for a in alerts):
        return "high"
    if alerts:
        return "medium"
    return "low"


def _wildlife_recommendations(alerts: list[dict]) -> list[str]:
    recs = []
    types = {a["alert_type"] for a in alerts}
    if "POACHING_ACTIVITY" in types:
        recs.append("Deploy rapid response rangers — preserve evidence")
    if "SNARE_DETECTED" in types:
        recs.append("Mark snare GPS for ranger removal sweep")
    if "COLLAR_SILENCE" in types:
        recs.append("Dispatch ground team to last known collar location")
    if "HUMAN_WILDLIFE_CONFLICT" in types:
        recs.append("Alert community warning system — advise stay indoors")
    return recs
