"""
Maritime Domain Logic

Covers: port security, vessel traffic management, maritime SAR,
offshore platform monitoring, coastal surveillance, EEZ patrol.

Key data sources: AIS, NMEA2000, AISstream, OpenSky (low-altitude coastal),
thermal (night patrol), Kraken (underwater/sonar).
"""

from __future__ import annotations

import math
from typing import Optional

# ── Vessel risk scoring ───────────────────────────────────────────────────────

_HIGH_RISK_VESSEL_TYPES = {
    "Tanker", "Chemical Tanker", "Gas Carrier", "Bulk Carrier",
    "Fishing", "Pleasure Craft",  # fishing fleets near restricted zones
}

_DARK_SHIP_MINUTES = 30       # AIS gap that triggers dark vessel alert
_ANCHOR_DRIFT_KNOTS = 0.5     # vessel motion while anchored = anchor drag alert
_CLOSE_APPROACH_NM = 0.25     # vessel-to-vessel proximity alert threshold


def assess_maritime_situation(entities: list[dict]) -> dict:
    """
    Assess risk and generate alerts for a set of maritime entities.

    Args:
        entities: List of entity dicts from the world model (AIS, NMEA2000, etc.)

    Returns:
        {
            "risk_level": "low" | "medium" | "high" | "critical",
            "alerts": [{"entity_id", "alert_type", "description", "severity"}],
            "vessel_count": int,
            "dark_vessel_count": int,
            "recommendations": [str],
        }
    """
    vessels = [e for e in entities if e.get("entity_type") in ("VESSEL", "MARITIME")]
    alerts = []
    dark_count = 0

    for vessel in vessels:
        meta = vessel.get("metadata", {})
        mmsi = vessel.get("entity_id", "unknown")

        # Dark vessel detection (AIS gap)
        ais_gap_min = meta.get("ais_gap_minutes", 0)
        if ais_gap_min >= _DARK_SHIP_MINUTES:
            dark_count += 1
            alerts.append({
                "entity_id": mmsi,
                "alert_type": "DARK_VESSEL",
                "description": f"No AIS for {ais_gap_min:.0f} min — possible transponder off",
                "severity": "high",
            })

        # Anchor drag
        status = meta.get("nav_status", "")
        speed_kn = meta.get("speed_kn", meta.get("sog_kn", 0)) or 0
        if "anchor" in str(status).lower() and float(speed_kn) > _ANCHOR_DRIFT_KNOTS:
            alerts.append({
                "entity_id": mmsi,
                "alert_type": "ANCHOR_DRAG",
                "description": f"Vessel anchored but moving at {speed_kn:.1f}kn",
                "severity": "high",
            })

        # High-risk vessel type in restricted zone
        vessel_type = meta.get("vessel_type", meta.get("ship_type", ""))
        restricted = meta.get("in_restricted_zone", False)
        if vessel_type in _HIGH_RISK_VESSEL_TYPES and restricted:
            alerts.append({
                "entity_id": mmsi,
                "alert_type": "RESTRICTED_ZONE_INTRUSION",
                "description": f"{vessel_type} in restricted zone",
                "severity": "critical",
            })

    # CPA (closest point of approach) check between vessels
    cpa_alerts = _check_cpa(vessels)
    alerts.extend(cpa_alerts)

    risk = _score_risk(alerts, dark_count)
    recommendations = _maritime_recommendations(risk, dark_count, len(alerts))

    return {
        "risk_level":       risk,
        "alerts":           alerts,
        "vessel_count":     len(vessels),
        "dark_vessel_count": dark_count,
        "recommendations":  recommendations,
    }


def plan_maritime_mission(
    situation: dict,
    area: Optional[dict] = None,
) -> dict:
    """
    Generate a maritime mission plan from a situation assessment.

    Returns a mission plan dict compatible with the fabric mission orchestrator.
    """
    risk = situation.get("risk_level", "low")
    alert_types = {a["alert_type"] for a in situation.get("alerts", [])}

    if "RESTRICTED_ZONE_INTRUSION" in alert_types or risk == "critical":
        mission_type = "MARITIME_SAR"
        pattern = "PARALLEL_TRACK"
        alt_m = 150
        objectives = ["Intercept and identify vessel", "Relay to coast guard"]
    elif "DARK_VESSEL" in alert_types:
        mission_type = "PATROL"
        pattern = "EXPANDING_SQUARE"
        alt_m = 200
        objectives = ["Locate dark vessel", "Confirm identity", "Document position"]
    elif "ANCHOR_DRAG" in alert_types:
        mission_type = "MONITOR"
        pattern = "ORBIT"
        alt_m = 120
        objectives = ["Monitor dragging vessel", "Alert harbor master"]
    else:
        mission_type = "SURVEY"
        pattern = "LAWNMOWER"
        alt_m = 250
        objectives = ["Survey vessel traffic", "Log AIS discrepancies"]

    return {
        "mission_type": mission_type,
        "pattern": pattern,
        "altitude_m": alt_m,
        "objectives": objectives,
        "domain": "maritime",
        "risk_level": risk,
        "area": area,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_cpa(vessels: list[dict]) -> list[dict]:
    alerts = []
    for i, a in enumerate(vessels):
        for b in vessels[i + 1:]:
            dist_nm = _distance_nm(a, b)
            if dist_nm is not None and dist_nm < _CLOSE_APPROACH_NM:
                alerts.append({
                    "entity_id": f"{a.get('entity_id')}/{b.get('entity_id')}",
                    "alert_type": "CLOSE_APPROACH",
                    "description": f"Vessels within {dist_nm:.2f} nm of each other",
                    "severity": "high",
                })
    return alerts


def _distance_nm(a: dict, b: dict) -> Optional[float]:
    pa = a.get("position") or {}
    pb = b.get("position") or {}
    if not (pa.get("lat") and pb.get("lat")):
        return None
    lat1, lon1 = math.radians(pa["lat"]), math.radians(pa["lon"])
    lat2, lon2 = math.radians(pb["lat"]), math.radians(pb["lon"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a_ = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a_))
    return c * 3440.065  # Earth radius in nautical miles


def _score_risk(alerts: list[dict], dark_count: int) -> str:
    critical = sum(1 for a in alerts if a["severity"] == "critical")
    high = sum(1 for a in alerts if a["severity"] == "high")
    if critical > 0 or dark_count >= 3:
        return "critical"
    if high >= 2 or dark_count >= 1:
        return "high"
    if alerts:
        return "medium"
    return "low"


def _maritime_recommendations(risk: str, dark_count: int, alert_count: int) -> list[str]:
    recs = []
    if dark_count:
        recs.append(f"Deploy UAV patrol to locate {dark_count} dark vessel(s)")
    if risk in ("high", "critical"):
        recs.append("Notify coast guard and port authority")
        recs.append("Activate expanded surveillance sweep pattern")
    if alert_count > 5:
        recs.append("Consider temporary anchorage restriction in affected zone")
    return recs
