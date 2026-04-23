"""
Construction Domain Logic

Covers: site progress monitoring, safety compliance (PPE detection, exclusion
zones), equipment utilization tracking, earthwork volume estimation,
perimeter security, material delivery verification.
"""

from __future__ import annotations

from typing import Optional

# ── Thresholds ────────────────────────────────────────────────────────────────

_PPE_VIOLATION_CONFIDENCE = 0.70   # model confidence to trigger PPE alert
_EXCLUSION_ZONE_BUFFER_M = 5.0     # safety buffer inside crane/excavation zone
_EQUIPMENT_IDLE_MINUTES = 30       # idling equipment utilization flag
_PROGRESS_DEVIATION_PCT = 15.0     # % behind schedule that triggers escalation


def assess_construction_situation(entities: list[dict]) -> dict:
    """
    Assess safety and progress status across a construction site.

    Returns:
        {
            "risk_level": str,
            "alerts": list[dict],
            "workers_detected": int,
            "equipment_count": int,
            "recommendations": list[str],
        }
    """
    alerts = []
    workers = [e for e in entities if e.get("classification") == "PERSON"]
    equipment = [e for e in entities if e.get("metadata", {}).get("asset_class") in (
        "CRANE", "EXCAVATOR", "LOADER", "CONCRETE_PUMP", "HEAVY_VEHICLE",
    )]
    zones = [e for e in entities if e.get("metadata", {}).get("zone_type") == "EXCLUSION"]

    # PPE compliance
    for worker in workers:
        meta = worker.get("metadata", {})
        ppe_violations = meta.get("ppe_violations", [])
        violation_conf = meta.get("ppe_confidence", 0)
        if ppe_violations and violation_conf >= _PPE_VIOLATION_CONFIDENCE:
            alerts.append({
                "entity_id": worker.get("entity_id"),
                "alert_type": "PPE_VIOLATION",
                "description": f"Missing PPE: {', '.join(ppe_violations)}",
                "severity": "high",
            })

    # Exclusion zone intrusion
    for worker in workers:
        for zone in zones:
            dist = _distance_m(worker, zone)
            if dist is not None and dist < _EXCLUSION_ZONE_BUFFER_M:
                alerts.append({
                    "entity_id": worker.get("entity_id"),
                    "alert_type": "EXCLUSION_ZONE_BREACH",
                    "description": "Worker inside equipment exclusion zone",
                    "severity": "critical",
                })
                break

    # Equipment idle time
    for eq in equipment:
        meta = eq.get("metadata", {})
        idle_min = meta.get("idle_minutes", 0)
        if idle_min >= _EQUIPMENT_IDLE_MINUTES:
            alerts.append({
                "entity_id": eq.get("entity_id"),
                "alert_type": "EQUIPMENT_IDLE",
                "description": f"{eq.get('callsign', 'Equipment')} idle {idle_min:.0f} min",
                "severity": "low",
            })

    # Schedule progress
    schedule_entities = [e for e in entities if e.get("metadata", {}).get("schedule_deviation_pct") is not None]
    for se in schedule_entities:
        dev = se.get("metadata", {}).get("schedule_deviation_pct", 0)
        if dev >= _PROGRESS_DEVIATION_PCT:
            alerts.append({
                "entity_id": se.get("entity_id"),
                "alert_type": "SCHEDULE_DEVIATION",
                "description": f"Zone {se.get('callsign')} is {dev:.0f}% behind schedule",
                "severity": "medium",
            })

    risk = _score_risk(alerts)
    return {
        "risk_level":       risk,
        "alerts":           alerts,
        "workers_detected": len(workers),
        "equipment_count":  len(equipment),
        "recommendations":  _construction_recommendations(alerts),
    }


def plan_construction_mission(
    situation: dict,
    area: Optional[dict] = None,
) -> dict:
    """Generate a construction monitoring mission."""
    alert_types = {a["alert_type"] for a in situation.get("alerts", [])}
    risk = situation.get("risk_level", "low")

    if "EXCLUSION_ZONE_BREACH" in alert_types or "PPE_VIOLATION" in alert_types:
        objectives = [
            "Locate and document safety violations",
            "Alert site safety officer",
            "Capture timestamped evidence imagery",
        ]
        pattern = "ORBIT"
        alt_m = 40
    else:
        objectives = [
            "Full-site progress survey",
            "Earthwork volume estimation",
            "Equipment utilization count",
            "Generate progress report",
        ]
        pattern = "LAWNMOWER"
        alt_m = 80

    return {
        "mission_type": "INSPECT",
        "pattern":      pattern,
        "altitude_m":   alt_m,
        "objectives":   objectives,
        "domain":       "construction",
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


def _construction_recommendations(alerts: list[dict]) -> list[str]:
    recs = []
    types = {a["alert_type"] for a in alerts}
    if "EXCLUSION_ZONE_BREACH" in types:
        recs.append("Stop crane/excavation operation — remove worker from zone")
    if "PPE_VIOLATION" in types:
        recs.append("Notify site safety officer — issue stop-work notice to violating crew")
    if "SCHEDULE_DEVIATION" in types:
        recs.append("Review resource allocation for behind-schedule zones")
    if "EQUIPMENT_IDLE" in types:
        recs.append("Review equipment scheduling to reduce idle time and fuel cost")
    return recs
