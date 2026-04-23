"""
Agriculture Domain Logic

Covers: crop health monitoring (NDVI), precision variable-rate spraying,
livestock tracking, field boundary mapping, irrigation status,
pest/disease outbreak detection.

Primary sources: multispectral camera (NDVI/NDRE), ISOBUS (tractor telemetry),
thermal (livestock), LoRaWAN (soil moisture sensors).
"""

from __future__ import annotations

from typing import Optional

# ── Thresholds ────────────────────────────────────────────────────────────────

_NDVI_STRESS_THRESHOLD = 0.35    # below this = vegetation stress
_NDVI_CRITICAL_THRESHOLD = 0.20  # severe stress / bare soil
_MOISTURE_LOW_PCT = 25.0         # soil moisture % (VWC) below which irrigation needed
_LIVESTOCK_TEMP_HIGH_C = 39.5    # fever threshold for cattle/sheep
_PEST_CONFIDENCE_THRESHOLD = 0.6


def assess_agriculture_situation(entities: list[dict]) -> dict:
    """
    Assess crop health, soil status, and livestock welfare.

    Args:
        entities: World model entities from multispectral, ISOBUS, LoRaWAN, thermal adapters

    Returns:
        {
            "risk_level": str,
            "alerts": list[dict],
            "fields_assessed": int,
            "stress_zone_count": int,
            "recommendations": list[str],
        }
    """
    alerts = []
    fields = [e for e in entities if e.get("metadata", {}).get("asset_class") == "FIELD"]
    livestock = [e for e in entities if e.get("metadata", {}).get("asset_class") == "LIVESTOCK"]
    soil_sensors = [e for e in entities if e.get("metadata", {}).get("asset_class") == "SOIL_SENSOR"]

    stress_zones = 0

    for field in fields:
        meta = field.get("metadata", {})
        eid = field.get("entity_id", "unknown")

        ndvi = meta.get("ndvi")
        if ndvi is not None:
            if ndvi < _NDVI_CRITICAL_THRESHOLD:
                stress_zones += 1
                alerts.append({
                    "entity_id": eid,
                    "alert_type": "CROP_FAILURE_RISK",
                    "description": f"NDVI {ndvi:.2f} — critical vegetation stress",
                    "severity": "critical",
                })
            elif ndvi < _NDVI_STRESS_THRESHOLD:
                stress_zones += 1
                alerts.append({
                    "entity_id": eid,
                    "alert_type": "CROP_STRESS",
                    "description": f"NDVI {ndvi:.2f} — vegetation stress detected",
                    "severity": "medium",
                })

        pest_conf = meta.get("pest_confidence", 0)
        pest_type = meta.get("pest_type", "unknown")
        if pest_conf >= _PEST_CONFIDENCE_THRESHOLD:
            alerts.append({
                "entity_id": eid,
                "alert_type": "PEST_DETECTION",
                "description": f"{pest_type} detected (confidence {pest_conf:.0%})",
                "severity": "high",
            })

    for animal in livestock:
        meta = animal.get("metadata", {})
        eid = animal.get("entity_id", "unknown")
        temp_c = meta.get("skin_temp_c") or meta.get("peak_temp_c")
        if temp_c and temp_c >= _LIVESTOCK_TEMP_HIGH_C:
            alerts.append({
                "entity_id": eid,
                "alert_type": "LIVESTOCK_FEVER",
                "description": f"Livestock temp {temp_c:.1f}°C — possible illness",
                "severity": "high",
            })

    for sensor in soil_sensors:
        meta = sensor.get("metadata", {})
        moisture = meta.get("soil_moisture_pct")
        if moisture is not None and moisture < _MOISTURE_LOW_PCT:
            alerts.append({
                "entity_id": sensor.get("entity_id", "unknown"),
                "alert_type": "IRRIGATION_NEEDED",
                "description": f"Soil moisture {moisture:.1f}% — below {_MOISTURE_LOW_PCT}% threshold",
                "severity": "medium",
            })

    risk = _score_risk(alerts)
    return {
        "risk_level":      risk,
        "alerts":          alerts,
        "fields_assessed": len(fields),
        "stress_zone_count": stress_zones,
        "recommendations": _ag_recommendations(alerts),
    }


def plan_agriculture_mission(
    situation: dict,
    area: Optional[dict] = None,
) -> dict:
    """Generate a precision agriculture mission plan."""
    alert_types = {a["alert_type"] for a in situation.get("alerts", [])}
    risk = situation.get("risk_level", "low")

    if "PEST_DETECTION" in alert_types:
        objectives = [
            "Map pest infestation boundary",
            "Generate variable-rate spray prescription",
            "Schedule applicator pass",
        ]
        pattern = "LAWNMOWER"
        alt_m = 30
    elif "CROP_STRESS" in alert_types or "CROP_FAILURE_RISK" in alert_types:
        objectives = [
            "Full-field NDVI/NDRE survey",
            "Identify stress zones with GPS polygons",
            "Recommend irrigation or fertilization",
        ]
        pattern = "LAWNMOWER"
        alt_m = 60
    else:
        objectives = ["Routine field health survey", "Update crop progress records"]
        pattern = "LAWNMOWER"
        alt_m = 80

    return {
        "mission_type": "PRECISION_AG",
        "pattern":      pattern,
        "altitude_m":   alt_m,
        "objectives":   objectives,
        "domain":       "agriculture",
        "risk_level":   risk,
        "area":         area,
    }


def _score_risk(alerts: list[dict]) -> str:
    if any(a["severity"] == "critical" for a in alerts):
        return "critical"
    if any(a["severity"] == "high" for a in alerts):
        return "high"
    if alerts:
        return "medium"
    return "low"


def _ag_recommendations(alerts: list[dict]) -> list[str]:
    recs = []
    types = {a["alert_type"] for a in alerts}
    if "PEST_DETECTION" in types:
        recs.append("Generate variable-rate spray prescription map")
        recs.append("Notify agronomist for species confirmation")
    if "CROP_FAILURE_RISK" in types:
        recs.append("Prioritize irrigation and fertilization in critical zones")
    if "LIVESTOCK_FEVER" in types:
        recs.append("Alert farm manager — individual animal follow-up required")
    if "IRRIGATION_NEEDED" in types:
        recs.append("Activate irrigation in affected zones")
    return recs
