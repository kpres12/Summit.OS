"""
Oil & Gas Domain Logic

Covers: pipeline right-of-way patrol, compressor station monitoring,
flare stack health, tank farm inspection, offshore platform surveillance,
spill detection, third-party intrusion on ROW.
"""

from __future__ import annotations

from typing import Optional

# ── Thresholds ────────────────────────────────────────────────────────────────

_FLARE_TEMP_ANOMALY_C = 200      # delta above nominal flare temp
_SPILL_AREA_THRESHOLD_M2 = 10.0  # minimum detected spill area
_GAS_PPM_THRESHOLD = 100.0       # methane/H2S ppm at survey altitude
_ROW_INTRUSION_BUFFER_M = 15.0   # encroachment within ROW boundary
_CORROSION_SCORE = 0.65


def assess_oilgas_situation(entities: list[dict]) -> dict:
    """
    Assess risk across oil & gas infrastructure.

    Returns:
        {
            "risk_level": str,
            "alerts": list[dict],
            "assets_assessed": int,
            "recommendations": list[str],
        }
    """
    alerts = []
    assets = [e for e in entities if e.get("metadata", {}).get("asset_class") in (
        "PIPELINE", "COMPRESSOR_STATION", "FLARE_STACK", "TANK_FARM",
        "OFFSHORE_PLATFORM", "WELLPAD",
    )]
    unknowns = [e for e in entities if _near_row(e, assets)]

    for asset in assets:
        meta = asset.get("metadata", {})
        eid = asset.get("entity_id", "unknown")
        asset_class = meta.get("asset_class", "")

        # Spill detection (thermal or optical)
        spill_m2 = meta.get("spill_area_m2", 0)
        if spill_m2 >= _SPILL_AREA_THRESHOLD_M2:
            alerts.append({
                "entity_id": eid,
                "alert_type": "SPILL_DETECTED",
                "description": f"Suspected spill {spill_m2:.0f} m² — thermal/optical anomaly",
                "severity": "critical",
            })

        # Gas concentration (thermal + methane sensor)
        gas_ppm = meta.get("gas_ppm", 0)
        if gas_ppm >= _GAS_PPM_THRESHOLD:
            alerts.append({
                "entity_id": eid,
                "alert_type": "GAS_CONCENTRATION",
                "description": f"Gas concentration {gas_ppm:.0f} ppm at asset",
                "severity": "critical" if gas_ppm >= 1000 else "high",
            })

        # Flare anomaly
        if asset_class == "FLARE_STACK":
            flare_delta = meta.get("flare_temp_delta_c", 0)
            if abs(flare_delta) >= _FLARE_TEMP_ANOMALY_C:
                alerts.append({
                    "entity_id": eid,
                    "alert_type": "FLARE_ANOMALY",
                    "description": f"Flare temp {flare_delta:+.0f}°C from nominal",
                    "severity": "high",
                })

        # Corrosion
        corrosion = meta.get("corrosion_score", 0)
        if corrosion >= _CORROSION_SCORE:
            alerts.append({
                "entity_id": eid,
                "alert_type": "CORROSION_DETECTED",
                "description": f"External corrosion (confidence {corrosion:.0%})",
                "severity": "high",
            })

    # ROW intrusion (third-party encroachment)
    for intruder in unknowns:
        alerts.append({
            "entity_id": intruder.get("entity_id"),
            "alert_type": "ROW_INTRUSION",
            "description": "Unauthorized activity near pipeline right-of-way",
            "severity": "high",
        })

    risk = _score_risk(alerts)
    return {
        "risk_level":      risk,
        "alerts":          alerts,
        "assets_assessed": len(assets),
        "recommendations": _oilgas_recommendations(alerts),
    }


def plan_oilgas_mission(
    situation: dict,
    area: Optional[dict] = None,
) -> dict:
    """Generate a pipeline/facility inspection mission."""
    alert_types = {a["alert_type"] for a in situation.get("alerts", [])}
    risk = situation.get("risk_level", "low")

    if "SPILL_DETECTED" in alert_types or "GAS_CONCENTRATION" in alert_types:
        return {
            "mission_type": "PIPELINE_PATROL",
            "pattern":      "DIRECT",
            "altitude_m":   60,
            "objectives":   [
                "Characterize spill/leak extent",
                "Relay coordinates to emergency response",
                "Document with thermal and RGB",
            ],
            "domain":       "oilgas",
            "risk_level":   risk,
            "area":         area,
        }

    return {
        "mission_type": "PIPELINE_PATROL",
        "pattern":      "LAWNMOWER",
        "altitude_m":   80,
        "objectives":   [
            "Systematic ROW patrol",
            "Thermal hot-spot scan",
            "Third-party intrusion check",
            "Generate inspection report",
        ],
        "domain":       "oilgas",
        "risk_level":   risk,
        "area":         area,
    }


def _near_row(entity: dict, assets: list[dict]) -> bool:
    import math
    if entity.get("metadata", {}).get("asset_class"):
        return False
    pos = entity.get("position") or {}
    if not pos.get("lat"):
        return False
    for asset in assets:
        ap = asset.get("position") or {}
        if not ap.get("lat"):
            continue
        lat1, lon1 = math.radians(pos["lat"]), math.radians(pos["lon"])
        lat2, lon2 = math.radians(ap["lat"]), math.radians(ap["lon"])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        d = 2 * math.asin(math.sqrt(a)) * 6_371_000
        if d <= _ROW_INTRUSION_BUFFER_M:
            return True
    return False


def _score_risk(alerts: list[dict]) -> str:
    if any(a["severity"] == "critical" for a in alerts):
        return "critical"
    if any(a["severity"] == "high" for a in alerts):
        return "high"
    if alerts:
        return "medium"
    return "low"


def _oilgas_recommendations(alerts: list[dict]) -> list[str]:
    recs = []
    types = {a["alert_type"] for a in alerts}
    if "SPILL_DETECTED" in types:
        recs.append("Activate spill response plan — notify NRC and state regulator")
    if "GAS_CONCENTRATION" in types:
        recs.append("Evacuate exclusion zone — notify fire department and PHMSA")
    if "ROW_INTRUSION" in types:
        recs.append("Dispatch ROW inspector — document intrusion for legal record")
    if "CORROSION_DETECTED" in types:
        recs.append("Schedule ILI run or cathodic protection survey")
    return recs
