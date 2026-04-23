"""
Utilities Domain Logic

Covers: electrical transmission line inspection, natural gas pipeline patrol,
bridge / dam structural inspection, water treatment facility monitoring,
telecom tower inspection.

Primary sensors: RGB camera, thermal (hot spots on lines), LiDAR (sag),
modbus (SCADA values), lorawan (leak sensors).
"""

from __future__ import annotations

from typing import Optional

# ── Thresholds ────────────────────────────────────────────────────────────────

_HOT_SPOT_DELTA_C = 15.0      # temp above ambient that flags a component
_SAG_CLEARANCE_M = 6.0        # minimum conductor-to-ground clearance (meters)
_LEAK_PPM_THRESHOLD = 50.0    # gas concentration ppm that triggers alert
_CORROSION_SCORE_HIGH = 0.7   # ML corrosion confidence threshold


def assess_utilities_situation(entities: list[dict]) -> dict:
    """
    Assess risk across utility infrastructure entities.

    Args:
        entities: World model entities from modbus, lorawan, thermal, rtsp adapters

    Returns:
        {
            "risk_level": str,
            "alerts": list[dict],
            "assets_inspected": int,
            "faults_detected": int,
            "recommendations": list[str],
        }
    """
    alerts = []
    assets = [e for e in entities if e.get("metadata", {}).get("asset_class") in (
        "POWER_LINE", "PIPELINE", "BRIDGE", "WATER", "TELECOM",
    )]

    for asset in assets:
        meta = asset.get("metadata", {})
        asset_class = meta.get("asset_class", "")
        eid = asset.get("entity_id", "unknown")

        # Thermal hot spot (transmission lines, transformers)
        if asset_class in ("POWER_LINE",):
            peak_c = meta.get("peak_temp_c", 0)
            ambient_c = meta.get("ambient_temp_c", 20)
            if peak_c - ambient_c >= _HOT_SPOT_DELTA_C:
                alerts.append({
                    "entity_id": eid,
                    "alert_type": "THERMAL_HOT_SPOT",
                    "description": f"Component {peak_c - ambient_c:.1f}°C above ambient",
                    "severity": "high" if peak_c - ambient_c >= 30 else "medium",
                })

        # Conductor sag (LiDAR)
        sag_clearance = meta.get("sag_clearance_m")
        if sag_clearance is not None and sag_clearance < _SAG_CLEARANCE_M:
            alerts.append({
                "entity_id": eid,
                "alert_type": "CONDUCTOR_SAG",
                "description": f"Ground clearance {sag_clearance:.1f}m — below {_SAG_CLEARANCE_M}m minimum",
                "severity": "critical" if sag_clearance < 4.0 else "high",
            })

        # Pipeline gas leak (LoRaWAN sensors or thermal)
        if asset_class == "PIPELINE":
            ppm = meta.get("gas_ppm", 0)
            if ppm >= _LEAK_PPM_THRESHOLD:
                alerts.append({
                    "entity_id": eid,
                    "alert_type": "GAS_LEAK",
                    "description": f"Gas concentration {ppm:.0f} ppm at sensor",
                    "severity": "critical" if ppm >= 500 else "high",
                })

        # Structural corrosion (ML confidence from vision model)
        corrosion = meta.get("corrosion_score", 0)
        if corrosion >= _CORROSION_SCORE_HIGH:
            alerts.append({
                "entity_id": eid,
                "alert_type": "STRUCTURAL_CORROSION",
                "description": f"Corrosion detected (confidence {corrosion:.0%})",
                "severity": "high",
            })

        # SCADA anomaly (modbus adapter)
        scada_fault = meta.get("scada_fault_code")
        if scada_fault:
            alerts.append({
                "entity_id": eid,
                "alert_type": "SCADA_FAULT",
                "description": f"SCADA fault code: {scada_fault}",
                "severity": "high",
            })

    risk = _score_risk(alerts)
    return {
        "risk_level":       risk,
        "alerts":           alerts,
        "assets_inspected": len(assets),
        "faults_detected":  len(alerts),
        "recommendations":  _utilities_recommendations(alerts),
    }


def plan_utilities_mission(
    situation: dict,
    asset_class: str = "POWER_LINE",
    area: Optional[dict] = None,
) -> dict:
    """Generate an inspection mission plan for a utility asset type."""
    alert_types = {a["alert_type"] for a in situation.get("alerts", [])}
    risk = situation.get("risk_level", "low")

    if "GAS_LEAK" in alert_types or "CONDUCTOR_SAG" in alert_types:
        pattern = "DIRECT"
        alt_m = 50
        objectives = ["Locate and document fault", "Assess severity", "Mark GPS coordinates"]
    else:
        pattern = "LAWNMOWER"
        alt_m = 80 if asset_class == "POWER_LINE" else 60
        objectives = [
            "Systematic thermal and RGB inspection",
            "Flag anomalies for maintenance crew",
            "Generate inspection report",
        ]

    return {
        "mission_type": "INSPECT",
        "pattern":       pattern,
        "altitude_m":    alt_m,
        "objectives":    objectives,
        "domain":        "utilities",
        "asset_class":   asset_class,
        "risk_level":    risk,
        "area":          area,
    }


def _score_risk(alerts: list[dict]) -> str:
    if any(a["severity"] == "critical" for a in alerts):
        return "critical"
    if any(a["severity"] == "high" for a in alerts):
        return "high"
    if alerts:
        return "medium"
    return "low"


def _utilities_recommendations(alerts: list[dict]) -> list[str]:
    recs = []
    types = {a["alert_type"] for a in alerts}
    if "GAS_LEAK" in types:
        recs.append("Isolate pipeline section — notify operations center and safety team")
    if "CONDUCTOR_SAG" in types or "THERMAL_HOT_SPOT" in types:
        recs.append("Dispatch line crew for physical inspection — consider outage window")
    if "STRUCTURAL_CORROSION" in types:
        recs.append("Schedule structural engineering assessment")
    if "SCADA_FAULT" in types:
        recs.append("Review SCADA fault log — escalate to control room")
    return recs
