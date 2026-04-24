"""
Flood Domain — Situation Assessment and Mission Planning

Covers: riverine flooding, flash flood, coastal surge, dam failure.
Sensor inputs: SAR inundation extent, water level gauges, rainfall accumulation,
               population density layers, infrastructure intersection.

Alert types:
  FLOOD_INUNDATION_DETECTED   — active flooding in area
  FLOOD_RISING_RATE           — water level rising faster than 10cm/hr
  FLOOD_INFRASTRUCTURE_RISK   — bridge/road/levee in flood zone
  FLOOD_EVACUATION_NEEDED     — population in inundation zone
  FLOOD_RECEDING              — water level dropping, recovery phase possible

Mission types:
  FLOOD_SURVEY     — SAR/optical mapping of inundation extent
  FLOOD_SAR        — search for stranded persons in flood zone
  FLOOD_LOGISTICS  — assess road/bridge passability for aid corridors
  FLOOD_MONITOR    — continuous water level / levee integrity watch
"""

from __future__ import annotations

from typing import Any


def assess_flood_situation(entities: list[dict[str, Any]]) -> dict[str, Any]:
    alerts = []
    rising_zones = []
    inundated_zones = []
    infrastructure_at_risk = []

    for e in entities:
        etype = e.get("type", "").upper()
        props = e.get("properties", e)

        # SAR-derived inundation detection
        if etype in ("SAR_INUNDATION", "FLOOD_EXTENT", "WATER_BODY"):
            area_km2  = float(props.get("area_km2", 0))
            pop_count = int(props.get("population_affected", 0))
            inundated_zones.append(e)
            sev = "HIGH" if area_km2 > 50 or pop_count > 1000 else "MEDIUM"
            alerts.append({
                "type": "FLOOD_INUNDATION_DETECTED",
                "entity_id": e.get("entity_id"),
                "severity": sev,
                "area_km2": area_km2,
                "population_affected": pop_count,
            })
            if pop_count > 500:
                alerts.append({
                    "type": "FLOOD_EVACUATION_NEEDED",
                    "entity_id": e.get("entity_id"),
                    "severity": "HIGH",
                    "population": pop_count,
                })

        # Water gauge telemetry
        elif etype in ("WATER_GAUGE", "STREAM_GAUGE", "FLOOD_SENSOR"):
            level_m      = float(props.get("water_level_m", 0))
            rise_cm_hr   = float(props.get("rise_rate_cm_hr", 0))
            flood_stage  = float(props.get("flood_stage_m", 3.0))
            if rise_cm_hr > 10:
                rising_zones.append(e)
                alerts.append({
                    "type": "FLOOD_RISING_RATE",
                    "entity_id": e.get("entity_id"),
                    "severity": "HIGH" if rise_cm_hr > 25 else "MEDIUM",
                    "rise_rate_cm_hr": rise_cm_hr,
                    "current_level_m": level_m,
                    "flood_stage_m": flood_stage,
                })
            elif rise_cm_hr < -5 and level_m > flood_stage:
                alerts.append({
                    "type": "FLOOD_RECEDING",
                    "entity_id": e.get("entity_id"),
                    "severity": "LOW",
                    "level_m": level_m,
                })

        # Infrastructure intersection
        elif etype in ("BRIDGE", "ROAD", "LEVEE", "DAM"):
            flood_score = float(props.get("flood_risk_score", 0))
            if flood_score > 0.6:
                infrastructure_at_risk.append(e)
                alerts.append({
                    "type": "FLOOD_INFRASTRUCTURE_RISK",
                    "entity_id": e.get("entity_id"),
                    "severity": "HIGH" if flood_score > 0.85 else "MEDIUM",
                    "asset_type": etype,
                    "flood_risk_score": flood_score,
                })

    return {
        "alerts": alerts,
        "inundated_zones": len(inundated_zones),
        "rising_zones": len(rising_zones),
        "infrastructure_at_risk": len(infrastructure_at_risk),
        "mission_recommended": len(inundated_zones) > 0 or len(rising_zones) > 0,
    }


def plan_flood_mission(situation: dict[str, Any], area: dict[str, Any]) -> dict[str, Any]:
    alerts = situation.get("alerts", [])
    types  = {a["type"] for a in alerts}

    if "FLOOD_EVACUATION_NEEDED" in types:
        return {
            "mission_type": "FLOOD_SAR",
            "priority": "IMMEDIATE",
            "pattern": "EXPANDING_SQUARE",
            "sensor_config": {"thermal": True, "optical": True, "sar": False},
            "notes": "Locate stranded persons in inundation zone for rescue coordination",
        }

    if "FLOOD_INUNDATION_DETECTED" in types:
        return {
            "mission_type": "FLOOD_SURVEY",
            "priority": "HIGH",
            "pattern": "PARALLEL_TRACK",
            "sensor_config": {"sar": True, "optical": True, "lidar": False},
            "notes": "Map inundation extent for emergency management EOC",
        }

    if "FLOOD_INFRASTRUCTURE_RISK" in types:
        return {
            "mission_type": "FLOOD_LOGISTICS",
            "priority": "HIGH",
            "pattern": "DIRECT",
            "sensor_config": {"optical": True, "thermal": False},
            "notes": "Assess bridge/road passability for supply convoys",
        }

    if "FLOOD_RISING_RATE" in types:
        return {
            "mission_type": "FLOOD_MONITOR",
            "priority": "MEDIUM",
            "pattern": "ORBIT",
            "sensor_config": {"optical": True, "thermal": False},
            "notes": "Continuous monitoring of gauge sites and levee integrity",
        }

    return {
        "mission_type": "FLOOD_SURVEY",
        "priority": "LOW",
        "pattern": "LAWNMOWER",
        "sensor_config": {"optical": True},
        "notes": "Precautionary area survey",
    }
