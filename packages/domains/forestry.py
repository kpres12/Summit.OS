"""
Forestry & Deforestation Domain — Situation Assessment and Mission Planning

Covers: illegal logging detection, deforestation monitoring, reforestation
        verification, forest fire risk, biodiversity corridor integrity.

Alert types:
  DEFORESTATION_DETECTED     — canopy loss exceeds threshold in AOI
  ILLEGAL_LOGGING_SUSPECTED  — fresh cut + vehicle tracks, no permit record
  FOREST_FIRE_RISK           — dry fuel load + high wind = elevated risk
  REFORESTATION_FAILURE      — planted area shows low survival rate
  BIODIVERSITY_CORRIDOR_GAP  — fragmentation detected in protected corridor

Mission types:
  DEFORESTATION_SURVEY   — canopy change mapping (NDVI comparison)
  LOGGING_INTERCEPT      — intercept/document active logging operation
  FIRE_RISK_PATROL       — patrol high-risk dry season zones
  REFORESTATION_ASSESS   — count survival trees, measure canopy cover
  CORRIDOR_INTEGRITY     — map fragmentation along biodiversity corridor
"""

from __future__ import annotations

from typing import Any

_DEFORESTATION_THRESHOLD_HA = 2.0   # hectares/day trigger
_NDVI_INTACT = 0.60                  # healthy closed-canopy NDVI
_NDVI_DEGRADED = 0.35               # degraded/sparse cover


def assess_forestry_situation(entities: list[dict[str, Any]]) -> dict[str, Any]:
    alerts = []
    logging_events = []
    deforestation_zones = []

    for e in entities:
        etype = e.get("type", "").upper()
        props = e.get("properties", e)

        if etype in ("CANOPY_CHANGE", "NDVI_ANOMALY", "FOREST_CHANGE"):
            loss_ha_day   = float(props.get("loss_rate_ha_day", 0))
            ndvi_current  = float(props.get("ndvi_current", 0.5))
            ndvi_prev     = float(props.get("ndvi_previous", 0.5))
            permit_valid  = bool(props.get("logging_permit_valid", False))
            vehicle_tracks = bool(props.get("vehicle_tracks_detected", False))

            if loss_ha_day > _DEFORESTATION_THRESHOLD_HA:
                deforestation_zones.append(e)
                sev = "HIGH" if loss_ha_day > 10 else "MEDIUM"
                alerts.append({
                    "type": "DEFORESTATION_DETECTED",
                    "entity_id": e.get("entity_id"),
                    "severity": sev,
                    "loss_ha_day": loss_ha_day,
                    "ndvi_delta": round(ndvi_current - ndvi_prev, 3),
                })
                if vehicle_tracks and not permit_valid:
                    logging_events.append(e)
                    alerts.append({
                        "type": "ILLEGAL_LOGGING_SUSPECTED",
                        "entity_id": e.get("entity_id"),
                        "severity": "HIGH",
                        "loss_ha_day": loss_ha_day,
                        "evidence": "vehicle_tracks_no_permit",
                    })

        elif etype in ("FIRE_RISK_ZONE", "DRY_FUEL_ZONE"):
            fwi = float(props.get("fire_weather_index", 0))  # FWI 0-100
            if fwi > 60:
                alerts.append({
                    "type": "FOREST_FIRE_RISK",
                    "entity_id": e.get("entity_id"),
                    "severity": "HIGH" if fwi > 80 else "MEDIUM",
                    "fire_weather_index": fwi,
                })

        elif etype in ("REFORESTATION_PLOT", "RESTORATION_AREA"):
            survival_pct = float(props.get("tree_survival_pct", 100))
            target_pct   = float(props.get("target_survival_pct", 70))
            if survival_pct < target_pct:
                alerts.append({
                    "type": "REFORESTATION_FAILURE",
                    "entity_id": e.get("entity_id"),
                    "severity": "MEDIUM",
                    "survival_pct": survival_pct,
                    "target_pct": target_pct,
                })

        elif etype in ("BIODIVERSITY_CORRIDOR", "WILDLIFE_CORRIDOR"):
            fragmentation = float(props.get("fragmentation_index", 0))
            if fragmentation > 0.4:
                alerts.append({
                    "type": "BIODIVERSITY_CORRIDOR_GAP",
                    "entity_id": e.get("entity_id"),
                    "severity": "MEDIUM" if fragmentation < 0.7 else "HIGH",
                    "fragmentation_index": fragmentation,
                })

    return {
        "alerts": alerts,
        "deforestation_zones": len(deforestation_zones),
        "logging_events": len(logging_events),
        "mission_recommended": len(deforestation_zones) > 0 or len(logging_events) > 0,
    }


def plan_forestry_mission(situation: dict[str, Any], area: dict[str, Any]) -> dict[str, Any]:
    alerts = situation.get("alerts", [])
    types  = {a["type"] for a in alerts}

    if "ILLEGAL_LOGGING_SUSPECTED" in types:
        return {
            "mission_type": "LOGGING_INTERCEPT",
            "priority": "HIGH",
            "pattern": "DIRECT",
            "sensor_config": {"optical": True, "thermal": True},
            "notes": "Document illegal activity — imagery for law enforcement referral",
        }

    if "DEFORESTATION_DETECTED" in types:
        return {
            "mission_type": "DEFORESTATION_SURVEY",
            "priority": "HIGH",
            "pattern": "LAWNMOWER",
            "sensor_config": {"optical": True, "multispectral": True},
            "notes": "Map canopy loss extent — generate NDVI change product",
        }

    if "FOREST_FIRE_RISK" in types:
        return {
            "mission_type": "FIRE_RISK_PATROL",
            "priority": "MEDIUM",
            "pattern": "PARALLEL_TRACK",
            "sensor_config": {"thermal": True, "optical": True},
            "notes": "Patrol high-FWI zone — early detection of ignition",
        }

    if "BIODIVERSITY_CORRIDOR_GAP" in types:
        return {
            "mission_type": "CORRIDOR_INTEGRITY",
            "priority": "MEDIUM",
            "pattern": "LAWNMOWER",
            "sensor_config": {"optical": True, "multispectral": True},
            "notes": "Map corridor fragmentation for conservation management",
        }

    return {
        "mission_type": "REFORESTATION_ASSESS",
        "priority": "LOW",
        "pattern": "LAWNMOWER",
        "sensor_config": {"optical": True, "multispectral": True},
        "notes": "Tree survival count and canopy cover assessment",
    }
