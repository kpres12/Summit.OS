"""
Traffic & Transportation Domain — Situation Assessment and Mission Planning

Covers: post-disaster road network assessment, evacuation corridor management,
        vehicle counting, congestion detection, accident/incident monitoring.

Alert types:
  TRAFFIC_CONGESTION         — vehicle density above threshold for road class
  TRAFFIC_INCIDENT           — stopped vehicles + pedestrian activity pattern
  TRAFFIC_BLOCKED_ROUTE      — key evacuation/supply route impassable
  TRAFFIC_CONTRAFLOW_NEEDED  — unidirectional evacuation flow required
  TRAFFIC_BRIDGE_CAPACITY    — bridge approach vehicle density near capacity

Mission types:
  TRAFFIC_SURVEY      — aerial vehicle counting and flow assessment
  TRAFFIC_INCIDENT    — investigate stopped vehicle cluster
  EVACUATION_MONITOR  — monitor evacuation corridor compliance and flow
  ROUTE_ASSESS        — check road surface/bridge passability
"""

from __future__ import annotations

from typing import Any

# Vehicles-per-km thresholds by road class
_CONGESTION_THRESHOLDS = {
    "highway":    {"medium": 40, "high": 80},
    "arterial":   {"medium": 25, "high": 50},
    "local":      {"medium": 15, "high": 30},
    "bridge":     {"medium": 20, "high": 40},
    "default":    {"medium": 20, "high": 45},
}


def assess_traffic_situation(entities: list[dict[str, Any]]) -> dict[str, Any]:
    alerts = []
    blocked_routes = []
    congested_segments = []

    for e in entities:
        etype = e.get("type", "").upper()
        props = e.get("properties", e)

        if etype in ("ROAD_SEGMENT", "HIGHWAY", "ARTERIAL", "BRIDGE", "ROUTE"):
            road_class   = props.get("road_class", "default").lower()
            v_per_km     = float(props.get("vehicle_density_per_km", 0))
            passable     = bool(props.get("passable", True))
            speed_kmh    = float(props.get("avg_speed_kmh", 60))
            incident_flag = bool(props.get("incident_detected", False))
            thresholds    = _CONGESTION_THRESHOLDS.get(road_class, _CONGESTION_THRESHOLDS["default"])

            if not passable:
                blocked_routes.append(e)
                alerts.append({
                    "type": "TRAFFIC_BLOCKED_ROUTE",
                    "entity_id": e.get("entity_id"),
                    "severity": "HIGH",
                    "road_class": road_class,
                    "reason": props.get("blockage_reason", "unknown"),
                })

            elif v_per_km >= thresholds["high"]:
                congested_segments.append(e)
                alerts.append({
                    "type": "TRAFFIC_CONGESTION",
                    "entity_id": e.get("entity_id"),
                    "severity": "HIGH",
                    "road_class": road_class,
                    "vehicle_density": v_per_km,
                    "avg_speed_kmh": speed_kmh,
                })
            elif v_per_km >= thresholds["medium"]:
                congested_segments.append(e)
                alerts.append({
                    "type": "TRAFFIC_CONGESTION",
                    "entity_id": e.get("entity_id"),
                    "severity": "MEDIUM",
                    "road_class": road_class,
                    "vehicle_density": v_per_km,
                })

            if incident_flag:
                alerts.append({
                    "type": "TRAFFIC_INCIDENT",
                    "entity_id": e.get("entity_id"),
                    "severity": "HIGH",
                    "road_class": road_class,
                    "speed_kmh": speed_kmh,
                })

            if road_class == "bridge" and v_per_km >= thresholds["medium"]:
                alerts.append({
                    "type": "TRAFFIC_BRIDGE_CAPACITY",
                    "entity_id": e.get("entity_id"),
                    "severity": "MEDIUM" if v_per_km < thresholds["high"] else "HIGH",
                    "vehicle_density": v_per_km,
                })

        # Evacuation contraflow assessment
        elif etype in ("EVACUATION_ZONE", "EVACUATION_CORRIDOR"):
            outbound_pct = float(props.get("outbound_vehicle_pct", 50))
            if outbound_pct < 30:
                alerts.append({
                    "type": "TRAFFIC_CONTRAFLOW_NEEDED",
                    "entity_id": e.get("entity_id"),
                    "severity": "HIGH",
                    "outbound_pct": outbound_pct,
                    "notes": "Insufficient outbound flow — consider contraflow lanes",
                })

    return {
        "alerts": alerts,
        "blocked_routes": len(blocked_routes),
        "congested_segments": len(congested_segments),
        "mission_recommended": len(blocked_routes) > 0 or len(congested_segments) > 2,
    }


def plan_traffic_mission(situation: dict[str, Any], area: dict[str, Any]) -> dict[str, Any]:
    alerts = situation.get("alerts", [])
    types  = {a["type"] for a in alerts}

    if "TRAFFIC_INCIDENT" in types:
        return {
            "mission_type": "TRAFFIC_INCIDENT",
            "priority": "HIGH",
            "pattern": "DIRECT",
            "sensor_config": {"optical": True, "thermal": True},
            "notes": "Investigate incident — assess injuries and lane blockage",
        }

    if "TRAFFIC_BLOCKED_ROUTE" in types:
        return {
            "mission_type": "ROUTE_ASSESS",
            "priority": "HIGH",
            "pattern": "DIRECT",
            "sensor_config": {"optical": True, "lidar": False},
            "notes": "Assess blockage extent and identify alternate corridor",
        }

    if "TRAFFIC_CONTRAFLOW_NEEDED" in types:
        return {
            "mission_type": "EVACUATION_MONITOR",
            "priority": "HIGH",
            "pattern": "PARALLEL_TRACK",
            "sensor_config": {"optical": True},
            "notes": "Monitor evacuation compliance along corridor — report to EOC",
        }

    return {
        "mission_type": "TRAFFIC_SURVEY",
        "priority": "MEDIUM",
        "pattern": "LAWNMOWER",
        "sensor_config": {"optical": True},
        "notes": "Vehicle counting and flow mapping for EOC situational awareness",
    }
