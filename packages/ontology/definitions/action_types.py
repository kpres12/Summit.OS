"""
Summit.OS Ontology — Action Type Definitions

Governed mutations. The ONLY way to change ontology state is via ActionRunner.execute().

Each ActionTypeDef has:
  - input_properties:  validated input schema
  - validators:        business rules that can reject the action
  - side_effects:      hooks that fire after successful mutation

Validators signature:   (inputs: dict, instance: ObjectInstance, store: ObjectStore) → Optional[str]
Side effects signature: (inputs: dict, instance: ObjectInstance, store: ObjectStore) → Optional[str]
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..store import ObjectStore
from ..types import ActionTypeDef, ObjectInstance, PropertyDef, PropertyKind


# ── validators ────────────────────────────────────────────────────────────────

def _asset_must_be_available(inputs: dict, instance: ObjectInstance, store: ObjectStore) -> Optional[str]:
    if instance.properties.get("status") not in ("AVAILABLE", None):
        return f"Asset '{instance.object_id}' is not available (status: {instance.properties.get('status')})"
    return None


def _alert_must_be_active(inputs: dict, instance: ObjectInstance, store: ObjectStore) -> Optional[str]:
    if instance.properties.get("acknowledged", False):
        return f"Alert '{instance.object_id}' is already acknowledged"
    return None


def _incident_must_be_active(inputs: dict, instance: ObjectInstance, store: ObjectStore) -> Optional[str]:
    status = instance.properties.get("status")
    if status in ("RESOLVED", "CLOSED"):
        return f"Incident '{instance.object_id}' is already {status}"
    return None


def _severity_must_escalate(inputs: dict, instance: ObjectInstance, store: ObjectStore) -> Optional[str]:
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    current = order.get(instance.properties.get("severity", "LOW"), 0)
    new_sev  = order.get(inputs.get("new_severity", "LOW"), 0)
    if new_sev <= current:
        return f"New severity '{inputs.get('new_severity')}' is not higher than current '{instance.properties.get('severity')}'"
    return None


def _observations_must_exist(inputs: dict, instance: ObjectInstance, store: ObjectStore) -> Optional[str]:
    obs_ids = inputs.get("obs_ids", [])
    if not obs_ids:
        return "At least one observation id is required"
    for oid in obs_ids:
        if store.get("Observation", oid) is None:
            return f"Observation '{oid}' not found in ontology store"
    return None


def _zone_geometry_valid(inputs: dict, instance: ObjectInstance, store: ObjectStore) -> Optional[str]:
    geometry_type = inputs.get("geometry_type", "CIRCLE")
    if geometry_type == "CIRCLE":
        if not inputs.get("radius_m") or float(inputs["radius_m"]) <= 0:
            return "CIRCLE zone requires radius_m > 0"
    elif geometry_type == "POLYGON":
        polygon = inputs.get("polygon", [])
        if len(polygon) < 3:
            return "POLYGON zone requires at least 3 vertices"
    return None


def _asset_state_transition_valid(inputs: dict, instance: ObjectInstance, store: ObjectStore) -> Optional[str]:
    current = instance.properties.get("status")
    new     = inputs.get("new_status")
    # Terminals — can't transition out of LOST
    if current == "LOST" and new != "MAINTENANCE":
        return f"Asset in LOST state can only transition to MAINTENANCE (requested: {new})"
    return None


# ── side effects ──────────────────────────────────────────────────────────────

def _link_asset_to_mission(inputs: dict, instance: ObjectInstance, store: ObjectStore) -> Optional[str]:
    """Create Asset → Mission link when a mission is dispatched."""
    asset_id   = inputs.get("asset_id")
    mission_id = instance.object_id
    if asset_id:
        from ..types import LinkInstance
        store._upsert_link(LinkInstance(
            link_type  = "asset_executing_mission",
            source_id  = asset_id,
            target_id  = mission_id,
        ))
        # Update asset status
        asset = store.get("Asset", asset_id)
        if asset:
            asset.properties["status"] = "ASSIGNED"
            store._upsert(asset)
        return f"Linked asset {asset_id} to mission {mission_id}"
    return None


def _link_alert_to_operator(inputs: dict, instance: ObjectInstance, store: ObjectStore) -> Optional[str]:
    """Create Alert → Operator link on acknowledgement."""
    operator_id = inputs.get("operator_id", "")
    if operator_id:
        from ..types import LinkInstance
        from datetime import datetime, timezone
        store._upsert_link(LinkInstance(
            link_type  = "alert_acknowledged_by_operator",
            source_id  = instance.object_id,
            target_id  = operator_id,
            properties = {
                "notes":            inputs.get("notes", ""),
                "acknowledged_at":  datetime.now(timezone.utc).isoformat(),
            },
        ))
        return f"Linked alert {instance.object_id} to operator {operator_id}"
    return None


def _create_incident_links(inputs: dict, instance: ObjectInstance, store: ObjectStore) -> Optional[str]:
    """Link observations to newly created incident."""
    from ..types import LinkInstance
    obs_ids = inputs.get("obs_ids", [])
    for oid in obs_ids:
        store._upsert_link(LinkInstance(
            link_type = "observation_part_of_incident",
            source_id = oid,
            target_id = instance.object_id,
        ))
    return f"Linked {len(obs_ids)} observations to incident {instance.object_id}"


def _cancel_asset_missions_on_offline(inputs: dict, instance: ObjectInstance, store: ObjectStore) -> Optional[str]:
    """If asset goes OFFLINE or MAINTENANCE, cancel its active missions."""
    new_status = inputs.get("new_status")
    if new_status in ("OFFLINE", "MAINTENANCE", "LOST"):
        links = store.links_from_object(instance.object_id, "asset_executing_mission")
        cancelled = 0
        for link in links:
            mission = store.get("Mission", link.target_id)
            if mission and mission.properties.get("status") in ("PENDING", "ACTIVE"):
                mission.properties["status"] = "CANCELLED"
                store._upsert(mission)
                cancelled += 1
        if cancelled:
            return f"Cancelled {cancelled} active missions for asset {instance.object_id}"
    return None


def _link_sitrep_to_incidents(inputs: dict, instance: ObjectInstance, store: ObjectStore) -> Optional[str]:
    """Link newly created SitRep to all active incidents."""
    from ..types import LinkInstance
    active_incidents = store.list("Incident", {"status": "ACTIVE"})
    for incident in active_incidents:
        store._upsert_link(LinkInstance(
            link_type = "sitrep_summarizes_incident",
            source_id = instance.object_id,
            target_id = incident.object_id,
        ))
    return f"Linked SitRep to {len(active_incidents)} active incidents"


# ── action type definitions ────────────────────────────────────────────────────

DISPATCH_MISSION = ActionTypeDef(
    name         = "dispatch_mission",
    display_name = "Dispatch Mission",
    description  = "Assign an asset to a mission. Validates asset availability. Creates Asset→Mission link.",
    target_type  = "Mission",
    input_properties = [
        PropertyDef("mission_type", PropertyKind.ENUM, required=True,
                    enum_values=["SEARCH", "SURVEY", "MONITOR", "PERIMETER",
                                 "INSPECT", "DELIVER", "ORBIT", "ESCORT", "INTERCEPT"]),
        PropertyDef("lat",          PropertyKind.FLOAT,  required=True),
        PropertyDef("lon",          PropertyKind.FLOAT,  required=True),
        PropertyDef("alt_m",        PropertyKind.FLOAT,  default=80.0),
        PropertyDef("asset_id",     PropertyKind.STRING, required=True),
        PropertyDef("priority",     PropertyKind.ENUM,
                    enum_values=["ROUTINE", "IMPORTANT", "URGENT", "CRITICAL"],
                    default="ROUTINE"),
        PropertyDef("rationale",    PropertyKind.STRING),
        PropertyDef("org_id",       PropertyKind.STRING),
    ],
    validators   = [_asset_must_be_available],
    side_effects = [_link_asset_to_mission],
)

ACKNOWLEDGE_ALERT = ActionTypeDef(
    name         = "acknowledge_alert",
    display_name = "Acknowledge Alert",
    description  = "Operator acknowledges an alert. Creates Alert→Operator link. Marks alert acknowledged.",
    target_type  = "Alert",
    input_properties = [
        PropertyDef("operator_id", PropertyKind.STRING, required=True),
        PropertyDef("notes",       PropertyKind.STRING),
    ],
    validators   = [_alert_must_be_active],
    side_effects = [_link_alert_to_operator],
)

ESCALATE_INCIDENT = ActionTypeDef(
    name         = "escalate_incident",
    display_name = "Escalate Incident",
    description  = "Escalate an incident to a higher severity level. Requires reason.",
    target_type  = "Incident",
    input_properties = [
        PropertyDef("new_severity", PropertyKind.ENUM, required=True,
                    enum_values=["MEDIUM", "HIGH", "CRITICAL"]),
        PropertyDef("reason",       PropertyKind.STRING, required=True),
        PropertyDef("escalated_by", PropertyKind.STRING),
    ],
    validators   = [_incident_must_be_active, _severity_must_escalate],
    side_effects = [],
)

CLOSE_INCIDENT = ActionTypeDef(
    name         = "close_incident",
    display_name = "Close Incident",
    description  = "Resolve and close an incident. Requires resolution notes.",
    target_type  = "Incident",
    input_properties = [
        PropertyDef("resolution_notes", PropertyKind.STRING, required=True),
        PropertyDef("closed_by",        PropertyKind.STRING, required=True),
        PropertyDef("status",           PropertyKind.STRING, default="CLOSED"),
    ],
    validators   = [_incident_must_be_active],
    side_effects = [],
)

CORRELATE_OBSERVATIONS = ActionTypeDef(
    name         = "correlate_observations",
    display_name = "Correlate Observations into Incident",
    description  = "Manually group observations into a new or existing incident.",
    target_type  = "Incident",
    input_properties = [
        PropertyDef("obs_ids",   PropertyKind.ARRAY,  required=True, item_kind=PropertyKind.STRING),
        PropertyDef("title",     PropertyKind.STRING, required=True),
        PropertyDef("severity",  PropertyKind.ENUM,   required=True,
                    enum_values=["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
        PropertyDef("domain",    PropertyKind.STRING),
        PropertyDef("lat",       PropertyKind.FLOAT),
        PropertyDef("lon",       PropertyKind.FLOAT),
        PropertyDef("org_id",    PropertyKind.STRING),
    ],
    validators   = [_observations_must_exist],
    side_effects = [_create_incident_links],
)

CREATE_ZONE = ActionTypeDef(
    name         = "create_zone",
    display_name = "Create Geographic Zone",
    description  = "Create a geofence, area of interest, or exclusion zone with geometry validation.",
    target_type  = "Zone",
    input_properties = [
        PropertyDef("name",          PropertyKind.STRING, required=True),
        PropertyDef("zone_type",     PropertyKind.ENUM,   required=True,
                    enum_values=["GEOFENCE", "EXCLUSION", "AOI", "SECTOR",
                                 "LANDING_ZONE", "RALLY_POINT", "SEARCH_AREA"]),
        PropertyDef("geometry_type", PropertyKind.ENUM,
                    enum_values=["CIRCLE", "POLYGON"], default="CIRCLE"),
        PropertyDef("center_lat",    PropertyKind.FLOAT),
        PropertyDef("center_lon",    PropertyKind.FLOAT),
        PropertyDef("radius_m",      PropertyKind.FLOAT),
        PropertyDef("polygon",       PropertyKind.ARRAY, item_kind=PropertyKind.OBJECT),
        PropertyDef("created_by",    PropertyKind.STRING),
        PropertyDef("org_id",        PropertyKind.STRING),
    ],
    validators   = [_zone_geometry_valid],
    side_effects = [],
)

UPDATE_ASSET_STATUS = ActionTypeDef(
    name         = "update_asset_status",
    display_name = "Update Asset Status",
    description  = "Change an asset's operational status. Enforces state-machine rules.",
    target_type  = "Asset",
    input_properties = [
        PropertyDef("new_status", PropertyKind.ENUM, required=True,
                    enum_values=["AVAILABLE", "ASSIGNED", "IN_FLIGHT", "RETURNING",
                                 "CHARGING", "MAINTENANCE", "OFFLINE", "LOST"]),
        PropertyDef("reason",     PropertyKind.STRING),
    ],
    validators   = [_asset_state_transition_valid],
    side_effects = [_cancel_asset_missions_on_offline],
)

GENERATE_SITREP = ActionTypeDef(
    name         = "generate_sitrep",
    display_name = "Generate Situation Report",
    description  = "Snapshot a SitRep and link it to all active incidents.",
    target_type  = "SitRep",
    input_properties = [
        PropertyDef("generated_at",       PropertyKind.DATETIME, required=True),
        PropertyDef("generated_by",       PropertyKind.ENUM,
                    enum_values=["kofa-template", "kofa-llm"],
                    default="kofa-template"),
        PropertyDef("time_window_s",      PropertyKind.INTEGER),
        PropertyDef("advisory_count",     PropertyKind.INTEGER),
        PropertyDef("highest_risk",       PropertyKind.ENUM,
                    enum_values=["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
        PropertyDef("summary",            PropertyKind.STRING),
        PropertyDef("recommended_action", PropertyKind.STRING),
        PropertyDef("findings",           PropertyKind.ARRAY, item_kind=PropertyKind.OBJECT),
        PropertyDef("org_id",             PropertyKind.STRING),
    ],
    validators   = [],
    side_effects = [_link_sitrep_to_incidents],
)


# ── registry ──────────────────────────────────────────────────────────────────

ALL_ACTION_TYPES = [
    DISPATCH_MISSION,
    ACKNOWLEDGE_ALERT,
    ESCALATE_INCIDENT,
    CLOSE_INCIDENT,
    CORRELATE_OBSERVATIONS,
    CREATE_ZONE,
    UPDATE_ASSET_STATUS,
    GENERATE_SITREP,
]
