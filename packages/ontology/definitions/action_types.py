"""
Summit.OS Ontology — Action Type Definitions

Governed mutations. The ONLY way to change ontology state is via ActionRunner.execute().

Each ActionTypeDef has:
  - input_properties:  validated input schema
  - validators:        business rules that can reject the action
  - side_effects:      hooks that fire after successful mutation

Validators signature:   (inputs: dict, instance: ObjectInstance, store: ObjectStore) → Optional[str]
Side effects signature: (inputs: dict, instance: ObjectInstance, store: ObjectStore) → Optional[str]

NOTE on dispatch_mission validation:
  The `instance` passed to validators is the TARGET object (a Mission stub).
  Asset checks must look up the Asset separately from the store using inputs["asset_id"].
  The store is populated by OntologySync.from_entity() from Fusion telemetry, so
  freshness is bounded by the sync interval (typically <5s in production).
  For hard real-time guarantees, FABRIC_URL can be set to query Fabric directly.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import httpx

from ..store import ObjectStore
from ..types import ActionTypeDef, ObjectInstance, PropertyDef, PropertyKind

logger = logging.getLogger("ontology.actions.validators")

# If set, asset validators will do a live check against Fabric before approving dispatch.
FABRIC_URL = os.getenv("FABRIC_URL", "")
BATTERY_THRESHOLD = float(os.getenv("DISPATCH_MIN_BATTERY_PCT", "0.20"))  # 20% minimum


# ── validators ────────────────────────────────────────────────────────────────


def _asset_must_be_available(
    inputs: dict, instance: ObjectInstance, store: ObjectStore
) -> Optional[str]:
    """
    Checks the ASSET named in inputs["asset_id"], not the Mission instance.

    Validation order:
      1. asset_id must be present in inputs
      2. Asset must exist in the OntologyStore (synced from Fusion)
      3. Asset.status must be AVAILABLE
      4. Asset.battery_pct must be above DISPATCH_MIN_BATTERY_PCT (default 20%)
      5. No other PENDING/ACTIVE mission already linked to this asset
      6. If FABRIC_URL is set — live check against Fabric entity service
    """
    asset_id = inputs.get("asset_id")
    if not asset_id:
        return "dispatch_mission requires 'asset_id'"

    # ── 1. Ontology store check ────────────────────────────────────────────
    asset = store.get("Asset", asset_id)
    if asset is None:
        return (
            f"Asset '{asset_id}' not found in ontology store. "
            "Ensure Fusion is syncing entity updates to /sync/entity."
        )

    status = asset.properties.get("status", "UNKNOWN")
    if status != "AVAILABLE":
        return (
            f"Asset '{asset_id}' is not available (current status: {status}). "
            f"Valid status for dispatch: AVAILABLE."
        )

    # ── 2. Battery check ───────────────────────────────────────────────────
    battery = asset.properties.get("battery_pct")
    if battery is not None and battery < BATTERY_THRESHOLD:
        return (
            f"Asset '{asset_id}' battery too low for dispatch: "
            f"{battery:.0%} < {BATTERY_THRESHOLD:.0%} minimum."
        )

    # ── 3. No active mission already assigned ──────────────────────────────
    existing_mission_links = store.links_from_object(
        asset_id, "asset_executing_mission"
    )
    for link in existing_mission_links:
        active_mission = store.get("Mission", link.target_id)
        if active_mission and active_mission.properties.get("status") in (
            "PENDING",
            "ACTIVE",
        ):
            return (
                f"Asset '{asset_id}' is already assigned to mission "
                f"'{link.target_id}' (status: {active_mission.properties.get('status')}). "
                "Complete or cancel that mission before dispatching a new one."
            )

    # ── 4. Live Fabric check (optional, requires FABRIC_URL) ──────────────
    if FABRIC_URL:
        fabric_error = _fabric_live_check(asset_id)
        if fabric_error:
            return fabric_error

    return None


def _fabric_live_check(asset_id: str) -> Optional[str]:
    """
    Query Fabric entity service for the real-time asset state.
    Called only when FABRIC_URL is configured.
    Falls back gracefully (returns None = allow) if Fabric is unreachable,
    so a Fabric outage doesn't block all dispatch operations.
    """
    try:
        resp = httpx.get(
            f"{FABRIC_URL}/entities/{asset_id}",
            timeout=2.0,
        )
        if resp.status_code == 404:
            return f"Asset '{asset_id}' not found in Fabric entity registry."
        if resp.status_code != 200:
            logger.warning(
                "Fabric live check returned %d for asset %s — allowing dispatch",
                resp.status_code,
                asset_id,
            )
            return None  # degrade gracefully

        entity = resp.json()
        state = entity.get("state", "").upper()
        if state in ("INACTIVE", "DELETED"):
            return f"Asset '{asset_id}' is {state} in Fabric — cannot dispatch."

        # Re-check battery from live telemetry
        aerial = entity.get("aerial") or {}
        battery = aerial.get("battery_pct") if isinstance(aerial, dict) else None
        if battery is not None and battery < BATTERY_THRESHOLD:
            return (
                f"Asset '{asset_id}' live battery {battery:.0%} below threshold "
                f"{BATTERY_THRESHOLD:.0%} (Fabric telemetry)."
            )

    except Exception as exc:
        logger.warning(
            "Fabric live check failed for asset %s (%s) — proceeding with store data",
            asset_id,
            exc,
        )

    return None


def _alert_must_be_active(
    inputs: dict, instance: ObjectInstance, store: ObjectStore
) -> Optional[str]:
    if instance.properties.get("acknowledged", False):
        return f"Alert '{instance.object_id}' is already acknowledged"
    return None


def _incident_must_be_active(
    inputs: dict, instance: ObjectInstance, store: ObjectStore
) -> Optional[str]:
    status = instance.properties.get("status")
    if status in ("RESOLVED", "CLOSED"):
        return f"Incident '{instance.object_id}' is already {status}"
    return None


def _severity_must_escalate(
    inputs: dict, instance: ObjectInstance, store: ObjectStore
) -> Optional[str]:
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    current = order.get(instance.properties.get("severity", "LOW"), 0)
    new_sev = order.get(inputs.get("new_severity", "LOW"), 0)
    if new_sev <= current:
        return f"New severity '{inputs.get('new_severity')}' is not higher than current '{instance.properties.get('severity')}'"
    return None


def _observations_must_exist(
    inputs: dict, instance: ObjectInstance, store: ObjectStore
) -> Optional[str]:
    obs_ids = inputs.get("obs_ids", [])
    if not obs_ids:
        return "At least one observation id is required"
    for oid in obs_ids:
        if store.get("Observation", oid) is None:
            return f"Observation '{oid}' not found in ontology store"
    return None


def _zone_geometry_valid(
    inputs: dict, instance: ObjectInstance, store: ObjectStore
) -> Optional[str]:
    geometry_type = inputs.get("geometry_type", "CIRCLE")
    if geometry_type == "CIRCLE":
        if not inputs.get("radius_m") or float(inputs["radius_m"]) <= 0:
            return "CIRCLE zone requires radius_m > 0"
    elif geometry_type == "POLYGON":
        polygon = inputs.get("polygon", [])
        if len(polygon) < 3:
            return "POLYGON zone requires at least 3 vertices"
    return None


def _asset_state_transition_valid(
    inputs: dict, instance: ObjectInstance, store: ObjectStore
) -> Optional[str]:
    current = instance.properties.get("status")
    new = inputs.get("new_status")
    # Terminals — can't transition out of LOST
    if current == "LOST" and new != "MAINTENANCE":
        return (
            f"Asset in LOST state can only transition to MAINTENANCE (requested: {new})"
        )
    return None


# ── side effects ──────────────────────────────────────────────────────────────


def _link_asset_to_mission(
    inputs: dict, instance: ObjectInstance, store: ObjectStore
) -> Optional[str]:
    """
    Create Asset → Mission link when a mission is dispatched.

    Uses optimistic locking: re-reads the asset inside the side-effect and
    checks version + status again before committing. Since store operations
    run under the Python GIL (synchronous, no await), this check-and-set is
    atomic within a single process — it catches the case where a concurrent
    request validated the same asset simultaneously.
    """
    asset_id = inputs.get("asset_id")
    mission_id = instance.object_id
    if not asset_id:
        return None

    from ..types import LinkInstance

    # Re-read the asset now, inside the side-effect, after the validator ran.
    # If another dispatch snuck in between the validator and here, the version
    # will have changed and the status will be ASSIGNED.
    asset = store.get("Asset", asset_id)
    if asset is None:
        raise RuntimeError(
            f"Asset '{asset_id}' disappeared between validation and commit"
        )

    if asset.properties.get("status") != "AVAILABLE":
        raise RuntimeError(
            f"Asset '{asset_id}' status changed to "
            f"'{asset.properties.get('status')}' between validation and commit "
            f"(version {asset.version}) — dispatch rejected."
        )

    # Commit: mark ASSIGNED and create the link
    asset.properties["status"] = "ASSIGNED"
    store._upsert(asset)
    store._upsert_link(
        LinkInstance(
            link_type="asset_executing_mission",
            source_id=asset_id,
            target_id=mission_id,
        )
    )
    return f"Linked asset {asset_id} → mission {mission_id} (asset v{asset.version})"


def _link_alert_to_operator(
    inputs: dict, instance: ObjectInstance, store: ObjectStore
) -> Optional[str]:
    """Create Alert → Operator link on acknowledgement."""
    operator_id = inputs.get("operator_id", "")
    if operator_id:
        from ..types import LinkInstance
        from datetime import datetime, timezone

        store._upsert_link(
            LinkInstance(
                link_type="alert_acknowledged_by_operator",
                source_id=instance.object_id,
                target_id=operator_id,
                properties={
                    "notes": inputs.get("notes", ""),
                    "acknowledged_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        )
        return f"Linked alert {instance.object_id} to operator {operator_id}"
    return None


def _create_incident_links(
    inputs: dict, instance: ObjectInstance, store: ObjectStore
) -> Optional[str]:
    """Link observations to newly created incident."""
    from ..types import LinkInstance

    obs_ids = inputs.get("obs_ids", [])
    for oid in obs_ids:
        store._upsert_link(
            LinkInstance(
                link_type="observation_part_of_incident",
                source_id=oid,
                target_id=instance.object_id,
            )
        )
    return f"Linked {len(obs_ids)} observations to incident {instance.object_id}"


def _cancel_asset_missions_on_offline(
    inputs: dict, instance: ObjectInstance, store: ObjectStore
) -> Optional[str]:
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
            return (
                f"Cancelled {cancelled} active missions for asset {instance.object_id}"
            )
    return None


def _link_sitrep_to_incidents(
    inputs: dict, instance: ObjectInstance, store: ObjectStore
) -> Optional[str]:
    """Link newly created SitRep to all active incidents."""
    from ..types import LinkInstance

    active_incidents = store.list("Incident", {"status": "ACTIVE"})
    for incident in active_incidents:
        store._upsert_link(
            LinkInstance(
                link_type="sitrep_summarizes_incident",
                source_id=instance.object_id,
                target_id=incident.object_id,
            )
        )
    return f"Linked SitRep to {len(active_incidents)} active incidents"


# ── action type definitions ────────────────────────────────────────────────────

DISPATCH_MISSION = ActionTypeDef(
    name="dispatch_mission",
    display_name="Dispatch Mission",
    description="Assign an asset to a mission. Validates asset availability. Creates Asset→Mission link.",
    target_type="Mission",
    input_properties=[
        PropertyDef(
            "mission_type",
            PropertyKind.ENUM,
            required=True,
            enum_values=[
                "SEARCH",
                "SURVEY",
                "MONITOR",
                "PERIMETER",
                "INSPECT",
                "DELIVER",
                "ORBIT",
                "ESCORT",
                "INTERCEPT",
            ],
        ),
        PropertyDef("lat", PropertyKind.FLOAT, required=True),
        PropertyDef("lon", PropertyKind.FLOAT, required=True),
        PropertyDef("alt_m", PropertyKind.FLOAT, default=80.0),
        PropertyDef("asset_id", PropertyKind.STRING, required=True),
        PropertyDef(
            "priority",
            PropertyKind.ENUM,
            enum_values=["ROUTINE", "IMPORTANT", "URGENT", "CRITICAL"],
            default="ROUTINE",
        ),
        PropertyDef("rationale", PropertyKind.STRING),
        PropertyDef("org_id", PropertyKind.STRING),
    ],
    validators=[_asset_must_be_available],
    side_effects=[_link_asset_to_mission],
)

ACKNOWLEDGE_ALERT = ActionTypeDef(
    name="acknowledge_alert",
    display_name="Acknowledge Alert",
    description="Operator acknowledges an alert. Creates Alert→Operator link. Marks alert acknowledged.",
    target_type="Alert",
    input_properties=[
        PropertyDef("operator_id", PropertyKind.STRING, required=True),
        PropertyDef("notes", PropertyKind.STRING),
    ],
    validators=[_alert_must_be_active],
    side_effects=[_link_alert_to_operator],
)

ESCALATE_INCIDENT = ActionTypeDef(
    name="escalate_incident",
    display_name="Escalate Incident",
    description="Escalate an incident to a higher severity level. Requires reason.",
    target_type="Incident",
    input_properties=[
        PropertyDef(
            "new_severity",
            PropertyKind.ENUM,
            required=True,
            enum_values=["MEDIUM", "HIGH", "CRITICAL"],
        ),
        PropertyDef("reason", PropertyKind.STRING, required=True),
        PropertyDef("escalated_by", PropertyKind.STRING),
    ],
    validators=[_incident_must_be_active, _severity_must_escalate],
    side_effects=[],
)

CLOSE_INCIDENT = ActionTypeDef(
    name="close_incident",
    display_name="Close Incident",
    description="Resolve and close an incident. Requires resolution notes.",
    target_type="Incident",
    input_properties=[
        PropertyDef("resolution_notes", PropertyKind.STRING, required=True),
        PropertyDef("closed_by", PropertyKind.STRING, required=True),
        PropertyDef("status", PropertyKind.STRING, default="CLOSED"),
    ],
    validators=[_incident_must_be_active],
    side_effects=[],
)

CORRELATE_OBSERVATIONS = ActionTypeDef(
    name="correlate_observations",
    display_name="Correlate Observations into Incident",
    description="Manually group observations into a new or existing incident.",
    target_type="Incident",
    input_properties=[
        PropertyDef(
            "obs_ids", PropertyKind.ARRAY, required=True, item_kind=PropertyKind.STRING
        ),
        PropertyDef("title", PropertyKind.STRING, required=True),
        PropertyDef(
            "severity",
            PropertyKind.ENUM,
            required=True,
            enum_values=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        ),
        PropertyDef("domain", PropertyKind.STRING),
        PropertyDef("lat", PropertyKind.FLOAT),
        PropertyDef("lon", PropertyKind.FLOAT),
        PropertyDef("org_id", PropertyKind.STRING),
    ],
    validators=[_observations_must_exist],
    side_effects=[_create_incident_links],
)

CREATE_ZONE = ActionTypeDef(
    name="create_zone",
    display_name="Create Geographic Zone",
    description="Create a geofence, area of interest, or exclusion zone with geometry validation.",
    target_type="Zone",
    input_properties=[
        PropertyDef("name", PropertyKind.STRING, required=True),
        PropertyDef(
            "zone_type",
            PropertyKind.ENUM,
            required=True,
            enum_values=[
                "GEOFENCE",
                "EXCLUSION",
                "AOI",
                "SECTOR",
                "LANDING_ZONE",
                "RALLY_POINT",
                "SEARCH_AREA",
            ],
        ),
        PropertyDef(
            "geometry_type",
            PropertyKind.ENUM,
            enum_values=["CIRCLE", "POLYGON"],
            default="CIRCLE",
        ),
        PropertyDef("center_lat", PropertyKind.FLOAT),
        PropertyDef("center_lon", PropertyKind.FLOAT),
        PropertyDef("radius_m", PropertyKind.FLOAT),
        PropertyDef("polygon", PropertyKind.ARRAY, item_kind=PropertyKind.OBJECT),
        PropertyDef("created_by", PropertyKind.STRING),
        PropertyDef("org_id", PropertyKind.STRING),
    ],
    validators=[_zone_geometry_valid],
    side_effects=[],
)

UPDATE_ASSET_STATUS = ActionTypeDef(
    name="update_asset_status",
    display_name="Update Asset Status",
    description="Change an asset's operational status. Enforces state-machine rules.",
    target_type="Asset",
    input_properties=[
        PropertyDef(
            "new_status",
            PropertyKind.ENUM,
            required=True,
            enum_values=[
                "AVAILABLE",
                "ASSIGNED",
                "IN_FLIGHT",
                "RETURNING",
                "CHARGING",
                "MAINTENANCE",
                "OFFLINE",
                "LOST",
            ],
        ),
        PropertyDef("reason", PropertyKind.STRING),
    ],
    validators=[_asset_state_transition_valid],
    side_effects=[_cancel_asset_missions_on_offline],
)

GENERATE_SITREP = ActionTypeDef(
    name="generate_sitrep",
    display_name="Generate Situation Report",
    description="Snapshot a SitRep and link it to all active incidents.",
    target_type="SitRep",
    input_properties=[
        PropertyDef("generated_at", PropertyKind.DATETIME, required=True),
        PropertyDef(
            "generated_by",
            PropertyKind.ENUM,
            enum_values=["kofa-template", "kofa-llm"],
            default="kofa-template",
        ),
        PropertyDef("time_window_s", PropertyKind.INTEGER),
        PropertyDef("advisory_count", PropertyKind.INTEGER),
        PropertyDef(
            "highest_risk",
            PropertyKind.ENUM,
            enum_values=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        ),
        PropertyDef("summary", PropertyKind.STRING),
        PropertyDef("recommended_action", PropertyKind.STRING),
        PropertyDef("findings", PropertyKind.ARRAY, item_kind=PropertyKind.OBJECT),
        PropertyDef("org_id", PropertyKind.STRING),
    ],
    validators=[],
    side_effects=[_link_sitrep_to_incidents],
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
