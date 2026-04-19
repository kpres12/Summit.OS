"""
Heli.OS Ontology — Link Type Definitions

All typed, directed relationships between object types.
Links are the semantic connective tissue of the ontology —
they express what things *mean* in relation to each other.

Naming convention: {source}_{verb}_{target}
e.g. "asset_executing_mission", "observation_triggered_alert"

Cardinality notation:
  ONE_TO_MANY  → one source can link to many targets
  MANY_TO_ONE  → many sources can link to one target
  MANY_TO_MANY → unrestricted
"""

from ..types import Cardinality, LinkTypeDef, PropertyDef, PropertyKind

# ── Asset links ───────────────────────────────────────────────────────────────

ASSET_EXECUTING_MISSION = LinkTypeDef(
    name="asset_executing_mission",
    display_name="Asset → executing → Mission",
    description="An asset is currently executing (or was assigned to) a mission.",
    source_type="Asset",
    target_type="Mission",
    cardinality=Cardinality.ONE_TO_MANY,
)

ASSET_MEMBER_OF_SWARM = LinkTypeDef(
    name="asset_member_of_swarm",
    display_name="Asset → member of → Swarm",
    description="An asset is a member of a coordinated swarm.",
    source_type="Asset",
    target_type="Swarm",
    cardinality=Cardinality.MANY_TO_ONE,
)

ASSET_PATROLLING_ZONE = LinkTypeDef(
    name="asset_patrolling_zone",
    display_name="Asset → patrolling → Zone",
    description="An asset is currently patrolling or monitoring a zone.",
    source_type="Asset",
    target_type="Zone",
    cardinality=Cardinality.MANY_TO_ONE,
)

ASSET_DETECTED_TRACK = LinkTypeDef(
    name="asset_detected_track",
    display_name="Asset → detected → Track",
    description="An asset's sensors contributed to a track.",
    source_type="Asset",
    target_type="Track",
    cardinality=Cardinality.MANY_TO_MANY,
)

# ── Observation links ─────────────────────────────────────────────────────────

OBSERVATION_TRIGGERED_ALERT = LinkTypeDef(
    name="observation_triggered_alert",
    display_name="Observation → triggered → Alert",
    description="An observation caused KOFA to generate an alert.",
    source_type="Observation",
    target_type="Alert",
    cardinality=Cardinality.MANY_TO_ONE,
)

OBSERVATION_PART_OF_INCIDENT = LinkTypeDef(
    name="observation_part_of_incident",
    display_name="Observation → part of → Incident",
    description="An observation has been correlated into an incident.",
    source_type="Observation",
    target_type="Incident",
    cardinality=Cardinality.MANY_TO_ONE,
)

OBSERVATION_FROM_ASSET = LinkTypeDef(
    name="observation_from_asset",
    display_name="Observation → from → Asset",
    description="The asset whose sensor produced this observation.",
    source_type="Observation",
    target_type="Asset",
    cardinality=Cardinality.MANY_TO_ONE,
)

# ── Alert links ───────────────────────────────────────────────────────────────

ALERT_ASSOCIATED_WITH_INCIDENT = LinkTypeDef(
    name="alert_associated_with_incident",
    display_name="Alert → associated with → Incident",
    description="An alert is associated with an incident.",
    source_type="Alert",
    target_type="Incident",
    cardinality=Cardinality.MANY_TO_ONE,
)

ALERT_ACKNOWLEDGED_BY_OPERATOR = LinkTypeDef(
    name="alert_acknowledged_by_operator",
    display_name="Alert → acknowledged by → Operator",
    description="An operator acknowledged (dismissed or acted on) this alert.",
    source_type="Alert",
    target_type="Operator",
    cardinality=Cardinality.MANY_TO_ONE,
    properties=[
        PropertyDef(
            "notes",
            PropertyKind.STRING,
            description="Operator notes at acknowledgement",
        ),
        PropertyDef("acknowledged_at", PropertyKind.DATETIME),
    ],
)

ALERT_ASSIGNED_TO_OPERATOR = LinkTypeDef(
    name="alert_assigned_to_operator",
    display_name="Alert → assigned to → Operator",
    description="An alert has been routed to a specific operator for action.",
    source_type="Alert",
    target_type="Operator",
    cardinality=Cardinality.MANY_TO_ONE,
)

# ── Mission links ──────────────────────────────────────────────────────────────

MISSION_COVERS_ZONE = LinkTypeDef(
    name="mission_covers_zone",
    display_name="Mission → covers → Zone",
    description="A mission's area of operation includes this zone.",
    source_type="Mission",
    target_type="Zone",
    cardinality=Cardinality.ONE_TO_MANY,
)

MISSION_ORIGINATED_FROM_OBSERVATION = LinkTypeDef(
    name="mission_originated_from_observation",
    display_name="Mission → originated from → Observation",
    description="The observation that triggered KOFA to generate this mission.",
    source_type="Mission",
    target_type="Observation",
    cardinality=Cardinality.ONE_TO_ONE,
)

MISSION_PART_OF_SWARM = LinkTypeDef(
    name="mission_part_of_swarm",
    display_name="Mission → part of → Swarm",
    description="A sector mission is part of a larger swarm operation.",
    source_type="Mission",
    target_type="Swarm",
    cardinality=Cardinality.MANY_TO_ONE,
)

# ── Operator links ────────────────────────────────────────────────────────────

OPERATOR_COMMANDING_MISSION = LinkTypeDef(
    name="operator_commanding_mission",
    display_name="Operator → commanding → Mission",
    description="An operator is the commanding authority for a mission.",
    source_type="Operator",
    target_type="Mission",
    cardinality=Cardinality.ONE_TO_MANY,
)

# ── Incident links ────────────────────────────────────────────────────────────

INCIDENT_WITHIN_ZONE = LinkTypeDef(
    name="incident_within_zone",
    display_name="Incident → within → Zone",
    description="An incident is occurring within a named zone.",
    source_type="Incident",
    target_type="Zone",
    cardinality=Cardinality.MANY_TO_ONE,
)

# ── SitRep links ──────────────────────────────────────────────────────────────

SITREP_SUMMARIZES_INCIDENT = LinkTypeDef(
    name="sitrep_summarizes_incident",
    display_name="SitRep → summarizes → Incident",
    description="A situation report covers this incident.",
    source_type="SitRep",
    target_type="Incident",
    cardinality=Cardinality.MANY_TO_MANY,
)

# ── Track links ───────────────────────────────────────────────────────────────

TRACK_CORRELATED_WITH_TRACK = LinkTypeDef(
    name="track_correlated_with_track",
    display_name="Track → correlated with → Track",
    description="Two tracks have been assessed as possibly representing the same entity.",
    source_type="Track",
    target_type="Track",
    cardinality=Cardinality.MANY_TO_MANY,
    properties=[
        PropertyDef(
            "correlation_score",
            PropertyKind.FLOAT,
            description="0–1 probability same entity",
        ),
    ],
)


# ── registry ──────────────────────────────────────────────────────────────────

ALL_LINK_TYPES = [
    ASSET_EXECUTING_MISSION,
    ASSET_MEMBER_OF_SWARM,
    ASSET_PATROLLING_ZONE,
    ASSET_DETECTED_TRACK,
    OBSERVATION_TRIGGERED_ALERT,
    OBSERVATION_PART_OF_INCIDENT,
    OBSERVATION_FROM_ASSET,
    ALERT_ASSOCIATED_WITH_INCIDENT,
    ALERT_ACKNOWLEDGED_BY_OPERATOR,
    ALERT_ASSIGNED_TO_OPERATOR,
    MISSION_COVERS_ZONE,
    MISSION_ORIGINATED_FROM_OBSERVATION,
    MISSION_PART_OF_SWARM,
    OPERATOR_COMMANDING_MISSION,
    INCIDENT_WITHIN_ZONE,
    SITREP_SUMMARIZES_INCIDENT,
    TRACK_CORRELATED_WITH_TRACK,
]
