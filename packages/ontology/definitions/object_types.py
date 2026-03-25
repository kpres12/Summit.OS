"""
Summit.OS Ontology — Object Type Definitions

These are the canonical real-world entities in Summit.OS.
Every row of data from every source eventually maps to one of these.

Object types and their key properties:

  Asset       — any physical platform (drone, sensor, ground vehicle, vessel)
  Mission     — a tasked operation dispatched to one or more assets
  Observation — a detection or sensing event (from ML inference or sensor)
  Alert       — an advisory surfaced to a human operator
  Incident    — a correlated cluster of observations representing a real event
  Zone        — a geographic area (geofence, AoI, exclusion zone, sector)
  Operator    — a human in the loop
  Swarm       — a coordinated group of assets on a decomposed mission
  SitRep      — a situation report snapshot at a point in time
  Track       — an entity being tracked (friendly/unknown/hostile)
"""

from ..types import ObjectTypeDef, PropertyDef, PropertyKind

# ── Asset ─────────────────────────────────────────────────────────────────────

ASSET = ObjectTypeDef(
    name         = "Asset",
    display_name = "Physical Asset",
    description  = "Any physical or virtual platform — drone, sensor, ground vehicle, vessel, or fixed station.",
    icon         = "drone",
    properties   = [
        PropertyDef("id",           PropertyKind.STRING,  required=True,  description="Stable unique asset identifier", index=True),
        PropertyDef("name",         PropertyKind.STRING,  required=True,  description="Human-readable name"),
        PropertyDef("asset_type",   PropertyKind.ENUM,    required=True,
                    enum_values=["UAV_MULTIROTOR", "UAV_FIXED_WING", "UAV_VTOL",
                                 "GROUND_VEHICLE", "VESSEL", "SENSOR_STATION",
                                 "GROUND_STATION", "UNKNOWN"],
                    description="Platform class", index=True),
        PropertyDef("domain",       PropertyKind.ENUM,    required=True,
                    enum_values=["AERIAL", "GROUND", "MARITIME", "FIXED", "CYBER"],
                    description="Operational domain", index=True),
        PropertyDef("status",       PropertyKind.ENUM,    required=True,
                    enum_values=["AVAILABLE", "ASSIGNED", "IN_FLIGHT", "RETURNING",
                                 "CHARGING", "MAINTENANCE", "OFFLINE", "LOST"],
                    default="AVAILABLE", index=True),
        PropertyDef("lat",          PropertyKind.FLOAT,   description="Last known latitude"),
        PropertyDef("lon",          PropertyKind.FLOAT,   description="Last known longitude"),
        PropertyDef("alt_m",        PropertyKind.FLOAT,   description="Altitude MSL (metres)"),
        PropertyDef("heading_deg",  PropertyKind.FLOAT,   description="True heading (degrees)"),
        PropertyDef("speed_mps",    PropertyKind.FLOAT,   description="Ground speed (m/s)"),
        PropertyDef("battery_pct",  PropertyKind.FLOAT,   description="Battery state-of-charge (0–1)"),
        PropertyDef("signal_db",    PropertyKind.FLOAT,   description="Uplink signal strength (dBm)"),
        PropertyDef("operator_id",  PropertyKind.STRING,  description="Assigned operator id", index=True),
        PropertyDef("org_id",       PropertyKind.STRING,  description="Owning organisation", index=True),
        PropertyDef("capabilities", PropertyKind.ARRAY,   item_kind=PropertyKind.STRING,
                    description="List of capabilities (e.g. thermal, lidar, loudspeaker)"),
        PropertyDef("firmware",     PropertyKind.STRING,  description="Firmware version"),
        PropertyDef("serial",       PropertyKind.STRING,  description="Hardware serial number"),
        PropertyDef("metadata",     PropertyKind.OBJECT,  description="Free-form adapter metadata"),
    ],
)

# ── Mission ───────────────────────────────────────────────────────────────────

MISSION = ObjectTypeDef(
    name         = "Mission",
    display_name = "Mission",
    description  = "A tasked operation dispatched to one or more assets.",
    icon         = "target",
    properties   = [
        PropertyDef("id",            PropertyKind.STRING, required=True,  index=True),
        PropertyDef("mission_type",  PropertyKind.ENUM,   required=True,
                    enum_values=["SEARCH", "SURVEY", "MONITOR", "PERIMETER",
                                 "INSPECT", "DELIVER", "ORBIT", "ESCORT", "INTERCEPT"],
                    index=True),
        PropertyDef("status",        PropertyKind.ENUM,   required=True,
                    enum_values=["PENDING", "ACTIVE", "COMPLETED", "FAILED",
                                 "CANCELLED", "PAUSED"],
                    default="PENDING", index=True),
        PropertyDef("priority",      PropertyKind.ENUM,
                    enum_values=["ROUTINE", "IMPORTANT", "URGENT", "CRITICAL"],
                    default="ROUTINE", index=True),
        PropertyDef("lat",           PropertyKind.FLOAT,  required=True),
        PropertyDef("lon",           PropertyKind.FLOAT,  required=True),
        PropertyDef("alt_m",         PropertyKind.FLOAT,  default=80.0),
        PropertyDef("asset_id",      PropertyKind.STRING, index=True),
        PropertyDef("swarm_id",      PropertyKind.STRING, index=True),
        PropertyDef("sector_id",     PropertyKind.STRING),
        PropertyDef("rationale",     PropertyKind.STRING),
        PropertyDef("created_at",    PropertyKind.DATETIME),
        PropertyDef("dispatched_at", PropertyKind.DATETIME),
        PropertyDef("completed_at",  PropertyKind.DATETIME),
        PropertyDef("outcome_prob",  PropertyKind.FLOAT,
                    description="KOFA predicted mission success probability (0–1)"),
        PropertyDef("org_id",        PropertyKind.STRING, index=True),
        PropertyDef("waypoints",     PropertyKind.ARRAY,  item_kind=PropertyKind.OBJECT),
        PropertyDef("metadata",      PropertyKind.OBJECT),
    ],
)

# ── Observation ───────────────────────────────────────────────────────────────

OBSERVATION = ObjectTypeDef(
    name         = "Observation",
    display_name = "Sensor Observation",
    description  = "A detection or sensing event — produced by ML inference or a sensor adapter.",
    icon         = "eye",
    properties   = [
        PropertyDef("id",          PropertyKind.STRING, required=True,  index=True),
        PropertyDef("class_label", PropertyKind.STRING, required=True,  index=True,
                    description="Detected class (e.g. 'smoke', 'person', 'vehicle')"),
        PropertyDef("confidence",  PropertyKind.FLOAT,  required=True,
                    description="Detection confidence (0–1)"),
        PropertyDef("risk_level",  PropertyKind.ENUM,
                    enum_values=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                    index=True),
        PropertyDef("domain",      PropertyKind.STRING, index=True,
                    description="KOFA domain classification (e.g. fire_smoke, person_sar)"),
        PropertyDef("lat",         PropertyKind.FLOAT),
        PropertyDef("lon",         PropertyKind.FLOAT),
        PropertyDef("alt_m",       PropertyKind.FLOAT),
        PropertyDef("sensor_id",   PropertyKind.STRING, index=True),
        PropertyDef("asset_id",    PropertyKind.STRING, index=True),
        PropertyDef("ts",          PropertyKind.DATETIME, index=True),
        PropertyDef("is_fp",       PropertyKind.BOOLEAN, default=False,
                    description="True if KOFA classified this as a false positive"),
        PropertyDef("features",    PropertyKind.OBJECT,
                    description="Raw 15-float feature vector from inference"),
        PropertyDef("metadata",    PropertyKind.OBJECT),
    ],
)

# ── Alert ─────────────────────────────────────────────────────────────────────

ALERT = ObjectTypeDef(
    name         = "Alert",
    display_name = "Operator Alert",
    description  = "An advisory generated by KOFA and surfaced to a human operator.",
    icon         = "bell",
    properties   = [
        PropertyDef("id",              PropertyKind.STRING,  required=True,  index=True),
        PropertyDef("severity",        PropertyKind.ENUM,    required=True,
                    enum_values=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                    index=True),
        PropertyDef("description",     PropertyKind.STRING,  required=True),
        PropertyDef("source",          PropertyKind.STRING,
                    description="Source entity/sensor that generated this alert"),
        PropertyDef("acknowledged",    PropertyKind.BOOLEAN, default=False, index=True),
        PropertyDef("acknowledged_by", PropertyKind.STRING),
        PropertyDef("acknowledged_at", PropertyKind.DATETIME),
        PropertyDef("incident_id",     PropertyKind.STRING, index=True),
        PropertyDef("ts",              PropertyKind.DATETIME, index=True),
        PropertyDef("org_id",          PropertyKind.STRING, index=True),
        PropertyDef("metadata",        PropertyKind.OBJECT),
    ],
)

# ── Incident ──────────────────────────────────────────────────────────────────

INCIDENT = ObjectTypeDef(
    name         = "Incident",
    display_name = "Incident",
    description  = "A correlated cluster of observations representing a real-world event.",
    icon         = "alert-triangle",
    properties   = [
        PropertyDef("id",               PropertyKind.STRING, required=True,  index=True),
        PropertyDef("title",            PropertyKind.STRING, required=True),
        PropertyDef("severity",         PropertyKind.ENUM,   required=True,
                    enum_values=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                    index=True),
        PropertyDef("status",           PropertyKind.ENUM,   required=True,
                    enum_values=["ACTIVE", "MONITORING", "RESOLVED", "CLOSED"],
                    default="ACTIVE", index=True),
        PropertyDef("domain",           PropertyKind.STRING, index=True,
                    description="KOFA domain (e.g. fire_smoke, flood_water)"),
        PropertyDef("lat",              PropertyKind.FLOAT),
        PropertyDef("lon",              PropertyKind.FLOAT),
        PropertyDef("first_seen",       PropertyKind.DATETIME),
        PropertyDef("last_seen",        PropertyKind.DATETIME),
        PropertyDef("obs_count",        PropertyKind.INTEGER, default=0),
        PropertyDef("resolution_notes", PropertyKind.STRING),
        PropertyDef("closed_by",        PropertyKind.STRING),
        PropertyDef("closed_at",        PropertyKind.DATETIME),
        PropertyDef("org_id",           PropertyKind.STRING, index=True),
        PropertyDef("metadata",         PropertyKind.OBJECT),
    ],
)

# ── Zone ──────────────────────────────────────────────────────────────────────

ZONE = ObjectTypeDef(
    name         = "Zone",
    display_name = "Geographic Zone",
    description  = "A named geographic area — geofence, area of interest, exclusion zone, or operational sector.",
    icon         = "map",
    properties   = [
        PropertyDef("id",            PropertyKind.STRING, required=True, index=True),
        PropertyDef("name",          PropertyKind.STRING, required=True),
        PropertyDef("zone_type",     PropertyKind.ENUM,   required=True,
                    enum_values=["GEOFENCE", "EXCLUSION", "AOI", "SECTOR",
                                 "LANDING_ZONE", "RALLY_POINT", "SEARCH_AREA"],
                    index=True),
        PropertyDef("geometry_type", PropertyKind.ENUM,
                    enum_values=["CIRCLE", "POLYGON"],
                    default="CIRCLE"),
        PropertyDef("center_lat",    PropertyKind.FLOAT),
        PropertyDef("center_lon",    PropertyKind.FLOAT),
        PropertyDef("radius_m",      PropertyKind.FLOAT),
        PropertyDef("polygon",       PropertyKind.ARRAY,  item_kind=PropertyKind.OBJECT,
                    description="List of {lat, lon} vertices for polygon zones"),
        PropertyDef("active",        PropertyKind.BOOLEAN, default=True, index=True),
        PropertyDef("created_by",    PropertyKind.STRING),
        PropertyDef("org_id",        PropertyKind.STRING, index=True),
        PropertyDef("metadata",      PropertyKind.OBJECT),
    ],
)

# ── Operator ──────────────────────────────────────────────────────────────────

OPERATOR = ObjectTypeDef(
    name         = "Operator",
    display_name = "Human Operator",
    description  = "A human in the loop — mission commander, sensor operator, or dispatch officer.",
    icon         = "user",
    properties   = [
        PropertyDef("id",                PropertyKind.STRING, required=True, index=True),
        PropertyDef("name",              PropertyKind.STRING, required=True),
        PropertyDef("role",              PropertyKind.ENUM,
                    enum_values=["OPS", "COMMAND", "DEV", "ADMIN"],
                    default="OPS", index=True),
        PropertyDef("org_id",            PropertyKind.STRING, index=True),
        PropertyDef("online",            PropertyKind.BOOLEAN, default=False, index=True),
        PropertyDef("active_since",      PropertyKind.DATETIME),
        PropertyDef("assigned_missions", PropertyKind.ARRAY, item_kind=PropertyKind.STRING),
    ],
)

# ── Swarm ─────────────────────────────────────────────────────────────────────

SWARM = ObjectTypeDef(
    name         = "Swarm",
    display_name = "Drone Swarm",
    description  = "A coordinated group of UAVs executing a spatially decomposed mission.",
    icon         = "grid",
    properties   = [
        PropertyDef("id",           PropertyKind.STRING, required=True, index=True),
        PropertyDef("mission_type", PropertyKind.STRING, required=True, index=True),
        PropertyDef("status",       PropertyKind.ENUM,
                    enum_values=["FORMING", "ACTIVE", "COMPLETED", "ABORTED"],
                    default="FORMING", index=True),
        PropertyDef("n_assets",     PropertyKind.INTEGER, required=True),
        PropertyDef("sector_count", PropertyKind.INTEGER),
        PropertyDef("center_lat",   PropertyKind.FLOAT),
        PropertyDef("center_lon",   PropertyKind.FLOAT),
        PropertyDef("radius_m",     PropertyKind.FLOAT),
        PropertyDef("created_at",   PropertyKind.DATETIME),
        PropertyDef("org_id",       PropertyKind.STRING, index=True),
    ],
)

# ── SitRep ────────────────────────────────────────────────────────────────────

SITREP = ObjectTypeDef(
    name         = "SitRep",
    display_name = "Situation Report",
    description  = "A structured situation report snapshot generated by KOFA.",
    icon         = "file-text",
    properties   = [
        PropertyDef("id",                  PropertyKind.STRING,  required=True, index=True),
        PropertyDef("generated_at",        PropertyKind.DATETIME, required=True),
        PropertyDef("generated_by",        PropertyKind.ENUM,
                    enum_values=["kofa-template", "kofa-llm"]),
        PropertyDef("time_window_s",       PropertyKind.INTEGER),
        PropertyDef("advisory_count",      PropertyKind.INTEGER),
        PropertyDef("highest_risk",        PropertyKind.ENUM,
                    enum_values=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                    index=True),
        PropertyDef("summary",             PropertyKind.STRING),
        PropertyDef("recommended_action",  PropertyKind.STRING),
        PropertyDef("findings",            PropertyKind.ARRAY, item_kind=PropertyKind.OBJECT),
        PropertyDef("org_id",              PropertyKind.STRING, index=True),
    ],
)

# ── Track ─────────────────────────────────────────────────────────────────────

TRACK = ObjectTypeDef(
    name         = "Track",
    display_name = "Entity Track",
    description  = "A sensor-fused track of a moving entity (friendly, unknown, or hostile).",
    icon         = "crosshair",
    properties   = [
        PropertyDef("id",                   PropertyKind.STRING, required=True, index=True),
        PropertyDef("state",                PropertyKind.ENUM,
                    enum_values=["TENTATIVE", "CONFIRMED", "COASTING", "DELETED"],
                    default="TENTATIVE", index=True),
        PropertyDef("class_label",          PropertyKind.STRING, index=True),
        PropertyDef("confidence",           PropertyKind.FLOAT),
        PropertyDef("lat",                  PropertyKind.FLOAT),
        PropertyDef("lon",                  PropertyKind.FLOAT),
        PropertyDef("alt_m",               PropertyKind.FLOAT),
        PropertyDef("speed_mps",            PropertyKind.FLOAT),
        PropertyDef("heading_deg",          PropertyKind.FLOAT),
        PropertyDef("first_seen",           PropertyKind.DATETIME),
        PropertyDef("last_seen",            PropertyKind.DATETIME),
        PropertyDef("hit_count",            PropertyKind.INTEGER, default=0),
        PropertyDef("contributing_sensors", PropertyKind.ARRAY, item_kind=PropertyKind.STRING),
        PropertyDef("org_id",               PropertyKind.STRING, index=True),
        PropertyDef("metadata",             PropertyKind.OBJECT),
    ],
)


# ── registry ──────────────────────────────────────────────────────────────────

ALL_OBJECT_TYPES = [ASSET, MISSION, OBSERVATION, ALERT, INCIDENT, ZONE, OPERATOR, SWARM, SITREP, TRACK]
