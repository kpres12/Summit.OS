"""SQLAlchemy table definitions for the tasking service."""
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Boolean,
    Float,
    JSON,
    Table,
)

metadata = MetaData()

# Legacy task table (kept for backward compatibility)
tasks = Table(
    "tasks",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("task_id", String(128), nullable=False, unique=True),
    Column("asset_id", String(128)),
    Column("action", String(256)),
    Column("status", String(32)),  # PENDING, ACTIVE, COMPLETED, FAILED
    Column("created_at", DateTime(timezone=True)),
    Column("started_at", DateTime(timezone=True)),
    Column("completed_at", DateTime(timezone=True)),
    Column("org_id", String(128), nullable=True, index=True),
)

assets = Table(
    "assets",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("asset_id", String(128), nullable=False, unique=True),
    Column("type", String(64), nullable=True),
    Column("capabilities", JSON, nullable=True),
    Column("battery", Float, nullable=True),
    Column("link", String(32), nullable=True),
    Column("constraints", JSON, nullable=True),
    Column("updated_at", DateTime(timezone=True)),
    Column("org_id", String(128), nullable=True, index=True),
)

missions = Table(
    "missions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("mission_id", String(128), nullable=False, unique=True),
    Column("name", String(256), nullable=True),
    Column("objectives", JSON, nullable=True),
    Column("area", JSON, nullable=True),
    Column("policy_ok", Boolean, nullable=False, default=False),
    Column("status", String(32), nullable=False),  # PLANNING, ACTIVE, COMPLETED, FAILED
    Column("created_at", DateTime(timezone=True)),
    Column("started_at", DateTime(timezone=True)),
    Column("completed_at", DateTime(timezone=True)),
    Column("org_id", String(128), nullable=True, index=True),
)

mission_assignments = Table(
    "mission_assignments",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("mission_id", String(128), nullable=False),
    Column("asset_id", String(128), nullable=False),
    Column("plan", JSON, nullable=True),  # waypoints/patterns
    Column(
        "status", String(32), nullable=False
    ),  # ASSIGNED, DISPATCHED, ACTIVE, COMPLETED, FAILED
    Column("org_id", String(128), nullable=True, index=True),
)

# Tiered response tables
tiered_missions = Table(
    "tiered_missions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("mission_id", String(128), nullable=False, unique=True),
    Column("alert_id", String(128), nullable=False),
    Column(
        "current_tier", String(32), nullable=False
    ),  # tier_1_verify, tier_2_suppress, etc.
    Column("tier_1_status", String(32), nullable=True),
    Column("tier_2_status", String(32), nullable=True),
    Column("tier_3_status", String(32), nullable=True),
    Column("verification_result", JSON, nullable=True),
    Column("intervention_result", JSON, nullable=True),
    Column("escalation_reason", String(512), nullable=True),
    Column("next_tier_eta", Float, nullable=True),
    Column("assets_deployed", JSON, nullable=True),  # list of asset_ids
    Column("fire_threshold", JSON, nullable=True),
    Column("created_at", DateTime(timezone=True)),
    Column("updated_at", DateTime(timezone=True)),
    Column("org_id", String(128), nullable=True, index=True),
)

drone_boxes = Table(
    "drone_boxes",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("box_id", String(128), nullable=False, unique=True),
    Column("location", JSON, nullable=False),  # lat, lon
    Column("firefly_id", String(128), nullable=False),
    Column("emberwing_id", String(128), nullable=False),
    Column("status", String(32), nullable=False),  # READY, DEPLOYED, MAINTENANCE
    Column("launch_sequence_delay", Float, nullable=False, default=2.0),
    Column("recovery_timeout", Float, nullable=False, default=1800),
    Column("weather_limits", JSON, nullable=True),
    Column("updated_at", DateTime(timezone=True)),
    Column("org_id", String(128), nullable=True, index=True),
)

interventions = Table(
    "interventions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("intervention_id", String(128), nullable=False, unique=True),
    Column("mission_id", String(128), nullable=False),
    Column("asset_id", String(128), nullable=False),
    Column("target_location", JSON, nullable=False),  # lat, lon
    Column("payload_config", JSON, nullable=False),
    Column("intervention_plan", JSON, nullable=True),
    Column(
        "status", String(32), nullable=False
    ),  # PLANNED, DEPLOYED, COMPLETED, FAILED
    Column("effectiveness", Float, nullable=True),  # 0.0 - 1.0 success rate
    Column("deployed_at", DateTime(timezone=True)),
    Column("completed_at", DateTime(timezone=True)),
    Column("org_id", String(128), nullable=True, index=True),
)
