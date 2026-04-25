"""
C2 Intel — Domain Models

C2-domain equivalents of Mira's signal/priority/source models.
These are the atomic types used throughout the c2-intel package.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class C2EventType(str, Enum):
    """Discrete event types observable in a distributed C2 environment."""

    # Comms state
    COMMS_DEGRADED = "comms_degraded"
    COMMS_RESTORED = "comms_restored"
    COMMS_DENIED = "comms_denied"

    # Link state (physical/RF layer — distinct from logical comms)
    LINK_DEGRADED = "link_degraded"
    LINK_LOST = "link_lost"

    # Asset state
    ASSET_ONLINE = "asset_online"
    ASSET_OFFLINE = "asset_offline"
    ASSET_DEGRADED = "asset_degraded"
    BATTERY_CRITICAL = "battery_critical"
    BATTERY_LOW = "battery_low"          # 15-30% — pre-critical window
    BATTERY_WARNING = "battery_warning"  # alias kept for backward compat
    RESOURCE_DEPLETED = "resource_depleted"

    # Sensor state
    SENSOR_LOSS = "sensor_loss"
    SENSOR_RESTORED = "sensor_restored"

    # Authority / mission
    AUTHORITY_DELEGATED = "authority_delegated"
    AUTHORITY_REVOKED = "authority_revoked"
    HANDOFF_INITIATED = "handoff_initiated"
    HANDOFF_COMPLETE = "handoff_complete"
    MISSION_ASSIGNED = "mission_assigned"
    MISSION_STARTED = "mission_started"
    MISSION_COMPLETED = "mission_completed"
    MISSION_ABORTED = "mission_aborted"
    TASK_CREATED = "task_created"
    TASK_APPROVED = "task_approved"
    TASK_FAILED = "task_failed"

    # Engagement state
    ENGAGEMENT_AUTHORIZED = "engagement_authorized"
    ENGAGEMENT_DENIED = "engagement_denied"
    ENGAGEMENT_COMPLETE = "engagement_complete"

    # Situational awareness
    ENTITY_DETECTED = "entity_detected"
    ENTITY_LOST = "entity_lost"
    ENTITY_REACQUIRED = "entity_reacquired"
    THREAT_IDENTIFIED = "threat_identified"
    THREAT_NEUTRALIZED = "threat_neutralized"
    GEOFENCE_BREACH = "geofence_breach"
    GEOFENCE_CLEARED = "geofence_cleared"

    # Environment
    WEATHER_ALERT = "weather_alert"
    AIRSPACE_CONFLICT = "airspace_conflict"

    # Node health
    NODE_DEGRADED = "node_degraded"
    NODE_FAILED = "node_failed"
    NODE_RECOVERED = "node_recovered"
    NODE_JOINED_MESH = "node_joined_mesh"
    NODE_LEFT_MESH = "node_left_mesh"

    # Mesh / peer
    PEER_OBSERVATION = "peer_observation"


class ObservationPriority(str, Enum):
    """
    Priority levels for C2 observations.

    Maps directly to Mira's SignalPriority so the priority engine
    can be ported without structural changes.
    """
    CRITICAL = "critical"  # P1: Immediate action (<2 min)
    HIGH = "high"          # P2: Urgent attention (<15 min)
    MEDIUM = "medium"      # P3: Monitor and track (<1 hr)
    LOW = "low"            # P4: Ambient awareness


class SensorSource(str, Enum):
    """Source that produced an observation."""

    # Direct sensors
    MAVLINK = "mavlink"
    ADS_B = "ads_b"
    RADAR = "radar"
    IFF = "iff"
    EO_IR = "eo_ir"
    SIGINT = "sigint"

    # Network / platform
    MESH_PEER = "mesh_peer"
    WORLD_STORE = "world_store"
    MQTT = "mqtt"

    # External / cooperative
    OPENSKY = "opensky"
    MANUAL = "manual"
    FUSED = "fused"
    UNKNOWN = "unknown"


class C2ActionType(str, Enum):
    """Actions the priority engine can trigger."""
    ESCALATE_COMMAND = "escalate_command"       # P1: push up chain of command
    DELEGATE_AUTHORITY = "delegate_authority"   # P1: trigger authority handoff
    SURFACE_TO_OPERATOR = "surface_to_operator" # P2: alert operator UI
    AUTO_TASK = "auto_task"                     # P2: generate task automatically
    GENERATE_BRIEF = "generate_brief"           # P2: produce situation brief
    BROADCAST_MESH = "broadcast_mesh"           # P1/P2: push to mesh peers
    LOG_ONLY = "log_only"                       # P3/P4: record, no alert


@dataclass
class C2Observation:
    """
    A discrete observation in the C2 domain.

    Equivalent to Mira's BuyingSignal — the atomic unit that flows through
    the deduplication, priority, and graph engines.
    """
    event_type: C2EventType
    node_id: str                            # node/asset this observation concerns
    title: str
    description: Optional[str] = None
    source: SensorSource = SensorSource.UNKNOWN
    confidence: float = 0.5                # 0-1
    score: int = 50                        # 0-100 composite urgency score
    priority: ObservationPriority = ObservationPriority.MEDIUM
    detected_at: Optional[datetime] = None
    event_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.now(timezone.utc)


__all__ = [
    "C2EventType",
    "ObservationPriority",
    "SensorSource",
    "C2ActionType",
    "C2Observation",
]
