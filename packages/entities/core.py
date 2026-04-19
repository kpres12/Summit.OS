"""
Heli.OS Core Entity Dataclasses

Pure-Python dataclasses mirroring the protobuf definitions.  Every field
supports JSON serialization via .to_dict() / .from_dict() for backward
compatibility with the existing REST services, while the protobuf wire
format is used for gRPC inter-service calls.
"""

from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


# ─── Enums ───────────────────────────────────────────────────


class EntityType(str, Enum):
    ASSET = "ASSET"
    TRACK = "TRACK"
    OBSERVATION = "OBSERVATION"
    ALERT = "ALERT"
    GEOFENCE = "GEOFENCE"
    MISSION = "MISSION"
    OBJECTIVE = "OBJECTIVE"
    SENSOR = "SENSOR"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    ZONE = "ZONE"


class EntityDomain(str, Enum):
    AERIAL = "AERIAL"
    GROUND = "GROUND"
    MARITIME = "MARITIME"
    FIXED = "FIXED"
    CYBER = "CYBER"


class LifecycleState(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    TENTATIVE = "TENTATIVE"
    COASTING = "COASTING"
    COMPLETED = "COMPLETED"
    DELETED = "DELETED"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PENDING_APPROVAL = "PENDING_APPROVAL"


class TaskPriority(str, Enum):
    ROUTINE = "ROUTINE"
    IMPORTANT = "IMPORTANT"
    URGENT = "URGENT"
    CRITICAL = "CRITICAL"


class TrackState(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    COASTING = "COASTING"
    DELETED = "DELETED"


# ─── Geometric Types ─────────────────────────────────────────


@dataclass
class GeoPoint:
    latitude: float = 0.0
    longitude: float = 0.0
    altitude_msl: float = 0.0
    altitude_agl: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeoPoint":
        return cls(
            **{
                k: d[k]
                for k in ("latitude", "longitude", "altitude_msl", "altitude_agl")
                if k in d
            }
        )


@dataclass
class GeoPolygon:
    vertices: List[GeoPoint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"vertices": [v.to_dict() for v in self.vertices]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeoPolygon":
        return cls(vertices=[GeoPoint.from_dict(v) for v in d.get("vertices", [])])


@dataclass
class GeoCircle:
    center: Optional[GeoPoint] = None
    radius_m: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": self.center.to_dict() if self.center else None,
            "radius_m": self.radius_m,
        }


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Quaternion:
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class BoundingBox:
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0


# ─── Entity Components ───────────────────────────────────────


@dataclass
class Provenance:
    source_id: str = ""
    source_type: str = ""
    org_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CovarianceMatrix:
    """6x6 covariance stored as upper-triangle (21 elements)."""

    values: List[float] = field(default_factory=lambda: [0.0] * 21)


@dataclass
class Kinematics:
    position: Optional[GeoPoint] = None
    velocity: Optional[Vector3] = None
    acceleration: Optional[Vector3] = None
    heading_deg: float = 0.0
    speed_mps: float = 0.0
    climb_rate: float = 0.0
    orientation: Optional[Quaternion] = None
    position_covariance: Optional[CovarianceMatrix] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.position:
            d["position"] = self.position.to_dict()
        if self.velocity:
            d["velocity"] = self.velocity.to_dict()
        d["heading_deg"] = self.heading_deg
        d["speed_mps"] = self.speed_mps
        d["climb_rate"] = self.climb_rate
        return d


@dataclass
class Relationship:
    entity_id: str = ""
    relationship: str = ""  # "parent", "child", "correlated_with", "assigned_to"


# ─── Domain Data ─────────────────────────────────────────────


@dataclass
class AerialData:
    altitude_agl: float = 0.0
    altitude_msl: float = 0.0
    airspeed_mps: float = 0.0
    flight_mode: str = ""
    battery_pct: float = 0.0
    link_quality: str = ""


@dataclass
class GroundData:
    terrain_type: str = ""
    wheel_speed: float = 0.0
    obstacle_clearance_m: float = 0.0


@dataclass
class MaritimeData:
    heading_deg: float = 0.0
    draft_m: float = 0.0
    sea_state: str = ""
    ais_mmsi: str = ""


@dataclass
class FixedData:
    sensor_ids: List[str] = field(default_factory=list)
    structure_type: str = ""


@dataclass
class SensorData:
    modality: str = ""  # "camera", "radar", "thermal", "lidar", "adsb"
    fov_deg: float = 0.0
    range_m: float = 0.0
    accuracy_m: float = 0.0
    refresh_hz: float = 0.0


# ─── The Entity ──────────────────────────────────────────────


@dataclass
class Entity:
    """The universal data type in Heli.OS.  Everything is an Entity."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: EntityType = EntityType.ASSET
    domain: EntityDomain = EntityDomain.AERIAL
    state: LifecycleState = LifecycleState.ACTIVE
    name: str = ""
    class_label: str = ""
    confidence: float = 1.0

    kinematics: Optional[Kinematics] = None
    provenance: Optional[Provenance] = None

    relationships: List[Relationship] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    ttl_seconds: int = 0

    # Domain data (at most one populated)
    aerial: Optional[AerialData] = None
    ground: Optional[GroundData] = None
    maritime: Optional[MaritimeData] = None
    fixed: Optional[FixedData] = None
    sensor: Optional[SensorData] = None

    # Alert fields
    severity: str = ""
    description: str = ""

    # Geofence fields
    boundary: Optional[GeoPolygon] = None
    circle_bound: Optional[GeoCircle] = None

    # Mission fields
    mission_status: str = ""
    assigned_asset_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict (for REST backward compat)."""
        d: Dict[str, Any] = {
            "id": self.id,
            "entity_type": self.entity_type.value,
            "domain": self.domain.value,
            "state": self.state.value,
            "name": self.name,
            "class_label": self.class_label,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "ttl_seconds": self.ttl_seconds,
        }
        if self.kinematics:
            d["kinematics"] = self.kinematics.to_dict()
        if self.provenance:
            d["provenance"] = self.provenance.to_dict()
        if self.relationships:
            d["relationships"] = [asdict(r) for r in self.relationships]
        if self.aerial:
            d["aerial"] = asdict(self.aerial)
        if self.ground:
            d["ground"] = asdict(self.ground)
        if self.maritime:
            d["maritime"] = asdict(self.maritime)
        if self.fixed:
            d["fixed"] = asdict(self.fixed)
        if self.sensor:
            d["sensor"] = asdict(self.sensor)
        if self.severity:
            d["severity"] = self.severity
        if self.description:
            d["description"] = self.description
        if self.mission_status:
            d["mission_status"] = self.mission_status
        if self.assigned_asset_ids:
            d["assigned_asset_ids"] = self.assigned_asset_ids
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Entity":
        """Deserialize from JSON dict."""
        kwargs: Dict[str, Any] = {}
        kwargs["id"] = d.get("id", str(uuid.uuid4()))
        if "entity_type" in d:
            kwargs["entity_type"] = EntityType(d["entity_type"])
        if "domain" in d:
            kwargs["domain"] = EntityDomain(d["domain"])
        if "state" in d:
            kwargs["state"] = LifecycleState(d["state"])
        for simple in (
            "name",
            "class_label",
            "confidence",
            "ttl_seconds",
            "severity",
            "description",
            "mission_status",
        ):
            if simple in d:
                kwargs[simple] = d[simple]
        if "metadata" in d:
            kwargs["metadata"] = d["metadata"]
        if "assigned_asset_ids" in d:
            kwargs["assigned_asset_ids"] = d["assigned_asset_ids"]
        if "kinematics" in d and d["kinematics"]:
            k = d["kinematics"]
            pos = GeoPoint.from_dict(k["position"]) if "position" in k else None
            vel = Vector3(**k["velocity"]) if "velocity" in k else None
            kwargs["kinematics"] = Kinematics(
                position=pos,
                velocity=vel,
                heading_deg=k.get("heading_deg", 0),
                speed_mps=k.get("speed_mps", 0),
                climb_rate=k.get("climb_rate", 0),
            )
        if "provenance" in d and d["provenance"]:
            kwargs["provenance"] = Provenance(**d["provenance"])
        if "aerial" in d and d["aerial"]:
            kwargs["aerial"] = AerialData(**d["aerial"])
        if "ground" in d and d["ground"]:
            kwargs["ground"] = GroundData(**d["ground"])
        if "maritime" in d and d["maritime"]:
            kwargs["maritime"] = MaritimeData(**d["maritime"])
        return cls(**kwargs)


@dataclass
class EntityBatch:
    entities: List[Entity] = field(default_factory=list)
    as_of: float = field(default_factory=time.time)


# ─── Telemetry ───────────────────────────────────────────────


@dataclass
class SensorReading:
    sensor_id: str = ""
    modality: str = ""
    value: float = 0.0
    unit: str = ""
    sampled_at: float = field(default_factory=time.time)


@dataclass
class TelemetryReport:
    entity_id: str = ""
    org_id: str = ""
    timestamp: float = field(default_factory=time.time)

    position: Optional[GeoPoint] = None
    velocity: Optional[Vector3] = None
    heading_deg: float = 0.0
    speed_mps: float = 0.0

    battery_pct: float = 0.0
    battery_voltage: float = 0.0
    signal_rssi_dbm: float = 0.0
    link_quality_pct: float = 0.0
    link_type: str = ""

    sensors: List[SensorReading] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ─── Tracks ──────────────────────────────────────────────────


@dataclass
class TrackObservation:
    observation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sensor_id: str = ""
    sensor_type: str = ""
    measured_position: Optional[GeoPoint] = None
    bearing_deg: float = 0.0
    range_m: float = 0.0
    confidence: float = 0.0
    observed_at: float = field(default_factory=time.time)
    detection_bbox: Optional[BoundingBox] = None
    class_label: str = ""


@dataclass
class TrackCorrelation:
    track_id_a: str = ""
    track_id_b: str = ""
    mahalanobis_dist: float = 0.0
    correlation_score: float = 0.0
    is_same_object: bool = False
    evaluated_at: float = field(default_factory=time.time)


@dataclass
class Track:
    track_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: TrackState = TrackState.TENTATIVE
    class_label: str = ""
    confidence: float = 0.0

    kinematics: Optional[Kinematics] = None
    observations: List[TrackObservation] = field(default_factory=list)
    correlations: List[TrackCorrelation] = field(default_factory=list)

    hits: int = 0
    misses: int = 0
    age_ticks: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    predicted_at: float = 0.0

    org_id: str = ""
    contributing_sensor_ids: List[str] = field(default_factory=list)
    entity_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "track_id": self.track_id,
            "state": self.state.value,
            "class_label": self.class_label,
            "confidence": self.confidence,
            "hits": self.hits,
            "misses": self.misses,
            "age_ticks": self.age_ticks,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "contributing_sensor_ids": self.contributing_sensor_ids,
        }
        if self.kinematics:
            d["kinematics"] = self.kinematics.to_dict()
        return d


# ─── Tasks ───────────────────────────────────────────────────


@dataclass
class Objective:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    type: str = ""  # "survey", "monitor", "intercept", "deliver", "inspect"
    location: Optional[GeoPoint] = None
    area: Optional[GeoPolygon] = None
    radius_m: float = 0.0
    parameters: Dict[str, str] = field(default_factory=dict)


@dataclass
class Waypoint:
    position: Optional[GeoPoint] = None
    speed_mps: float = 5.0
    action: str = "WAYPOINT"
    loiter_s: float = 0.0
    params: Dict[str, str] = field(default_factory=dict)


@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mission_id: str = ""
    asset_id: str = ""
    org_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.ROUTINE

    objective: Optional[Objective] = None
    plan: List[Waypoint] = field(default_factory=list)

    behavior_tree: str = ""  # BT template name

    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    failure_reason: str = ""

    policy_approved: bool = False
    policy_violations: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class TaskAssignment:
    task_id: str = ""
    asset_id: str = ""
    plan: List[Waypoint] = field(default_factory=list)
    behavior_tree: str = ""
