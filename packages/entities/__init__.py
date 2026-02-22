"""
Summit.OS Unified Entity Model

Every object in Summit.OS is an Entity. This module provides the canonical
Python dataclasses that mirror the protobuf definitions and are used by
all services for inter-service communication.

Usage:
    from packages.entities import Entity, EntityType, EntityDomain
    from packages.entities import GeoPoint, Kinematics, Provenance
"""

from packages.entities.core import (
    # Enums
    EntityType,
    EntityDomain,
    LifecycleState,
    TaskStatus,
    TaskPriority,
    TrackState,
    # Geometric types
    GeoPoint,
    GeoPolygon,
    GeoCircle,
    Vector3,
    Quaternion,
    BoundingBox,
    # Entity components
    Provenance,
    Kinematics,
    Relationship,
    # Domain data
    AerialData,
    GroundData,
    MaritimeData,
    FixedData,
    SensorData,
    # The Entity
    Entity,
    EntityBatch,
    # Telemetry
    SensorReading,
    TelemetryReport,
    # Tracks
    TrackObservation,
    TrackCorrelation,
    Track,
    # Tasks
    Objective,
    Waypoint,
    Task,
    TaskAssignment,
)

__all__ = [
    "EntityType", "EntityDomain", "LifecycleState", "TaskStatus",
    "TaskPriority", "TrackState",
    "GeoPoint", "GeoPolygon", "GeoCircle", "Vector3", "Quaternion",
    "BoundingBox",
    "Provenance", "Kinematics", "Relationship",
    "AerialData", "GroundData", "MaritimeData", "FixedData", "SensorData",
    "Entity", "EntityBatch",
    "SensorReading", "TelemetryReport",
    "TrackObservation", "TrackCorrelation", "Track",
    "Objective", "Waypoint", "Task", "TaskAssignment",
]
