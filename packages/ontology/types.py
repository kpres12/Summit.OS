"""
Summit.OS Ontology — Base Type Definitions

Every concept in Summit.OS is expressed as one of three primitives:

  ObjectTypeDef   — the schema for a class of real-world things
                    (Asset, Mission, Alert, Incident, Zone, …)

  LinkTypeDef     — a typed, directed relationship between two object types
                    (Asset → Mission: "executing")

  ActionTypeDef   — a governed mutation; the ONLY way to change ontology state.
                    Includes input schema, validation rules, and side-effect hooks.

Runtime instances:

  ObjectInstance  — a live instance of an ObjectTypeDef (has a stable id + versioned properties)
  LinkInstance    — a live instance of a LinkTypeDef (source_id → target_id with optional props)
  AuditEntry      — immutable record of every ActionType execution
  ActionResult    — returned from ActionRunner.execute()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ── property kinds ─────────────────────────────────────────────────────────────


class PropertyKind(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    GEO = "geo_point"  # {"lat": float, "lon": float}
    ENUM = "enum"
    ARRAY = "array"
    OBJECT = "object"  # free-form dict


# ── property definition ────────────────────────────────────────────────────────


@dataclass
class PropertyDef:
    name: str
    kind: PropertyKind
    required: bool = False
    default: Any = None
    description: str = ""
    enum_values: List[str] = field(default_factory=list)  # for ENUM kind
    item_kind: Optional[PropertyKind] = None  # for ARRAY kind
    index: bool = False  # whether the store should index this property


# ── object type definition ─────────────────────────────────────────────────────


@dataclass
class ObjectTypeDef:
    """
    Schema for a class of real-world entities.
    Every ObjectInstance must conform to this definition.
    """

    name: str  # canonical name, e.g. "Asset"
    display_name: str  # human label, e.g. "Physical Asset"
    description: str
    properties: List[PropertyDef]
    primary_key: str = "id"  # which property is the stable, unique identifier
    icon: str = ""  # optional UI icon hint


# ── link type definition ───────────────────────────────────────────────────────


class Cardinality(str, Enum):
    ONE_TO_ONE = "ONE_TO_ONE"
    ONE_TO_MANY = "ONE_TO_MANY"
    MANY_TO_ONE = "MANY_TO_ONE"
    MANY_TO_MANY = "MANY_TO_MANY"


@dataclass
class LinkTypeDef:
    """
    A typed, directed relationship between two ObjectTypes.
    e.g. Asset → Mission with link_name "executing"
    """

    name: str  # snake_case, e.g. "asset_executing_mission"
    display_name: str
    description: str
    source_type: str  # ObjectTypeDef.name
    target_type: str  # ObjectTypeDef.name
    cardinality: Cardinality = Cardinality.MANY_TO_MANY
    properties: List[PropertyDef] = field(default_factory=list)


# ── action type definition ─────────────────────────────────────────────────────


@dataclass
class ActionTypeDef:
    """
    A governed, audited mutation.
    All state changes to ontology objects MUST go through an ActionType.

    Validators:   List[Callable[[dict, 'OntologyStore'], Optional[str]]]
                  Each returns None (ok) or a human-readable error string.

    Side effects: List[Callable[[dict, 'ObjectInstance', 'OntologyStore'], None]]
                  Called after validation passes and instance is updated.
    """

    name: str
    display_name: str
    description: str
    target_type: str  # ObjectTypeDef.name this action applies to
    input_properties: List[PropertyDef]
    validators: List[Callable] = field(default_factory=list)
    side_effects: List[Callable] = field(default_factory=list)
    requires_approval: bool = False


# ── runtime instances ──────────────────────────────────────────────────────────


@dataclass
class ObjectInstance:
    """A live instance of an ObjectTypeDef."""

    object_type: str  # ObjectTypeDef.name
    object_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def get(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)

    def to_dict(self) -> dict:
        return {
            "object_type": self.object_type,
            "object_id": self.object_id,
            "properties": self.properties,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class LinkInstance:
    """A live instance of a LinkTypeDef."""

    link_type: str  # LinkTypeDef.name
    source_id: str  # ObjectInstance.object_id
    target_id: str  # ObjectInstance.object_id
    link_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "link_type": self.link_type,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "link_id": self.link_id,
            "properties": self.properties,
            "created_at": self.created_at,
        }


# ── audit trail ───────────────────────────────────────────────────────────────


@dataclass
class AuditEntry:
    """Immutable record of an action execution."""

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_name: str = ""
    target_type: str = ""
    object_id: str = ""
    actor_id: str = ""  # operator or service that triggered it
    inputs: Dict[str, Any] = field(default_factory=dict)
    outcome: str = "success"  # "success" | "rejected" | "error"
    rejection_reason: str = ""
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "action_name": self.action_name,
            "target_type": self.target_type,
            "object_id": self.object_id,
            "actor_id": self.actor_id,
            "inputs": self.inputs,
            "outcome": self.outcome,
            "rejection_reason": self.rejection_reason,
            "ts": self.ts,
        }


# ── action result ──────────────────────────────────────────────────────────────


@dataclass
class ActionResult:
    success: bool
    object_instance: Optional[ObjectInstance] = None
    audit_entry: Optional[AuditEntry] = None
    error: str = ""
    side_effect_log: List[str] = field(default_factory=list)
