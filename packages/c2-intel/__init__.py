"""
c2-intel — C2 Intelligence Engine

Core entity resolution, graph, priority, and deduplication for Heli.OS TA1.

Ported and remapped from Mira Signals' commercial intelligence engine.
All modifications live here; Mira Signals codebase is read-only.

Components:
  resolver  — entity deconfliction across sensor sources
  graph     — NetworkX relationship graph (effect radius, path finding, lookalike)
  priority  — composite condition scoring + TA1 simulation layer
  dedup     — observation deduplication across redundant sensor streams
"""

from .models import (
    C2EventType,
    ObservationPriority,
    SensorSource,
    C2ActionType,
    C2Observation,
)
from .resolver import C2EntityResolver, EntityMatch, normalize_entity_id
from .graph import C2EntityGraph, NodeType, EdgeType, get_c2_graph, build_graph_from_world_store
from .priority import C2PriorityMatrix, CONDITION_BASE_PRIORITY
from .dedup import ObservationDeduplicator, generate_observation_fingerprint

__all__ = [
    # Models
    "C2EventType", "ObservationPriority", "SensorSource", "C2ActionType", "C2Observation",
    # Resolver
    "C2EntityResolver", "EntityMatch", "normalize_entity_id",
    # Graph
    "C2EntityGraph", "NodeType", "EdgeType", "get_c2_graph", "build_graph_from_world_store",
    # Priority
    "C2PriorityMatrix", "CONDITION_BASE_PRIORITY",
    # Dedup
    "ObservationDeduplicator", "generate_observation_fingerprint",
]
