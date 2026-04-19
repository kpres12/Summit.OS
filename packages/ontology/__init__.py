"""
Heli.OS Ontology

The semantic layer and operational backbone of Heli.OS.

Equivalent in function to Palantir Foundry's Ontology:
  - Object Types:  canonical real-world entities (Asset, Mission, Alert, Incident, …)
  - Link Types:    typed relationships between objects (Asset→executing→Mission)
  - Action Types:  the ONLY way to change state (dispatch_mission, acknowledge_alert, …)
  - Object Store:  live instances, indexed, event-emitting
  - Query:         filter + graph traversal + AI-ready semantic summary
  - Sync:          bridges raw service data into the ontology

Quick start:
    from packages.ontology import get_registry, get_store, get_action_runner, OntologyQuery

    reg   = get_registry()          # type definitions
    store = get_store()             # live instances
    runner = get_action_runner()    # governed mutations
    q     = OntologyQuery()         # query interface

    # Dispatch a mission (governed action)
    result = runner.execute(
        "dispatch_mission",
        object_id="",           # "" = create new Mission
        inputs={
            "mission_type": "SURVEY",
            "lat": 37.77,
            "lon": -122.41,
            "asset_id": "drone-001",
            "priority": "HIGH",
            "rationale": "Smoke reported by sensor grid",
        },
        actor_id="operator-42",
    )

    # Query live state
    alerts = q.unacknowledged_alerts()
    context = q.semantic_summary()    # inject into LLM prompt
"""

from .actions import get_action_runner, get_audit_log, recent_audit
from .query import OntologyQuery
from .registry import get_registry
from .store import get_store
from .sync import get_sync

__all__ = [
    "get_registry",
    "get_store",
    "get_action_runner",
    "get_audit_log",
    "recent_audit",
    "OntologyQuery",
    "get_sync",
]
