"""
Summit.OS Ontology Registry

The single authoritative catalog of all ObjectTypes, LinkTypes, and ActionTypes.
All definitions are registered at import time from packages/ontology/definitions/.

Usage:
    from packages.ontology.registry import get_registry

    reg = get_registry()
    obj_type = reg.get_object_type("Asset")
    link_type = reg.get_link_type("asset_executing_mission")
    action    = reg.get_action("dispatch_mission")
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .types import ActionTypeDef, LinkTypeDef, ObjectTypeDef

logger = logging.getLogger("ontology.registry")


class OntologyRegistry:
    """
    Thread-safe catalog of all type definitions.
    Populated once at startup; read-only at runtime.
    """

    def __init__(self) -> None:
        self._object_types: Dict[str, ObjectTypeDef]  = {}
        self._link_types:   Dict[str, LinkTypeDef]    = {}
        self._action_types: Dict[str, ActionTypeDef]  = {}

    # ── registration ──────────────────────────────────────────────────────────

    def register_object_type(self, defn: ObjectTypeDef) -> None:
        if defn.name in self._object_types:
            logger.warning("ObjectType '%s' already registered — overwriting", defn.name)
        self._object_types[defn.name] = defn
        logger.debug("Registered ObjectType: %s", defn.name)

    def register_link_type(self, defn: LinkTypeDef) -> None:
        if defn.name in self._link_types:
            logger.warning("LinkType '%s' already registered — overwriting", defn.name)
        self._link_types[defn.name] = defn
        logger.debug("Registered LinkType: %s", defn.name)

    def register_action_type(self, defn: ActionTypeDef) -> None:
        if defn.name in self._action_types:
            logger.warning("ActionType '%s' already registered — overwriting", defn.name)
        self._action_types[defn.name] = defn
        logger.debug("Registered ActionType: %s", defn.name)

    # ── lookup ────────────────────────────────────────────────────────────────

    def get_object_type(self, name: str) -> Optional[ObjectTypeDef]:
        return self._object_types.get(name)

    def get_link_type(self, name: str) -> Optional[LinkTypeDef]:
        return self._link_types.get(name)

    def get_action(self, name: str) -> Optional[ActionTypeDef]:
        return self._action_types.get(name)

    # ── introspection ─────────────────────────────────────────────────────────

    def list_object_types(self) -> List[ObjectTypeDef]:
        return list(self._object_types.values())

    def list_link_types(self) -> List[LinkTypeDef]:
        return list(self._link_types.values())

    def list_action_types(self) -> List[ActionTypeDef]:
        return list(self._action_types.values())

    def links_from(self, object_type: str) -> List[LinkTypeDef]:
        """All link types whose source is the given object type."""
        return [lt for lt in self._link_types.values() if lt.source_type == object_type]

    def links_to(self, object_type: str) -> List[LinkTypeDef]:
        """All link types whose target is the given object type."""
        return [lt for lt in self._link_types.values() if lt.target_type == object_type]

    def actions_for(self, object_type: str) -> List[ActionTypeDef]:
        """All action types that target the given object type."""
        return [a for a in self._action_types.values() if a.target_type == object_type]

    def summary(self) -> dict:
        return {
            "object_types": len(self._object_types),
            "link_types":   len(self._link_types),
            "action_types": len(self._action_types),
            "types": {
                "objects": [t.name for t in self.list_object_types()],
                "links":   [t.name for t in self.list_link_types()],
                "actions": [t.name for t in self.list_action_types()],
            },
        }


# ── singleton ──────────────────────────────────────────────────────────────────

_registry: OntologyRegistry | None = None


def get_registry() -> OntologyRegistry:
    global _registry
    if _registry is None:
        _registry = OntologyRegistry()
        _bootstrap(_registry)
    return _registry


def _bootstrap(registry: OntologyRegistry) -> None:
    """Load all Summit.OS definitions into the registry."""
    from .definitions import register_all
    register_all(registry)
    logger.info(
        "Ontology bootstrapped — %d object types, %d link types, %d action types",
        len(registry._object_types),
        len(registry._link_types),
        len(registry._action_types),
    )
