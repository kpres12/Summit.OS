"""Register all Summit.OS ontology definitions into the registry."""

from __future__ import annotations

from .action_types import ALL_ACTION_TYPES
from .link_types import ALL_LINK_TYPES
from .object_types import ALL_OBJECT_TYPES


def register_all(registry) -> None:
    """Called once at startup by OntologyRegistry._bootstrap()."""
    for obj_type in ALL_OBJECT_TYPES:
        registry.register_object_type(obj_type)

    for link_type in ALL_LINK_TYPES:
        registry.register_link_type(link_type)

    for action_type in ALL_ACTION_TYPES:
        registry.register_action_type(action_type)

    # Register property indexes on the store
    from ..store import get_store
    store = get_store()
    for obj_type in ALL_OBJECT_TYPES:
        for prop in obj_type.properties:
            if prop.index:
                store.register_index(obj_type.name, prop.name)
