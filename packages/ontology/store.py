"""
Summit.OS Ontology Object Store

In-memory store for ObjectInstances and LinkInstances.

  - Keyed by (object_type, object_id) for objects
  - Keyed by (link_type, source_id, target_id) for links
  - Secondary indexes on indexed properties for fast filter queries
  - Event callbacks on create/update/delete

All writes go through ActionRunner (see actions.py).
Direct upsert is available for internal sync only (_upsert / _upsert_link).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from .types import LinkInstance, ObjectInstance

logger = logging.getLogger("ontology.store")


class ObjectStore:
    """
    Thread-safe (GIL-level) in-memory store for the live ontology graph.
    """

    def __init__(self) -> None:
        # Primary storage: (object_type, object_id) → ObjectInstance
        self._objects: Dict[Tuple[str, str], ObjectInstance] = {}

        # Links: (link_type, source_id, target_id) → LinkInstance
        self._links: Dict[Tuple[str, str, str], LinkInstance] = {}

        # Secondary index: object_type → {prop_name → {value → set(object_id)}}
        self._indexes: Dict[str, Dict[str, Dict[Any, set]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set))
        )

        # Event listeners: event_name → list of callbacks
        # Events: "object.created", "object.updated", "object.deleted",
        #         "link.created", "link.deleted"
        self._listeners: Dict[str, List[Callable]] = defaultdict(list)

    # ── object CRUD ───────────────────────────────────────────────────────────

    def _upsert(self, instance: ObjectInstance) -> ObjectInstance:
        """
        Internal: create or update an ObjectInstance.
        Called by ActionRunner after validation, and by OntologySync.
        External callers should use ActionRunner.execute() instead.
        """
        key = (instance.object_type, instance.object_id)
        existing = self._objects.get(key)

        if existing:
            # Preserve created_at, bump version
            instance.created_at = existing.created_at
            instance.version    = existing.version + 1
            instance.updated_at = datetime.now(timezone.utc).isoformat()
            self._deindex(existing)
            self._objects[key] = instance
            self._index(instance)
            self._emit("object.updated", instance)
        else:
            self._objects[key] = instance
            self._index(instance)
            self._emit("object.created", instance)

        return instance

    def get(self, object_type: str, object_id: str) -> Optional[ObjectInstance]:
        return self._objects.get((object_type, object_id))

    def delete(self, object_type: str, object_id: str) -> bool:
        key = (object_type, object_id)
        instance = self._objects.pop(key, None)
        if instance:
            self._deindex(instance)
            self._emit("object.deleted", instance)
            return True
        return False

    def list(
        self,
        object_type: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> List[ObjectInstance]:
        """
        List instances of a type.
        filters: {prop_name: value} — all must match (AND logic).
        Indexed properties use secondary index; others do linear scan.
        """
        if filters:
            # Start with the smallest indexed result set
            candidate_ids: Optional[set] = None
            unindexed_filters: Dict[str, Any] = {}

            for prop, value in filters.items():
                idx = self._indexes.get(object_type, {}).get(prop)
                if idx is not None:
                    matching = idx.get(value, set())
                    candidate_ids = matching if candidate_ids is None else candidate_ids & matching
                else:
                    unindexed_filters[prop] = value

            if candidate_ids is not None:
                candidates = [
                    self._objects[(object_type, oid)]
                    for oid in candidate_ids
                    if (object_type, oid) in self._objects
                ]
            else:
                candidates = [
                    inst for (ot, _), inst in self._objects.items()
                    if ot == object_type
                ]

            # Apply unindexed filters
            if unindexed_filters:
                candidates = [
                    inst for inst in candidates
                    if all(inst.properties.get(k) == v for k, v in unindexed_filters.items())
                ]
        else:
            candidates = [
                inst for (ot, _), inst in self._objects.items()
                if ot == object_type
            ]

        return candidates[offset : offset + limit]

    def count(self, object_type: str, filters: Optional[Dict[str, Any]] = None) -> int:
        return len(self.list(object_type, filters, limit=100_000))

    def all_of_type(self, object_type: str) -> List[ObjectInstance]:
        return [inst for (ot, _), inst in self._objects.items() if ot == object_type]

    # ── link CRUD ─────────────────────────────────────────────────────────────

    def _upsert_link(self, link: LinkInstance) -> LinkInstance:
        key = (link.link_type, link.source_id, link.target_id)
        if key not in self._links:
            self._links[key] = link
            self._emit("link.created", link)
        return self._links[key]

    def get_link(self, link_type: str, source_id: str, target_id: str) -> Optional[LinkInstance]:
        return self._links.get((link_type, source_id, target_id))

    def delete_link(self, link_type: str, source_id: str, target_id: str) -> bool:
        link = self._links.pop((link_type, source_id, target_id), None)
        if link:
            self._emit("link.deleted", link)
            return True
        return False

    def links_from_object(
        self,
        source_id: str,
        link_type: Optional[str] = None,
    ) -> List[LinkInstance]:
        return [
            link for (lt, src, _), link in self._links.items()
            if src == source_id and (link_type is None or lt == link_type)
        ]

    def links_to_object(
        self,
        target_id: str,
        link_type: Optional[str] = None,
    ) -> List[LinkInstance]:
        return [
            link for (lt, _, tgt), link in self._links.items()
            if tgt == target_id and (link_type is None or lt == link_type)
        ]

    # ── indexing ──────────────────────────────────────────────────────────────

    def register_index(self, object_type: str, prop_name: str) -> None:
        """Called by definitions to mark a property as indexed."""
        # Ensure the index key exists (defaultdict handles the rest)
        _ = self._indexes[object_type][prop_name]

    def _index(self, instance: ObjectInstance) -> None:
        type_idx = self._indexes.get(instance.object_type)
        if not type_idx:
            return
        for prop_name in type_idx:
            value = instance.properties.get(prop_name)
            if value is not None:
                self._indexes[instance.object_type][prop_name][value].add(instance.object_id)

    def _deindex(self, instance: ObjectInstance) -> None:
        type_idx = self._indexes.get(instance.object_type)
        if not type_idx:
            return
        for prop_name in type_idx:
            value = instance.properties.get(prop_name)
            if value is not None:
                self._indexes[instance.object_type][prop_name][value].discard(instance.object_id)

    # ── event system ──────────────────────────────────────────────────────────

    def on(self, event: str, callback: Callable) -> None:
        self._listeners[event].append(callback)

    def _emit(self, event: str, payload: Any) -> None:
        for cb in self._listeners.get(event, []):
            try:
                cb(payload)
            except Exception as exc:
                logger.warning("Event listener error (%s): %s", event, exc)

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        type_counts: Dict[str, int] = defaultdict(int)
        for (ot, _) in self._objects:
            type_counts[ot] += 1
        link_counts: Dict[str, int] = defaultdict(int)
        for (lt, _, _) in self._links:
            link_counts[lt] += 1
        return {
            "total_objects": len(self._objects),
            "total_links":   len(self._links),
            "by_type":       dict(type_counts),
            "by_link":       dict(link_counts),
        }


# ── singleton ──────────────────────────────────────────────────────────────────

_store: ObjectStore | None = None


def get_store() -> ObjectStore:
    global _store
    if _store is None:
        _store = ObjectStore()
    return _store
