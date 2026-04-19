"""
Heli.OS Ontology Query Interface

Fluent builder for filtering objects and traversing the link graph.

Basic usage:
    from packages.ontology.query import OntologyQuery

    q = OntologyQuery()

    # All CRITICAL alerts
    alerts = q.objects("Alert").where(severity="CRITICAL").all()

    # Assets linked to a mission
    assets = q.objects("Asset").linked_from("asset_executing_mission", mission_id).all()

    # One hop: find all observations that triggered an alert
    obs_ids = q.traverse(
        start_type="Alert",
        start_id=alert_id,
        link_type="observation_triggered_alert",
        direction="inbound",
    )

    # Summary for an AI agent
    context = q.semantic_summary()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .store import get_store
from .types import LinkInstance, ObjectInstance

logger = logging.getLogger("ontology.query")


class ObjectQuery:
    """Fluent query for a single object type."""

    def __init__(self, object_type: str) -> None:
        self._type = object_type
        self._filters: Dict[str, Any] = {}
        self._limit = 500
        self._offset = 0

    def where(self, **kwargs) -> "ObjectQuery":
        self._filters.update(kwargs)
        return self

    def limit(self, n: int) -> "ObjectQuery":
        self._limit = n
        return self

    def offset(self, n: int) -> "ObjectQuery":
        self._offset = n
        return self

    def linked_from(self, link_type: str, source_id: str) -> "ObjectQuery":
        """Filter to only objects that are the TARGET of a specific link from source_id."""
        store = get_store()
        links = store.links_from_object(source_id, link_type=link_type)
        target_ids = {link.target_id for link in links}
        # Add a special filter that will be applied in .all()
        self._filters["__linked_target_ids"] = target_ids
        return self

    def linked_to(self, link_type: str, target_id: str) -> "ObjectQuery":
        """Filter to only objects that are the SOURCE of a specific link to target_id."""
        store = get_store()
        links = store.links_to_object(target_id, link_type=link_type)
        source_ids = {link.source_id for link in links}
        self._filters["__linked_source_ids"] = source_ids
        return self

    def all(self) -> List[ObjectInstance]:
        store = get_store()

        # Extract special in-memory filters
        linked_target_ids = self._filters.pop("__linked_target_ids", None)
        linked_source_ids = self._filters.pop("__linked_source_ids", None)

        results = store.list(
            self._type, self._filters or None, self._limit, self._offset
        )

        if linked_target_ids is not None:
            results = [r for r in results if r.object_id in linked_target_ids]
        if linked_source_ids is not None:
            results = [r for r in results if r.object_id in linked_source_ids]

        return results

    def first(self) -> Optional[ObjectInstance]:
        results = self.limit(1).all()
        return results[0] if results else None

    def count(self) -> int:
        return len(self.all())

    def ids(self) -> List[str]:
        return [r.object_id for r in self.all()]


class OntologyQuery:
    """
    Entry point for all ontology queries.
    Stateless — create a new instance per request or reuse freely.
    """

    def objects(self, object_type: str) -> ObjectQuery:
        return ObjectQuery(object_type)

    def get(self, object_type: str, object_id: str) -> Optional[ObjectInstance]:
        return get_store().get(object_type, object_id)

    # ── graph traversal ───────────────────────────────────────────────────────

    def traverse(
        self,
        start_type: str,
        start_id: str,
        link_type: str,
        direction: str = "outbound",  # "outbound" | "inbound"
        target_type: Optional[str] = None,
    ) -> List[ObjectInstance]:
        """
        One-hop traversal from start_id along link_type.

        direction="outbound" → start_id is SOURCE, return TARGETs
        direction="inbound"  → start_id is TARGET, return SOURCEs
        """
        store = get_store()
        results = []

        if direction == "outbound":
            links = store.links_from_object(start_id, link_type)
            for link in links:
                obj = store.get(target_type or "", link.target_id)
                if obj is None and target_type is None:
                    # Try all types
                    for (ot, oid), inst in store._objects.items():
                        if oid == link.target_id:
                            obj = inst
                            break
                if obj:
                    results.append(obj)
        else:
            links = store.links_to_object(start_id, link_type)
            for link in links:
                obj = store.get(start_type, link.source_id)
                if obj is None:
                    for (ot, oid), inst in store._objects.items():
                        if oid == link.source_id:
                            obj = inst
                            break
                if obj:
                    results.append(obj)

        return results

    def neighbors(self, object_id: str) -> Dict[str, List[ObjectInstance]]:
        """
        Return all directly linked objects in both directions, grouped by link type.
        Useful for building a local graph view of a single entity.
        """
        store = get_store()
        out_links = store.links_from_object(object_id)
        in_links = store.links_to_object(object_id)

        result: Dict[str, List[ObjectInstance]] = {}

        for link in out_links:
            for (ot, oid), inst in store._objects.items():
                if oid == link.target_id:
                    result.setdefault(f"→{link.link_type}", []).append(inst)

        for link in in_links:
            for (ot, oid), inst in store._objects.items():
                if oid == link.source_id:
                    result.setdefault(f"←{link.link_type}", []).append(inst)

        return result

    # ── semantic summary (for AI agents) ─────────────────────────────────────

    def semantic_summary(self, max_items_per_type: int = 5) -> str:
        """
        Return a compact natural-language summary of the current ontology state.
        Designed to be injected into an LLM prompt as context.
        """
        store = get_store()
        stats = store.stats()
        lines = ["## Heli.OS Ontology — Live State\n"]

        # Object counts
        lines.append("### Object counts")
        for otype, count in stats["by_type"].items():
            lines.append(f"  {otype}: {count} instances")

        lines.append("\n### Recent critical objects")
        for otype in ("Alert", "Incident", "Mission", "Asset"):
            instances = store.all_of_type(otype)
            critical = [
                i
                for i in instances
                if i.properties.get("severity") in ("CRITICAL", "HIGH")
                or i.properties.get("status") in ("ACTIVE", "IN_PROGRESS")
            ][:max_items_per_type]
            if critical:
                lines.append(
                    f"\n**{otype}** (showing {len(critical)} active/critical):"
                )
                for inst in critical:
                    props = inst.properties
                    label = (
                        props.get("title")
                        or props.get("name")
                        or props.get("description", "")
                    )
                    sev = props.get("severity") or props.get("status") or ""
                    lines.append(f"  - [{sev}] {inst.object_id[:8]}… {label[:80]}")

        lines.append(f"\n### Links: {stats['total_links']} total")
        for lt, cnt in stats["by_link"].items():
            lines.append(f"  {lt}: {cnt}")

        return "\n".join(lines)

    # ── convenience accessors ─────────────────────────────────────────────────

    def active_incidents(self) -> List[ObjectInstance]:
        return self.objects("Incident").where(status="ACTIVE").all()

    def unacknowledged_alerts(self) -> List[ObjectInstance]:
        return self.objects("Alert").where(acknowledged=False).all()

    def available_assets(self) -> List[ObjectInstance]:
        return self.objects("Asset").where(status="AVAILABLE").all()

    def active_missions(self) -> List[ObjectInstance]:
        return self.objects("Mission").where(status="ACTIVE").all()
