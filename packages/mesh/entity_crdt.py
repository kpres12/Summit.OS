"""
Entity CRDT for Summit.OS Mesh

Bridges the canonical Entity model (packages/entities/core.py) with the
mesh CRDT system (packages/mesh/crdt.py) for distributed state sync.

Each entity is stored as an LWWRegister keyed by entity_id.
Merge semantics: latest provenance.updated_at wins.

Usage:
    from packages.mesh.entity_crdt import EntityCRDTMap

    emap = EntityCRDTMap(node_id="node-01")
    emap.put(entity)

    # On receiving remote state:
    emap.merge_remote(remote_entity_dict)

    # Get all entities:
    for entity in emap.values():
        world_store.merge_remote(entity.to_dict(), source="mesh")
"""
from __future__ import annotations

import json
import time
import logging
from typing import Any, Callable, Dict, List, Optional

from packages.entities.core import Entity
from packages.mesh.crdt import LWWRegister, CRDTStore

logger = logging.getLogger("mesh.entity_crdt")


class EntityCRDTMap:
    """
    A CRDT-backed map of entity_id -> Entity.

    Each entity is wrapped in an LWWRegister. When two nodes have
    conflicting updates to the same entity, the one with the latest
    provenance.updated_at timestamp wins.

    This is the bridge between the mesh layer and the WorldStore.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._store = CRDTStore(node_id)
        self._on_merge_callbacks: List[Callable] = []

    def put(self, entity: Entity) -> None:
        """
        Insert or update an entity in the CRDT map.

        Serializes the entity to a dict and stores it in an LWWRegister
        with the entity's provenance timestamp.
        """
        ts = entity.provenance.updated_at if entity.provenance else time.time()
        reg = self._store.get_register(f"entity:{entity.id}")
        reg.set(entity.to_dict(), t=ts)

    def get(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity dict by ID, or None if not present."""
        key = f"entity:{entity_id}"
        if key not in self._store.registers:
            return None
        return self._store.registers[key].get()

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get deserialized Entity by ID."""
        d = self.get(entity_id)
        if d is None:
            return None
        return Entity.from_dict(d)

    def remove(self, entity_id: str) -> bool:
        """Remove entity from CRDT map (set to tombstone)."""
        key = f"entity:{entity_id}"
        if key in self._store.registers:
            # Set to None with current timestamp to tombstone
            self._store.registers[key].set(None, t=time.time())
            return True
        return False

    def values(self) -> List[Entity]:
        """Get all non-tombstoned entities."""
        entities = []
        for key, reg in self._store.registers.items():
            if not key.startswith("entity:"):
                continue
            val = reg.get()
            if val is None:
                continue
            try:
                entities.append(Entity.from_dict(val))
            except Exception as e:
                logger.debug(f"Failed to deserialize entity from CRDT: {e}")
        return entities

    def entity_ids(self) -> List[str]:
        """Get all entity IDs in the map."""
        ids = []
        for key, reg in self._store.registers.items():
            if key.startswith("entity:") and reg.get() is not None:
                ids.append(key[len("entity:"):])
        return ids

    @property
    def count(self) -> int:
        return len(self.entity_ids())

    # ── Merge ──────────────────────────────────────────────────

    def merge_remote(self, entity_dict: Dict[str, Any]) -> bool:
        """
        Merge a remote entity update.

        Returns True if the remote entity was newer and replaced local.
        """
        entity_id = entity_dict.get("id")
        if not entity_id:
            return False

        # Get remote timestamp
        prov = entity_dict.get("provenance") or {}
        remote_ts = prov.get("updated_at", time.time())

        key = f"entity:{entity_id}"
        remote_reg = LWWRegister(
            node_id=f"remote",
            value=entity_dict,
            timestamp=remote_ts,
        )

        if key in self._store.registers:
            merged = self._store.registers[key].merge(remote_reg)
            was_updated = merged.timestamp == remote_ts and merged.value == entity_dict
            self._store.registers[key] = merged
        else:
            self._store.registers[key] = LWWRegister(
                node_id=self.node_id,
                value=entity_dict,
                timestamp=remote_ts,
            )
            was_updated = True

        if was_updated:
            for cb in self._on_merge_callbacks:
                try:
                    cb(entity_dict)
                except Exception as e:
                    logger.error(f"Merge callback error: {e}")

        return was_updated

    def merge_store(self, remote_store: CRDTStore) -> int:
        """
        Merge an entire remote CRDTStore.

        Returns count of entities that were updated.
        """
        updated = 0
        self._store.merge(remote_store)

        # Notify for each entity register that changed
        for key, reg in remote_store.registers.items():
            if key.startswith("entity:") and reg.get() is not None:
                for cb in self._on_merge_callbacks:
                    try:
                        cb(reg.get())
                    except Exception:
                        pass
                updated += 1

        return updated

    def on_merge(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for when remote entities are merged in."""
        self._on_merge_callbacks.append(callback)

    # ── Serialization (for sync protocol) ──────────────────────

    def to_sync_payload(self) -> Dict[str, Any]:
        """Serialize CRDT state for network transmission."""
        payload = {}
        for key, reg in self._store.registers.items():
            if key.startswith("entity:"):
                payload[key] = reg.to_dict()
        return payload

    def apply_sync_payload(self, payload: Dict[str, Any]) -> int:
        """
        Apply a sync payload from a remote node.

        Returns count of entities updated.
        """
        updated = 0
        for key, reg_data in payload.items():
            if not key.startswith("entity:"):
                continue

            remote_reg = LWWRegister.from_dict(reg_data)

            if key in self._store.registers:
                merged = self._store.registers[key].merge(remote_reg)
                if merged.timestamp == remote_reg.timestamp:
                    updated += 1
                self._store.registers[key] = merged
            else:
                self._store.registers[key] = LWWRegister(
                    self.node_id, remote_reg.value, remote_reg.timestamp,
                )
                updated += 1

        return updated

    @property
    def crdt_store(self) -> CRDTStore:
        """Access the underlying CRDTStore (for SyncProtocol)."""
        return self._store

    def stats(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "entity_count": self.count,
            "register_count": len(self._store.registers),
        }
