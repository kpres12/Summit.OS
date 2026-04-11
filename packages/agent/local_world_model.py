"""
Local World Model — Summit.OS Edge Agent

Subset of the WorldStore that runs on-asset. Stores entity states in a
bounded in-memory dict (max 200 entities, LRU eviction). Syncs with the
central Fabric when link is available; operates standalone when offline.

On reconnect, merges local changes into the central store via last-write-wins
(timestamp-based). No CRDT needed at this scale — single-asset view.
"""

from __future__ import annotations

import time
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("agent.local_world_model")


class LocalWorldModel:
    """
    Bounded in-memory entity store for edge assets.

    Stores up to max_entities states keyed by entity_id. When at capacity,
    evicts the least-recently-used entry. Each entry carries an updated_at
    timestamp to support delta sync.
    """

    def __init__(self, asset_id: str, max_entities: int = 200) -> None:
        self.asset_id = asset_id
        self.max_entities = max_entities
        # OrderedDict preserves insertion/access order for LRU semantics.
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    # ── Write ────────────────────────────────────────────────

    def upsert(self, entity_id: str, state: dict) -> None:
        """
        Insert or update an entity's state.

        Stamps updated_at with the current epoch. If the store is at
        capacity the least-recently-used entry is evicted first.
        """
        if entity_id in self._store:
            # Move to end (most-recently-used position).
            self._store.move_to_end(entity_id)
        elif len(self._store) >= self.max_entities:
            evicted_id, _ = self._store.popitem(last=False)
            logger.debug("LRU eviction: entity_id=%s", evicted_id)

        entry = dict(state)
        entry["entity_id"] = entity_id
        entry["updated_at"] = time.time()
        entry.setdefault("asset_id", self.asset_id)
        self._store[entity_id] = entry

    # ── Read ─────────────────────────────────────────────────

    def get(self, entity_id: str) -> Optional[dict]:
        """Return entity state or None if not found."""
        entry = self._store.get(entity_id)
        if entry is not None:
            self._store.move_to_end(entity_id)
        return entry

    def all_entities(self) -> List[dict]:
        """Return a snapshot list of all stored entity states."""
        return list(self._store.values())

    def get_sync_delta(self, since_ts: float) -> List[dict]:
        """
        Return entities whose updated_at is strictly after since_ts.

        Used to build the delta payload for push_to_fabric.
        """
        return [e for e in self._store.values() if e.get("updated_at", 0.0) > since_ts]

    # ── Fabric Sync ──────────────────────────────────────────

    async def push_to_fabric(self, fabric_url: str) -> int:
        """
        POST the full sync delta to the central Fabric entity ingest endpoint.

        Returns the number of entities successfully synced. The delta covers
        all entities updated since the last successful push (tracked via
        _last_push_ts). Falls back to a full push on first call.
        """
        since = getattr(self, "_last_push_ts", 0.0)
        delta = self.get_sync_delta(since)
        if not delta:
            return 0

        url = f"{fabric_url.rstrip('/')}/api/v1/entities/batch"
        payload = {
            "asset_id": self.asset_id,
            "entities": delta,
            "synced_at": time.time(),
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            self._last_push_ts: float = time.time()
            logger.info("Pushed %d entities to fabric", len(delta))
            return len(delta)
        except httpx.HTTPError as exc:
            logger.warning("push_to_fabric failed: %s", exc)
            return 0

    async def pull_from_fabric(self, fabric_url: str) -> int:
        """
        GET nearby/relevant entities from the central Fabric and merge them.

        Returns the number of entities pulled. Uses last-write-wins: only
        overwrites a local entry if the remote updated_at is newer.
        """
        url = f"{fabric_url.rstrip('/')}/api/v1/entities"
        params: Dict[str, Any] = {"asset_id": self.asset_id, "nearby": True}

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
            remote_entities: List[dict] = resp.json().get("entities", [])
        except httpx.HTTPError as exc:
            logger.warning("pull_from_fabric failed: %s", exc)
            return 0

        merged = 0
        for remote in remote_entities:
            eid = remote.get("entity_id")
            if not eid:
                continue
            local = self._store.get(eid)
            # LWW: remote wins if newer or not present locally.
            if local is None or remote.get("updated_at", 0.0) > local.get("updated_at", 0.0):
                # Bypass the public upsert to avoid re-stamping updated_at.
                if eid in self._store:
                    self._store.move_to_end(eid)
                elif len(self._store) >= self.max_entities:
                    evicted_id, _ = self._store.popitem(last=False)
                    logger.debug("LRU eviction (pull): entity_id=%s", evicted_id)
                self._store[eid] = dict(remote)
                merged += 1

        logger.info("Pulled %d new/updated entities from fabric", merged)
        return merged
