"""
Summit.OS Mesh Sync — prioritised WorldStore delta sync for degraded networks.

Enables Summit.OS nodes to stay synchronised over contested or low-bandwidth
links (BATMAN-adv mesh, satellite, LTE with 90% packet loss).

Key design decisions:
  - Delta-only: never replicate full state, only changes (bytes not megabytes)
  - Priority tiers: CRITICAL > WARNING > ACTIVE > telemetry
  - Store-and-forward: queue changes locally, drain on reconnect
  - Bandwidth-adaptive: back-pressure when link is saturated
  - Peer discovery: uses MQTT retained topics to find peer nodes

Each node publishes its entity deltas to: mesh/node/{node_id}/delta
Each node subscribes to:              mesh/node/+/delta

On reconnect, nodes exchange a lightweight state vector (entity_id → version)
and sync only the divergent entities — similar to CRDT anti-entropy.

Environment variables:
    MESH_NODE_ID        - unique node identifier (default: hostname)
    MESH_ENABLED        - "true" to enable (default: "false")
    MESH_SYNC_INTERVAL  - seconds between sync cycles (default: 5)
    MESH_MAX_QUEUE      - max pending deltas before dropping low-priority (default: 500)
    MESH_BANDWIDTH_BPS  - target bandwidth cap in bytes/sec (default: 10000 = 10KB/s)
    MQTT_HOST/PORT      - MQTT broker
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Deque, Dict, List, Optional, Set

logger = logging.getLogger("summit.fabric.mesh_sync")

MESH_NODE_ID = os.getenv("MESH_NODE_ID", socket.gethostname())
MESH_SYNC_INTERVAL = float(os.getenv("MESH_SYNC_INTERVAL", "5"))
MESH_MAX_QUEUE = int(os.getenv("MESH_MAX_QUEUE", "500"))
MESH_BANDWIDTH_BPS = int(os.getenv("MESH_BANDWIDTH_BPS", "10000"))


# ── Priority tiers ────────────────────────────────────────────────────────────


class SyncPriority(IntEnum):
    CRITICAL = 0  # Alerts, CRITICAL state entities — always first
    WARNING = 1  # WARNING state entities
    MISSION = 2  # Mission / tasking updates
    ACTIVE = 3  # Normal active entity telemetry
    LOW = 4  # Inactive / low-value telemetry


def _entity_priority(entity: Dict) -> SyncPriority:
    state = entity.get("state", "")
    etype = entity.get("entity_type", "")
    if state == "CRITICAL":
        return SyncPriority.CRITICAL
    if etype == "ALERT":
        return SyncPriority.CRITICAL
    if state == "WARNING":
        return SyncPriority.WARNING
    if etype == "MISSION":
        return SyncPriority.MISSION
    if state == "ACTIVE":
        return SyncPriority.ACTIVE
    return SyncPriority.LOW


@dataclass(order=True)
class DeltaEntry:
    priority: int  # SyncPriority value (lower = higher priority)
    ts: float = field(compare=False)  # Timestamp for ordering within same priority
    entity_id: str = field(compare=False)
    version: int = field(compare=False)
    payload: bytes = field(compare=False)
    entity_type: str = field(compare=False, default="")

    def size(self) -> int:
        return len(self.payload)


# ── State vector for anti-entropy ─────────────────────────────────────────────


class StateVector:
    """
    Lightweight summary of a node's entity versions.
    Exchanged during reconnect to identify divergent entities.
    """

    def __init__(self):
        self._versions: Dict[str, int] = {}  # entity_id → version

    def update(self, entity_id: str, version: int) -> None:
        self._versions[entity_id] = max(self._versions.get(entity_id, 0), version)

    def diff(self, other: "StateVector") -> List[str]:
        """Return entity_ids where our version > other's version."""
        result = []
        for eid, ver in self._versions.items():
            if ver > other._versions.get(eid, -1):
                result.append(eid)
        return result

    def to_dict(self) -> Dict:
        return dict(self._versions)

    @classmethod
    def from_dict(cls, d: Dict) -> "StateVector":
        sv = cls()
        sv._versions = {k: int(v) for k, v in d.items()}
        return sv


# ── Bandwidth token bucket ────────────────────────────────────────────────────


class BandwidthBucket:
    """Token bucket for outbound bandwidth shaping."""

    def __init__(self, rate_bps: int):
        self.rate_bps = rate_bps
        self._tokens = float(rate_bps)
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        dt = now - self._last_refill
        self._tokens = min(float(self.rate_bps), self._tokens + dt * self.rate_bps)
        self._last_refill = now

    def consume(self, n_bytes: int) -> bool:
        """Returns True if bandwidth is available and consumes tokens."""
        self._refill()
        if self._tokens >= n_bytes:
            self._tokens -= n_bytes
            return True
        return False

    async def wait_for(self, n_bytes: int) -> None:
        """Block until bandwidth is available."""
        while not self.consume(n_bytes):
            await asyncio.sleep(0.1)


# ── Mesh sync engine ──────────────────────────────────────────────────────────


class MeshSync:
    """
    Manages prioritised delta sync between Summit.OS mesh nodes.

    Plugs into the WorldStore via subscription callbacks — call
    `on_entity_update(entity)` whenever an entity changes locally.
    """

    def __init__(
        self,
        node_id: str = MESH_NODE_ID,
        sync_interval: float = MESH_SYNC_INTERVAL,
        max_queue: int = MESH_MAX_QUEUE,
        bandwidth_bps: int = MESH_BANDWIDTH_BPS,
    ):
        self.node_id = node_id
        self.sync_interval = sync_interval
        self.max_queue = max_queue

        self._queue: Deque[DeltaEntry] = deque()
        self._queue_size_bytes = 0
        self._bandwidth = BandwidthBucket(bandwidth_bps)
        self._state_vector = StateVector()
        self._known_peers: Set[str] = set()

        # MQTT client (set by caller)
        self._mqtt_client = None

        # Stats
        self._stats = {
            "deltas_queued": 0,
            "deltas_sent": 0,
            "deltas_dropped": 0,
            "bytes_sent": 0,
            "peers_seen": 0,
        }

    def set_mqtt_client(self, client) -> None:
        self._mqtt_client = client

    # ── Inbound: entity changed locally ──────────────────────────────────────

    def on_entity_update(self, entity: Dict) -> None:
        """Call this when a local entity changes. Queues it for replication."""
        entity_id = entity.get("entity_id", "")
        if not entity_id:
            return

        version = int(entity.get("version", 0) or time.time() * 1000)
        priority = _entity_priority(entity)

        payload = json.dumps(entity, separators=(",", ":")).encode()

        entry = DeltaEntry(
            priority=int(priority),
            ts=time.time(),
            entity_id=entity_id,
            version=version,
            payload=payload,
            entity_type=entity.get("entity_type", ""),
        )

        # Deduplicate: replace older entry for same entity if lower priority
        self._dedup_insert(entry)
        self._state_vector.update(entity_id, version)
        self._stats["deltas_queued"] += 1

    def _dedup_insert(self, entry: DeltaEntry) -> None:
        """Insert entry, dropping a lower-priority duplicate if queue is full."""
        # Check if entity already in queue — replace if newer
        for i, existing in enumerate(self._queue):
            if existing.entity_id == entry.entity_id:
                if entry.version >= existing.version:
                    self._queue_size_bytes -= existing.size()
                    self._queue[i] = entry
                    self._queue_size_bytes += entry.size()
                return  # Older version — discard

        # Queue not full — just append
        if len(self._queue) < self.max_queue:
            self._queue.append(entry)
            self._queue_size_bytes += entry.size()
            return

        # Queue full — drop lowest priority entry to make room
        # Find the lowest-priority, oldest entry
        worst_idx = -1
        worst_pri = entry.priority
        for i in range(len(self._queue) - 1, -1, -1):
            if self._queue[i].priority > worst_pri:
                worst_idx = i
                worst_pri = self._queue[i].priority
                break

        if worst_idx >= 0:
            dropped = self._queue[worst_idx]
            self._queue_size_bytes -= dropped.size()
            del self._queue[worst_idx]
            self._queue.append(entry)
            self._queue_size_bytes += entry.size()
            self._stats["deltas_dropped"] += 1
            logger.debug(
                f"Queue full — dropped {dropped.entity_id} ({dropped.entity_type}, pri={dropped.priority})"
            )
        else:
            self._stats["deltas_dropped"] += 1
            logger.debug(
                f"Queue full — dropped incoming {entry.entity_id} (lower priority)"
            )

    # ── Outbound: drain queue over MQTT ──────────────────────────────────────

    async def _drain_queue(self) -> int:
        """Drain the queue, respecting bandwidth limits. Returns bytes sent."""
        if not self._mqtt_client or not self._queue:
            return 0

        sent_bytes = 0
        topic_base = f"mesh/node/{self.node_id}/delta"

        # Sort by priority for this drain cycle
        entries = sorted(self._queue, key=lambda e: (e.priority, e.ts))
        self._queue.clear()
        self._queue_size_bytes = 0

        for entry in entries:
            await self._bandwidth.wait_for(entry.size())
            try:
                self._mqtt_client.publish(
                    topic_base, entry.payload, qos=1, retain=False
                )
                sent_bytes += entry.size()
                self._stats["deltas_sent"] += 1
                self._stats["bytes_sent"] += entry.size()
            except Exception as e:
                logger.warning(f"MQTT publish failed for {entry.entity_id}: {e}")
                # Re-queue on send failure (best effort)
                if len(self._queue) < self.max_queue:
                    self._queue.append(entry)
                    self._queue_size_bytes += entry.size()

        return sent_bytes

    # ── Anti-entropy: state vector exchange ──────────────────────────────────

    async def _publish_state_vector(self) -> None:
        """Broadcast our state vector so peers can identify what they're missing."""
        if not self._mqtt_client:
            return
        topic = f"mesh/node/{self.node_id}/state_vector"
        payload = json.dumps(
            {
                "node_id": self.node_id,
                "ts": time.time(),
                "versions": self._state_vector.to_dict(),
            }
        ).encode()
        try:
            self._mqtt_client.publish(topic, payload, qos=0, retain=True)
        except Exception as e:
            logger.debug(f"State vector publish failed: {e}")

    def on_peer_state_vector(self, peer_id: str, peer_sv_dict: Dict) -> List[str]:
        """
        Called when we receive a peer's state vector.
        Returns list of entity_ids we have that peer is missing.
        """
        if peer_id not in self._known_peers:
            self._known_peers.add(peer_id)
            self._stats["peers_seen"] += 1
            logger.info(f"New mesh peer: {peer_id}")

        peer_sv = StateVector.from_dict(peer_sv_dict)
        missing_on_peer = self._state_vector.diff(peer_sv)
        return missing_on_peer

    # ── Main sync loop ────────────────────────────────────────────────────────

    async def run(self) -> None:
        logger.info(
            f"MeshSync starting — node_id={self.node_id}, interval={self.sync_interval}s, bw={MESH_BANDWIDTH_BPS}B/s"
        )
        while True:
            try:
                await self._drain_queue()
                await self._publish_state_vector()

                if (
                    self._stats["deltas_sent"] % 100 == 0
                    and self._stats["deltas_sent"] > 0
                ):
                    logger.info(
                        f"MeshSync stats: sent={self._stats['deltas_sent']} "
                        f"dropped={self._stats['deltas_dropped']} "
                        f"bytes={self._stats['bytes_sent']} "
                        f"peers={len(self._known_peers)}"
                    )
            except Exception as e:
                logger.error(f"MeshSync cycle error: {e}")

            await asyncio.sleep(self.sync_interval)

    def stats(self) -> Dict:
        return {
            **self._stats,
            "queue_depth": len(self._queue),
            "queue_bytes": self._queue_size_bytes,
            "peer_count": len(self._known_peers),
            "node_id": self.node_id,
        }
