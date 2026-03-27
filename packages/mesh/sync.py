"""
Anti-Entropy Synchronization Protocol for Summit.OS Mesh

Implements a pull-based anti-entropy protocol where nodes periodically:
1. Exchange state digests (hashes of CRDT keys + versions)
2. Identify missing or stale entries
3. Exchange only the deltas needed to converge

This minimizes bandwidth in contested/degraded network environments
while guaranteeing eventual consistency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from packages.mesh.crdt import CRDTStore, LWWRegister, PNCounter, ORSet

logger = logging.getLogger("mesh.sync")


@dataclass
class StateDigest:
    """
    Compact representation of a node's CRDT state.
    Used for efficient comparison during anti-entropy rounds.
    """

    node_id: str
    # key -> hash of serialized value
    register_digests: Dict[str, str] = field(default_factory=dict)
    counter_digests: Dict[str, str] = field(default_factory=dict)
    set_digests: Dict[str, str] = field(default_factory=dict)
    timestamp: float = 0.0

    @classmethod
    def from_store(cls, store: CRDTStore) -> "StateDigest":
        """Build digest from a CRDT store."""
        digest = cls(node_id=store.node_id, timestamp=time.time())

        for key, reg in store.registers.items():
            h = hashlib.sha256(
                json.dumps(reg.to_dict(), sort_keys=True).encode()
            ).hexdigest()[:16]
            digest.register_digests[key] = h

        for key, ctr in store.counters.items():
            h = hashlib.sha256(
                json.dumps(ctr.to_dict(), sort_keys=True).encode()
            ).hexdigest()[:16]
            digest.counter_digests[key] = h

        for key, s in store.sets.items():
            h = hashlib.sha256(
                json.dumps(s.to_dict(), sort_keys=True).encode()
            ).hexdigest()[:16]
            digest.set_digests[key] = h

        return digest

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "registers": self.register_digests,
            "counters": self.counter_digests,
            "sets": self.set_digests,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "StateDigest":
        return cls(
            node_id=data["node_id"],
            register_digests=data.get("registers", {}),
            counter_digests=data.get("counters", {}),
            set_digests=data.get("sets", {}),
            timestamp=data.get("timestamp", 0.0),
        )


@dataclass
class SyncDelta:
    """
    A delta containing only the CRDT entries that differ between two nodes.
    """

    source_node_id: str
    registers: Dict[str, Dict] = field(default_factory=dict)
    counters: Dict[str, Dict] = field(default_factory=dict)
    sets: Dict[str, Dict] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.registers and not self.counters and not self.sets

    def to_dict(self) -> Dict:
        return {
            "source": self.source_node_id,
            "registers": self.registers,
            "counters": self.counters,
            "sets": self.sets,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SyncDelta":
        return cls(
            source_node_id=data["source"],
            registers=data.get("registers", {}),
            counters=data.get("counters", {}),
            sets=data.get("sets", {}),
        )


class SyncProtocol:
    """
    Anti-entropy synchronization protocol.

    Usage:
        proto = SyncProtocol(my_store)

        # Initiator side:
        my_digest = proto.create_digest()
        # send my_digest to peer, receive peer_digest back
        delta = proto.compute_delta(peer_digest)
        # send delta to peer

        # Receiver side:
        # receive delta from peer
        proto.apply_delta(delta)
    """

    def __init__(self, store: CRDTStore):
        self.store = store
        self.sync_count: int = 0
        self.last_sync_time: float = 0.0
        self.bytes_sent: int = 0
        self.bytes_received: int = 0

    def create_digest(self) -> StateDigest:
        """Create a digest of current state for comparison."""
        return StateDigest.from_store(self.store)

    def compute_delta(self, remote_digest: StateDigest) -> SyncDelta:
        """
        Compare local state with a remote digest and produce a delta
        containing entries the remote is missing or has stale.
        """
        local_digest = self.create_digest()
        delta = SyncDelta(source_node_id=self.store.node_id)

        # Check registers
        for key, local_hash in local_digest.register_digests.items():
            remote_hash = remote_digest.register_digests.get(key)
            if remote_hash != local_hash:
                # Remote is missing or stale — include our version
                delta.registers[key] = self.store.registers[key].to_dict()

        # Check counters
        for key, local_hash in local_digest.counter_digests.items():
            remote_hash = remote_digest.counter_digests.get(key)
            if remote_hash != local_hash:
                delta.counters[key] = self.store.counters[key].to_dict()

        # Check sets
        for key, local_hash in local_digest.set_digests.items():
            remote_hash = remote_digest.set_digests.get(key)
            if remote_hash != local_hash:
                delta.sets[key] = self.store.sets[key].to_dict()

        return delta

    def apply_delta(self, delta: SyncDelta):
        """
        Apply a received delta by merging each CRDT entry.

        Because all our data structures are CRDTs, merge is safe
        regardless of message ordering or duplication.
        """
        # Merge registers
        for key, reg_data in delta.registers.items():
            remote_reg = LWWRegister.from_dict(reg_data)
            if key in self.store.registers:
                self.store.registers[key] = self.store.registers[key].merge(remote_reg)
            else:
                self.store.registers[key] = LWWRegister(
                    self.store.node_id, remote_reg.value, remote_reg.timestamp
                )

        # Merge counters
        for key, ctr_data in delta.counters.items():
            remote_ctr = PNCounter.from_dict(ctr_data)
            if key in self.store.counters:
                self.store.counters[key] = self.store.counters[key].merge(remote_ctr)
            else:
                merged = PNCounter(self.store.node_id)
                merged.p = remote_ctr.p
                merged.n = remote_ctr.n
                self.store.counters[key] = merged

        # Merge sets
        for key, set_data in delta.sets.items():
            remote_set = ORSet.from_dict(set_data)
            if key in self.store.sets:
                self.store.sets[key] = self.store.sets[key].merge(remote_set)
            else:
                merged = ORSet(self.store.node_id)
                merged.elements = {k: set(v) for k, v in remote_set.elements.items()}
                merged.tombstones = set(remote_set.tombstones)
                self.store.sets[key] = merged

        self.sync_count += 1
        self.last_sync_time = time.time()
        logger.debug(
            f"Applied delta from {delta.source_node_id}: "
            f"{len(delta.registers)} regs, {len(delta.counters)} ctrs, {len(delta.sets)} sets"
        )

    def needs_sync(self, remote_digest: StateDigest) -> bool:
        """Quick check if sync is needed."""
        local_digest = self.create_digest()
        return (
            local_digest.register_digests != remote_digest.register_digests
            or local_digest.counter_digests != remote_digest.counter_digests
            or local_digest.set_digests != remote_digest.set_digests
        )

    def get_stats(self) -> Dict:
        """Get sync protocol statistics."""
        return {
            "sync_count": self.sync_count,
            "last_sync": self.last_sync_time,
            "store_keys": {
                "registers": len(self.store.registers),
                "counters": len(self.store.counters),
                "sets": len(self.store.sets),
            },
        }
