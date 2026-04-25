"""
Conflict-Free Replicated Data Types (CRDTs) for Heli.OS Mesh

Provides eventually-consistent distributed data structures that can be
replicated across mesh nodes without coordination. Supports:

- LWWRegister: Last-Writer-Wins register for single values
- GCounter: Grow-only counter (distributed increment)
- PNCounter: Positive-Negative counter (increment + decrement)
- ORSet: Observed-Remove set (add + remove elements)

All CRDTs implement a merge() operation that is commutative, associative,
and idempotent — guaranteeing convergence regardless of message ordering.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Set


class LWWRegister:
    """
    Last-Writer-Wins Register.

    Stores a single value with a timestamp. On merge, the value with
    the highest timestamp wins. Ties broken by node_id.
    """

    def __init__(self, node_id: str, value: Any = None, timestamp: float = 0.0):
        self.node_id = node_id
        self.value = value
        self.timestamp = timestamp

    def set(self, value: Any, t: float | None = None):
        """Set value with current timestamp."""
        t = t or time.time()
        if t >= self.timestamp:
            self.value = value
            self.timestamp = t

    def get(self) -> Any:
        return self.value

    def merge(self, other: "LWWRegister") -> "LWWRegister":
        """Merge with another LWWRegister. Returns new merged register."""
        if other.timestamp > self.timestamp:
            return LWWRegister(self.node_id, other.value, other.timestamp)
        elif other.timestamp == self.timestamp:
            # Tie-break by node_id (deterministic)
            winner = max(self.node_id, other.node_id)
            if winner == other.node_id:
                return LWWRegister(self.node_id, other.value, other.timestamp)
        return LWWRegister(self.node_id, self.value, self.timestamp)

    def to_dict(self) -> Dict:
        return {
            "value": self.value,
            "timestamp": self.timestamp,
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "LWWRegister":
        return cls(data["node_id"], data["value"], data["timestamp"])


class GCounter:
    """
    Grow-only Counter (G-Counter).

    Each node maintains its own counter. The global count is the sum
    of all node counts. Only supports increment.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.counts: Dict[str, int] = {node_id: 0}

    def increment(self, amount: int = 1):
        """Increment this node's count."""
        self.counts[self.node_id] = self.counts.get(self.node_id, 0) + amount

    def value(self) -> int:
        """Get the global counter value."""
        return sum(self.counts.values())

    def merge(self, other: "GCounter") -> "GCounter":
        """Merge: take max of each node's count."""
        result = GCounter(self.node_id)
        all_nodes = set(self.counts.keys()) | set(other.counts.keys())
        for node in all_nodes:
            result.counts[node] = max(
                self.counts.get(node, 0),
                other.counts.get(node, 0),
            )
        return result

    def to_dict(self) -> Dict:
        return {"node_id": self.node_id, "counts": dict(self.counts)}

    @classmethod
    def from_dict(cls, data: Dict) -> "GCounter":
        c = cls(data["node_id"])
        c.counts = dict(data["counts"])
        return c


class PNCounter:
    """
    Positive-Negative Counter.

    Composed of two G-Counters: one for increments, one for decrements.
    Value = P.value() - N.value()
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.p = GCounter(node_id)  # positive
        self.n = GCounter(node_id)  # negative

    def increment(self, amount: int = 1):
        self.p.increment(amount)

    def decrement(self, amount: int = 1):
        self.n.increment(amount)

    def value(self) -> int:
        return self.p.value() - self.n.value()

    def merge(self, other: "PNCounter") -> "PNCounter":
        result = PNCounter(self.node_id)
        result.p = self.p.merge(other.p)
        result.n = self.n.merge(other.n)
        return result

    def to_dict(self) -> Dict:
        return {"node_id": self.node_id, "p": self.p.to_dict(), "n": self.n.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict) -> "PNCounter":
        c = cls(data["node_id"])
        c.p = GCounter.from_dict(data["p"])
        c.n = GCounter.from_dict(data["n"])
        return c


class ORSet:
    """
    Observed-Remove Set (OR-Set).

    Elements can be added and removed. Each add generates a unique tag.
    Remove deletes all known tags for an element. Concurrent add + remove
    results in the element being present (add wins).

    Internal representation: elements map to sets of unique tags.
    Tombstones track removed tags.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        # element -> set of unique tags (each add creates a new tag)
        self.elements: Dict[Any, Set[str]] = {}
        # Tags that have been removed
        self.tombstones: Set[str] = set()

    def _make_tag(self) -> str:
        return f"{self.node_id}:{uuid.uuid4().hex[:12]}"

    def add(self, element: Any):
        """Add an element with a unique tag."""
        tag = self._make_tag()
        if element not in self.elements:
            self.elements[element] = set()
        self.elements[element].add(tag)

    def remove(self, element: Any):
        """Remove all observed instances of element."""
        if element in self.elements:
            # Move all tags to tombstones
            self.tombstones.update(self.elements[element])
            del self.elements[element]

    def contains(self, element: Any) -> bool:
        """Check if element is in the set."""
        if element not in self.elements:
            return False
        # Element exists if it has at least one non-tombstoned tag
        live_tags = self.elements[element] - self.tombstones
        return len(live_tags) > 0

    def values(self) -> Set[Any]:
        """Get all elements currently in the set."""
        result = set()
        for elem, tags in self.elements.items():
            if tags - self.tombstones:
                result.add(elem)
        return result

    def merge(self, other: "ORSet") -> "ORSet":
        """
        Merge two OR-Sets.

        - Union of all tags per element
        - Union of all tombstones
        - An element is present if it has any non-tombstoned tag
        """
        result = ORSet(self.node_id)
        result.tombstones = self.tombstones | other.tombstones

        # Union all elements and their tags
        all_elements = set(self.elements.keys()) | set(other.elements.keys())
        for elem in all_elements:
            tags = set()
            if elem in self.elements:
                tags |= self.elements[elem]
            if elem in other.elements:
                tags |= other.elements[elem]
            # Keep only non-tombstoned tags
            live_tags = tags - result.tombstones
            if live_tags:
                result.elements[elem] = live_tags

        return result

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "elements": {str(k): list(v) for k, v in self.elements.items()},
            "tombstones": list(self.tombstones),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ORSet":
        s = cls(data["node_id"])
        s.elements = {k: set(v) for k, v in data["elements"].items()}
        s.tombstones = set(data["tombstones"])
        return s


@dataclass
class CRDTStore:
    """
    A collection of CRDTs representing a node's replicated state.

    Used by MeshPeer to manage all distributed state for a node.
    """

    node_id: str
    registers: Dict[str, LWWRegister] = field(default_factory=dict)
    counters: Dict[str, PNCounter] = field(default_factory=dict)
    sets: Dict[str, ORSet] = field(default_factory=dict)

    def get_register(self, key: str) -> LWWRegister:
        if key not in self.registers:
            self.registers[key] = LWWRegister(self.node_id)
        return self.registers[key]

    def get_counter(self, key: str) -> PNCounter:
        if key not in self.counters:
            self.counters[key] = PNCounter(self.node_id)
        return self.counters[key]

    def get_set(self, key: str) -> ORSet:
        if key not in self.sets:
            self.sets[key] = ORSet(self.node_id)
        return self.sets[key]

    def merge(self, other: "CRDTStore"):
        """Merge another store into this one."""
        for key, reg in other.registers.items():
            if key in self.registers:
                self.registers[key] = self.registers[key].merge(reg)
            else:
                self.registers[key] = LWWRegister(
                    self.node_id, reg.value, reg.timestamp
                )

        for key, ctr in other.counters.items():
            if key in self.counters:
                self.counters[key] = self.counters[key].merge(ctr)
            else:
                merged = PNCounter(self.node_id)
                merged.p = ctr.p
                merged.n = ctr.n
                self.counters[key] = merged

        for key, s in other.sets.items():
            if key in self.sets:
                self.sets[key] = self.sets[key].merge(s)
            else:
                merged = ORSet(self.node_id)
                merged.elements = {k: set(v) for k, v in s.elements.items()}
                merged.tombstones = set(s.tombstones)
                self.sets[key] = merged
