"""
Swarm Task CRDT — Heli.OS

Conflict-free replicated data type for distributed task assignment.
Uses OR-Set semantics: tasks can be assigned to assets and reassigned;
the assignment with the highest (timestamp, node_id) wins per task.

Designed to work over Meshtastic mesh where messages may be delayed,
duplicated, or reordered. Converges to consistent state regardless of
message order.
"""

from __future__ import annotations

import time
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger("swarm.task_crdt")

# Internal entry: (asset_id, timestamp, originating_node_id, done)
_Entry = Tuple[str, float, str, bool]


class TaskCRDT:
    """
    Last-write-wins task assignment register, replicated across swarm nodes.

    Each task_id maps to the assignment that has the greatest
    (timestamp, node_id) tuple — node_id breaks ties deterministically.

    Completed tasks are retained in the store with a done=True flag so
    convergence is preserved when stale ASSIGN messages arrive late.
    """

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        # task_id → (asset_id, ts, originating_node_id, done)
        self._store: Dict[str, _Entry] = {}

    # ── Mutations ────────────────────────────────────────────

    def assign(self, task_id: str, asset_id: str) -> None:
        """Record that task_id is assigned to asset_id (from this node)."""
        ts = time.time()
        existing = self._store.get(task_id)
        if existing is None or self._dominates(ts, self.node_id, existing[1], existing[2]):
            self._store[task_id] = (asset_id, ts, self.node_id, False)
            logger.debug("CRDT assign: task=%s asset=%s ts=%.4f", task_id, asset_id, ts)

    def complete(self, task_id: str, asset_id: str) -> None:
        """Mark task_id as completed by asset_id."""
        ts = time.time()
        existing = self._store.get(task_id)
        if existing is None or self._dominates(ts, self.node_id, existing[1], existing[2]):
            self._store[task_id] = (asset_id, ts, self.node_id, True)
            logger.debug("CRDT complete: task=%s asset=%s", task_id, asset_id)

    def unassign(self, task_id: str) -> None:
        """
        Remove the current assignment for task_id.

        Implemented as a tombstone-style assign to the empty string so
        the operation replicates cleanly during merge.
        """
        ts = time.time()
        existing = self._store.get(task_id)
        if existing is None or self._dominates(ts, self.node_id, existing[1], existing[2]):
            self._store[task_id] = ("", ts, self.node_id, False)
            logger.debug("CRDT unassign: task=%s", task_id)

    # ── Reads ────────────────────────────────────────────────

    def get_assignment(self, task_id: str) -> Optional[str]:
        """
        Return the current asset_id assigned to task_id, or None.

        Returns None for completed or tombstoned (empty asset_id) entries.
        """
        entry = self._store.get(task_id)
        if entry is None:
            return None
        asset_id, _, _, done = entry
        if done or not asset_id:
            return None
        return asset_id

    def get_all_assignments(self) -> Dict[str, str]:
        """
        Return a dict of task_id → asset_id for all active (non-complete)
        assignments with a non-empty asset_id.
        """
        return {
            tid: entry[0]
            for tid, entry in self._store.items()
            if entry[0] and not entry[3]
        }

    # ── Merge ────────────────────────────────────────────────

    def merge(self, remote: "TaskCRDT") -> None:
        """
        Merge remote CRDT state into self using LWW per task.

        For each task_id the entry with the higher (ts, node_id) wins.
        Idempotent and commutative — safe to call multiple times with the
        same remote state.
        """
        for task_id, remote_entry in remote._store.items():
            r_asset, r_ts, r_node, r_done = remote_entry
            local_entry = self._store.get(task_id)

            if local_entry is None:
                self._store[task_id] = remote_entry
            else:
                _, l_ts, l_node, _ = local_entry
                if self._dominates(r_ts, r_node, l_ts, l_node):
                    self._store[task_id] = remote_entry

    # ── Serialization ────────────────────────────────────────

    def to_payload(self) -> dict:
        """Return a JSON-serializable dict for mesh broadcast."""
        return {
            "node_id": self.node_id,
            "assignments": {
                tid: {
                    "asset_id": e[0],
                    "ts": e[1],
                    "origin_node": e[2],
                    "done": e[3],
                }
                for tid, e in self._store.items()
            },
        }

    @classmethod
    def from_payload(cls, payload: dict, node_id: str) -> "TaskCRDT":
        """Deserialize a payload produced by to_payload()."""
        crdt = cls(node_id=node_id)
        for tid, v in payload.get("assignments", {}).items():
            crdt._store[tid] = (
                v["asset_id"],
                float(v["ts"]),
                v["origin_node"],
                bool(v["done"]),
            )
        return crdt

    # ── Internal ─────────────────────────────────────────────

    @staticmethod
    def _dominates(ts_a: float, node_a: str, ts_b: float, node_b: str) -> bool:
        """Return True if (ts_a, node_a) strictly dominates (ts_b, node_b)."""
        if ts_a > ts_b:
            return True
        if ts_a == ts_b and node_a > node_b:
            return True
        return False
