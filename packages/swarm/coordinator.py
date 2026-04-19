"""
Swarm Coordinator — Heli.OS

Peer-to-peer swarm coordination without a central server.
Leader election uses a scored vote:
  score = battery_pct * 0.6 + link_quality * 0.4

The node with the highest score becomes leader. All nodes broadcast their
score every 5s. The leader runs SwarmPlanner.replan() and broadcasts the
TaskCRDT state. Non-leaders merge incoming CRDT state.

Designed to work over Meshtastic (low-bandwidth, high-latency) mesh.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Dict, Optional

from packages.swarm.task_crdt import TaskCRDT
from packages.swarm.swarm_planner import SwarmPlanner

logger = logging.getLogger("swarm.coordinator")

_BROADCAST_INTERVAL = 5.0   # seconds between score + CRDT broadcasts
_PEER_TIMEOUT = 15.0         # seconds before a silent peer is considered gone


class _PeerScore:
    """Holds the last-known score and announcement time for a peer."""

    __slots__ = ("peer_id", "score", "last_seen")

    def __init__(self, peer_id: str, score: float) -> None:
        self.peer_id = peer_id
        self.score = score
        self.last_seen: float = time.time()

    def refresh(self, score: float) -> None:
        self.score = score
        self.last_seen = time.time()


class SwarmCoordinator:
    """
    Decentralised leader election and task distribution for a UAV swarm.

    Each node maintains its own score derived from battery and link quality.
    Every 5 seconds each node:
      1. Broadcasts its score + current CRDT state (via broadcast_fn).
      2. If it is the leader, calls replan() and broadcasts the result.

    Incoming peer announcements are processed via on_peer_announcement().
    """

    def __init__(
        self,
        node_id: str,
        task_crdt: TaskCRDT,
        planner: SwarmPlanner,
        broadcast_fn: Callable,
    ) -> None:
        self.node_id = node_id
        self._crdt = task_crdt
        self._planner = planner
        self._broadcast = broadcast_fn

        self._battery_pct: float = 100.0
        self._link_quality: float = 1.0
        self._own_score: float = self._compute_score(100.0, 1.0)

        # peer_id → _PeerScore
        self._peers: Dict[str, _PeerScore] = {}

        self._running = False

    # ── Self-state ───────────────────────────────────────────

    def update_self(self, battery_pct: float, link_quality: float) -> None:
        """Update this node's battery and link-quality metrics."""
        self._battery_pct = battery_pct
        self._link_quality = link_quality
        self._own_score = self._compute_score(battery_pct, link_quality)

    # ── Peer Handling ────────────────────────────────────────

    async def on_peer_announcement(
        self, peer_id: str, score: float, crdt_payload: dict
    ) -> None:
        """
        Handle an incoming swarm state broadcast from a peer.

        Updates the peer's score record and merges the remote CRDT into
        the local store. Should be called from the mesh receive callback.
        """
        if peer_id == self.node_id:
            return  # Ignore own reflections

        peer = self._peers.get(peer_id)
        if peer is None:
            self._peers[peer_id] = _PeerScore(peer_id, score)
            logger.info("New swarm peer: %s (score=%.2f)", peer_id, score)
        else:
            peer.refresh(score)

        # Merge remote CRDT
        try:
            remote_crdt = TaskCRDT.from_payload(crdt_payload, peer_id)
            self._crdt.merge(remote_crdt)
        except Exception as exc:
            logger.warning("CRDT merge failed from peer %s: %s", peer_id, exc)

    def is_leader(self) -> bool:
        """
        Return True if this node has the highest (or joint-highest) score.

        Stale peers (last_seen > _PEER_TIMEOUT) are excluded from the
        comparison so a crashed leader is replaced promptly.
        """
        now = time.time()
        for peer in self._peers.values():
            if now - peer.last_seen > _PEER_TIMEOUT:
                continue
            if peer.score > self._own_score:
                return False
            if peer.score == self._own_score and peer.peer_id > self.node_id:
                # Tie-break by node_id lexicographic order (larger wins)
                return False
        return True

    def get_leader_id(self) -> Optional[str]:
        """
        Return the peer_id of the current leader, or self.node_id if self is leader.
        """
        if self.is_leader():
            return self.node_id

        now = time.time()
        best_peer_id: Optional[str] = None
        best_score: float = self._own_score

        for peer in self._peers.values():
            if now - peer.last_seen > _PEER_TIMEOUT:
                continue
            if peer.score > best_score or (
                peer.score == best_score and peer.peer_id > (best_peer_id or self.node_id)
            ):
                best_score = peer.score
                best_peer_id = peer.peer_id

        return best_peer_id or self.node_id

    # ── Main Loop ────────────────────────────────────────────

    async def run(self) -> None:
        """
        Background coordination loop.

        Broadcasts own score + CRDT every 5 s. When this node is leader,
        also triggers SwarmPlanner.replan() and broadcasts the updated CRDT.
        """
        self._running = True
        logger.info("SwarmCoordinator %s started", self.node_id)

        while self._running:
            try:
                await self._broadcast_state()

                if self.is_leader():
                    logger.debug("Node %s is leader — running replan", self.node_id)
                    self._planner.replan()
                    # Broadcast post-replan CRDT so followers converge
                    await self._broadcast_state()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Coordinator loop error: %s", exc, exc_info=True)

            await asyncio.sleep(_BROADCAST_INTERVAL)

        logger.info("SwarmCoordinator %s stopped", self.node_id)

    async def _broadcast_state(self) -> None:
        """Serialise and broadcast own score + CRDT payload."""
        payload = {
            "type": "swarm_state",
            "peer_id": self.node_id,
            "score": self._own_score,
            "crdt": self._crdt.to_payload(),
            "ts": time.time(),
        }
        try:
            await self._broadcast(payload)
        except Exception as exc:
            logger.warning("Broadcast failed: %s", exc)

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _compute_score(battery_pct: float, link_quality: float) -> float:
        """Compute election score: battery weighted 60%, link quality 40%."""
        return (battery_pct / 100.0) * 0.6 + link_quality * 0.4
