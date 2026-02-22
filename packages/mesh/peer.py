"""
Mesh Peer Networking for Summit.OS

Implements a gossip-based mesh network where nodes:
1. Discover peers via UDP broadcast
2. Exchange heartbeats for liveness detection
3. Synchronize CRDT state via anti-entropy protocol
4. Detect and handle network partitions

Designed for contested/degraded environments where:
- Nodes may lose connectivity temporarily
- No central coordinator is available
- State must converge eventually when partitions heal
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum

from packages.mesh.crdt import CRDTStore, LWWRegister
from packages.mesh.transport import TransportManager, MessageType

logger = logging.getLogger("mesh.peer")


class PeerState(str, Enum):
    ALIVE = "alive"
    SUSPECT = "suspect"
    DEAD = "dead"


@dataclass
class PeerInfo:
    """Information about a known peer."""
    peer_id: str
    address: Tuple[str, int]  # (host, port)
    state: PeerState = PeerState.ALIVE
    last_heartbeat: float = 0.0
    last_sync: float = 0.0
    incarnation: int = 0  # Monotonic counter for refuting suspicion
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Version vector for anti-entropy
    version_vector: Dict[str, int] = field(default_factory=dict)


class MeshPeer:
    """
    A node in the Summit.OS mesh network.

    Provides:
    - Peer discovery (UDP broadcast or seed list)
    - SWIM-style failure detection (heartbeat + suspicion)
    - Anti-entropy state synchronization via CRDTs
    - Partition detection and healing
    """

    def __init__(
        self,
        node_id: str | None = None,
        bind_host: str = "0.0.0.0",
        bind_port: int = 9100,
        # Timing
        heartbeat_interval: float = 1.0,
        suspicion_timeout: float = 5.0,
        dead_timeout: float = 15.0,
        sync_interval: float = 2.0,
        # Discovery
        seed_peers: list[Tuple[str, int]] | None = None,
        broadcast_port: int = 9100,
    ):
        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.heartbeat_interval = heartbeat_interval
        self.suspicion_timeout = suspicion_timeout
        self.dead_timeout = dead_timeout
        self.sync_interval = sync_interval
        self.seed_peers = seed_peers or []
        self.broadcast_port = broadcast_port

        # State
        self.peers: Dict[str, PeerInfo] = {}
        self.crdt_store = CRDTStore(self.node_id)
        self.incarnation: int = 0
        self.version: int = 0

        # Callbacks
        self._on_peer_join: List[Callable] = []
        self._on_peer_leave: List[Callable] = []
        self._on_state_change: List[Callable] = []

        # Partition tracking
        self._partition_detected = False
        self._last_reachable_count = 0

        # Running state
        self._running = False

        # Transport layer
        self._transport_mgr: Optional[TransportManager] = None

    # ── Lifecycle ───────────────────────────────────────────

    async def start(self) -> None:
        """Start the mesh peer with real UDP transport."""
        if self._running:
            return

        self._transport_mgr = TransportManager(
            bind_host=self.bind_host,
            bind_port=self.bind_port,
        )

        # Register message handlers
        self._transport_mgr.register_handler(
            MessageType.HEARTBEAT, lambda data, addr: self.process_heartbeat(data)
        )
        self._transport_mgr.register_handler(
            MessageType.SYNC_RESPONSE, lambda data, addr: self.process_sync(data)
        )
        self._transport_mgr.register_handler(
            MessageType.PING, lambda data, addr: self._handle_ping(addr)
        )

        await self._transport_mgr.start()
        self._running = True

        # Schedule periodic tasks
        self._transport_mgr.schedule_periodic(
            self._heartbeat_loop, self.heartbeat_interval
        )
        self._transport_mgr.schedule_periodic(
            self._sync_loop, self.sync_interval
        )
        self._transport_mgr.schedule_periodic(
            self._liveness_loop, self.heartbeat_interval
        )

        # Connect to seed peers
        for addr in self.seed_peers:
            self._transport_mgr.send(
                MessageType.JOIN, self._make_heartbeat(), addr
            )

        logger.info(f"MeshPeer {self.node_id} started on {self.bind_host}:{self.bind_port}")

    async def stop(self) -> None:
        """Stop the mesh peer."""
        self._running = False

        # Notify peers we're leaving
        if self._transport_mgr:
            leave_msg = {"type": "leave", "node_id": self.node_id}
            for peer in self.alive_peers():
                self._transport_mgr.send(MessageType.LEAVE, leave_msg, peer.address)
            await self._transport_mgr.stop()

        logger.info(f"MeshPeer {self.node_id} stopped")

    async def _heartbeat_loop(self) -> None:
        """Send heartbeats to all alive peers."""
        msg = self._make_heartbeat()
        for peer in self.alive_peers() + self.suspect_peers():
            if self._transport_mgr:
                self._transport_mgr.send(MessageType.HEARTBEAT, msg, peer.address)

    async def _sync_loop(self) -> None:
        """Trigger anti-entropy sync with a random alive peer."""
        import random
        alive = self.alive_peers()
        if alive and self._transport_mgr:
            target = random.choice(alive)
            sync_req = self._make_sync_request(target.version_vector)
            self._transport_mgr.send(MessageType.SYNC_REQUEST, sync_req, target.address)

    async def _liveness_loop(self) -> None:
        """Run failure detection."""
        self.check_liveness()

    def _handle_ping(self, addr: Tuple[str, int]) -> None:
        """Respond to a ping."""
        if self._transport_mgr:
            self._transport_mgr.send(
                MessageType.PING_ACK,
                {"node_id": self.node_id, "timestamp": time.time()},
                addr,
            )

    # ── Event Registration ──────────────────────────────────

    def on_peer_join(self, callback: Callable[[str, PeerInfo], None]):
        self._on_peer_join.append(callback)

    def on_peer_leave(self, callback: Callable[[str, PeerInfo], None]):
        self._on_peer_leave.append(callback)

    def on_state_change(self, callback: Callable[[str, Any], None]):
        self._on_state_change.append(callback)

    # ── Core Protocol Messages ──────────────────────────────

    def _make_heartbeat(self) -> Dict:
        """Create a heartbeat message."""
        return {
            "type": "heartbeat",
            "node_id": self.node_id,
            "incarnation": self.incarnation,
            "timestamp": time.time(),
            "address": (self.bind_host, self.bind_port),
            "peer_count": len(self.alive_peers()),
            "version": self.version,
        }

    def _make_sync_request(self, target_version: Dict[str, int]) -> Dict:
        """Create an anti-entropy sync request."""
        return {
            "type": "sync_request",
            "node_id": self.node_id,
            "version_vector": self._get_version_vector(),
            "target_vector": target_version,
        }

    def _make_sync_response(self, requested_keys: List[str]) -> Dict:
        """Create sync response with requested CRDT states."""
        state_data = {}
        for key in requested_keys:
            if key in self.crdt_store.registers:
                state_data[key] = {
                    "type": "register",
                    "data": self.crdt_store.registers[key].to_dict(),
                }
        return {
            "type": "sync_response",
            "node_id": self.node_id,
            "state": state_data,
            "version_vector": self._get_version_vector(),
        }

    def _get_version_vector(self) -> Dict[str, int]:
        """Get current version vector."""
        vv = {self.node_id: self.version}
        for pid, peer in self.peers.items():
            if peer.version_vector:
                vv[pid] = peer.version_vector.get(pid, 0)
        return vv

    # ── Heartbeat Processing ────────────────────────────────

    def process_heartbeat(self, message: Dict):
        """Process an incoming heartbeat from a peer."""
        sender_id = message["node_id"]
        if sender_id == self.node_id:
            return

        now = time.time()

        if sender_id not in self.peers:
            # New peer discovered
            peer = PeerInfo(
                peer_id=sender_id,
                address=tuple(message["address"]),
                state=PeerState.ALIVE,
                last_heartbeat=now,
                incarnation=message.get("incarnation", 0),
            )
            self.peers[sender_id] = peer
            logger.info(f"Peer discovered: {sender_id} at {peer.address}")
            for cb in self._on_peer_join:
                cb(sender_id, peer)
        else:
            peer = self.peers[sender_id]
            peer.last_heartbeat = now
            peer.incarnation = max(peer.incarnation, message.get("incarnation", 0))

            # Revive suspect peers
            if peer.state == PeerState.SUSPECT:
                peer.state = PeerState.ALIVE
                logger.info(f"Peer {sender_id} revived from suspect")
            elif peer.state == PeerState.DEAD:
                peer.state = PeerState.ALIVE
                logger.info(f"Peer {sender_id} rejoined from dead")
                for cb in self._on_peer_join:
                    cb(sender_id, peer)

        # Update version from peer
        peer.version_vector[sender_id] = message.get("version", 0)

    # ── Failure Detection ───────────────────────────────────

    def check_liveness(self):
        """
        SWIM-style failure detection.

        Alive → Suspect (after suspicion_timeout without heartbeat)
        Suspect → Dead (after dead_timeout without heartbeat)
        """
        now = time.time()

        for pid, peer in list(self.peers.items()):
            elapsed = now - peer.last_heartbeat

            if peer.state == PeerState.ALIVE and elapsed > self.suspicion_timeout:
                peer.state = PeerState.SUSPECT
                logger.warning(f"Peer {pid} is SUSPECT (no heartbeat for {elapsed:.1f}s)")

            elif peer.state == PeerState.SUSPECT and elapsed > self.dead_timeout:
                peer.state = PeerState.DEAD
                logger.warning(f"Peer {pid} is DEAD (no heartbeat for {elapsed:.1f}s)")
                for cb in self._on_peer_leave:
                    cb(pid, peer)

        # Partition detection
        alive = len(self.alive_peers())
        if self._last_reachable_count > 0 and alive == 0 and not self._partition_detected:
            self._partition_detected = True
            logger.error("NETWORK PARTITION DETECTED — all peers unreachable")
        elif alive > 0 and self._partition_detected:
            self._partition_detected = False
            logger.info("PARTITION HEALED — peers reachable again")
        self._last_reachable_count = alive

    # ── Anti-Entropy Sync ───────────────────────────────────

    def process_sync(self, message: Dict):
        """Process incoming sync data and merge CRDTs."""
        sender_id = message["node_id"]
        state_data = message.get("state", {})

        for key, entry in state_data.items():
            if entry["type"] == "register":
                remote_reg = LWWRegister.from_dict(entry["data"])
                if key in self.crdt_store.registers:
                    self.crdt_store.registers[key] = (
                        self.crdt_store.registers[key].merge(remote_reg)
                    )
                else:
                    self.crdt_store.registers[key] = LWWRegister(
                        self.node_id, remote_reg.value, remote_reg.timestamp
                    )

                # Notify state change
                for cb in self._on_state_change:
                    cb(key, self.crdt_store.registers[key].get())

        # Update version vector for sender
        if sender_id in self.peers:
            self.peers[sender_id].version_vector = message.get("version_vector", {})

    # ── State Access ────────────────────────────────────────

    def set_state(self, key: str, value: Any):
        """Set a replicated state value (LWW register)."""
        reg = self.crdt_store.get_register(key)
        reg.set(value)
        self.version += 1

    def get_state(self, key: str) -> Any:
        """Get a replicated state value."""
        reg = self.crdt_store.registers.get(key)
        return reg.get() if reg else None

    def alive_peers(self) -> List[PeerInfo]:
        """Get list of alive peers."""
        return [p for p in self.peers.values() if p.state == PeerState.ALIVE]

    def suspect_peers(self) -> List[PeerInfo]:
        return [p for p in self.peers.values() if p.state == PeerState.SUSPECT]

    def is_partitioned(self) -> bool:
        return self._partition_detected

    def get_mesh_status(self) -> Dict:
        """Get mesh network status summary."""
        return {
            "node_id": self.node_id,
            "peers": {
                "alive": len(self.alive_peers()),
                "suspect": len(self.suspect_peers()),
                "dead": len([p for p in self.peers.values() if p.state == PeerState.DEAD]),
            },
            "partitioned": self._partition_detected,
            "version": self.version,
            "crdt_keys": (
                list(self.crdt_store.registers.keys()) +
                list(self.crdt_store.counters.keys()) +
                list(self.crdt_store.sets.keys())
            ),
        }
