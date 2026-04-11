"""
Link Monitor — Summit.OS Fabric

Tracks per-peer link quality metrics: round-trip time, packet loss, and a
composite link_score (0.0–1.0) used by the swarm coordinator for leader election.

Probes are sent via registered probe_fn. Results fed back via on_probe_response().
"""

import asyncio
import logging
import time
import uuid
from typing import Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_S = 1.0
_RTT_WINDOW = 20        # rolling average over last N samples
_LOSS_WINDOW = 20       # rolling window for packet loss calculation


class _PeerStats:
    """Per-peer link statistics."""

    __slots__ = (
        "rtt_samples",
        "loss_window",
        "pending_probes",  # probe_id -> sent_at
    )

    def __init__(self):
        self.rtt_samples: List[float] = []      # recent RTT values (ms)
        self.loss_window: List[bool] = []        # True=received, False=lost
        self.pending_probes: Dict[str, float] = {}  # probe_id -> sent_at

    @property
    def avg_rtt_ms(self) -> float:
        if not self.rtt_samples:
            return 0.0
        return sum(self.rtt_samples) / len(self.rtt_samples)

    @property
    def loss_rate(self) -> float:
        if not self.loss_window:
            return 0.0
        lost = sum(1 for v in self.loss_window if not v)
        return lost / len(self.loss_window)

    def record_rtt(self, rtt_ms: float) -> None:
        self.rtt_samples.append(rtt_ms)
        if len(self.rtt_samples) > _RTT_WINDOW:
            self.rtt_samples.pop(0)

    def record_result(self, received: bool) -> None:
        self.loss_window.append(received)
        if len(self.loss_window) > _LOSS_WINDOW:
            self.loss_window.pop(0)


class LinkMonitor:
    """
    Monitors link quality per peer using active RTT probes.

    link_score formula: 0.7*(1 - loss_rate) + 0.3*(1 - min(rtt_ms/500, 1.0))
    Score range: 0.0 (link dead) to 1.0 (perfect link).
    """

    def __init__(
        self,
        node_id: str,
        probe_fn: Callable[[str, bytes], Awaitable[None]] = None,
        probe_interval_s: float = 2.0,
    ):
        self._node_id = node_id
        self._probe_fn = probe_fn
        self._probe_interval_s = probe_interval_s
        self._peers: Dict[str, _PeerStats] = {}
        self._lock = asyncio.Lock()
        self._running = False

    def _get_or_create_peer(self, peer_id: str) -> _PeerStats:
        if peer_id not in self._peers:
            self._peers[peer_id] = _PeerStats()
        return self._peers[peer_id]

    # ------------------------------------------------------------------
    # Probe response handlers
    # ------------------------------------------------------------------

    async def on_probe_response(
        self, peer_id: str, rtt_ms: float, probe_id: str
    ) -> None:
        """Called when a probe response arrives from *peer_id*."""
        async with self._lock:
            stats = self._get_or_create_peer(peer_id)
            stats.pending_probes.pop(probe_id, None)
            stats.record_rtt(rtt_ms)
            stats.record_result(received=True)
        logger.debug(
            "LinkMonitor.on_probe_response: peer=%s probe=%s rtt=%.2fms score=%.3f",
            peer_id,
            probe_id,
            rtt_ms,
            self.get_link_score(peer_id),
        )

    async def on_probe_timeout(self, peer_id: str, probe_id: str) -> None:
        """Called when a probe to *peer_id* times out."""
        async with self._lock:
            stats = self._get_or_create_peer(peer_id)
            stats.pending_probes.pop(probe_id, None)
            stats.record_result(received=False)
        logger.debug(
            "LinkMonitor.on_probe_timeout: peer=%s probe=%s score=%.3f",
            peer_id,
            probe_id,
            self.get_link_score(peer_id),
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def get_link_score(self, peer_id: str) -> float:
        """
        Return composite link score 0.0–1.0.
        0.7*(1 - loss_rate) + 0.3*(1 - min(rtt_ms/500, 1.0))
        Returns 0.0 for unknown peers.
        """
        stats = self._peers.get(peer_id)
        if stats is None:
            return 0.0
        loss_component = 1.0 - stats.loss_rate
        rtt_component = 1.0 - min(stats.avg_rtt_ms / 500.0, 1.0)
        return 0.7 * loss_component + 0.3 * rtt_component

    def get_all_scores(self) -> Dict[str, float]:
        """Return link scores for all known peers."""
        return {peer_id: self.get_link_score(peer_id) for peer_id in self._peers}

    # ------------------------------------------------------------------
    # Background probe loop
    # ------------------------------------------------------------------

    async def _probe_peer(self, peer_id: str) -> None:
        """Send a single probe and schedule a timeout callback."""
        if self._probe_fn is None:
            return

        probe_id = str(uuid.uuid4())
        probe_payload = probe_id.encode()
        sent_at = time.monotonic()

        async with self._lock:
            stats = self._get_or_create_peer(peer_id)
            stats.pending_probes[probe_id] = sent_at

        try:
            await self._probe_fn(peer_id, probe_payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LinkMonitor: probe send failed for peer=%s: %s", peer_id, exc)
            await self.on_probe_timeout(peer_id, probe_id)
            return

        # Schedule timeout check
        await asyncio.sleep(_PROBE_TIMEOUT_S)
        async with self._lock:
            still_pending = probe_id in self._peers.get(peer_id, _PeerStats()).pending_probes
        if still_pending:
            await self.on_probe_timeout(peer_id, probe_id)

    async def run(self) -> None:
        """
        Background loop: probe all known peers every *probe_interval_s* seconds.
        Times out each probe after 1 second.
        """
        self._running = True
        logger.info("LinkMonitor.run: probe loop started (node=%s)", self._node_id)
        try:
            while self._running:
                peers_snapshot = list(self._peers.keys())
                if peers_snapshot:
                    probe_tasks = [
                        asyncio.create_task(self._probe_peer(peer_id))
                        for peer_id in peers_snapshot
                    ]
                    await asyncio.gather(*probe_tasks, return_exceptions=True)
                await asyncio.sleep(self._probe_interval_s)
        except asyncio.CancelledError:
            logger.info("LinkMonitor.run: loop cancelled")
            raise

    def stop(self) -> None:
        """Signal the background loop to stop."""
        self._running = False

    def add_peer(self, peer_id: str) -> None:
        """Register a peer to be monitored."""
        if peer_id not in self._peers:
            self._peers[peer_id] = _PeerStats()
            logger.debug("LinkMonitor.add_peer: registered peer=%s", peer_id)

    def remove_peer(self, peer_id: str) -> None:
        """Remove a peer from monitoring."""
        self._peers.pop(peer_id, None)
        logger.debug("LinkMonitor.remove_peer: removed peer=%s", peer_id)
