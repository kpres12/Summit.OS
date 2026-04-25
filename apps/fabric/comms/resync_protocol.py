"""
Resync Protocol — Heli.OS Fabric

After a link reconnects, drains the PriorityQueue in batches and delivers to
a transport callback. Tracks per-peer last-seen sequence numbers so duplicate
messages are skipped (sequence deduplication, not CRDT).
"""

import asyncio
import logging
from collections import deque
from typing import Awaitable, Callable, Dict, List

from .priority_queue import PriorityQueue

logger = logging.getLogger(__name__)

_SEEN_SET_MAX = 10_000
_PURGE_INTERVAL_S = 5.0
_BATCH_SIZE = 20


class ResyncProtocol:
    """
    Manages post-reconnect drain of the PriorityQueue and per-peer deduplication
    of inbound messages.
    """

    def __init__(
        self,
        queue: PriorityQueue,
        send_fn: Callable[[str, bytes], Awaitable[None]],
    ):
        self._queue = queue
        self._send_fn = send_fn

        # per-peer seen-message dedup: peer_id -> deque of msg_ids (bounded)
        self._seen: Dict[str, deque] = {}
        self._seen_set: Dict[str, set] = {}

        # per-peer last sequence number
        self._last_seq: Dict[str, int] = {}

        # topic-prefix -> handler callable
        self._handlers: List[tuple] = []  # list of (prefix, handler)

        self._running = False

    # ------------------------------------------------------------------
    # Resync (outbound drain on reconnect)
    # ------------------------------------------------------------------

    async def start_resync(self, peer_id: str) -> None:
        """
        Drain the priority queue and deliver all pending messages to *peer_id*
        via send_fn. Tracks sequence numbers per peer to detect drift.
        """
        logger.info("ResyncProtocol.start_resync: starting drain for peer=%s", peer_id)
        drained = 0
        while True:
            batch = await self._queue.get_batch(n=_BATCH_SIZE)
            if not batch:
                break
            for msg in batch:
                try:
                    # Compose delivery bytes: include topic as a simple header
                    # separated by newline so the receiver can parse it.
                    topic_header = msg["topic"].encode() + b"\n"
                    wire_payload = topic_header + msg["payload"]
                    await self._send_fn(peer_id, wire_payload)
                    await self._queue.ack(msg["id"])

                    # Update sequence tracking
                    seq = self._last_seq.get(peer_id, 0) + 1
                    self._last_seq[peer_id] = seq
                    drained += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "ResyncProtocol.start_resync: send failed for msg_id=%s: %s",
                        msg["id"],
                        exc,
                    )
                    await self._queue.nack(msg["id"])

        logger.info(
            "ResyncProtocol.start_resync: drained %d messages to peer=%s", drained, peer_id
        )

    # ------------------------------------------------------------------
    # Inbound message handling with deduplication
    # ------------------------------------------------------------------

    async def on_message_received(self, peer_id: str, msg: dict) -> None:
        """
        Process an inbound message from *peer_id*.  Deduplicates via a bounded
        seen-set (max 10k entries, evict oldest on overflow).  Dispatches to
        registered handlers.
        """
        msg_id = msg.get("id")
        if msg_id is None:
            logger.debug("ResyncProtocol: dropping message with no id from peer=%s", peer_id)
            return

        # Per-peer dedup structures
        if peer_id not in self._seen_set:
            self._seen[peer_id] = deque()
            self._seen_set[peer_id] = set()

        seen_set = self._seen_set[peer_id]
        seen_deque = self._seen[peer_id]

        if msg_id in seen_set:
            logger.debug(
                "ResyncProtocol: duplicate msg_id=%s from peer=%s — skipping", msg_id, peer_id
            )
            return

        # Add to seen, evict oldest if over limit
        seen_deque.append(msg_id)
        seen_set.add(msg_id)
        while len(seen_set) > _SEEN_SET_MAX:
            evicted = seen_deque.popleft()
            seen_set.discard(evicted)

        # Route to handlers
        topic = msg.get("topic", "")
        for prefix, handler in self._handlers:
            if topic.startswith(prefix):
                try:
                    result = handler(msg)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "ResyncProtocol: handler error for topic=%s: %s", topic, exc
                    )

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register_handler(self, topic_prefix: str, handler: Callable) -> None:
        """Register a handler for messages whose topic starts with *topic_prefix*."""
        self._handlers.append((topic_prefix, handler))
        logger.debug(
            "ResyncProtocol.register_handler: registered handler for prefix='%s'", topic_prefix
        )

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Background maintenance loop.
        Every 5 seconds: purge expired messages from the queue and log depth at DEBUG.
        """
        self._running = True
        logger.info("ResyncProtocol.run: background loop started")
        try:
            while self._running:
                await asyncio.sleep(_PURGE_INTERVAL_S)
                try:
                    await self._queue.purge_expired()
                    depth = await self._queue.depth()
                    logger.debug("ResyncProtocol queue depth: %s", depth)
                except Exception as exc:  # noqa: BLE001
                    logger.error("ResyncProtocol.run: maintenance error: %s", exc)
        except asyncio.CancelledError:
            logger.info("ResyncProtocol.run: loop cancelled")
            raise

    def stop(self) -> None:
        """Signal the background loop to stop."""
        self._running = False
