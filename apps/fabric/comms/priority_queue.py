"""
Tactical Priority Message Queue — Summit.OS Fabric

SQLite WAL-backed store-and-forward queue for degraded/intermittent links.
Messages survive service restarts. Higher priority messages drain first.

Priority levels:
  0 — CRITICAL (C2, emergency abort, kill-switch)
  1 — HIGH     (mission update, alert, status change)
  2 — NORMAL   (telemetry, position, sensor data)
  3 — LOW      (bulk transfer, logs, analytics)

Schema: messages(id TEXT PK, priority INT, topic TEXT, payload BLOB,
                 created_at REAL, ttl_s REAL, retry_count INT, sent_at REAL)
WAL mode ensures concurrent readers don't block writers.
"""

import asyncio
import logging
import os
import sqlite3
import time
import uuid
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_PRIORITY_LABELS = {0: "CRITICAL", 1: "HIGH", 2: "NORMAL", 3: "LOW"}

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    priority    INTEGER NOT NULL DEFAULT 2,
    topic       TEXT NOT NULL,
    payload     BLOB NOT NULL,
    created_at  REAL NOT NULL,
    ttl_s       REAL NOT NULL DEFAULT 300.0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    sent_at     REAL
);
CREATE INDEX IF NOT EXISTS idx_priority_created ON messages (priority ASC, created_at ASC);
"""


class PriorityQueue:
    """SQLite WAL-backed priority message queue for degraded-link store-and-forward."""

    def __init__(self, db_path: str = None):
        self._db_path = db_path or os.environ.get(
            "SUMMIT_MSGQ_DB", "/tmp/summit_msgq.db"
        )
        self._lock = asyncio.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_CREATE_TABLE)
        conn.commit()
        self._conn = conn
        logger.debug("PriorityQueue: DB initialised at %s (WAL)", self._db_path)

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._init_db()
        return self._conn  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def put(
        self,
        topic: str,
        payload: bytes,
        priority: int = 2,
        ttl_s: float = 300.0,
    ) -> str:
        """Insert a message. Returns the generated message ID."""
        msg_id = str(uuid.uuid4())
        now = time.time()
        async with self._lock:
            self._get_conn().execute(
                "INSERT INTO messages (id, priority, topic, payload, created_at, ttl_s, retry_count, sent_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 0, NULL)",
                (msg_id, priority, topic, payload, now, ttl_s),
            )
            self._get_conn().commit()
        logger.debug("PriorityQueue.put: id=%s priority=%d topic=%s", msg_id, priority, topic)
        return msg_id

    async def get_batch(self, n: int = 20) -> List[dict]:
        """
        Return up to *n* pending messages ordered by priority ASC, created_at ASC.
        Expired messages (created_at + ttl_s < now) are skipped.
        """
        now = time.time()
        async with self._lock:
            rows = self._get_conn().execute(
                "SELECT id, priority, topic, payload, created_at, ttl_s, retry_count "
                "FROM messages "
                "WHERE (created_at + ttl_s) >= ? AND sent_at IS NULL "
                "ORDER BY priority ASC, created_at ASC "
                "LIMIT ?",
                (now, n),
            ).fetchall()

        result = []
        for row in rows:
            result.append(
                {
                    "id": row[0],
                    "priority": row[1],
                    "topic": row[2],
                    "payload": row[3],
                    "created_at": row[4],
                    "ttl_s": row[5],
                    "retry_count": row[6],
                }
            )
        return result

    async def ack(self, msg_id: str) -> None:
        """Mark message as sent. Deletes after 3 accumulated acks."""
        now = time.time()
        async with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT retry_count FROM messages WHERE id = ?", (msg_id,)
            ).fetchone()
            if row is None:
                return
            ack_count = row[0] + 1
            if ack_count >= 3:
                conn.execute("DELETE FROM messages WHERE id = ?", (msg_id,))
                logger.debug("PriorityQueue.ack: deleted id=%s after 3 acks", msg_id)
            else:
                conn.execute(
                    "UPDATE messages SET sent_at = ?, retry_count = ? WHERE id = ?",
                    (now, ack_count, msg_id),
                )
            conn.commit()

    async def nack(self, msg_id: str) -> None:
        """
        Increment retry_count and apply exponential backoff by postponing created_at.
        Backoff: created_at += 2^retry_count * 10 seconds.
        """
        async with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT retry_count, created_at FROM messages WHERE id = ?", (msg_id,)
            ).fetchone()
            if row is None:
                return
            retry_count = row[0]
            created_at = row[1]
            backoff = (2 ** retry_count) * 10
            new_created_at = created_at + backoff
            conn.execute(
                "UPDATE messages SET retry_count = ?, created_at = ?, sent_at = NULL WHERE id = ?",
                (retry_count + 1, new_created_at, msg_id),
            )
            conn.commit()
        logger.debug(
            "PriorityQueue.nack: id=%s retry=%d backoff=%.1fs",
            msg_id,
            retry_count + 1,
            backoff,
        )

    async def depth(self) -> Dict[str, int]:
        """Return per-priority message counts keyed by label name."""
        async with self._lock:
            rows = self._get_conn().execute(
                "SELECT priority, COUNT(*) FROM messages WHERE sent_at IS NULL GROUP BY priority"
            ).fetchall()

        counts = {label: 0 for label in _PRIORITY_LABELS.values()}
        for priority, count in rows:
            label = _PRIORITY_LABELS.get(priority, str(priority))
            counts[label] = count
        return counts

    async def purge_expired(self) -> int:
        """Delete expired messages. Returns count of deleted rows."""
        now = time.time()
        async with self._lock:
            cur = self._get_conn().execute(
                "DELETE FROM messages WHERE (created_at + ttl_s) < ?", (now,)
            )
            self._get_conn().commit()
            deleted = cur.rowcount
        if deleted:
            logger.debug("PriorityQueue.purge_expired: removed %d expired messages", deleted)
        return deleted


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_queue_instance: Optional[PriorityQueue] = None
_queue_lock = asyncio.Lock()


async def get_queue(db_path: str = None) -> PriorityQueue:
    """Return the module-level singleton PriorityQueue."""
    global _queue_instance
    if _queue_instance is None:
        # asyncio.Lock cannot be acquired at import time, so we guard with a simple check first.
        _queue_instance = PriorityQueue(db_path=db_path)
    return _queue_instance
