"""
Replay Persistence — Heli.OS Tasking

Persists mission replay snapshots to SQLite for long-term storage and
post-mission analysis. The in-memory replay_router only holds 1 hour;
this persists indefinitely (bounded by disk).

Schema: replay_snapshots(id INTEGER PK, mission_id TEXT, ts REAL, snapshot BLOB)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import zlib
from datetime import datetime, timezone
from typing import List

logger = logging.getLogger("tasking.replay_persistence")

_DEFAULT_DB = os.getenv("SUMMIT_REPLAY_DB", "./replay.db")


class ReplayPersistence:
    """Persists mission replay snapshots to SQLite with ZLIB compression."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or _DEFAULT_DB
        self._init_db()

    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS replay_snapshots (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    mission_id  TEXT    NOT NULL,
                    ts          REAL    NOT NULL,
                    snapshot    BLOB    NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_replay_mission_ts ON replay_snapshots (mission_id, ts)"
            )
            conn.commit()

    # ------------------------------------------------------------------
    def save_snapshot(self, mission_id: str, snapshot: dict) -> None:
        """Compress and persist a single snapshot."""
        ts_iso = snapshot.get("ts_iso", datetime.now(timezone.utc).isoformat())
        try:
            ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).timestamp()
        except Exception:
            import time
            ts = time.time()

        raw = json.dumps(snapshot).encode()
        compressed = zlib.compress(raw, level=6)

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO replay_snapshots (mission_id, ts, snapshot) VALUES (?, ?, ?)",
                (mission_id, ts, compressed),
            )
            conn.commit()

    # ------------------------------------------------------------------
    def load_timeline(self, mission_id: str) -> List[dict]:
        """Load all snapshots for a mission, ordered by timestamp."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT snapshot FROM replay_snapshots WHERE mission_id=? ORDER BY ts ASC",
                (mission_id,),
            ).fetchall()

        result: List[dict] = []
        for row in rows:
            try:
                decompressed = zlib.decompress(row[0])
                result.append(json.loads(decompressed))
            except Exception as exc:
                logger.error("Failed to decompress snapshot for mission=%s: %s", mission_id, exc)

        return result

    # ------------------------------------------------------------------
    def list_missions(self) -> List[str]:
        """Return distinct mission_ids with stored replay data."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT mission_id FROM replay_snapshots ORDER BY mission_id"
            ).fetchall()
        return [row[0] for row in rows]

    # ------------------------------------------------------------------
    def delete_mission(self, mission_id: str) -> int:
        """Delete all snapshots for a mission. Returns number of rows deleted."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM replay_snapshots WHERE mission_id=?", (mission_id,)
            )
            conn.commit()
            deleted = cursor.rowcount

        logger.info("Deleted %d snapshots for mission_id=%s", deleted, mission_id)
        return deleted
