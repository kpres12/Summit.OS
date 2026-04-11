"""
packages/utm/cache.py — Persistent offline cache for FAA UTM data.

Backs the NOTAM and Facility Map clients with a SQLite database so that:
  1. Data fetched while online survives process restarts
  2. Operations can continue in degraded/offline mode using last-known data
  3. A background refresh task keeps the cache warm while connectivity exists

The cache operates in two tiers:
  FRESH  — data fetched within the configured TTL (NOTAM: 5min, FacMap: 24h)
  STALE  — older data, served with a staleness warning in AirspaceResult.errors
  ABSENT — no data ever fetched for this area (no cache entry exists)

Staleness warnings are surfaced to the operator console but do NOT block missions.
Only a hard denial from the policy engine blocks a mission.

Database location:
  Default: /var/lib/summit/utm_cache.db
  Override: SUMMIT_UTM_DB env var

Schema:
  notam_cache   (cache_key, fetched_at, expires_at, data_json)
  facility_cache(cache_key, fetched_at, expires_at, data_json)
  prefetch_areas(area_id, lat, lon, radius_nm, last_fetched, label)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator, List, Optional

logger = logging.getLogger("utm.cache")

_DB_PATH = os.getenv(
    "SUMMIT_UTM_DB",
    os.path.join(
        os.getenv("SUMMIT_DATA_DIR", "/var/lib/summit"),
        "utm_cache.db",
    ),
)

# TTLs (seconds)
_NOTAM_TTL_FRESH    = 300       # 5 minutes — NOTAMs change frequently
_NOTAM_TTL_STALE    = 3600 * 4  # 4 hours — stale but usable in degraded mode
_FACILITY_TTL_FRESH = 86400     # 24 hours — facility maps change rarely
_FACILITY_TTL_STALE = 86400 * 7 # 7 days  — very stale but still directionally correct


_SCHEMA = """
CREATE TABLE IF NOT EXISTS notam_cache (
    cache_key   TEXT PRIMARY KEY,
    fetched_at  REAL NOT NULL,
    expires_at  REAL NOT NULL,
    data_json   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facility_cache (
    cache_key   TEXT PRIMARY KEY,
    fetched_at  REAL NOT NULL,
    expires_at  REAL NOT NULL,
    data_json   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prefetch_areas (
    area_id      TEXT PRIMARY KEY,
    label        TEXT NOT NULL,
    lat          REAL NOT NULL,
    lon          REAL NOT NULL,
    radius_nm    REAL NOT NULL,
    last_fetched REAL
);

CREATE INDEX IF NOT EXISTS idx_notam_expires   ON notam_cache    (expires_at);
CREATE INDEX IF NOT EXISTS idx_facility_expires ON facility_cache (expires_at);
"""


def _ensure_db_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


@contextmanager
def _db(path: str = _DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    _ensure_db_dir(path)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
        yield conn
    finally:
        conn.close()


class UTMCache:
    """
    SQLite-backed persistent cache for NOTAM and UAS Facility Map data.

    Thread-safe for reads. Writes use SQLite's built-in serialization.
    """

    def __init__(self, db_path: str = _DB_PATH) -> None:
        self._db = db_path
        # Ensure schema exists on startup
        try:
            with _db(self._db):
                pass
        except Exception as exc:
            logger.warning("UTM cache init failed (%s) — falling back to memory", exc)
            self._db = ":memory:"

    # ── NOTAM cache ───────────────────────────────────────────────────────────

    def get_notams(self, cache_key: str) -> tuple[Optional[List], bool]:
        """
        Returns (data, is_fresh).
          data=None  → no entry or too stale to use
          is_fresh=False → stale data, surface warning to operator
        """
        return self._get("notam_cache", cache_key, _NOTAM_TTL_FRESH, _NOTAM_TTL_STALE)

    def set_notams(self, cache_key: str, data: List) -> None:
        self._set("notam_cache", cache_key, data, _NOTAM_TTL_FRESH)

    # ── Facility Map cache ────────────────────────────────────────────────────

    def get_facility(self, cache_key: str) -> tuple[Optional[List], bool]:
        return self._get("facility_cache", cache_key, _FACILITY_TTL_FRESH, _FACILITY_TTL_STALE)

    def set_facility(self, cache_key: str, data: List) -> None:
        self._set("facility_cache", cache_key, data, _FACILITY_TTL_FRESH)

    # ── Prefetch areas ────────────────────────────────────────────────────────

    def add_prefetch_area(
        self,
        area_id: str,
        label: str,
        lat: float,
        lon: float,
        radius_nm: float,
    ) -> None:
        """Register an area to be pre-fetched by the background refresh task."""
        with _db(self._db) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO prefetch_areas
                  (area_id, label, lat, lon, radius_nm, last_fetched)
                VALUES (?, ?, ?, ?, ?, NULL)
                """,
                (area_id, label, lat, lon, radius_nm),
            )
            conn.commit()
        logger.info("Registered prefetch area: %s (%.4f, %.4f r=%.1fnm)", label, lat, lon, radius_nm)

    def get_prefetch_areas(self) -> List[dict]:
        with _db(self._db) as conn:
            rows = conn.execute("SELECT * FROM prefetch_areas").fetchall()
            return [dict(r) for r in rows]

    def mark_prefetched(self, area_id: str) -> None:
        with _db(self._db) as conn:
            conn.execute(
                "UPDATE prefetch_areas SET last_fetched = ? WHERE area_id = ?",
                (time.time(), area_id),
            )
            conn.commit()

    # ── Cache stats ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        now = time.time()
        with _db(self._db) as conn:
            notam_total  = conn.execute("SELECT COUNT(*) FROM notam_cache").fetchone()[0]
            notam_fresh  = conn.execute(
                "SELECT COUNT(*) FROM notam_cache WHERE ? - fetched_at < ?",
                (now, _NOTAM_TTL_FRESH),
            ).fetchone()[0]
            fac_total    = conn.execute("SELECT COUNT(*) FROM facility_cache").fetchone()[0]
            fac_fresh    = conn.execute(
                "SELECT COUNT(*) FROM facility_cache WHERE ? - fetched_at < ?",
                (now, _FACILITY_TTL_FRESH),
            ).fetchone()[0]
            areas        = conn.execute("SELECT COUNT(*) FROM prefetch_areas").fetchone()[0]
        return {
            "notam_entries": notam_total,
            "notam_fresh": notam_fresh,
            "facility_entries": fac_total,
            "facility_fresh": fac_fresh,
            "prefetch_areas": areas,
            "db_path": self._db,
        }

    def purge_expired(self) -> int:
        """Remove entries older than the stale TTL. Returns rows deleted."""
        now = time.time()
        deleted = 0
        with _db(self._db) as conn:
            c1 = conn.execute(
                "DELETE FROM notam_cache WHERE ? - fetched_at > ?",
                (now, _NOTAM_TTL_STALE),
            )
            c2 = conn.execute(
                "DELETE FROM facility_cache WHERE ? - fetched_at > ?",
                (now, _FACILITY_TTL_STALE),
            )
            conn.commit()
            deleted = c1.rowcount + c2.rowcount
        if deleted:
            logger.debug("UTM cache: purged %d expired entries", deleted)
        return deleted

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get(
        self,
        table: str,
        key: str,
        fresh_ttl: float,
        stale_ttl: float,
    ) -> tuple[Optional[List], bool]:
        now = time.time()
        try:
            with _db(self._db) as conn:
                row = conn.execute(
                    f"SELECT fetched_at, data_json FROM {table} WHERE cache_key = ?",
                    (key,),
                ).fetchone()
                if not row:
                    return None, False
                age = now - row["fetched_at"]
                if age > stale_ttl:
                    return None, False  # too old to trust
                data = json.loads(row["data_json"])
                is_fresh = age <= fresh_ttl
                return data, is_fresh
        except Exception as exc:
            logger.debug("UTM cache read error: %s", exc)
            return None, False

    def _set(self, table: str, key: str, data: Any, ttl: float) -> None:
        now = time.time()
        try:
            with _db(self._db) as conn:
                conn.execute(
                    f"""
                    INSERT OR REPLACE INTO {table} (cache_key, fetched_at, expires_at, data_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (key, now, now + ttl, json.dumps(data, default=str)),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("UTM cache write error: %s", exc)


# ── Background refresh task ────────────────────────────────────────────────────

async def run_cache_refresh(cache: UTMCache, interval_seconds: float = 240.0) -> None:
    """
    Asyncio task that periodically refreshes cached UTM data for all
    registered prefetch areas. Run at service startup:

        asyncio.create_task(run_cache_refresh(utm_cache))

    This keeps the offline cache warm even before the first operator
    checks airspace — so when internet drops mid-operation, fresh data
    is already on disk.
    """
    import asyncio

    logger.info("UTM cache refresh task started (interval=%.0fs)", interval_seconds)

    while True:
        try:
            areas = cache.get_prefetch_areas()
            if areas:
                from packages.utm.airspace import AirspaceChecker
                checker = AirspaceChecker(auto_laanc=False)

                for area in areas:
                    try:
                        await checker.check(
                            lat=area["lat"],
                            lon=area["lon"],
                            radius_nm=area["radius_nm"],
                        )
                        cache.mark_prefetched(area["area_id"])
                        logger.debug(
                            "Refreshed UTM cache for area '%s'", area["label"]
                        )
                    except Exception as exc:
                        logger.debug("Prefetch failed for '%s': %s", area["label"], exc)

            cache.purge_expired()

        except Exception as exc:
            logger.warning("UTM cache refresh error: %s", exc)

        await asyncio.sleep(interval_seconds)


# Module-level singleton
_cache: Optional[UTMCache] = None

def get_cache() -> UTMCache:
    global _cache
    if _cache is None:
        _cache = UTMCache()
    return _cache
