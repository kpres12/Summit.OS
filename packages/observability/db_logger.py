"""
packages/observability/db_logger.py — Structured DB logging for Heli.OS services.

Three layers:
  1. PostgresLogHandler  — Python logging.Handler that writes log records to the
                           service_logs table.  Attach to the root logger and every
                           service's log records land in Postgres automatically.

  2. slow_query_listener — SQLAlchemy event listener that logs queries exceeding
                           SLOW_QUERY_THRESHOLD_MS (default 200 ms) as WARNING.

  3. init_db_logging()   — One-call setup: creates the service_logs table, wires
                           the handler to the given loggers, and returns the handler
                           so the caller can close it on shutdown.

Usage (in a service's lifespan):
    from packages.observability.db_logger import init_db_logging, close_db_logging

    handler = await init_db_logging(database_url=db_url, service="tasking")
    # ...
    await close_db_logging(handler)

The table is auto-created on first use. Failures to write are logged as warnings
and never propagate to the calling code (same contract as the audit middleware).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("summit.observability.db_logger")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SLOW_QUERY_THRESHOLD_MS: int = int(os.getenv("SLOW_QUERY_THRESHOLD_MS", "200"))
_LOG_LEVEL_FILTER: int = logging.WARNING   # only WARNING+ by default; override via env
_LOG_LEVEL_ENV = os.getenv("DB_LOG_LEVEL", "WARNING").upper()
try:
    _LOG_LEVEL_FILTER = getattr(logging, _LOG_LEVEL_ENV, logging.WARNING)
except AttributeError:
    pass

_BATCH_SIZE: int = int(os.getenv("DB_LOG_BATCH_SIZE", "100"))
_FLUSH_INTERVAL: float = float(os.getenv("DB_LOG_FLUSH_INTERVAL", "10"))  # seconds

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS service_logs (
    id          BIGSERIAL    PRIMARY KEY,
    log_id      UUID         NOT NULL DEFAULT gen_random_uuid(),
    timestamp   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    service     VARCHAR(64)  NOT NULL,
    level       VARCHAR(16)  NOT NULL,
    logger_name VARCHAR(128),
    message     TEXT,
    exc_text    TEXT,
    extra       JSONB        DEFAULT '{}'
);
"""

_CREATE_IDX_TS     = "CREATE INDEX IF NOT EXISTS idx_svclog_ts      ON service_logs(timestamp DESC);"
_CREATE_IDX_SVC    = "CREATE INDEX IF NOT EXISTS idx_svclog_service  ON service_logs(service);"
_CREATE_IDX_LEVEL  = "CREATE INDEX IF NOT EXISTS idx_svclog_level    ON service_logs(level);"

_INSERT_SQL = """
INSERT INTO service_logs
    (log_id, timestamp, service, level, logger_name, message, exc_text, extra)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
"""


# ---------------------------------------------------------------------------
# Helper: asyncpg URL normalisation
# ---------------------------------------------------------------------------

def _to_asyncpg_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return url


# ---------------------------------------------------------------------------
# PostgresLogHandler
# ---------------------------------------------------------------------------

class PostgresLogHandler(logging.Handler):
    """
    Async-friendly logging handler that batches records and flushes to Postgres.

    Because Python's logging.Handler.emit() is synchronous, we push records
    onto a deque and a background asyncio task drains them.  If the event loop
    isn't running yet (e.g. module import time), records are silently dropped
    rather than blocking or raising.
    """

    def __init__(
        self,
        service: str,
        pool,                      # asyncpg.Pool
        level: int = _LOG_LEVEL_FILTER,
        flush_interval: float = _FLUSH_INTERVAL,
        batch_size: int = _BATCH_SIZE,
    ) -> None:
        super().__init__(level=level)
        self._service = service
        self._pool = pool
        self._buffer: deque[Dict[str, Any]] = deque(maxlen=10_000)
        self._flush_interval = flush_interval
        self._batch_size = batch_size
        self._flush_task: Optional[asyncio.Task] = None

    # -- public API ----------------------------------------------------------

    def start(self) -> None:
        """Start the background flush task. Call from inside a running event loop."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._flush_task = loop.create_task(self._flush_loop())
        except RuntimeError:
            pass

    async def aclose(self) -> None:
        """Flush remaining records and stop the background task."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            self._flush_task = None
        await self._flush_all()

    # -- logging.Handler contract --------------------------------------------

    def emit(self, record: logging.LogRecord) -> None:
        try:
            exc_text: Optional[str] = None
            if record.exc_info:
                exc_text = self.formatException(record.exc_info)

            entry: Dict[str, Any] = {
                "log_id":      str(uuid.uuid4()),
                "timestamp":   datetime.fromtimestamp(record.created, tz=timezone.utc),
                "service":     self._service,
                "level":       record.levelname,
                "logger_name": record.name,
                "message":     record.getMessage(),
                "exc_text":    exc_text,
                "extra":       {},
            }

            # Capture any extra fields the caller attached
            _standard = {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            }
            for k, v in record.__dict__.items():
                if k not in _standard and not k.startswith("_"):
                    try:
                        json.dumps(v)   # only keep JSON-serialisable values
                        entry["extra"][k] = v
                    except (TypeError, ValueError):
                        entry["extra"][k] = str(v)

            self._buffer.append(entry)

            # Flush immediately if batch is full
            if len(self._buffer) >= self._batch_size:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self._flush_all())
                except RuntimeError:
                    pass

        except Exception:
            self.handleError(record)

    # -- internal ------------------------------------------------------------

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self._flush_interval)
            await self._flush_all()

    async def _flush_all(self) -> None:
        if not self._buffer or self._pool is None:
            return
        batch: List[Dict[str, Any]] = []
        while self._buffer:
            batch.append(self._buffer.popleft())
        if not batch:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    _INSERT_SQL,
                    [
                        (
                            r["log_id"],
                            r["timestamp"],
                            r["service"],
                            r["level"],
                            r["logger_name"],
                            r["message"],
                            r["exc_text"],
                            json.dumps(r["extra"]),
                        )
                        for r in batch
                    ],
                )
        except Exception as exc:
            logger.warning("DB log flush failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Slow-query SQLAlchemy listener
# ---------------------------------------------------------------------------

def slow_query_listener(
    service: str,
    threshold_ms: int = SLOW_QUERY_THRESHOLD_MS,
    log: Optional[logging.Logger] = None,
):
    """
    Return a SQLAlchemy 'after_cursor_execute' event listener that logs
    queries slower than *threshold_ms* at WARNING level.

    Usage:
        from sqlalchemy import event
        event.listen(engine.sync_engine, "after_cursor_execute", slow_query_listener("tasking"))
    """
    _log = log or logging.getLogger(f"{service}.slow_query")

    def _listener(conn, cursor, statement, parameters, context, executemany):
        elapsed_ms = getattr(context, "_query_start_time_ms", None)
        if elapsed_ms is None:
            return
        duration = int((time.monotonic_ns() - elapsed_ms) / 1_000_000)
        if duration >= threshold_ms:
            short = statement.strip().replace("\n", " ")[:120]
            _log.warning(
                "Slow query (%dms > %dms): %s", duration, threshold_ms, short,
                extra={"query_duration_ms": duration, "sql_snippet": short},
            )

    return _listener


def slow_query_before_listener():
    """Companion 'before_cursor_execute' listener that stamps the start time."""
    def _listener(conn, cursor, statement, parameters, context, executemany):
        context._query_start_time_ms = time.monotonic_ns()
    return _listener


def attach_slow_query_logging(engine, service: str, threshold_ms: int = SLOW_QUERY_THRESHOLD_MS) -> None:
    """
    Wire slow-query listeners onto a SQLAlchemy async engine.

    Call after the engine is created:
        attach_slow_query_logging(state.engine, "tasking")
    """
    try:
        from sqlalchemy import event as _sa_event

        sync_engine = engine.sync_engine if hasattr(engine, "sync_engine") else engine
        _sa_event.listen(sync_engine, "before_cursor_execute", slow_query_before_listener())
        _sa_event.listen(sync_engine, "after_cursor_execute",  slow_query_listener(service, threshold_ms))
        logger.info("Slow-query logging attached (service=%s threshold=%dms)", service, threshold_ms)
    except Exception as exc:
        logger.warning("Slow-query listener not attached (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# init / close
# ---------------------------------------------------------------------------

async def init_db_logging(
    database_url: str,
    service: str,
    loggers: Optional[Sequence[str]] = None,
    level: int = _LOG_LEVEL_FILTER,
) -> Optional[PostgresLogHandler]:
    """
    Create the asyncpg pool, ensure the service_logs table, wire up the
    PostgresLogHandler, and start its flush background task.

    Args:
        database_url: Postgres DSN (postgresql:// or postgresql+asyncpg://)
        service:      Service name tag (e.g. "tasking", "api-gateway")
        loggers:      Logger names to attach the handler to.
                      Defaults to [""] (root logger) which catches everything.
        level:        Minimum log level to capture.

    Returns the handler so the caller can close it on shutdown, or None if
    init failed (non-fatal — the service continues without DB logging).
    """
    try:
        import asyncpg  # type: ignore

        dsn = _to_asyncpg_url(database_url)
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3)

        async with pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE_SQL)
            await conn.execute(_CREATE_IDX_TS)
            await conn.execute(_CREATE_IDX_SVC)
            await conn.execute(_CREATE_IDX_LEVEL)

        handler = PostgresLogHandler(service=service, pool=pool, level=level)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(fmt)

        target_loggers = loggers if loggers is not None else [""]
        for log_name in target_loggers:
            logging.getLogger(log_name).addHandler(handler)

        handler.start()
        logger.info(
            "DB logging initialised (service=%s level=%s table=service_logs)",
            service, logging.getLevelName(level),
        )
        return handler

    except Exception as exc:
        logger.warning("DB logging init failed (non-fatal): %s", exc)
        return None


async def close_db_logging(handler: Optional[PostgresLogHandler]) -> None:
    """Flush remaining records and close the asyncpg pool."""
    if handler is None:
        return
    try:
        await handler.aclose()
        if handler._pool is not None:
            await handler._pool.close()
    except Exception as exc:
        logger.warning("DB logging close error (non-fatal): %s", exc)
