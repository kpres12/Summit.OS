"""
routers/audit.py — Audit log query router for Heli.OS API Gateway.

Endpoints (all under /v1/audit, all require ADMIN role):
  GET /v1/audit/logs              — paginated HTTP audit log query
  GET /v1/audit/logs/{event_id}   — single audit event by UUID
  GET /v1/audit/stats             — event-type summary for a time window
  GET /v1/audit/service-logs      — paginated service/app log query
  GET /v1/audit/service-logs/stats — level counts + slow-query summary
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from middleware.rbac import require_role

logger = logging.getLogger("api-gateway.audit_router")

audit_router = APIRouter(prefix="/v1/audit", tags=["audit"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class AuditEntry(BaseModel):
    id: int
    event_id: str
    timestamp: datetime
    event_type: str
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    ip_address: Optional[str] = None
    method: Optional[str] = None
    path: Optional[str] = None
    status_code: Optional[int] = None
    duration_ms: Optional[int] = None
    extra: dict = {}


class AuditPage(BaseModel):
    entries: list[AuditEntry]
    total: int
    page: int
    page_size: int


class AuditStats(BaseModel):
    window_hours: int
    total_events: int
    auth_failure_count: int
    access_denied_count: int
    api_error_count: int
    event_type_counts: dict[str, int]


class ServiceLogEntry(BaseModel):
    id: int
    log_id: str
    timestamp: datetime
    service: str
    level: str
    logger_name: Optional[str] = None
    message: Optional[str] = None
    exc_text: Optional[str] = None
    extra: dict = {}


class ServiceLogPage(BaseModel):
    entries: list[ServiceLogEntry]
    total: int
    page: int
    page_size: int


class ServiceLogStats(BaseModel):
    window_hours: int
    total: int
    by_level: dict[str, int]
    by_service: dict[str, int]
    slow_queries: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_extra(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


def _row_to_entry(row: Any) -> AuditEntry:
    return AuditEntry(
        id=row["id"],
        event_id=str(row["event_id"]),
        timestamp=row["timestamp"],
        event_type=row["event_type"],
        user_id=row["user_id"],
        user_email=row["user_email"],
        ip_address=row["ip_address"],
        method=row["method"],
        path=row["path"],
        status_code=row["status_code"],
        duration_ms=row["duration_ms"],
        extra=_parse_extra(row["extra"]),
    )


# ---------------------------------------------------------------------------
# GET /v1/audit/logs
# ---------------------------------------------------------------------------


@audit_router.get("/logs", response_model=AuditPage)
async def query_audit_logs(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=500, description="Results per page"),
    user_id: Optional[str] = Query(None, description="Filter by user_id"),
    event_type: Optional[str] = Query(
        None, description="Filter by event_type (e.g. AUTH_FAILURE)"
    ),
    from_ts: Optional[datetime] = Query(
        None, description="Start of time range (ISO 8601)"
    ),
    to_ts: Optional[datetime] = Query(None, description="End of time range (ISO 8601)"),
    _role: Any = Depends(require_role("ADMIN")),
) -> AuditPage:
    """
    Query the audit log with optional filters.

    Requires ADMIN role (or RBAC_ENFORCE=false for dev environments).
    Results are ordered newest-first.
    """
    from middleware.audit import _audit_pool

    if _audit_pool is None:
        raise HTTPException(status_code=503, detail="Audit log unavailable")

    conditions: list[str] = []
    args: list[Any] = []
    idx = 1

    if user_id:
        conditions.append(f"user_id = ${idx}")
        args.append(user_id)
        idx += 1
    if event_type:
        conditions.append(f"event_type = ${idx}")
        args.append(event_type.upper())
        idx += 1
    if from_ts:
        conditions.append(f"timestamp >= ${idx}")
        args.append(from_ts)
        idx += 1
    if to_ts:
        conditions.append(f"timestamp <= ${idx}")
        args.append(to_ts)
        idx += 1

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    # LIMIT and OFFSET cannot be parameterized via $N in asyncpg; use explicit
    # int() casts. Both values are already validated by FastAPI Query(ge=1, le=500).
    safe_limit: int = int(page_size)
    safe_offset: int = int((page - 1) * page_size)

    count_sql = f"SELECT COUNT(*) FROM audit_log {where}"
    data_sql = f"""
        SELECT id, event_id, timestamp, event_type, user_id, user_email,
               ip_address, method, path, status_code, duration_ms, extra
        FROM audit_log {where}
        ORDER BY timestamp DESC
        LIMIT {safe_limit} OFFSET {safe_offset}
    """

    try:
        async with _audit_pool.acquire() as conn:
            total_row = await conn.fetchrow(count_sql, *args)
            total = int(total_row[0]) if total_row else 0
            rows = await conn.fetch(data_sql, *args)
    except Exception as exc:
        logger.error("Audit log query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Audit log query failed")

    return AuditPage(
        entries=[_row_to_entry(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# GET /v1/audit/logs/{event_id}
# ---------------------------------------------------------------------------


@audit_router.get("/logs/{event_id}", response_model=AuditEntry)
async def get_audit_event(
    event_id: str,
    _role: Any = Depends(require_role("ADMIN")),
) -> AuditEntry:
    """Retrieve a single audit event by its UUID."""
    from middleware.audit import _audit_pool

    if _audit_pool is None:
        raise HTTPException(status_code=503, detail="Audit log unavailable")

    try:
        async with _audit_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM audit_log WHERE event_id = $1::uuid",
                event_id,
            )
    except Exception as exc:
        logger.error("Audit event fetch failed: %s", exc)
        raise HTTPException(status_code=500, detail="Audit log query failed")

    if row is None:
        raise HTTPException(status_code=404, detail="Audit event not found")

    return _row_to_entry(row)


# ---------------------------------------------------------------------------
# GET /v1/audit/stats
# ---------------------------------------------------------------------------


@audit_router.get("/stats", response_model=AuditStats)
async def audit_stats(
    hours: int = Query(24, ge=1, le=720, description="Lookback window in hours"),
    _role: Any = Depends(require_role("ADMIN")),
) -> AuditStats:
    """
    Return event-type counts for the given lookback window.

    Useful for dashboards and detecting auth failure spikes.
    """
    from middleware.audit import _audit_pool

    if _audit_pool is None:
        raise HTTPException(status_code=503, detail="Audit log unavailable")

    try:
        async with _audit_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT event_type, COUNT(*) AS cnt
                FROM audit_log
                WHERE timestamp >= NOW() - ($1 * INTERVAL '1 hour')
                GROUP BY event_type
                ORDER BY cnt DESC
                """,
                hours,
            )
    except Exception as exc:
        logger.error("Audit stats query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Audit stats query failed")

    counts: dict[str, int] = {row["event_type"]: int(row["cnt"]) for row in rows}

    return AuditStats(
        window_hours=hours,
        total_events=sum(counts.values()),
        auth_failure_count=counts.get("AUTH_FAILURE", 0),
        access_denied_count=counts.get("ACCESS_DENIED", 0),
        api_error_count=counts.get("API_ERROR", 0),
        event_type_counts=counts,
    )


# ---------------------------------------------------------------------------
# Service log helpers
# ---------------------------------------------------------------------------

async def _get_svc_pool():
    """Reuse the audit pool — both tables live in the same DB."""
    from middleware.audit import _audit_pool
    return _audit_pool


def _row_to_svc_entry(row: Any) -> ServiceLogEntry:
    return ServiceLogEntry(
        id=row["id"],
        log_id=str(row["log_id"]),
        timestamp=row["timestamp"],
        service=row["service"],
        level=row["level"],
        logger_name=row.get("logger_name"),
        message=row.get("message"),
        exc_text=row.get("exc_text"),
        extra=_parse_extra(row.get("extra")),
    )


# ---------------------------------------------------------------------------
# GET /v1/audit/service-logs
# ---------------------------------------------------------------------------


@audit_router.get("/service-logs", response_model=ServiceLogPage)
async def query_service_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    service: Optional[str] = Query(None, description="Filter by service name (e.g. 'tasking')"),
    level: Optional[str] = Query(None, description="Filter by level (ERROR, WARNING, INFO, ...)"),
    from_ts: Optional[datetime] = Query(None, description="Start of time range (ISO 8601)"),
    to_ts: Optional[datetime] = Query(None, description="End of time range (ISO 8601)"),
    q: Optional[str] = Query(None, description="Full-text search in message (case-insensitive)"),
    _role: Any = Depends(require_role("ADMIN")),
) -> ServiceLogPage:
    """
    Query structured service logs from all Heli.OS backend services.

    Rows include stack traces (exc_text) for ERROR/CRITICAL entries and
    any extra fields the caller attached (e.g. query_duration_ms for slow queries).
    """
    pool = await _get_svc_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Service log DB unavailable")

    conditions: list[str] = []
    args: list[Any] = []
    idx = 1

    if service:
        conditions.append(f"service = ${idx}")
        args.append(service)
        idx += 1
    if level:
        conditions.append(f"level = ${idx}")
        args.append(level.upper())
        idx += 1
    if from_ts:
        conditions.append(f"timestamp >= ${idx}")
        args.append(from_ts)
        idx += 1
    if to_ts:
        conditions.append(f"timestamp <= ${idx}")
        args.append(to_ts)
        idx += 1
    if q:
        conditions.append(f"message ILIKE ${idx}")
        args.append(f"%{q}%")
        idx += 1

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    safe_limit = int(page_size)
    safe_offset = int((page - 1) * page_size)

    count_sql = f"SELECT COUNT(*) FROM service_logs {where}"
    data_sql = f"""
        SELECT id, log_id, timestamp, service, level, logger_name, message, exc_text, extra
        FROM service_logs {where}
        ORDER BY timestamp DESC
        LIMIT {safe_limit} OFFSET {safe_offset}
    """

    try:
        async with pool.acquire() as conn:
            total_row = await conn.fetchrow(count_sql, *args)
            total = int(total_row[0]) if total_row else 0
            rows = await conn.fetch(data_sql, *args)
    except Exception as exc:
        logger.error("Service log query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Service log query failed")

    return ServiceLogPage(
        entries=[_row_to_svc_entry(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# GET /v1/audit/service-logs/stats
# ---------------------------------------------------------------------------


@audit_router.get("/service-logs/stats", response_model=ServiceLogStats)
async def service_log_stats(
    hours: int = Query(24, ge=1, le=720, description="Lookback window in hours"),
    _role: Any = Depends(require_role("ADMIN")),
) -> ServiceLogStats:
    """
    Summary counts for service logs: by level, by service, and slow-query count.

    Slow queries are WARNING rows where extra->>'query_duration_ms' is present.
    """
    pool = await _get_svc_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Service log DB unavailable")

    try:
        async with pool.acquire() as conn:
            level_rows = await conn.fetch(
                """
                SELECT level, COUNT(*) AS cnt FROM service_logs
                WHERE timestamp >= NOW() - ($1 * INTERVAL '1 hour')
                GROUP BY level ORDER BY cnt DESC
                """,
                hours,
            )
            svc_rows = await conn.fetch(
                """
                SELECT service, COUNT(*) AS cnt FROM service_logs
                WHERE timestamp >= NOW() - ($1 * INTERVAL '1 hour')
                GROUP BY service ORDER BY cnt DESC
                """,
                hours,
            )
            slow_row = await conn.fetchrow(
                """
                SELECT COUNT(*) AS cnt FROM service_logs
                WHERE timestamp >= NOW() - ($1 * INTERVAL '1 hour')
                  AND extra ? 'query_duration_ms'
                """,
                hours,
            )
    except Exception as exc:
        logger.error("Service log stats query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Service log stats query failed")

    by_level = {r["level"]: int(r["cnt"]) for r in level_rows}
    by_service = {r["service"]: int(r["cnt"]) for r in svc_rows}
    slow_queries = int(slow_row["cnt"]) if slow_row else 0

    return ServiceLogStats(
        window_hours=hours,
        total=sum(by_level.values()),
        by_level=by_level,
        by_service=by_service,
        slow_queries=slow_queries,
    )
