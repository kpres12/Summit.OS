"""
routers/audit.py — Audit log query router for Summit.OS API Gateway.

Endpoints (all under /v1/audit, all require ADMIN role):
  GET /v1/audit/logs              — paginated log query
  GET /v1/audit/logs/{event_id}   — single event by UUID
  GET /v1/audit/stats             — event-type summary for a time window
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
