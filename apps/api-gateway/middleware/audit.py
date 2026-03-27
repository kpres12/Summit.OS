"""
Audit logging middleware for the Summit API Gateway.

Captures security-relevant events and writes them to an append-only
audit_log table in Postgres via asyncpg. Non-blocking: a failure to
write an audit record NEVER fails the originating request.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger("api-gateway.audit")

# ---------------------------------------------------------------------------
# Module-level pool — initialised by init_audit_log()
# ---------------------------------------------------------------------------
_audit_pool: Optional[asyncpg.Pool] = None

# ---------------------------------------------------------------------------
# DDL — table + indexes (idempotent)
# ---------------------------------------------------------------------------
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id          BIGSERIAL PRIMARY KEY,
    event_id    UUID         NOT NULL DEFAULT gen_random_uuid(),
    timestamp   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    event_type  VARCHAR(64)  NOT NULL,
    user_id     VARCHAR(128),
    user_email  VARCHAR(256),
    ip_address  VARCHAR(64),
    method      VARCHAR(16),
    path        TEXT,
    status_code INTEGER,
    duration_ms INTEGER,
    extra       JSONB        DEFAULT '{}'
);
"""

_CREATE_IDX_TIMESTAMP = (
    "CREATE INDEX IF NOT EXISTS idx_audit_timestamp  ON audit_log(timestamp DESC);"
)
_CREATE_IDX_USER_ID = (
    "CREATE INDEX IF NOT EXISTS idx_audit_user_id    ON audit_log(user_id);"
)
_CREATE_IDX_EVENT_TYPE = (
    "CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log(event_type);"
)

_INSERT_SQL = """
INSERT INTO audit_log
    (event_id, timestamp, event_type, user_id, user_email,
     ip_address, method, path, status_code, duration_ms, extra)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
"""

# ---------------------------------------------------------------------------
# Paths to skip (health checks, metrics, static assets)
# ---------------------------------------------------------------------------
_SKIP_PREFIXES = ("/health", "/metrics", "/favicon", "/static", "/_next")


# ---------------------------------------------------------------------------
# Pool lifecycle
# ---------------------------------------------------------------------------


def _to_asyncpg_url(url: str) -> str:
    """asyncpg wants plain postgresql:// — strip the +asyncpg dialect if present."""
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return url


async def init_audit_log(database_url: str) -> None:
    """Create the asyncpg pool and ensure the audit_log table exists."""
    global _audit_pool
    try:
        dsn = _to_asyncpg_url(database_url)
        _audit_pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
        async with _audit_pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE_SQL)
            await conn.execute(_CREATE_IDX_TIMESTAMP)
            await conn.execute(_CREATE_IDX_USER_ID)
            await conn.execute(_CREATE_IDX_EVENT_TYPE)
        logger.info("Audit log initialised (pool ready, table ensured)")
    except Exception as exc:
        logger.warning("Audit log init failed (non-fatal): %s", exc)
        _audit_pool = None


async def close_audit_log() -> None:
    """Gracefully close the asyncpg pool on shutdown."""
    global _audit_pool
    if _audit_pool is not None:
        try:
            await _audit_pool.close()
        except Exception as exc:
            logger.warning("Audit pool close error (non-fatal): %s", exc)
        finally:
            _audit_pool = None


async def prune_old_entries(retention_days: int = 90) -> int:
    """
    Delete audit_log rows older than *retention_days*.

    Returns the number of rows deleted. Safe to call on a schedule — no-op
    if the pool is not initialised. Logs on completion.
    """
    if _audit_pool is None:
        return 0
    try:
        async with _audit_pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM audit_log WHERE timestamp < NOW() - ($1 * INTERVAL '1 day')",
                retention_days,
            )
        # asyncpg returns "DELETE N" as a string
        deleted = int(result.split()[-1]) if result else 0
        if deleted:
            logger.info(
                "Audit retention: pruned %d rows older than %d days",
                deleted,
                retention_days,
            )
        return deleted
    except Exception as exc:
        logger.warning("Audit retention prune failed (non-fatal): %s", exc)
        return 0


# ---------------------------------------------------------------------------
# JWT claim extraction (no verification — gateway already verified elsewhere)
# ---------------------------------------------------------------------------


def _extract_jwt_claims(authorization: str | None) -> tuple[str | None, str | None]:
    """
    Return (user_id, user_email) from a Bearer JWT by base64-decoding the
    payload segment. Never raises; returns (None, None) on any failure.
    """
    if not authorization:
        return None, None
    try:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return None, None
        parts = token.split(".")
        if len(parts) != 3:
            return None, None
        # Base64url-decode the payload; add padding as needed
        payload_b64 = parts[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        claims: dict[str, Any] = json.loads(payload_bytes)
        # Common JWT claim names for user identity
        user_id = claims.get("sub") or claims.get("user_id") or claims.get("uid")
        user_email = (
            claims.get("email") or claims.get("preferred_username") or claims.get("upn")
        )
        return (
            str(user_id) if user_id else None,
            str(user_email) if user_email else None,
        )
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Event type classification
# ---------------------------------------------------------------------------


def _classify_event(method: str, path: str, status: int) -> str:
    m = method.upper()
    p = path.lower()

    # Auth paths — check failure first so it takes priority over specific types
    is_auth_path = p.startswith("/auth/") or p.startswith("/api/auth/")

    if is_auth_path and status in (401, 403):
        return "AUTH_FAILURE"

    if m in ("POST", "GET") and (p in ("/auth/login", "/api/auth/login")):
        return "AUTH_LOGIN"

    if m == "POST" and p in ("/auth/logout", "/api/auth/logout"):
        return "AUTH_LOGOUT"

    if m == "GET" and (p == "/auth/callback" or p == "/api/auth/callback"):
        return "AUTH_CALLBACK"

    if m == "POST" and "/mfa/totp" in p:
        return "AUTH_MFA_TOTP"

    if m == "POST" and "/mfa/webauthn" in p:
        return "AUTH_MFA_WEBAUTHN"

    # Non-auth 401/403
    if status in (401, 403):
        return "ACCESS_DENIED"

    # 5xx errors
    if status >= 500:
        return "API_ERROR"

    # Mission operations (2xx only)
    if status >= 200 and status < 300:
        if m == "POST" and (
            p in ("/missions", "/api/missions")
            or p in ("/v1/missions", "/api/v1/missions")
        ):
            return "MISSION_CREATE"

        if m in ("PUT", "PATCH") and ("/missions/" in p or "/v1/missions/" in p):
            return "MISSION_UPDATE"

        if m == "POST" and "/dispatch" in p:
            return "MISSION_DISPATCH"

        if m == "POST" and "/sessions/revoke" in p:
            return "SESSION_REVOKE"

    # Entity access
    if m == "GET" and (
        "/entities/" in p
        or p.startswith("/entities/")
        or "/api/entities/" in p
        or p.startswith("/api/entities/")
    ):
        return "ENTITY_ACCESS"

    # Admin actions (any method, any status)
    if "/admin" in p:
        return "ADMIN_ACTION"

    return "GENERAL"


# ---------------------------------------------------------------------------
# IP address extraction
# ---------------------------------------------------------------------------


def _get_ip(request: Request) -> str | None:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        # Take the leftmost (original client) address
        return forwarded_for.split(",")[0].strip()
    client = request.client
    if client:
        return client.host
    return None


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class AuditLogMiddleware(BaseHTTPMiddleware):
    """
    Starlette BaseHTTPMiddleware that writes one row to audit_log for every
    request that is not a health/metrics/static asset path.

    Failures in audit writing are logged as warnings and never propagate to
    the caller.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        path = request.url.path

        # Fast path: skip non-auditable paths
        for prefix in _SKIP_PREFIXES:
            if path.startswith(prefix):
                return await call_next(request)

        start_ns = time.monotonic_ns()
        response: Response = await call_next(request)
        duration_ms = int((time.monotonic_ns() - start_ns) / 1_000_000)

        # Fire-and-forget audit write — never awaited inline so it doesn't
        # add latency to the response, and exceptions are swallowed.
        try:
            await self._write_audit(request, response.status_code, duration_ms)
        except Exception as exc:
            logger.warning("Audit write error (non-fatal): %s", exc)

        return response

    async def _write_audit(
        self,
        request: Request,
        status_code: int,
        duration_ms: int,
    ) -> None:
        if _audit_pool is None:
            logger.debug("Audit pool not initialised — skipping audit write")
            return

        method = request.method
        path = request.url.path
        authorization = request.headers.get("authorization")

        user_id, user_email = _extract_jwt_claims(authorization)
        ip_address = _get_ip(request)
        event_type = _classify_event(method, path, status_code)
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        # Build a minimal extra dict (query string, for example)
        extra: dict[str, Any] = {}
        qs = str(request.url.query)
        if qs:
            extra["query_string"] = qs

        try:
            async with _audit_pool.acquire() as conn:
                await conn.execute(
                    _INSERT_SQL,
                    event_id,
                    timestamp,
                    event_type,
                    user_id,
                    user_email,
                    ip_address,
                    method,
                    path,
                    status_code,
                    duration_ms,
                    json.dumps(extra),
                )
        except Exception as exc:
            logger.warning("Audit DB insert failed (non-fatal): %s", exc)
