"""
apps/api-gateway/routers/orgs.py — Organization management (Enterprise, SUPER_ADMIN only).

Endpoints:
    GET    /v1/admin/orgs            — list all organizations
    POST   /v1/admin/orgs            — create a new organization
    GET    /v1/admin/orgs/{org_id}   — get a single organization
    PATCH  /v1/admin/orgs/{org_id}   — update name / active flag
    DELETE /v1/admin/orgs/{org_id}   — deactivate (soft delete)

All endpoints require:
    - ENTERPRISE_MULTI_TENANT=true
    - Role: SUPER_ADMIN  (enforced via X-Summit-Role header or JWT roles claim)

These endpoints are intentionally NOT proxied to downstream services — the API
Gateway owns the organizations table directly.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("api-gateway.orgs")

ENTERPRISE_MULTI_TENANT: bool = (
    os.getenv("ENTERPRISE_MULTI_TENANT", "false").lower() == "true"
)

router = APIRouter(prefix="/v1/admin/orgs", tags=["Organizations (Enterprise)"])

# Injected at startup by init_orgs_router()
_session_factory = None
_orgs_table = None


def init_orgs_router(session_factory, orgs_table) -> None:
    """Wire in the DB session factory and table ref from the main lifespan."""
    global _session_factory, _orgs_table
    _session_factory = session_factory
    _orgs_table = orgs_table


# ── Auth guard ──────────────────────────────────────────────────────────────


def _require_super_admin(request: Request) -> None:
    """Verify the caller has SUPER_ADMIN role. Raises 403 otherwise."""
    # Accept role from explicit header (internal/service calls) or JWT claims
    role_header = (
        request.headers.get("X-Summit-Role", "")
        or request.headers.get("x-summit-role", "")
    )
    if role_header.upper() == "SUPER_ADMIN":
        return

    # Try JWT roles claim
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            from jose import jwt as _jwt

            claims = _jwt.get_unverified_claims(auth[7:])
            roles = claims.get("roles") or claims.get("role") or []
            if isinstance(roles, str):
                roles = [roles]
            if "SUPER_ADMIN" in [r.upper() for r in roles]:
                return
        except Exception:
            pass

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Organization management requires SUPER_ADMIN role.",
    )


def _enterprise_only() -> None:
    if not ENTERPRISE_MULTI_TENANT:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization management is an Enterprise feature. Set ENTERPRISE_MULTI_TENANT=true.",
        )


# ── Helpers ─────────────────────────────────────────────────────────────────


async def _get_session() -> AsyncSession:
    if _session_factory is None:
        raise HTTPException(status_code=503, detail="Database not initialized.")
    async with _session_factory() as session:
        return session


async def _fetch_org(session: AsyncSession, org_id: str) -> Optional[dict]:
    result = await session.execute(
        sa.select(_orgs_table).where(_orgs_table.c.org_id == org_id)
    )
    row = result.fetchone()
    return dict(row._mapping) if row else None


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get("", summary="List all organizations")
async def list_orgs(request: Request):
    _enterprise_only()
    _require_super_admin(request)

    async with _session_factory() as session:
        result = await session.execute(
            sa.select(_orgs_table).order_by(_orgs_table.c.created_at.desc())
        )
        rows = [dict(r._mapping) for r in result.fetchall()]

    # Serialize datetime fields
    for r in rows:
        for k in ("created_at", "updated_at"):
            if r.get(k) and hasattr(r[k], "isoformat"):
                r[k] = r[k].isoformat()

    return {"organizations": rows, "total": len(rows)}


@router.post("", status_code=status.HTTP_201_CREATED, summary="Create organization")
async def create_org(request: Request):
    _enterprise_only()
    _require_super_admin(request)

    body = await request.json()
    org_id = (body.get("org_id") or "").strip().lower()
    name = (body.get("name") or "").strip()

    if not org_id or not name:
        raise HTTPException(status_code=400, detail="org_id and name are required.")

    import re

    if not re.match(r"^[a-z0-9][a-z0-9\-]{1,126}$", org_id):
        raise HTTPException(
            status_code=400,
            detail="org_id must be lowercase alphanumeric with hyphens (e.g. acme-corp).",
        )

    async with _session_factory() as session:
        # Check for duplicate
        existing = await _fetch_org(session, org_id)
        if existing:
            raise HTTPException(
                status_code=409, detail=f"Organization '{org_id}' already exists."
            )

        now = datetime.now(timezone.utc)
        await session.execute(
            _orgs_table.insert().values(
                org_id=org_id,
                name=name,
                plan=body.get("plan", "enterprise"),
                active=True,
                created_at=now,
                updated_at=now,
                metadata=body.get("metadata"),
            )
        )
        await session.commit()

    logger.info("Created organization: %s (%s)", org_id, name)
    return {"org_id": org_id, "name": name, "active": True}


@router.get("/{org_id}", summary="Get organization")
async def get_org(org_id: str, request: Request):
    _enterprise_only()
    _require_super_admin(request)

    async with _session_factory() as session:
        org = await _fetch_org(session, org_id)

    if not org:
        raise HTTPException(status_code=404, detail=f"Organization '{org_id}' not found.")

    for k in ("created_at", "updated_at"):
        if org.get(k) and hasattr(org[k], "isoformat"):
            org[k] = org[k].isoformat()

    return org


@router.patch("/{org_id}", summary="Update organization")
async def update_org(org_id: str, request: Request):
    _enterprise_only()
    _require_super_admin(request)

    body = await request.json()
    updates: dict = {}
    if "name" in body:
        updates["name"] = body["name"].strip()
    if "active" in body:
        updates["active"] = bool(body["active"])
    if "metadata" in body:
        updates["metadata"] = body["metadata"]

    if not updates:
        raise HTTPException(status_code=400, detail="No updatable fields provided.")

    updates["updated_at"] = datetime.now(timezone.utc)

    async with _session_factory() as session:
        org = await _fetch_org(session, org_id)
        if not org:
            raise HTTPException(status_code=404, detail=f"Organization '{org_id}' not found.")

        await session.execute(
            _orgs_table.update()
            .where(_orgs_table.c.org_id == org_id)
            .values(**updates)
        )
        await session.commit()

    logger.info("Updated organization: %s %s", org_id, list(updates.keys()))
    return {"org_id": org_id, "updated": list(updates.keys())}


@router.delete("/{org_id}", summary="Deactivate organization")
async def deactivate_org(org_id: str, request: Request):
    _enterprise_only()
    _require_super_admin(request)

    if org_id == "default":
        raise HTTPException(status_code=400, detail="Cannot deactivate the default organization.")

    async with _session_factory() as session:
        org = await _fetch_org(session, org_id)
        if not org:
            raise HTTPException(status_code=404, detail=f"Organization '{org_id}' not found.")

        await session.execute(
            _orgs_table.update()
            .where(_orgs_table.c.org_id == org_id)
            .values(active=False, updated_at=datetime.now(timezone.utc))
        )
        await session.commit()

    logger.warning("Deactivated organization: %s", org_id)
    return {"org_id": org_id, "active": False}
