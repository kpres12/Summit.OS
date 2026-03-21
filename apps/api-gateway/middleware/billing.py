"""
billing.py — API key enforcement middleware for Summit.OS api-gateway.

Provides:
  - OrgContext dataclass returned by require_api_key
  - require_api_key FastAPI dependency
  - init_billing_tables(engine) coroutine called from lifespan

When the env var API_KEY_ENFORCE is not "true" (the default), require_api_key
is a no-op that returns a free-tier OrgContext so all existing tests keep passing.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Field-level encryption for PII columns (org name)
try:
    _pkg_root = str(Path(__file__).resolve().parents[4])
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    from packages.security.field_encryption import encrypt_field, decrypt_field
except Exception:
    def encrypt_field(v: str) -> str: return v  # type: ignore[misc]
    def decrypt_field(v: str) -> str: return v  # type: ignore[misc]

from fastapi import HTTPException, Request
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    select,
    update,
)
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger("api-gateway.billing")

# ---------------------------------------------------------------------------
# Table definitions — use a local MetaData to avoid circular imports with main
# ---------------------------------------------------------------------------

_metadata = MetaData()

api_keys_table = Table(
    "api_keys",
    _metadata,
    Column("key_hash", String(64), primary_key=True),
    Column("org_id", String(128), nullable=False),
    Column("tier", String(32), nullable=False, default="free"),
    Column("created_at", DateTime(timezone=True)),
    Column("last_used_at", DateTime(timezone=True), nullable=True),
    Column("active", Boolean, nullable=False, default=True),
)

orgs_table = Table(
    "orgs",
    _metadata,
    Column("org_id", String(128), primary_key=True),
    Column("name", String(256)),
    Column("tier", String(32), nullable=False, default="free"),
    Column("stripe_customer_id", String(128), nullable=True),
    Column("stripe_subscription_id", String(128), nullable=True),
    Column("subscription_status", String(32), nullable=False, default="active"),
    Column("entity_limit", Integer, nullable=False, default=10),
    Column("operator_limit", Integer, nullable=False, default=1),
    Column("created_at", DateTime(timezone=True)),
)

# ---------------------------------------------------------------------------
# Tier defaults
# ---------------------------------------------------------------------------

TIER_DEFAULTS: dict[str, dict] = {
    "free":       {"entity_limit": 10,   "operator_limit": 1},
    "pro":        {"entity_limit": 500,  "operator_limit": 5},
    "org":        {"entity_limit": 5000, "operator_limit": -1},
    "enterprise": {"entity_limit": -1,   "operator_limit": -1},
}

_FREE_DEFAULTS = TIER_DEFAULTS["free"]


# ---------------------------------------------------------------------------
# OrgContext
# ---------------------------------------------------------------------------

@dataclass
class OrgContext:
    org_id: str
    tier: str
    entity_limit: int
    operator_limit: int


_FREE_ORG_CONTEXT = OrgContext(
    org_id="anonymous",
    tier="free",
    entity_limit=_FREE_DEFAULTS["entity_limit"],
    operator_limit=_FREE_DEFAULTS["operator_limit"],
)

# ---------------------------------------------------------------------------
# Module-level session factory — set by init_billing_tables
# ---------------------------------------------------------------------------

_SessionLocal: Optional[sessionmaker] = None  # type: ignore[type-arg]


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

async def init_billing_tables(engine: AsyncEngine) -> None:
    """Create billing tables and store a session factory for later use."""
    global _SessionLocal
    async with engine.begin() as conn:
        await conn.run_sync(_metadata.create_all)
    _SessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    logger.info("Billing tables ready.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _hash_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

async def require_api_key(request: Request) -> OrgContext:
    """
    FastAPI dependency that enforces API key authentication.

    Resolution order:
      1. X-API-Key header
      2. Authorization: Bearer sk_... header (must start with 'sk_')

    Returns OrgContext on success.
    Raises HTTP 401 for missing/invalid key.
    Raises HTTP 402 if subscription is cancelled.

    When API_KEY_ENFORCE != "true", returns a free-tier OrgContext immediately
    so existing tests are unaffected.
    """
    enforce = os.getenv("API_KEY_ENFORCE", "false").lower() == "true"
    if not enforce:
        return _FREE_ORG_CONTEXT

    # Extract raw key from headers
    raw_key: Optional[str] = None

    x_api_key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")
    if x_api_key:
        raw_key = x_api_key
    else:
        auth = request.headers.get("Authorization") or request.headers.get("authorization")
        if auth and auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1]
            if token.startswith("sk_"):
                raw_key = token

    if not raw_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    if _SessionLocal is None:
        logger.warning("Billing session factory not initialised — falling back to free tier")
        return _FREE_ORG_CONTEXT

    key_hash = _hash_key(raw_key)

    async with _SessionLocal() as session:
        # Look up the key
        key_row_result = await session.execute(
            select(api_keys_table).where(
                api_keys_table.c.key_hash == key_hash,
                api_keys_table.c.active == True,  # noqa: E712
            )
        )
        key_row = key_row_result.first()

        if key_row is None:
            raise HTTPException(status_code=401, detail="Invalid or inactive API key")

        org_id: str = key_row.org_id

        # Stamp last_used_at (best-effort, non-fatal)
        try:
            await session.execute(
                update(api_keys_table)
                .where(api_keys_table.c.key_hash == key_hash)
                .values(last_used_at=datetime.now(timezone.utc))
            )
            await session.commit()
        except Exception as exc:
            logger.warning("Failed to update last_used_at: %s", exc)

        # Look up org
        org_row_result = await session.execute(
            select(orgs_table).where(orgs_table.c.org_id == org_id)
        )
        org_row = org_row_result.first()

    if org_row is None:
        # Key exists but no org record — treat as free tier
        return OrgContext(
            org_id=org_id,
            tier="free",
            entity_limit=_FREE_DEFAULTS["entity_limit"],
            operator_limit=_FREE_DEFAULTS["operator_limit"],
        )

    if org_row.subscription_status == "cancelled":
        raise HTTPException(status_code=402, detail="Subscription cancelled")

    tier: str = org_row.tier or "free"
    defaults = TIER_DEFAULTS.get(tier, _FREE_DEFAULTS)

    return OrgContext(
        org_id=org_id,
        tier=tier,
        entity_limit=org_row.entity_limit if org_row.entity_limit is not None else defaults["entity_limit"],
        operator_limit=org_row.operator_limit if org_row.operator_limit is not None else defaults["operator_limit"],
    )
