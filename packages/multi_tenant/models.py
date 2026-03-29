"""
packages/multi_tenant/models.py — Organization data model.

SQLAlchemy table definition + Pydantic schemas for the organizations table.
The table is created/managed by the API Gateway (it owns org lifecycle).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import sqlalchemy as sa
from pydantic import BaseModel, Field

# ── SQLAlchemy table ────────────────────────────────────────────────────────

ORG_TABLE_NAME = "summit_organizations"

organizations = sa.Table(
    ORG_TABLE_NAME,
    sa.MetaData(),
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("org_id", sa.String(128), nullable=False, unique=True, index=True),
    sa.Column("name", sa.String(256), nullable=False),
    sa.Column("plan", sa.String(32), nullable=False, server_default="enterprise"),
    sa.Column("active", sa.Boolean, nullable=False, server_default=sa.text("true")),
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    ),
    sa.Column("updated_at", sa.DateTime(timezone=True)),
    sa.Column("metadata", sa.JSON, nullable=True),
)


# ── Pydantic schemas ────────────────────────────────────────────────────────


class OrgCreate(BaseModel):
    org_id: str = Field(..., min_length=2, max_length=128, pattern=r"^[a-z0-9][a-z0-9\-]*$")
    name: str = Field(..., min_length=1, max_length=256)
    plan: str = Field(default="enterprise")
    metadata: Optional[dict] = None


class OrgResponse(BaseModel):
    org_id: str
    name: str
    plan: str
    active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Optional[dict] = None

    model_config = {"from_attributes": True}


class OrgUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=256)
    active: Optional[bool] = None
    metadata: Optional[dict] = None
