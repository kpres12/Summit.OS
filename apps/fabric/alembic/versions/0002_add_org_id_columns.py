"""Add org_id columns for Enterprise multi-tenancy.

Revision ID: 0002
Revises: 0001_registry_init
Create Date: 2026-03-28

Adds:
  - org_id column (nullable, indexed) to:
      tasks, assets, missions, mission_assignments,
      tiered_missions, drone_boxes, interventions  (tasking service)
      approvals                                     (api-gateway)
  - summit_organizations table                      (api-gateway / multi-tenant pkg)

Safe to run against Community deployments — all columns are nullable so
existing rows keep working with org_id=NULL (treated as "default" at runtime).
"""

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001_registry_init"
branch_labels = None
depends_on = None

# Tables that live in the tasking service schema
_TASKING_TABLES = [
    "tasks",
    "assets",
    "missions",
    "mission_assignments",
    "tiered_missions",
    "drone_boxes",
    "interventions",
]

# Tables that live in the api-gateway schema
_GATEWAY_TABLES = [
    "approvals",
]


def upgrade() -> None:
    # ── Add org_id to tasking + gateway tables ──────────────────────────────
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = inspector.get_table_names()
    for table in _TASKING_TABLES + _GATEWAY_TABLES:
        # Guard: skip if table doesn't exist (single-service deployments omit tasking/gateway)
        if table not in existing_tables:
            continue
        existing_cols = [c["name"] for c in inspector.get_columns(table)]
        if "org_id" not in existing_cols:
            op.add_column(table, sa.Column("org_id", sa.String(128), nullable=True))
            op.create_index(f"ix_{table}_org_id", table, ["org_id"])

    # ── Create summit_organizations table ───────────────────────────────────
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if "summit_organizations" not in inspector.get_table_names():
        op.create_table(
            "summit_organizations",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("org_id", sa.String(128), nullable=False, unique=True),
            sa.Column("name", sa.String(256), nullable=False),
            sa.Column(
                "plan",
                sa.String(32),
                nullable=False,
                server_default="enterprise",
            ),
            sa.Column(
                "active",
                sa.Boolean,
                nullable=False,
                server_default=sa.text("true"),
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("metadata", sa.JSON, nullable=True),
        )
        op.create_index(
            "ix_summit_organizations_org_id", "summit_organizations", ["org_id"]
        )


def downgrade() -> None:
    # Drop summit_organizations
    op.drop_index("ix_summit_organizations_org_id", table_name="summit_organizations")
    op.drop_table("summit_organizations")

    # Drop org_id columns
    for table in _TASKING_TABLES + _GATEWAY_TABLES:
        op.drop_index(f"ix_{table}_org_id", table_name=table)
        op.drop_column(table, "org_id")
