"""create nodes and coverages tables

Revision ID: 0001_registry_init
Revises: 
Create Date: 2025-10-15 05:59:00
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0001_registry_init'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        'nodes',
        sa.Column('id', sa.String(length=128), primary_key=True),
        sa.Column('type', sa.String(length=32), nullable=False),
        sa.Column('pubkey', sa.String(length=4096)),
        sa.Column('fw_version', sa.String(length=64)),
        sa.Column('location', sa.JSON()),
        sa.Column('capabilities', sa.JSON()),
        sa.Column('comm', sa.JSON()),
        sa.Column('policy', sa.JSON()),
        sa.Column('status', sa.String(length=32), server_default='OFFLINE'),
        sa.Column('last_seen', sa.DateTime(timezone=True)),
        sa.Column('retired', sa.Boolean(), server_default=sa.text('false')),
        sa.Column('created_at', sa.DateTime(timezone=True)),
        sa.Column('updated_at', sa.DateTime(timezone=True)),
    )
    op.create_index('ix_nodes_status', 'nodes', ['status'])

    op.create_table(
        'coverages',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('node_id', sa.String(length=128), nullable=False),
        sa.Column('viewshed_geojson', sa.JSON()),
        sa.Column('version', sa.String(length=128)),
        sa.Column('updated_at', sa.DateTime(timezone=True)),
    )
    op.create_index('ix_coverages_node_id', 'coverages', ['node_id'])


def downgrade() -> None:
    op.drop_index('ix_coverages_node_id', table_name='coverages')
    op.drop_table('coverages')
    op.drop_index('ix_nodes_status', table_name='nodes')
    op.drop_table('nodes')
