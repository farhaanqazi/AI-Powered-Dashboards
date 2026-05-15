"""create dashboard_records

Revision ID: 0001
Revises:
Create Date: 2026-05-15
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dashboard_records",
        sa.Column("session_key", sa.String(length=255), primary_key=True),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("original_filename", sa.Text(), nullable=False, server_default=""),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_dashboard_records_expires_at", "dashboard_records", ["expires_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_dashboard_records_expires_at", table_name="dashboard_records")
    op.drop_table("dashboard_records")
