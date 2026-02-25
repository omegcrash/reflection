# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""Add composite index for API key validation

This migration adds a composite index on (tenant_id, key_hash) to the
api_keys table for faster API key validation queries.

Revision ID: 004_api_key_composite_index
Revises: 003_session_history
Create Date: 2026-02-03
"""
from alembic import op
import sqlalchemy as sa

# Revision identifiers
revision = '004_api_key_composite_index'
down_revision = '003_session_history'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add composite index for API key validation."""
    # This composite index optimizes the common query pattern:
    # SELECT * FROM api_keys WHERE tenant_id = ? AND key_hash = ?
    #
    # Without this index, PostgreSQL would need to use one of the
    # single-column indexes and then filter. With the composite
    # index, it can satisfy the query directly.
    op.create_index(
        'idx_api_keys_tenant_hash',
        'api_keys',
        ['tenant_id', 'key_hash'],
        unique=False,
        postgresql_using='btree'
    )


def downgrade() -> None:
    """Remove composite index."""
    op.drop_index('idx_api_keys_tenant_hash', table_name='api_keys')
