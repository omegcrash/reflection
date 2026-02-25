# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""Add memories table for tenant-scoped memory storage

Revision ID: 002
Revises: 001
Create Date: 2026-02-02

Phase 4: Memory & Performance Optimizations
- Tenant-scoped memories with database-native search
- Indexes optimized for tenant isolation queries
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create memories table
    op.create_table(
        'memories',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('uuid_generate_v4()'), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('key', sa.String(255), nullable=False),
        sa.Column('value', sa.Text(), nullable=False),
        sa.Column('category', sa.String(100), server_default='general', nullable=False),
        sa.Column('importance', sa.Integer(), server_default='5', nullable=False),
        sa.Column('access_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('source', sa.String(50), server_default='api', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
    )
    
    # CRITICAL: Indexes for tenant isolation
    # These indexes ensure that tenant_id filtering is O(log n), not O(n)
    
    # Primary lookup: tenant + key (unique constraint for UPSERT)
    op.create_index(
        'idx_memories_tenant_key',
        'memories',
        ['tenant_id', 'key'],
        unique=True
    )
    
    # Category filtering within tenant
    op.create_index(
        'idx_memories_tenant_category',
        'memories',
        ['tenant_id', 'category']
    )
    
    # Importance-based retrieval (for "most important memories" queries)
    op.create_index(
        'idx_memories_tenant_importance',
        'memories',
        ['tenant_id', 'importance']
    )
    
    # Full-text search support (PostgreSQL)
    # Create a GIN index on key and value for ILIKE performance
    op.execute("""
        CREATE INDEX idx_memories_search 
        ON memories 
        USING gin (to_tsvector('english', key || ' ' || value))
    """)
    
    # Add trigger for updated_at
    op.execute("""
        CREATE OR REPLACE FUNCTION update_memories_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        
        CREATE TRIGGER trigger_memories_updated_at
        BEFORE UPDATE ON memories
        FOR EACH ROW
        EXECUTE FUNCTION update_memories_updated_at();
    """)


def downgrade() -> None:
    # Drop trigger and function
    op.execute('DROP TRIGGER IF EXISTS trigger_memories_updated_at ON memories')
    op.execute('DROP FUNCTION IF EXISTS update_memories_updated_at()')
    
    # Drop indexes
    op.execute('DROP INDEX IF EXISTS idx_memories_search')
    op.drop_index('idx_memories_tenant_importance', table_name='memories')
    op.drop_index('idx_memories_tenant_category', table_name='memories')
    op.drop_index('idx_memories_tenant_key', table_name='memories')
    
    # Drop table
    op.drop_table('memories')
