# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""Session history audit table

Revision ID: 003_session_history
Revises: 002_memories_table
Create Date: 2026-02-02

v1.2.0 Authentication Enhancement - Session audit trail.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '003_session_history'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade():
    """Create session_history table for audit trail."""
    
    # Session history table
    op.create_table(
        'session_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, 
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('jti', sa.String(64), nullable=False, comment='JWT ID of the session'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('NOW()')),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('ip_address', sa.String(45), nullable=True, 
                  comment='Client IP (IPv4 or IPv6)'),
        sa.Column('user_agent', sa.String(500), nullable=True),
        sa.Column('device_name', sa.String(200), nullable=True),
        sa.Column('revocation_reason', sa.String(100), nullable=True,
                  comment='Why session was revoked: logout, logout_all, token_reuse, expired'),
        
        # Foreign keys
        sa.ForeignKeyConstraint(['user_id'], ['tenant_users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
    )
    
    # Indexes for efficient querying
    op.create_index(
        'idx_session_history_user_created',
        'session_history',
        ['user_id', 'created_at'],
        postgresql_using='btree',
    )
    
    op.create_index(
        'idx_session_history_tenant_created',
        'session_history',
        ['tenant_id', 'created_at'],
        postgresql_using='btree',
    )
    
    op.create_index(
        'idx_session_history_jti',
        'session_history',
        ['jti'],
        unique=True,
    )
    
    # Partial index for active sessions (not yet revoked)
    op.execute("""
        CREATE INDEX idx_session_history_active 
        ON session_history (user_id, created_at DESC)
        WHERE revoked_at IS NULL
    """)
    
    # Add comment to table
    op.execute("""
        COMMENT ON TABLE session_history IS 
        'Audit trail of user sessions for security and compliance (v1.2.0)'
    """)


def downgrade():
    """Drop session_history table."""
    op.drop_index('idx_session_history_active')
    op.drop_index('idx_session_history_jti')
    op.drop_index('idx_session_history_tenant_created')
    op.drop_index('idx_session_history_user_created')
    op.drop_table('session_history')
