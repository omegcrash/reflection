# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Data layer for Reflection.

Provides database engine management, ORM models, and repository pattern
for tenant-isolated CRUD operations. Supports both PostgreSQL (production)
and SQLite (development/single-node) via SQLAlchemy async.
"""

from .models import (
    AgentModel,
    APIKeyModel,
    Base,
    ConversationModel,
    MemoryModel,
    MessageModel,
    TenantModel,
    TenantUserModel,
    UsageRecordModel,
)
from .postgres import close_database, get_database, get_db_session, init_database
from .redis import close_redis, get_redis, init_redis
from .repositories import (
    AgentRepository,
    APIKeyRepository,
    ConversationRepository,
    MessageRepository,
    TenantRepository,
    TenantUserRepository,
    UsageEvent,
    UsageRepository,
)

__all__ = [
    # Engine lifecycle
    "init_database",
    "close_database",
    "get_database",
    "get_db_session",
    # Redis lifecycle
    "init_redis",
    "close_redis",
    "get_redis",
    # ORM models
    "Base",
    "TenantModel",
    "TenantUserModel",
    "APIKeyModel",
    "AgentModel",
    "ConversationModel",
    "MessageModel",
    "UsageRecordModel",
    "MemoryModel",
    # Repositories
    "TenantRepository",
    "TenantUserRepository",
    "APIKeyRepository",
    "AgentRepository",
    "UsageRepository",
    "UsageEvent",
    "ConversationRepository",
    "MessageRepository",
]
