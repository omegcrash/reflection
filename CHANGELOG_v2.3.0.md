# Reflection v2.3.0

Released: February 2026

## Embedded Familiar Upgraded to v2.6.0

Replaced the embedded familiar v1.5.8 with the clean v2.6.0 build. All 29 symbols
imported by `reflection` and `reflection_core` resolve correctly. Four modules
not present in v2.6.0 (`tool_router`, `skill_bus`, `workflow_state`, `guidance`)
were carried forward from the previous embedded copy with compatibility stubs.

## Runtime Bugs Fixed (from boot testing)

- **ConversationRepository + MessageRepository** — added to `data/repositories.py`.
  Both were imported by orchestrator, chat_routes, and export_handlers but did not exist.
- **TenantTier.PRO** — added as enum alias in both `tenants/models.py` and
  `core/executor.py`. `quota_service.py` referenced `PRO` but only `PROFESSIONAL` existed.
- **record_usage()** — added to `tenants/quotas.py`. Services orchestrator called it
  but the function was missing. Implemented as async fire-and-forget task.
- **quota_middleware import path** — fixed `gateway/quota_middleware.py` to import
  from `tenants.quota_service` instead of nonexistent `gateway.quota_service`.
- **get_current_user** — added FastAPI dependency to `gateway/auth.py`. Three route
  files used it as `Depends(get_current_user)` but it did not exist.
- **UUID/SQLite binding** — changed `_new_uuid()` to return `str(uuid.uuid4())` and
  fixed all 8 repository `id=` assignments. SQLite's aiosqlite driver cannot bind
  Python UUID objects to `String(36)` columns.
- **ConversationModel.status** — removed from ConversationRepository.create (column
  does not exist on the model).
- **sa.delete** — fixed to use imported `delete` function directly.

## Alembic Migration Chain Fixed

Fixed `003_session_history` migration: `down_revision` pointed to filename
`'20260202_000001_002_memories_table'` instead of revision ID `'002'`.

## Boot Test Results

- FastAPI app: 89 routes, zero import errors
- Embedded familiar: v2.6.0, 29/29 symbols resolve
- SQLite round-trip: tenant → conversation → messages → stats → delete ✓
