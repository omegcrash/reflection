# Reflection v2.2.0 — Release Notes

**Date:** 2026-02-06

## What Changed

### Data Layer Created (Critical Fix)
The application-crashing missing `reflection/data/` package now exists:
- **`postgres.py`** — Async engine management. PostgreSQL via asyncpg in production, SQLite via aiosqlite for development. Auto-creates tables from ORM metadata in SQLite mode.
- **`redis.py`** — Redis connection with in-memory fallback. `_InMemoryRedis` class satisfies the same interface when Redis is unavailable.
- **`models.py`** — 8 ORM models matching Alembic migrations 001-003: TenantModel, TenantUserModel, APIKeyModel, AgentModel, ConversationModel, MessageModel, UsageRecordModel, MemoryModel.
- **`repositories.py`** — TenantRepository, TenantUserRepository, APIKeyRepository, AgentRepository, UsageRepository, and UsageEvent pydantic model. Every method name traced from actual call sites across the gateway.

### Tenant CLI Wired to Database
- `reflection tenant create` now persists via TenantRepository, checks slug uniqueness, and optionally creates admin user with `--admin-email`.
- `reflection tenant list` queries the database and displays a formatted table.
- No more "Database integration pending" stubs.

### Scripts Merged
- `quickstart.sh` removed. Its subcommands are now in `install.sh`:
  - `./install.sh` — guided installer (unchanged)
  - `./install.sh setup` — install deps + create .env
  - `./install.sh dev` — start dev server with hot reload
  - `./install.sh test` — run pytest
  - `./install.sh docker` — docker compose up
  - `./install.sh shell` — interactive Python shell

### Cleanup (from prior session, included in this release)
- All name redactions fixed
- All license headers unified to MIT
- `aiosqlite` added to dependencies
- `.env.example` JWT default pre-generated for dev mode
- 30+ dev doc artifacts stripped
- Version unified to 2.2.0

## Upgrade

Replace your existing Reflection directory.
Run `./install.sh setup` to install the new aiosqlite dependency.
For existing PostgreSQL deployments, no migration needed — the data layer maps to the same Alembic schema.
