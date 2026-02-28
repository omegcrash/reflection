# Reflection — Enterprise Multi-Tenant AI Platform

- **Package**: `reflection` v2.0.0
- **Language**: Python >=3.11
- **Build**: hatchling
- **License**: MIT

## Structure

```
reflection/             # Main package (FastAPI app)
  auth/                 # Authentication + JWT
  cli.py                # Typer CLI entry point
  core/                 # Core logic
  gateway/              # API gateway / routing
  jobs/                 # Background jobs
  observability/        # OpenTelemetry + Prometheus
  routing/              # Request routing
  services/             # Business logic services
  tenants/              # Tenant management
  tenant_wrappers/      # Familiar tenant adapters
  data/                 # Data layer
reflection_core/        # Shared core utilities
  security/             # Security primitives
  exceptions/           # Exception types
tests/                  # Pytest test suite
alembic/                # DB migrations
docker-compose.yml      # Local dev stack
Dockerfile              # Container build
pyproject.toml          # Project metadata + tool config
```

## Commands

```bash
# Install (editable, with dev extras)
pip install -e ".[dev]"

# Run tests
pytest tests

# Lint
ruff check reflection reflection_core

# Type check (strict mode)
mypy reflection reflection_core

# Format check
ruff format --check reflection reflection_core

# DB migrations
alembic upgrade head

# Run server
reflection serve
```

## Key conventions

- **Line length**: 100 (ruff + black + mypy)
- **Ruff rules**: E, F, I, N, W, UP, B, C4, SIM (E501 + B008 ignored)
- **Mypy**: strict mode enabled — all functions must have type annotations
- **Async tests**: `asyncio_mode = "auto"`
- **Target Python**: 3.11 — 3.10 syntax is not sufficient
- **Depends on**: `familiar-agent>=1.4.0` (from PyPI)
- **Web framework**: FastAPI + Uvicorn
- **Database**: PostgreSQL (asyncpg) + SQLAlchemy async + Alembic migrations
- **Entry point**: `reflection` CLI (typer)
- **Observability**: OpenTelemetry + Prometheus + structlog
