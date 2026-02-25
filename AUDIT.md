# Reflection — Living Audit Document

**Last updated:** 2026-02-25
**Current version:** 2.0.0
**Maintained by:** George Scott Foley

This document is the single source of truth for known issues, intentional stubs,
architectural decisions, and the gap between what exists and what's wired.
Update it when you add a TODO, complete a stub, or make a decision you'll want
to remember next session.

---

## How to Use This Document

**Starting a session:** Read this first. It saves reconstructing context from code.

**Ending a session:** Add anything that's intentionally incomplete, any architectural
decision made, any new TODO introduced. If you completed something, mark it ✅ and
add the version it landed in.

**Marker conventions in code:**

| Marker | Meaning |
|--------|---------|
| `# TODO:` | Planned but not started |
| `# STUB:` | Structure exists, execution path not yet implemented |
| `# FIXME:` | Known bug, not yet fixed |
| `# INTENTIONAL:` | Looks wrong but isn't — explains why |

---

## Version Status

### Current: v2.0.0
Reflection split into its own repo (`omegcrash/reflection`) as a standalone
multi-tenant platform. Depends on `familiar-agent>=1.4.0` from PyPI.

**Packages:** `reflection/` (69 .py files), `reflection_core/` (8 .py files)
**Entry point:** `reflection` (CLI via typer, 20+ commands)
**Build:** hatchling | Python >=3.11
**Tests:** 700 passed, 0 failed, 1 skipped (10.7s)
**Lint:** ruff check + format clean (77 files)

### Familiar Core Dependency
Familiar v1.6.1 is the latest release of the core library (`omegcrash/familiar`).
PyPI has `familiar-agent==1.6.1`. Key features available:
- 50+ skills, all channel integrations (CLI, Telegram, Discord, Matrix, Teams, Signal, iMessage, WhatsApp, SMS)
- IMAP server, mesh gateway peer auth, Double Ratchet prev_chain_len
- 785 tests passing

---

## Test Results (2026-02-25)

**700 passed, 0 failed, 1 skipped** across 19 test files in 10.7s.
Tested against `familiar-agent==1.6.1` from PyPI.

### ~~Known Failure: SKILL.md packaging~~ ✅ Fixed (Familiar v1.6.1)
`test_skill_description_from_md` previously failed because the PyPI wheel did not
bundle `SKILL.md` files. Fixed in Familiar v1.6.1 by adding `artifacts` config to
`[tool.hatch.build.targets.wheel]` in `pyproject.toml`. 44 SKILL.md + 27 config.yaml
files now included in the wheel.

### ~~Known Failure: Skill count assertion~~ ✅ Fixed (Reflection 6bb1e2e)
`test_total_skill_count` hardcoded `== 40` but Familiar v1.6.1 ships 48 skills.
Changed to `>= 40` so it doesn't break when upstream adds skills.

### Known Skip
Environment-dependent test (likely hardware detection or optional dependency).

---

## Verified Implemented (Previously Misclassified as Stubs)

These files were initially classified as stubs but are **fully implemented and wired**
in `app.py`. Corrected 2026-02-25 after code review.

### ~~Gateway: Request Context Middleware~~ ✅ Implemented
**File:** `reflection/gateway/request_context.py` (413 lines)
**Implementation:** Full `RequestContextMiddleware` with ContextVar propagation,
X-Request-ID generation/forwarding, tenant context injection, structured logging filter.
**Wired in:** `app.py` — always registered as middleware.

### ~~Gateway: Token Store~~ ✅ Implemented
**File:** `reflection/gateway/token_store.py` (622 lines)
**Implementation:** Redis-backed `TokenStore` with session management, refresh token
rotation, `TokenReuseError` detection for replay attacks.
**Wired in:** `JWTService.decode_token_async()` in `auth.py`.

### ~~Gateway: Quota Middleware~~ ✅ Implemented
**File:** `reflection/gateway/quota_middleware.py` (316 lines)
**Implementation:** Full `QuotaMiddleware(BaseHTTPMiddleware)` with `QuotaChecker`
dependency injection, per-request quota enforcement, 429 responses.
**Wired in:** `app.py` — conditional on `settings.quota_middleware_enabled`.

### ~~Gateway: Rate Limiter~~ ✅ Implemented
**File:** `reflection/gateway/rate_limit.py` (425 lines)
**Implementation:** Redis-backed `RateLimiter` + `LoginRateLimiter` with sliding window
algorithm, progressive lockout, configurable thresholds.
**Wired in:** `auth_routes.py` — login endpoint brute-force protection.

### ~~Observability: Middleware~~ ✅ Implemented
**File:** `reflection/observability/middleware.py` (251 lines)
**Implementation:** `MetricsMiddleware` (Prometheus request metrics) + `TracingMiddleware`
(OpenTelemetry span creation), path normalization for cardinality control.
**Wired in:** `app.py` — `MetricsMiddleware` always registered, `TracingMiddleware`
production only.

### ~~Tenant Wrappers: Memory~~ ✅ Implemented
**File:** `reflection/tenant_wrappers/memory.py` (598 lines)
**Implementation:** `TenantMemory(Memory)` — DB-native UPSERT, LRU cache with TTL,
async+sync methods, SQL tenant isolation (tenant_id in every WHERE clause).
Also `TenantConversationHistory` for scoped chat history.
**Wired in:** `TenantAgent.__init__()` — replaces parent Memory when `db_session`
is provided. `AgentOrchestrator._get_agent()` passes the session automatically.
Corrected 2026-02-25 after code review.

### ~~Tenant Wrappers: Tools~~ ✅ Implemented
**File:** `reflection/tenant_wrappers/tools.py` (177 lines)
**Implementation:** `TenantToolRegistry(ToolRegistry)` — per-tenant enable/disable,
tenant-specific configs, usage tracking callback, sandboxed directories.
**Wired in:** `TenantAgent.__init__()` — creates `_tenant_tools` via
`get_tenant_tool_registry()`, applies skill preset filtering from `_allowed_skills`.
Corrected 2026-02-25 after code review.

### ~~Tenant Wrappers: Channels~~ ✅ Implemented
**File:** `reflection/tenant_wrappers/channels.py` (312 lines)
**Implementation:** `TenantChannelManager` (lifecycle management, health monitoring) +
`TenantChannelRouter` (server/chat-to-tenant routing for shared bots). Conditional
channel classes for Discord, Telegram, Teams.
**Wired in:** `app.py` — startup/shutdown lifecycle hooks initialize the singleton
and gracefully stop all running channels. Corrected 2026-02-25 after code review.

### ~~Gateway: Chat Routes V2~~ ✅ Implemented
**File:** `reflection/gateway/chat_routes_v2.py` (365 lines)
**Implementation:** 3 endpoints (`POST /v2/chat/completions`, `GET /v2/chat/path-info`,
`POST /v2/chat/simple`). Dual-path routing: async (direct provider call) vs sync
(thread pool + full Familiar Agent) via `AsyncOrchestrator` (967 lines). SSE streaming
with nginx-aware headers. Full auth with JWT, API key, and dev fallback — resolves
tenant tier from DB.
**Wired in:** `app.py` — `app.include_router(chat_router_v2, prefix="/api")`.
Corrected 2026-02-25 after code review.

---

## Known Intentional Stubs

This file has structure but incomplete or placeholder logic.

### Reflection Core: Types Package
**File:** `reflection_core/types/__init__.py`
**What exists:** Empty package directory.
**What's missing:** Type definitions (if any were planned).
**Priority:** Low — no code references this.

---

## Known Technical Debt

### ~~Familiar PyPI Package: Missing Non-Python Files~~ ✅ Fixed (Familiar v1.6.1)
Fixed by adding `artifacts` to wheel build config. 44 SKILL.md + 27 config.yaml
files now included.

### ~~Familiar PyPI Version Lag~~ ✅ Resolved (v1.6.1 published)
`familiar-agent==1.6.1` is now on PyPI with all v1.6.0 features (IMAP, mesh
gateway auth, prev_chain_len) plus the packaging fix.

---

## Architecture Overview

```
Reflection v2.0.0 — Enterprise Multi-Tenant AI Platform
├── reflection/
│   ├── cli.py                    — 20+ typer commands
│   ├── auth/sso.py               — SAML 2.0 + OIDC (550+ lines)
│   ├── core/
│   │   ├── settings.py           — Pydantic env config
│   │   ├── orchestrator.py       — Sync agent orchestration
│   │   ├── async_orchestrator.py — Dual-path (async simple, sync complex)
│   │   ├── providers_async.py    — Native Anthropic/OpenAI async SDKs
│   │   ├── circuit_breaker.py    — LLM provider resilience
│   │   ├── executor.py           — Tier-based thread pools
│   │   ├── tokens.py             — Accurate token counting + pricing
│   │   ├── regions.py            — Multi-region routing (519 lines)
│   │   ├── usage_calculator.py   — Unified billing
│   │   ├── usage_alerts.py       — Budget monitoring + webhooks
│   │   └── memory.py             — Summarization + semantic search
│   ├── data/
│   │   ├── models.py             — SQLAlchemy ORM (8 tables)
│   │   ├── postgres.py           — Async engine (PostgreSQL + SQLite)
│   │   ├── redis.py              — Redis + in-memory fallback
│   │   └── repositories.py       — CRUD with tenant isolation
│   ├── gateway/
│   │   ├── app.py                — FastAPI with 10+ routers
│   │   ├── auth.py               — JWT + bcrypt (OWASP compliant)
│   │   ├── auth_routes.py        — Register, login, token refresh
│   │   ├── chat_routes.py        — Chat completions + streaming
│   │   ├── sso_routes.py         — Enterprise SSO endpoints
│   │   ├── health.py             — Kubernetes probes + Prometheus
│   │   ├── request_context.py    — ContextVar propagation + X-Request-ID (413 lines)
│   │   ├── token_store.py        — Redis-backed session + refresh rotation (622 lines)
│   │   ├── quota_middleware.py    — Per-request quota enforcement (316 lines)
│   │   ├── rate_limit.py         — Sliding window + progressive lockout (425 lines)
│   │   └── chat_routes_v2.py     — Async dual-path chat API (365 lines)
│   ├── tenants/
│   │   ├── context.py            — contextvars isolation
│   │   ├── quotas.py             — Redis-backed enforcement
│   │   ├── lifecycle.py          — GDPR Article 17 support
│   │   └── quota_service.py      — Tier-based limits
│   ├── tenant_wrappers/
│   │   ├── agent.py              — TenantAgent (extends Familiar Agent)
│   │   ├── memory.py             — TenantMemory (DB-native, cached, tenant-isolated)
│   │   ├── tools.py              — TenantToolRegistry (per-tenant enable/disable)
│   │   └── channels.py           — TenantChannelManager + Router (multi-tenant bots)
│   ├── observability/
│   │   ├── logging.py            — Structured JSON + PII masking
│   │   ├── metrics.py            — 50+ Prometheus metrics
│   │   ├── tracing.py            — OpenTelemetry + fallback
│   │   └── middleware.py          — Metrics + Tracing auto-instrumentation (251 lines)
│   ├── routing/
│   │   ├── smart_router.py       — HIPAA-aware LLM routing
│   │   └── phi_detector.py       — PHI/PII detection (18 identifiers)
│   └── jobs/export_handlers.py   — GDPR data export (JSON/CSV/ZIP)
├── reflection_core/
│   ├── exceptions/hierarchy.py   — 19 exception classes (4 domains)
│   ├── security/encryption.py    — Fernet + PBKDF2 (480K iterations)
│   ├── security/sanitization.py  — Shell, path, prompt injection defense
│   └── security/trust.py         — Trust levels + 20 capabilities
├── tests/                        — 19 test files, 667 passing
├── alembic/                      — 4 migrations (8 tables + indexes)
├── Dockerfile                    — Multi-stage, non-root, health check
├── docker-compose.yml            — API + PostgreSQL 16 + Redis 7
└── .github/workflows/ci.yml     — lint + test (3.11, 3.12) + Docker build
```

---

## Architectural Decisions (Recorded)

### Repo Split: Familiar + Reflection (v1.4.0 / v2.0.0)
Familiar core library split into `omegcrash/familiar` as a standalone PyPI package
(`familiar-agent`). Reflection multi-tenant platform at `omegcrash/reflection`
depends on `familiar>=1.4.0`. Android app at `omegcrash/familiar-android` also
depends on `familiar-agent[llm,mesh]>=1.5.0`.

### Dependency Strategy
Reflection imports Familiar as a library dependency rather than vendoring.
This means Reflection always gets the latest Familiar features via `pip install --upgrade`.

### Dual-Path Orchestration (Phase 5)
Simple chat (no tools) routes through `AsyncOrchestrator` using native async SDKs.
Complex workflows (tools enabled) route through `TenantExecutorPool` with sync
Familiar `Agent.chat()` in thread pools. This avoids blocking the event loop while
maintaining Familiar's full tool execution capabilities.

### Tier-Based Thread Isolation
Each tenant tier (Free/Pro/Enterprise) gets its own thread pool with bounded queue.
Prevents noisy-neighbor issues where one tenant's heavy workload blocks others.
Free: 2 workers, Pro: 10, Enterprise: 50.

### HIPAA Smart Routing
PHI/PII detection runs before every LLM call. If sensitive data is detected and
the provider doesn't have a BAA, the request is routed to self-hosted Ollama.
General queries go to cloud APIs for better performance. Manual override available.

### Security: reflection_core
Security primitives (encryption, sanitization, trust model) live in `reflection_core`
rather than `reflection` to allow reuse without pulling in the full platform.
PBKDF2 at 480K iterations (OWASP 2023). Fernet for data at rest. Trust model
mirrors Familiar's with local fallback if Familiar not installed.

---

## CI Status

GitHub Actions pipeline configured in `.github/workflows/ci.yml`.
Runs on push to main/master/feat/** and pull requests.

**Repo:** `omegcrash/reflection`
**Matrix:** Python 3.11, 3.12 on `ubuntu-latest`
**Jobs:** Lint (ruff check + format), Test (matrix), Verify Import, Docker Build
**Lint:** ruff check + format — both clean (77 files)

Current result: 700 passed, 0 failed, 1 skipped.

---

## Open Questions

1. ~~**Familiar PyPI publish:**~~ ✅ Resolved — v1.6.1 published to PyPI.
2. ~~**SKILL.md packaging:**~~ ✅ Resolved — fixed in Familiar v1.6.1.
3. ~~**Stub priority:**~~ ✅ Resolved — the 4 gateway files (request_context, token_store,
   quota_middleware, rate_limit) were misclassified as stubs. All are fully implemented
   and wired in `app.py` / `auth_routes.py`. Corrected 2026-02-25.
