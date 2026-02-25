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

## Known Intentional Stubs

These files have structure but incomplete or placeholder logic. They are in the
request path and should be prioritized for implementation.

### Gateway: Request Context Middleware
**File:** `reflection/gateway/request_context.py`
**What exists:** Module file with imports.
**What's missing:** Full request context propagation (X-Request-ID, tenant context injection).
**Impact:** Request tracing incomplete without this middleware.
**Priority:** High — in the request path.

### Gateway: Token Store
**File:** `reflection/gateway/token_store.py`
**What exists:** Module file with imports.
**What's missing:** Token storage and revocation implementation (Redis-backed).
**Impact:** Token revocation (listed in CHANGELOG v1.2.0) not functional.
**Priority:** High — security feature.

### Gateway: Quota Middleware
**File:** `reflection/gateway/quota_middleware.py`
**What exists:** Module file with imports.
**What's missing:** Per-request quota enforcement in the middleware layer.
**Impact:** Quota checks may only happen at the service layer, not at the gateway.
**Priority:** Medium — quotas still enforced by QuotaService, this is defense-in-depth.

### Gateway: Rate Limiter
**File:** `reflection/gateway/rate_limit.py`
**What exists:** Module file with imports.
**What's missing:** Auth endpoint rate limiting implementation.
**Impact:** Brute-force protection on login/register endpoints.
**Priority:** High — security feature.

### Gateway: Chat Routes V2
**File:** `reflection/gateway/chat_routes_v2.py`
**What exists:** Shell/scaffold for enhanced v2 chat API.
**What's missing:** Full v2 implementation (streaming, enhanced tool use).
**Priority:** Medium — v1 chat routes are functional.

### Tenant Wrappers: Memory
**File:** `reflection/tenant_wrappers/memory.py`
**What exists:** Module file with imports.
**What's missing:** Tenant-scoped memory wrapper around Familiar's memory system.
**Impact:** Memory operations may not be properly tenant-isolated.
**Priority:** Medium — depends on whether TenantAgent handles this internally.

### Tenant Wrappers: Tools
**File:** `reflection/tenant_wrappers/tools.py`
**What exists:** Module file with imports.
**What's missing:** Tenant-scoped tool registry wrapper.
**Impact:** Tool permissions may not be properly tenant-isolated.
**Priority:** Medium.

### Tenant Wrappers: Channels
**File:** `reflection/tenant_wrappers/channels.py`
**What exists:** Module file with imports.
**What's missing:** Multi-channel support wrapper for tenant-scoped channel management.
**Priority:** Low — channels work via Familiar directly.

### Observability: Middleware
**File:** `reflection/observability/middleware.py`
**What exists:** Module file with imports.
**What's missing:** Auto-instrumentation middleware for logging/tracing per request.
**Impact:** Observability requires manual instrumentation without this.
**Priority:** Medium.

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
│   │   └── [4 stubs]             — request_context, token_store, quota, rate_limit
│   ├── tenants/
│   │   ├── context.py            — contextvars isolation
│   │   ├── quotas.py             — Redis-backed enforcement
│   │   ├── lifecycle.py          — GDPR Article 17 support
│   │   └── quota_service.py      — Tier-based limits
│   ├── tenant_wrappers/
│   │   ├── agent.py              — TenantAgent (extends Familiar Agent)
│   │   └── [3 stubs]             — memory, tools, channels
│   ├── observability/
│   │   ├── logging.py            — Structured JSON + PII masking
│   │   ├── metrics.py            — 50+ Prometheus metrics
│   │   ├── tracing.py            — OpenTelemetry + fallback
│   │   └── [1 stub]              — middleware
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
3. **Stub priority:** The 4 gateway stubs (request_context, token_store, quota_middleware,
   rate_limit) are in the request path. Should these be wired before any new features?
