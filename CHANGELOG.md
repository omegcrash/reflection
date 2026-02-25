# Changelog

All notable changes to Reflection will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-03

### 🎉 Initial Release

Reflection 1.0 combines Familiar Core v2.5.0 with the enterprise multi-tenant wrapper.

### Added

#### Enterprise Layer (Reflection)
- **Multi-Tenant Architecture**
  - Complete tenant isolation at database and execution level
  - TenantAgent wrapper with scoped memory
  - O(1) LRU agent pool for efficient tenant management

- **Authentication & Authorization**
  - JWT-based authentication with refresh tokens
  - API key authentication for services
  - Role-based access control (Admin/Member/Viewer)

- **Quota System (Phase 6)**
  - Tier-based limits (Free/Pro/Enterprise)
  - Redis-backed distributed quota storage
  - Per-tenant rate limiting (requests/min, tokens/day)
  - Cost tracking and budget enforcement
  - Admin override capabilities

- **Tiered Executor Pools (Phase 2)**
  - Separate thread pools per tenant tier
  - Fair resource allocation
  - Queue management with overflow protection
  - Timeout handling

- **Dual-Path Orchestration (Phase 5)**
  - Async path for simple conversations (~80% of requests)
  - Sync path for complex tool workflows (~20% of requests)
  - SDK-based async LLM providers (Anthropic, OpenAI)
  - Streaming support via SSE

- **Database Persistence (Phase 3-4)**
  - PostgreSQL with async SQLAlchemy
  - Alembic migrations
  - Repository pattern for data access
  - Conversation and message storage
  - Usage tracking for billing

- **Observability (Phase 7)**
  - Prometheus-compatible metrics registry
  - 50+ metrics covering HTTP, LLM, quotas, database
  - Structured JSON logging with context propagation
  - Distributed tracing (OpenTelemetry compatible)
  - Sensitive data masking
  - Audit logging for security events

- **Testing Framework (Phase 8)**
  - Pytest configuration with markers
  - Test factories for data generation
  - Unit, integration, E2E, load, and security tests
  - Load testing utilities with result aggregation

#### Familiar Core v2.5.0
- **Agent System**
  - Multi-provider LLM support (Anthropic, OpenAI, Ollama)
  - Tool execution loop with security controls
  - Streaming responses
  - Memory and conversation history

- **20+ Skills**
  - Browser automation
  - Email (send/receive)
  - Calendar management
  - Google Drive integration
  - Knowledge base
  - Voice and video processing
  - SMS messaging
  - Task management
  - And more...

- **Multi-Channel Support**
  - Discord
  - Telegram
  - Slack
  - Microsoft Teams
  - WhatsApp
  - iMessage
  - CLI

- **Security**
  - Trust levels (Untrusted → Trusted)
  - Capability-based permissions
  - Budget controls per user
  - Input sanitization
  - Secrets detection and masking

- **Compliance**
  - HIPAA configuration profile
  - SOC2 readiness
  - Audit logging
  - Encryption at rest

### Architecture

```
Reflection (Enterprise)
├── FastAPI Gateway
├── Dual-Path Orchestrator
├── Tiered Executor Pools
├── Quota System
└── Observability Stack
         │
         ▼
Familiar Core (Agent)
├── Agent.chat()
├── Tool Registry
├── Skills (20+)
├── Channels (7+)
└── Security Model
```

### API

#### Chat Endpoints
- `POST /api/v2/chat` - Non-streaming chat
- `POST /api/v2/chat/stream` - SSE streaming chat

#### Authentication
- `POST /api/v1/auth/login` - Get tokens
- `POST /api/v1/auth/refresh` - Refresh token
- `POST /api/v1/auth/api-keys` - Manage API keys

#### Quotas
- `GET /api/v1/quotas/usage` - Current usage
- `GET /api/v1/quotas/tiers` - Tier definitions

#### Health
- `GET /health` - Liveness
- `GET /ready` - Readiness
- `GET /metrics/prometheus` - Prometheus metrics

### Configuration

Environment variables:
- `DATABASE_URL` - PostgreSQL connection
- `REDIS_URL` - Redis connection
- `JWT_SECRET_KEY` - JWT signing key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `OPENAI_API_KEY` - OpenAI API key

### Requirements

- Python 3.11+
- PostgreSQL 15+
- Redis 7+

### Known Limitations

- Streaming end-to-end needs additional work for complex tool workflows
- Tool permissions between Reflection tiers and Familiar Capabilities need manual sync
- Memory isolation relies on key prefixing (tenant-scoped keys)

---

## [2.0.0] - 2026-06-01

### Platform Maturity

v2.0.0 is a major release introducing API versioning, Enterprise SSO, and multi-region infrastructure.

### ⚠️ Breaking Changes
- **Minimum Python 3.12**: Upgrade from 3.11
- **Minimum PostgreSQL 16**: Upgrade from 14
- **API v1 Deprecated**: Sunset December 2026

### Added
- **API Versioning Infrastructure**:
  - Versioned API paths (`/api/v1/*`, `/api/v2/*`)
  - `APIVersion` enum (V1, V2) with version info
  - Version negotiation via `Accept-Version` header
  - Deprecation headers (RFC 8594) for v1 endpoints
  - `VersionedAPIRouter` for version-specific routes
  - `GET /api/versions` - List all API versions
  - `GET /api/versions/{version}` - Version details

- **Enterprise SSO (SAML 2.0)**:
  - `SAMLConfig` for IdP configuration
  - AuthnRequest generation and Response parsing
  - Attribute mapping for user claims
  - Session management with configurable duration
  - `POST /api/v2/sso/config/saml` - Configure SAML
  - `POST /api/v2/sso/login/saml` - Initiate SAML login
  - `POST /api/v2/sso/callback/saml` - Handle SAML assertion

- **Enterprise SSO (OpenID Connect)**:
  - `OIDCConfig` with automatic endpoint discovery
  - Authorization code flow with PKCE support
  - Token exchange and userinfo retrieval
  - Claim mapping for user attributes
  - `POST /api/v2/sso/config/oidc` - Configure OIDC
  - `POST /api/v2/sso/login/oidc` - Initiate OIDC login
  - `GET /api/v2/sso/callback/oidc` - Handle OIDC callback

- **SSO Session Management**:
  - `SSOUser` dataclass for IdP user info
  - `SSOSession` with expiration tracking
  - `SSOService` for unified SAML/OIDC handling
  - `GET /api/v2/sso/session/{id}` - Get session info
  - `DELETE /api/v2/sso/session/{id}` - Revoke session

- **Multi-Region Infrastructure**:
  - `RegionCode` enum (us-east-1, eu-west-1, etc.)
  - `Region` dataclass with coordinates and features
  - `RegionService` for tenant region assignment
  - Latency-based routing support
  - Data residency configuration for GDPR
  - Failover region configuration

- **Supported SSO Providers**:
  - Okta, Azure AD, Google Workspace
  - Auth0, OneLogin, Ping Identity
  - Custom SAML/OIDC providers

### Changed
- API v1 endpoints now return deprecation headers
- Job routes available at both `/api/v1/` and `/api/v2/`
- Lifecycle routes available at both `/api/v1/` and `/api/v2/`

### Deprecated
- API v1 (`/api/v1/*`) - Will sunset December 2026
- Use `/api/v2/*` for all new integrations

### API
- `APIVersion`, `VersionInfo`, `VersionContext` for versioning
- `SSOProtocol`, `SSOProvider`, `SSOUser`, `SSOSession` for SSO
- `SAMLConfig`, `OIDCConfig` for SSO configuration
- `RegionCode`, `Region`, `RegionConfig` for regions

---

## [1.5.0] - 2026-05-15

### Production Readiness

v1.5.0 adds production-ready features for GDPR compliance and tenant management, preparing the foundation for v2.0.0.

### Added
- **Background Job System**: Redis-backed async job queue
  - `JobService` for creating, tracking, and managing background jobs
  - Job status tracking: pending, running, completed, failed, cancelled, expired
  - Priority levels: low, normal, high, critical
  - Automatic retry with exponential backoff
  - Progress reporting and cancellation support
  - `JobContext` for handler communication

- **Data Export (GDPR Compliance)**:
  - `tenant_data_export`: Full tenant data export (Article 20 - Data Portability)
  - `user_data_export`: User-specific export (Article 15 - Right of Access)
  - Export formats: JSON, ZIP (with CSV)
  - Configurable data inclusion (conversations, messages, agents, usage)

- **Tenant Lifecycle Management**:
  - `TenantLifecycleService` for state transitions
  - States: active, suspended, pending_deletion, deleted
  - Suspension reasons: billing, policy, security, admin, requested
  - GDPR Article 17 - Right to Erasure with grace period
  - `TenantSuspendedError`, `TenantDeletedError` exceptions
  - Bulk suspension support

- **New API Endpoints**:
  - `POST /api/v1/jobs` - Create background job
  - `GET /api/v1/jobs/{id}` - Get job status
  - `POST /api/v1/jobs/{id}/cancel` - Cancel job
  - `GET /api/v1/jobs` - List tenant jobs
  - `POST /api/v1/jobs/export/tenant` - Export tenant data
  - `POST /api/v1/jobs/export/user` - Export user data
  - `GET /api/v1/admin/tenants/{id}/status` - Get tenant status
  - `POST /api/v1/admin/tenants/{id}/suspend` - Suspend tenant
  - `POST /api/v1/admin/tenants/{id}/reactivate` - Reactivate tenant
  - `POST /api/v1/admin/tenants/{id}/schedule-deletion` - Schedule deletion
  - `POST /api/v1/admin/tenants/{id}/cancel-deletion` - Cancel deletion

### API
- `Job`, `JobStatus`, `JobPriority`, `JobContext` for job management
- `TenantStatus`, `SuspensionReason`, `TenantLifecycleEvent` for lifecycle
- `ExportConfig` for data export configuration

---

## [1.4.0] - 2026-05-01

### Observability & Scale

v1.4.0 enhances observability with request ID propagation, comprehensive health checks, and tenant-labeled Prometheus metrics.

### Added
- **RequestContextMiddleware**: Full request context management
  - Generates or accepts `X-Request-ID` header
  - Propagates request ID through logging via context variables
  - Tracks request timing, client IP, and user agent
  - Sanitizes request IDs to prevent log injection
- **Enhanced Health Checks**:
  - `GET /health/live` - Kubernetes liveness probe (process running?)
  - `GET /health/ready` - Kubernetes readiness probe (can serve traffic?)
  - Component-level health with latency tracking
  - Detailed status: healthy, degraded, unhealthy
- **Prometheus Metrics with Tenant Labels**:
  - `familiar_executor_active_tasks{tier="..."}` - Active tasks by tier
  - `familiar_executor_queued_tasks{tier="..."}` - Queued tasks by tier
  - `familiar_executor_completed_total{tier="..."}` - Completed tasks by tier
  - `familiar_executor_queue_utilization{tier="..."}` - Queue utilization ratio
  - `familiar_executor_worker_utilization{tier="..."}` - Worker utilization ratio
  - `familiar_executor_avg_latency_seconds{tier="..."}` - Average task latency
  - `familiar_app_info{version="...",environment="..."}` - Application info
  - `familiar_uptime_seconds` - Application uptime
- **RequestContextFilter**: Logging filter for structured logs with request context
- FastAPI dependencies: `get_request_id_dependency`, `get_request_context_dependency`

### Changed
- Replaced inline request ID middleware with `RequestContextMiddleware`
- Health checks now return component-specific latency measurements
- `/health/ready` distinguishes between unhealthy (503) and degraded (200) states
- Executor pool metrics now include utilization ratios and latency

### API
- New `RequestContext` dataclass for request-scoped data
- New `ComponentHealth` model for individual component health
- New `LivenessResponse` and `ReadinessResponse` models

---

## [1.3.0] - 2026-04-01

### Billing Accuracy

v1.3.0 provides unified token counting and budget monitoring for accurate billing across all code paths.

### Added
- **UsageCalculator**: Unified token counting class that ensures consistent billing
  - Single source of truth for token calculations
  - Supports provider-reported, calculated, and estimated counts
  - Includes cost calculation with per-model pricing
- **UsageAlertService**: Budget monitoring and alerting system
  - Configurable thresholds (default: 80% warning, 95% critical, 100% exceeded)
  - Multiple notification handlers (webhook, log, callback)
  - Cooldown to prevent alert spam
  - Real-time budget status API
- **UsageRecord**: Complete usage record for billing with detailed breakdown
- **AggregatedUsage**: Usage aggregation across multiple requests
- Budget alert checking integrated into orchestrator

### Changed
- `chat()` method now uses unified `UsageCalculator`
- `stream()` method now uses unified `UsageCalculator` (was using simple count)
- `_record_usage()` now accepts `cost_usd` parameter
- `_record_usage()` triggers budget alert checks automatically

### Fixed
- **Token counting inconsistency**: `chat()` and `stream()` now use identical calculation logic
- Stream method was not counting system prompt or history tokens

### API
- New `AlertType` enum for different usage metrics
- New `AlertSeverity` enum (info, warning, critical, exceeded)
- New `AlertThresholds` configuration dataclass

---

## [1.2.0] - 2026-03-01

### Authentication Enhancement

v1.2.0 enhances authentication security with token revocation, refresh token rotation, and session management.

### Added
- **Token Revocation System**: Redis-backed `TokenStore` for immediate token invalidation
- **Session Management Endpoints**:
  - `GET /auth/sessions` - List all active sessions
  - `POST /auth/logout` - Logout current session  
  - `POST /auth/logout-all` - Logout all sessions
  - `DELETE /auth/sessions/{id}` - Revoke specific session
- `SessionHistoryModel` for session audit trail
- `SessionInfo` dataclass for session metadata
- Async JWT methods: `decode_token_async`, `refresh_access_token_async`, `create_tokens_with_session`

### Changed
- **Refresh Token Rotation**: Refresh tokens are now single-use
- Token refresh returns new refresh token (clients must store it)
- Login now registers session with IP and user agent tracking

### Security
- Token reuse detection: All sessions revoked if stolen token is reused
- `TokenRevokedError` exception for revoked tokens
- Session activity tracking (last_used_at updates)

### Database
- New `session_history` table with audit indexes

---

## [1.1.0] - 2026-02-15

### Security Hardening

v1.1.0 addresses critical security issues identified in the code review.

### Added
- JWT secret key validation at startup
- `AuthRateLimiter` class for auth-specific rate limiting
- `auto_create_tenants` and `allow_x_tenant_id_auth` settings
- Production safety checks in Settings class

### Changed
- X-Tenant-ID authentication gated to development mode only
- Error messages sanitized (no stack traces to users)
- Request IDs added to error logs for support reference

### Security
- Reject weak JWT secrets (< 32 chars, known defaults)
- Rate limiting on auth endpoints (5/min, lockout after 10)
- Production environment blocks dev-only features

---

## Future Versions

See [ROADMAP.md](ROADMAP.md) for detailed implementation plans.

### [2.1.0] - Target: August 2026 — Enhanced Enterprise
**Priority: MEDIUM**
- Directory sync (SCIM 2.0)
- Advanced audit logging
- Custom role definitions
- Webhook notifications

### [3.0.0] - Target: 2027 — Next Generation
**Priority: LOW**
- GraphQL API
- Real-time subscriptions
- Federated multi-cluster deployment
- AI model fine-tuning per tenant
