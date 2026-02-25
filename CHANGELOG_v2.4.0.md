# Reflection v2.4.0 — Familiar 2.6.3 Reintegration

**Release Date:** February 2026

## Summary

Full reintegration of Familiar v2.6.3 into Reflection. Brings the embedded
Familiar core to parity with standalone, adds 18 missing skills (including the
complete nonprofit suite), and introduces tenant-scoped data isolation and
skill presets for one-API-call tenant provisioning.

## New Features

### Skill Presets
- New `skill_preset` field on TenantConfig: `"general"`, `"nonprofit"`, `"healthcare"`, `"enterprise"`
- Each preset activates a curated bundle of skills with an appropriate persona
- Nonprofit preset: donor CRM, grant tracker, bookkeeping, documents, reports, workflows, meetings
- Healthcare preset: PHI detection, audit, RBAC, encryption + HIPAA routing
- Enterprise preset: full productivity suite with knowledge management

### 18 New Skills (from Familiar 2.6.3)
**Nonprofit Suite:**
- `bookkeeping` — Fund accounting, donation receipts, 990-prep data export
- `workflows` — 5 nonprofit workflow templates (donor onboarding, grant lifecycle, board prep, year-end, monthly close)
- `documents` — Thank-you letters, board reports, grant narratives (DOCX/PDF generation)
- `reports` — Board packets, donor summaries, financial snapshots (XLSX/CSV)
- `contacts` — Contact management with organization directory
- `meetings` — Meeting scheduling, agendas, minutes

**Enterprise/Security:**
- `audit` — Audit trail viewing and compliance reporting
- `rbac` — Role-based access control management
- `encryption` — Key management, encrypted storage operations
- `sessions` — Session management, active session tracking
- `user_management` — User CRUD, role assignment
- `phi_detection` — PHI/PII detection as standalone skill

**Utilities:**
- `websearch` — Web search integration
- `smart_search` — Semantic search across memory and documents
- `filereader` — File content extraction
- `knowledge_base` — Persistent knowledge base management
- `notifications` — Push notifications, alerts
- `transcription` — Audio/video transcription

### Tenant Data Isolation
- Skills now use `_get_data_dir()` for dynamic path resolution
- `set_tenant_data_root(tenant_id)` in `familiar.core.paths` scopes all data to `~/.familiar/tenants/{id}/data/`
- TenantAgent calls `set_tenant_data_root()` before agent initialization
- Standalone Familiar behavior unchanged (default `~/.familiar/data/`)

## Core Sync

Embedded Familiar core updated to v2.6.3 parity:
- Graceful provider detection (checks both API key AND package availability)
- Native urllib Ollama fallback (zero external dependencies)
- Guarded `psutil` import prevents crash on minimal installs
- `atomic_write_json` / `safe_load_json` for crash-safe persistence
- All 15 installation fixes from the v2.6.3 patch set

## Compatibility

- All 18 cross-boundary imports verified against Familiar 2.6.3
- `skill_bus.py` (Mother-only module with ChainExecutor) preserved during core sync
- SmartLLMRouter (PHI routing) and ModelRouter (provider failover) continue to compose without conflict
- reflection_core shared types library unchanged

## Breaking Changes

None. Existing TenantConfig with `skill_preset` unset defaults to `"general"` (all skills enabled, original behavior).
