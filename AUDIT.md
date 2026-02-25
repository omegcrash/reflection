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

**Packages:** `reflection/`, `reflection_core/`
**Entry point:** `reflection` (CLI via typer)
**Build:** hatchling | Python >=3.11

### Familiar Core Dependency
Familiar v1.6.0 is the latest release of the core library (`omegcrash/familiar`).
Key features available via the dependency:
- IMAP server (STORE, COPY, EXPUNGE, CREATE, DELETE, RENAME)
- Mesh gateway peer auth with routing table exchange
- Double Ratchet prev_chain_len tracking
- 50+ skills, all channel integrations
- 785 tests passing

---

## Repo Split History

The original monorepo (`omegcrash/familiar-reflection-ai`) was split at v1.4.0:

- **`omegcrash/familiar`** — Core library, published as `familiar-agent` on PyPI
- **`omegcrash/reflection`** — Multi-tenant platform (this repo)
- **`omegcrash/familiar-android`** — GrapheneOS mobile app

The old `familiar-reflection-ai` GitHub name redirects to `omegcrash/familiar`.

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

---

## CI Status

GitHub Actions pipeline configured in `.github/workflows/ci.yml` (if present).

**Repo:** `omegcrash/reflection`
**Build:** hatchling
**Python:** >=3.11

---

## Open Questions

(None currently — this is a fresh audit for the standalone repo.)
