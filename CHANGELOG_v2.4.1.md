# Reflection v2.4.1 — Installation Parity

**Release Date:** February 2026

## Summary

Brings Reflection's installation experience up to parity with
standalone Familiar. Adds a `run.sh` quick-start launcher, `--nonprofit`
installer flag, and expands the CLI installer with nonprofit and
enterprise use-case options.

## New: `run.sh` Quick-Start Launcher

Familiar-style zero-config start. Just run `./run.sh` and it:

- Auto-detects Python 3.11+ (cascades through 3.12 → 3.11 → 3)
- Creates and activates a virtual environment if needed
- Installs dependencies on first run
- Loads `.env` if present
- **Auto-detects LLM provider** (Familiar-style cascade):
  - `ANTHROPIC_API_KEY` set? → Anthropic
  - `OPENAI_API_KEY` set? → OpenAI
  - Ollama installed? → Ollama (auto-starts, auto-pulls model)
  - Nothing? → Helpful error with setup instructions
- Auto-detects database (Docker Postgres → SQLite fallback)
- Auto-detects Redis (Docker → in-memory fallback)
- Generates ephemeral JWT secret if not configured
- Starts the API server with hot reload

The goal: `export ANTHROPIC_API_KEY=sk-ant-... && ./run.sh` gets you
a running Reflection instance in under 30 seconds, no Docker required.

## New: `--nonprofit` Installer Flag

```bash
./install.sh --nonprofit      # Pre-selects nonprofit, prompts for provider
./install.sh --healthcare     # Pre-selects healthcare (HIPAA)
./install.sh --enterprise     # Pre-selects enterprise
```

These shortcuts skip the use-case prompt and go directly to provider
selection. The nonprofit flag activates the nonprofit skill preset
(donor CRM, grant tracker, bookkeeping, document generation, reports,
workflows, meetings) with the 501(c)(3) persona.

## Updated: CLI Installer

The interactive CLI installer now offers five use-case options:

1. 🏛️ Nonprofit / 501(c)(3) — donor CRM, grants, bookkeeping
2. 🏥 Healthcare / Medical — HIPAA compliant
3. 🔒 Maximum Privacy — everything self-hosted
4. 🏢 Enterprise — full productivity suite
5. 💼 General Business — API-powered (default)

Selecting nonprofit or enterprise now writes `SKILL_PRESET=nonprofit`
(or `enterprise`) to `.env`, which TenantConfig reads at startup.

The completion message for nonprofit includes getting-started examples:
- "Log a $500 gift from Jane Smith"
- "Show upcoming grant deadlines"
- "Prepare the board packet for next Tuesday"

## Platform Coverage (v2.4.1 vs Familiar 2.6.3)

| Feature              | Familiar 2.6.3 | Mother 2.4.1 |
|----------------------|:---:|:---:|
| Quick-start launcher | ✓ run.sh | ✓ run.sh |
| Provider auto-detect | ✓ | ✓ |
| --nonprofit flag     | ✓ | ✓ |
| Windows              | ✓ | — |
| Raspberry Pi         | ✓ | — |
| Docker Compose       | — | ✓ |
| GUI installer        | ✓ | ✓ |
| CLI installer        | — | ✓ |
| systemd service      | ✓ | — |
| Standalone binary    | — | ✓ |
| Dev subcommands      | — | ✓ |

Windows and Raspberry Pi remain out of scope for Reflection (it
requires PostgreSQL and async Python, which are enterprise deployment
concerns). Familiar continues to serve those platforms for personal use.
