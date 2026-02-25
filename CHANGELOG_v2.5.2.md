# CHANGELOG — Reflection v2.5.2 / Familiar v2.7.2

## Progressive Capability + Moltbot Onboarding Parity

Convergence Pipeline Week 1 + Week 2 complete.

### Week 1 (P0 — shipped in v2.5.1/v2.7.1, included here)

**BUG-1 (P0):** Channel not passed to `get_system_prompt()` in `chat()`.
Fixed: Article 7 channel guidance now active for all 10 channels.

**BUG-2 (P1):** `chat_stream()` bypassed constitution entirely.
Fixed: full constitutional integration mirroring the `chat()` path.

### Week 1 (P0 — Onboarding + Proactive)

**Onboarding Wizard** `familiar/onboard.py` (726 LOC, NEW)
- `python -m familiar --onboard` or auto-prompted on first run
- 5-step flow: Provider → Channel → Name → Briefing → Activate
- Telegram: guided BotFather setup, waits for /start, sends test message
- Discord, WhatsApp, Signal, CLI paths
- Writes config.yaml + .env (chmod 600)
- Instant Win: first message over configured channel (Article 1)

**Proactive Presence Loop** `skills/proactive/skill.py` (+197 LOC)
- `heartbeat_check()` — 4-6hr interval, quiet hours respected, actionable gate
- `generate_eod_summary()` — end-of-day digest with task completion + tomorrow preview
- `run_heartbeat()`, `run_eod_summary()` — scheduler callbacks
- SILENCE_GUIDANCE enforced: empty return = no message sent

### Week 2 (P1 — Progressive Capability)

**Progressive Disclosure** `familiar/core/progressive.py` (340 LOC, NEW)
- `SkillUsageTracker` — per-skill invocation counters, discovery state, milestone tracking
- `STARTER_SKILLS` — 5 skills enabled at onboarding (calendar, tasks, email, websearch, proactive)
- `DISCOVERABLE_SKILLS` — tiered skill catalog (Tier 2 at 10, Tier 3 at 50, Tier 4 at 100)
- `suggest_skill_for_tool()` — contextual suggestion when tool-not-found
- `check_milestone()` — achievement messages at 10, 50, 100, 250, 500, 1000 interactions
- `generate_weekly_digest()` — one unused skill surfaced per week
- Wired into `tools.py` (tool-not-found path) and `agent.py` (success tracking + milestone check)

**Integration Points:**
- `tools.py`: tool-not-found now suggests enabling disabled skills (Article 4)
- `agent.py`: successful tool execution tracked per-skill
- `agent.py`: post-response milestone check with achievement append
- `agent.py`: `_get_skill_for_tool()` helper for skill↔tool mapping
- `core/__init__.py`: full progressive module exports
- `proactive/skill.py`: weekly capability digest tool + scheduler callback

### Version Summary

| Component | Version | Note |
|-----------|---------|------|
| Reflection | 2.5.2 | Full monorepo |
| Familiar core | 2.7.2 | Edition 0.0.2 (Phase 3 Memory) |
| Constitution | v2.7.7 | No changes |

### Convergence Pipeline Status

| Dimension | Status |
|-----------|--------|
| Personality | ✅ CLOSED (constitution v2.7.0) |
| Channel Intimacy | ✅ CLOSED (constitution + bugfixes v2.7.1) |
| Instant Win | ✅ CLOSED (onboarding wizard v2.7.1) |
| Proactive Presence | ✅ CLOSED (briefing + heartbeat + EOD v2.7.1) |
| Progressive Capability | ✅ CLOSED (starter set + suggestions + milestones + digest v2.7.2) |

**All 5 UX dimensions closed. Moltbot parity achieved.**

### LOC

| Metric | Count |
|--------|-------|
| Reflection total | 145,401 |
| New: onboard.py | 726 |
| New: progressive.py | 340 |
| Modified: proactive/skill.py | 367 → 595 (+228) |
| Modified: agent.py | 1,417 → 1,470 (+53) |
| Modified: tools.py | 757 → 766 (+9) |
| Modified: __main__.py | 327 → 352 (+25) |

---

George Scott Foley · February 2026 · MIT License
