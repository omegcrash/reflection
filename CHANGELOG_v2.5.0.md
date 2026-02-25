# Changelog v2.5.0 — Constitutional Personality Layer

## Phase 1: The Familiar Constitution

The Familiar Constitution — eight articles of binding design law — is now 
implemented as a production system prompt injection layer. Every response 
the agent generates is shaped by the constitutional ethic.

### New: `familiar/core/constitution.py`

The core constitutional personality module. Translates Articles 0-7 into 
dynamic, per-user, per-channel system prompts.

**Key components:**

- **ConstitutionalIdentity**: Builds constitutional system prompts combining 
  Article 0 (Sacred Exchange), Articles 1-7 (calibrated to relationship stage), 
  channel-specific delivery guidance, and optional tenant persona overlays.

- **RelationshipStage**: Three-stage trust calibration (Articles 3, 5).
  - Stage 1 (0-10 interactions): Professional, helpful, no presumed familiarity.
  - Stage 2 (10-50 interactions): Warmer tone, references shared history.
  - Stage 3 (50+ interactions): Trusted colleague, anticipates needs.

- **InteractionTracker**: Persistent per-user interaction counter for 
  relationship stage progression. Stores in `~/.familiar/data/interaction_counts.json`.

- **ChannelPersonality**: Channel-specific delivery guidance (Article 7).
  Telegram, WhatsApp, Discord, Slack, Signal, iMessage, Email, CLI, Teams, API.

- **Silence Guidance**: Proactive messages include explicit guidance on when 
  silence is also service (Article 2).

- **Constitutional Presets**: Role-specific overlays for Reflection tenant 
  skill presets (nonprofit, healthcare, enterprise).

### Modified: `familiar/core/agent.py`

- `get_system_prompt()` now uses constitutional identity as primary prompt 
  source with legacy fallback.
- Accepts `channel` and `is_proactive` parameters.
- Automatically records interactions for relationship stage progression.

### Modified: `familiar/core/context.py`

- `IdentityProcessor` now uses constitutional prompt generation.
- Accepts `channel`, `user_id`, `is_proactive`, and `persona_overlay` kwargs.
- Legacy fallback preserved if constitution module unavailable.

### Modified: `reflection/tenants/models.py`

- `SKILL_PRESETS` now include `constitutional_preset` key mapping each 
  preset to its constitutional persona overlay.

### Modified: `reflection/tenant_wrappers/agent.py`

- `_apply_tenant_config()` now injects constitutional persona overlay 
  from preset into tenant instances via `get_constitutional_preset()`.

### New: `tests/test_constitution.py`

Comprehensive test suite covering:
- Relationship stage progression and thresholds
- Interaction tracker persistence and user separation
- Constitutional prompt generation per stage
- Channel-specific guidance injection
- Proactive silence guidance
- Tenant preset verification
- Serialization

### Constitutional Articles Implemented

| Article | Title | Implementation |
|---------|-------|---------------|
| 0 | The Sacred Exchange | Always-present foundational ethic |
| 1 | The Person Is Sacred | Service-over-showcase directive |
| 2 | Anticipate from Gratitude | Proactive behavior guidance + silence |
| 3 | Every Relationship Unique | Stage-calibrated personalization |
| 4 | Serve Humility (Hot Dog) | Response scale matching |
| 5 | Earn Trust | Stage progression, reciprocity |
| 6 | Protection Is Harvest Prayer | Silent security directive |
| 7 | The Last Inch | Channel formatting, delivery ceremony |

### Prompt Metrics

- Constitutional prompt length: ~3,300 characters
- Zero additional latency (string assembly only)
- No new dependencies required
- Full backward compatibility (legacy fallback on ImportError)
