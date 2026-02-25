# Changelog v2.5.1 — Constitutional Integration Bugfixes

## Bug Fixes

Two integration bugs discovered during code review that prevented the 
constitutional personality layer from fully activating across all code paths.

### BUG-1 (P0): Channel not passed to get_system_prompt() in chat()

**File:** `familiar/core/agent.py`, line 528  
**Impact:** All channel-specific formatting guidance (Article 7) was silently 
ignored. Every Telegram, WhatsApp, Signal, and Discord message received CLI 
channel guidance instead of their constitutional channel personality.

The `channel` variable was available in scope but the call to 
`get_system_prompt()` omitted the `channel=` keyword argument, causing it 
to default to `"default"`.

**Fix:** Added `channel=channel` to the `get_system_prompt()` call in `chat()`.

### BUG-2 (P1): chat_stream() bypassed constitution entirely

**File:** `familiar/core/agent.py`, line 1149  
**Impact:** Streaming responses had no relationship stage tracking, no channel 
guidance, and no constitutional personality. Users on streaming-enabled channels 
would be stuck at Stage 1 permanently because `record_interaction()` was never 
called.

**Fix:** Added full constitutional integration to `chat_stream()`:
- User ID resolution from `user_context` (mirrors `chat()` path)
- Session creation via `get_or_create_session()`
- Interaction recording for relationship stage progression
- Constitutional system prompt with channel and session pass-through
- `user_context` propagated to non-streaming fallback path

### Articles Affected

| Article | Issue | Status |
|---------|-------|--------|
| 3 (Unique Relationship) | Stage tracking inactive on streaming path | **FIXED** |
| 5 (Earn Trust) | Interactions not counted in chat_stream() | **FIXED** |
| 7 (The Last Inch) | Channel guidance never delivered in chat() | **FIXED** |

### Version Bump

- Familiar: 2.6.3 → 2.7.1
- Reflection: 2.5.0 → 2.5.1
- VERSION file: 2.5.1 (Mother), 2.7.1 (Familiar)
- pyproject.toml: 2.5.1

### Mesh Versioning

Per constitutional mesh versioning (Phase 2 Trust = v2.5.1):
- v2.7.0: Phase 1 Discovery — constitutional layer deployed
- **v2.5.1: Phase 2 Trust — channel trust and stage tracking activated**
- v2.7.2: Phase 3 Memory (pending)
- v2.7.3: Phase 4 Delegation (pending)
