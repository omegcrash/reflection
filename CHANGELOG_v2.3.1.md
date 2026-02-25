# Reflection v2.3.1

Released: February 2026

## Smoke Test Fixes

Bugs found by the 8-phase smoke test that exercises the full API path
(tenant → user → JWT → API key → agent → conversation → usage → delete):

- **ConversationModel.updated_at** — column doesn't exist; model has
  `last_message_at`. Fixed 3 `order_by` clauses and 1 assignment in
  `ConversationRepository`.
- **bcrypt 5.x / passlib incompatibility** — bcrypt 5.0 removed the
  `__about__` attribute that passlib uses for version detection, breaking
  password hashing entirely. Pinned `bcrypt>=4.0.0,<5.0.0` in pyproject.toml.

## Smoke Test Results (all pass)

1. SQLite init + FastAPI (92 routes) + Familiar v2.6.1
2. Tenant create + User create + password hash/verify + JWT issue/decode
3. API key create + validate
4. Agent create
5. Conversation create + messages + stats update + list queries
6. Usage record + tenant aggregate + monthly spend
7. Cascade delete (conversation + messages) + commit
8. Tenant listing
