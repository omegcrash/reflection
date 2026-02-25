# Reflection

**Enterprise Multi-Tenant AI Companion Platform**

*"From one, many. From chaos, order — at scale."*

Reflection extends [Familiar](https://github.com/familiar-ai/familiar) with enterprise multi-tenancy capabilities. All core AI functionality comes from Familiar; this package adds the enterprise layer.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Reflection                           │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │         Tenant Wrappers (~1,200 lines)                 │ │
│  │                                                        │ │
│  │  TenantAgent        Per-tenant agent instances         │ │
│  │  TenantMemory       Tenant-scoped memory               │ │
│  │  TenantToolRegistry Per-tenant tool permissions        │ │
│  │  TenantChannels     Multi-tenant Discord/Telegram      │ │
│  │                                                        │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │         Familiar Core (~103,000 lines)            │ │ │
│  │  │                                                   │ │ │
│  │  │  Agent         Full agent loop                    │ │ │
│  │  │  Providers     Anthropic, OpenAI, Ollama          │ │ │
│  │  │  ToolRegistry  50+ built-in tools                 │ │ │
│  │  │  Memory        Episodic + semantic memory         │ │ │
│  │  │  Channels      Discord, Telegram, Teams, etc      │ │ │
│  │  │  Skills        Browser, calendar, email, etc      │ │ │
│  │  │  Guardrails    Safety + compliance                │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              Enterprise Layer                          │ │
│  │                                                        │ │
│  │  PostgreSQL    Tenants, users, conversations, usage    │ │
│  │  Redis         Quotas, sessions, distributed cache     │ │
│  │  FastAPI       REST API + SSE streaming                │ │
│  │  JWT Auth      Access + refresh tokens, bcrypt         │ │
│  │  Quotas        Rate limiting, token budgets            │ │
│  │  Billing       Usage tracking, metering                │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from reflection import get_tenant_agent, TenantToolRegistry
from uuid import uuid4

# Create a tenant
tenant_id = uuid4()

# Get an agent for this tenant
agent = get_tenant_agent(tenant_id, tenant_config={
    "agent_name": "Acme Assistant",
    "default_provider": "anthropic",
})

# Chat (automatically isolated to tenant)
response = agent.chat("Hello!", user_id="user123")
print(response)

# Per-tenant tool permissions
registry = TenantToolRegistry(tenant_id)
registry.enable_tool("web_search")
registry.disable_tool("shell_execute")
```

## Features

### Multi-Tenancy
- **Tenant Isolation**: Each tenant has isolated agents, memory, and data
- **Per-Tenant Configuration**: Custom models, providers, quotas per tenant
- **Shared Infrastructure**: One deployment serves many tenants

### Enterprise Components
- **PostgreSQL Persistence**: Conversations, messages, usage stored durably
- **JWT Authentication**: Secure API access with refresh tokens
- **Quota Management**: Rate limits, token budgets, concurrent request limits
- **Usage Tracking**: Detailed metering for billing

### Tenant Wrappers

| Wrapper | Extends | Purpose |
|---------|---------|---------|
| `TenantAgent` | `familiar.Agent` | Per-tenant agent instances |
| `TenantMemory` | `familiar.Memory` | Tenant-scoped memory |
| `TenantToolRegistry` | `familiar.ToolRegistry` | Per-tenant tool permissions |
| `TenantDiscordChannel` | `familiar.DiscordChannel` | Multi-tenant Discord bots |
| `TenantTelegramChannel` | `familiar.TelegramChannel` | Multi-tenant Telegram bots |

## Installation

### Quick Install (Recommended)

```bash
# 1. Download Reflection
curl -L https://github.com/familiar-ai/reflection/archive/refs/heads/main.zip -o reflection.zip
unzip reflection.zip
cd reflection

# 2. Run installer (GUI wizard)
pip install -e .
reflection install

# That's it! Browser will open to http://localhost:8000
```

The installer will:
- ✅ Detect your hardware
- ✅ Ask about your use case (Healthcare/Business/Privacy)
- ✅ Install dependencies (Ollama for self-hosted, or configure API)
- ✅ Set up databases
- ✅ Launch Reflection

**Total time:** 5-10 minutes

### Manual Installation

```bash
# Install Familiar (dependency)
pip install familiar-ai

# Install Reflection
pip install reflection

# Or from source
git clone https://github.com/familiar-ai/reflection
cd reflection
pip install -e .
```

## Configuration

```bash
# Required
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost/reflection"
export REDIS_URL="redis://localhost:6379"
export JWT_SECRET_KEY="your-secret-key"

# LLM Providers (at least one)
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# Optional
export ENVIRONMENT="production"
export DEBUG="false"
```

## Running

### Docker Compose (Recommended)

```bash
docker-compose up -d
```

### Manual

```bash
# Initialize database
reflection db init
reflection db migrate

# Start server
reflection serve --host 0.0.0.0 --port 8000
```

## API

### Chat Completion

```bash
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "stream": false}'
```

### Streaming

```bash
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message": "Write a haiku", "stream": true}'
```

## Project Structure

```
reflection/
├── reflection/
│   ├── __init__.py         # Main exports
│   ├── tenant_wrappers/    # Tenant isolation layer
│   │   ├── agent.py        # TenantAgent
│   │   ├── memory.py       # TenantMemory
│   │   ├── tools.py        # TenantToolRegistry
│   │   └── channels.py     # TenantDiscord/Telegram
│   ├── core/               # Enterprise core
│   │   ├── orchestrator.py # DB-backed agent loop
│   │   ├── memory.py       # Enterprise memory service
│   │   └── settings.py     # Configuration
│   ├── tenants/            # Tenant management
│   │   ├── context.py      # Request-scoped context
│   │   ├── quotas.py       # Rate limiting
│   │   └── models.py       # Tenant models
│   ├── data/               # Database layer
│   │   ├── models.py       # SQLAlchemy models
│   │   ├── repositories/   # Data access
│   │   └── postgres.py     # Connection management
│   └── gateway/            # API layer
│       ├── app.py          # FastAPI app
│       ├── auth.py         # JWT authentication
│       └── routes.py       # API endpoints
├── reflection_core/           # Shared primitives
│   ├── security/           # Trust, sanitization
│   └── exceptions/         # Error hierarchy
├── tests/                  # Test suite
├── alembic/                # Database migrations
└── docker-compose.yml      # Deployment config
```

## Relationship to Familiar

Reflection is designed as a **thin enterprise wrapper** around Familiar:

- **Familiar**: Core AI capabilities (agents, tools, memory, providers, channels)
- **Reflection**: Multi-tenancy, persistence, billing, quotas

This separation means:
- Familiar improvements flow through automatically
- Enterprise features are clearly isolated
- Testing is focused on tenant isolation, not AI capabilities
- Development is faster (no wheel reinvention)

## License

MIT License

Commercial licensing available. Contact: licensing@familiar.ai

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
