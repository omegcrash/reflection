#!/usr/bin/env bash
# Reflection — Quick Start
# Just run: ./run.sh
#
# Auto-detects configuration and starts the API server.
# For first-time setup, use: ./install.sh
set -e
cd "$(dirname "$0")"

echo "🐍 Reflection v2.0.0"
echo "========================"

# ── Find Python ──────────────────────────────────────────────
PYTHON=""
for p in python3.12 python3.11 python3; do
    if command -v "$p" &>/dev/null; then
        ver=$($p -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON="$p"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "❌ Python 3.11+ not found"
    echo "   macOS:  brew install python@3.12"
    echo "   Ubuntu: sudo apt install python3.12"
    exit 1
fi
echo "✓ Python: $($PYTHON --version 2>&1)"

# ── Activate venv if it exists ────────────────────────────────
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment active"
elif [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment active"
fi

# ── Check if deps are installed ───────────────────────────────
if ! $PYTHON -c "import fastapi" 2>/dev/null; then
    echo ""
    echo "Dependencies not installed. Running first-time setup..."
    echo ""
    if [ -d ".venv" ] || [ -d "venv" ]; then
        pip install -e . -q 2>/dev/null || pip install -e . --break-system-packages -q
    else
        echo "Creating virtual environment..."
        $PYTHON -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip -q
        pip install -e . -q
    fi
    echo "✓ Dependencies installed"
fi

# ── Load .env if present ──────────────────────────────────────
if [ -f ".env" ]; then
    set -a
    source .env 2>/dev/null || true
    set +a
    echo "✓ Loaded .env"
else
    echo "⚠ No .env file — using defaults"
    echo "  Run ./install.sh for guided setup"
fi

# ── Detect LLM provider (Familiar-style cascade) ──────────────
PROVIDER=""
if [ -n "$LLM_DEFAULT_PROVIDER" ]; then
    PROVIDER="$LLM_DEFAULT_PROVIDER"
elif [ -n "$ANTHROPIC_API_KEY" ] || [ -n "$LLM_ANTHROPIC_API_KEY" ]; then
    PROVIDER="anthropic"
elif [ -n "$OPENAI_API_KEY" ] || [ -n "$LLM_OPENAI_API_KEY" ]; then
    PROVIDER="openai"
elif command -v ollama &>/dev/null; then
    PROVIDER="ollama"
    # Auto-start Ollama if not running
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "Starting Ollama..."
        ollama serve > /dev/null 2>&1 &
        sleep 2
    fi
    MODEL_COUNT=$(ollama list 2>/dev/null | tail -n +2 | wc -l)
    if [ "$MODEL_COUNT" -eq 0 ]; then
        # Detect hardware for the shared setup script
        TOTAL_RAM_MB=0
        IS_PI=false
        if [[ -f /proc/meminfo ]]; then
            TOTAL_RAM_MB=$(awk '/MemTotal/ {printf "%d", $2/1024}' /proc/meminfo)
        elif command -v sysctl &>/dev/null; then
            TOTAL_RAM_MB=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%d", $1/1024/1024}')
        fi
        if [[ -f /proc/device-tree/model ]] && grep -qi "raspberry" /proc/device-tree/model 2>/dev/null; then
            IS_PI=true
        fi

        # Source the shared model setup (try both possible locations)
        SCRIPT_DIR="$(dirname "$0")"
        if [[ -f "$SCRIPT_DIR/familiar/scripts/model_setup.sh" ]]; then
            source "$SCRIPT_DIR/familiar/scripts/model_setup.sh"
        elif [[ -f "$SCRIPT_DIR/scripts/model_setup.sh" ]]; then
            source "$SCRIPT_DIR/scripts/model_setup.sh"
        fi

        if type run_model_setup &>/dev/null; then
            if ! run_model_setup; then
                PROVIDER=""
            else
                # Re-source .env in case setup wrote to it
                if [[ -f ".env" ]]; then set -a; source .env 2>/dev/null || true; set +a; fi
            fi
        else
            echo "⚠ Model setup script not found. Pulling llama3.2..."
            ollama pull llama3.2
        fi
    fi
fi

if [ -n "$PROVIDER" ]; then
    echo "✓ LLM provider: $PROVIDER"
    export LLM_DEFAULT_PROVIDER="${LLM_DEFAULT_PROVIDER:-$PROVIDER}"
else
    echo ""
    echo "⚠ No LLM provider detected. Pick one:"
    echo ""
    echo "  Option 1 (API):   export ANTHROPIC_API_KEY='sk-ant-...'"
    echo "  Option 2 (API):   export OPENAI_API_KEY='sk-...'"
    echo "  Option 3 (Local): curl -fsSL https://ollama.ai/install.sh | sh"
    echo ""
    echo "  Then run ./run.sh again"
    exit 1
fi

# ── Detect database ───────────────────────────────────────────
if [ -z "$DATABASE_URL" ]; then
    # No database configured — check if Docker services are running
    if command -v docker &>/dev/null && docker ps --format '{{.Names}}' 2>/dev/null | grep -q postgres; then
        export DATABASE_URL="postgresql+asyncpg://familiar:familiar@localhost:5432/reflection"
        echo "✓ PostgreSQL (Docker)"
    else
        # Fall back to SQLite for quick start
        export DATABASE_URL="sqlite+aiosqlite:///$(pwd)/data/familiar.db"
        mkdir -p data
        echo "✓ SQLite (development mode)"
    fi
else
    echo "✓ Database configured"
fi

# ── Redis (optional) ─────────────────────────────────────────
if [ -z "$REDIS_URL" ]; then
    if command -v docker &>/dev/null && docker ps --format '{{.Names}}' 2>/dev/null | grep -q redis; then
        export REDIS_URL="redis://localhost:6379/0"
        echo "✓ Redis (Docker)"
    else
        echo "~ Redis not available (using in-memory fallback)"
    fi
fi

# ── Generate JWT secret if missing ────────────────────────────
if [ -z "$SECURITY_JWT_SECRET_KEY" ]; then
    export SECURITY_JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(64))")
    echo "✓ Generated JWT secret (ephemeral — add to .env for persistence)"
fi

# ── Show preset if set ────────────────────────────────────────
if [ -n "$SKILL_PRESET" ] && [ "$SKILL_PRESET" != "general" ]; then
    echo "✓ Skill preset: $SKILL_PRESET"
fi

# ── Show HIPAA status ─────────────────────────────────────────
if [ "$HIPAA_COMPLIANT" = "true" ]; then
    echo "✓ HIPAA mode: PHI → ${PHI_PROVIDER_NAME:-ollama}"
fi

# ── Start ─────────────────────────────────────────────────────
echo ""
echo "Starting Reflection..."
echo "  API:  http://localhost:8000"
echo "  Docs: http://localhost:8000/docs"
echo ""

exec uvicorn reflection.gateway.app:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --reload \
    --log-level info
