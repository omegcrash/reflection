#!/bin/bash
# Reflection - One-Line Installer
# Usage:
#   curl -fsSL https://install.reflection.ai | bash
#   OR just: ./install.sh
#
# Detects GUI availability and launches the appropriate installer.
# Works on Linux, macOS, and WSL.

set -e

GREEN='\033[92m'
CYAN='\033[96m'
YELLOW='\033[93m'
RED='\033[91m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════╗"
echo -e "║       🐍  Reflection Installer  🐍           ║"
echo -e "║    Enterprise Multi-Tenant AI Companion Platform      ║"
echo -e "╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ── Find Python 3.11+ ────────────────────────────────────────

PYTHON=""
for cmd in python3.12 python3.11 python3; do
    if command -v "$cmd" &>/dev/null; then
        version=$($cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}✗ Python 3.11+ not found${NC}"
    echo ""
    echo "  Install Python 3.11+:"
    echo "    macOS:   brew install python@3.12"
    echo "    Ubuntu:  sudo apt install python3.12"
    echo "    Fedora:  sudo dnf install python3.12"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Found $PYTHON ($version)"

# ── Find project directory ────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/installer/setup_wizard.py" ]; then
    PROJECT_DIR="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/setup_wizard.py" ]; then
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
else
    echo -e "${RED}✗ Cannot find Reflection project directory${NC}"
    echo "  Run this script from the project root: ./install.sh"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Project: $PROJECT_DIR"

# ── Helper: activate venv or create it ────────────────────────

activate_venv() {
    if [ ! -d "$PROJECT_DIR/.venv" ]; then
        echo -e "  ${YELLOW}Creating virtual environment...${NC}"
        $PYTHON -m venv "$PROJECT_DIR/.venv"
    fi
    source "$PROJECT_DIR/.venv/bin/activate"
}

# ── Subcommand routing ────────────────────────────────────────
# ./install.sh          → guided installer (default)
# ./install.sh dev      → start dev server
# ./install.sh test     → run tests
# ./install.sh docker   → docker-compose up
# ./install.sh shell    → interactive Python shell
# ./install.sh setup    → install deps only

SUBCOMMAND="${1:-}"
case "$SUBCOMMAND" in
    setup)
        echo -e "\n  ${CYAN}Setting up development environment...${NC}\n"
        activate_venv
        pip install --upgrade pip -q
        pip install -e ".[dev]" 2>/dev/null || pip install -e .
        if [ ! -f "$PROJECT_DIR/.env" ] && [ -f "$PROJECT_DIR/.env.example" ]; then
            cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
            echo -e "  ${GREEN}✓${NC} Created .env from .env.example"
            echo -e "  ${YELLOW}Edit .env to add your API keys${NC}"
        fi
        echo -e "\n  ${GREEN}✓ Setup complete!${NC}"
        echo -e "\n  Next: ${CYAN}./install.sh dev${NC}"
        exit 0
        ;;
    dev)
        echo -e "\n  ${CYAN}Starting development server...${NC}\n"
        activate_venv
        if ! pip show reflection &>/dev/null 2>&1; then
            echo -e "  ${YELLOW}Installing dependencies...${NC}"
            pip install --upgrade pip -q
            pip install -e . -q
        fi
        echo -e "  API:  ${GREEN}http://localhost:8000${NC}"
        echo -e "  Docs: ${GREEN}http://localhost:8000/docs${NC}\n"
        exec uvicorn reflection.gateway.app:app --reload --host 0.0.0.0 --port 8000
        ;;
    test)
        echo -e "\n  ${CYAN}Running tests...${NC}\n"
        activate_venv
        exec pytest tests/ -v --tb=short
        ;;
    docker)
        if ! command -v docker &>/dev/null; then
            echo -e "  ${RED}✗ Docker not found${NC}"
            exit 1
        fi
        echo -e "\n  ${CYAN}Starting with Docker Compose...${NC}\n"
        if [ ! -f "$PROJECT_DIR/.env" ] && [ -f "$PROJECT_DIR/.env.example" ]; then
            cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
            echo -e "  ${YELLOW}Created .env — add API keys before use${NC}"
        fi
        docker compose -f "$PROJECT_DIR/docker-compose.yml" up -d
        echo -e "\n  ${GREEN}✓ Services started${NC}"
        echo -e "  API:        ${GREEN}http://localhost:8000${NC}"
        echo -e "  PostgreSQL: ${GREEN}localhost:5432${NC}"
        echo -e "  Redis:      ${GREEN}localhost:6379${NC}"
        echo -e "\n  Logs: ${CYAN}docker compose logs -f${NC}"
        exit 0
        ;;
    shell)
        echo -e "\n  ${CYAN}Opening Python shell...${NC}\n"
        activate_venv
        exec $PYTHON -i -c "
import reflection as me
import reflection_core
from reflection.core.settings import get_settings
print('Reflection shell ready — type help(me) for modules')
"
        ;;
    help|--help|-h)
        echo ""
        echo "Usage: ./install.sh [command]"
        echo ""
        echo "Commands:"
        echo "  (none)    Guided installer (GUI or CLI)"
        echo "  setup     Install deps and create .env"
        echo "  dev       Start dev server with hot reload"
        echo "  test      Run test suite"
        echo "  docker    Start with Docker Compose"
        echo "  shell     Interactive Python shell"
        echo "  --cli     Force CLI installer (no GUI)"
        echo ""
        echo "Shortcuts:"
        echo "  --nonprofit   Install with nonprofit preset (donor CRM, grants, bookkeeping)"
        echo "  --healthcare  Install with HIPAA-compliant healthcare preset"
        echo "  --enterprise  Install with full enterprise productivity suite"
        echo ""
        exit 0
        ;;
esac

# ── Detect display and choose installer ───────────────────────

# Shortcut flags: --nonprofit, --healthcare, --enterprise
# These bypass the GUI and go straight to CLI with use-case pre-selected
for shortcut in nonprofit healthcare enterprise; do
    if [[ "$*" == *"--$shortcut"* ]]; then
        echo -e "\n  ${CYAN}Installing with ${shortcut} preset...${NC}\n"
        activate_venv
        if ! pip show reflection &>/dev/null 2>&1; then
            echo -e "  ${YELLOW}Installing dependencies...${NC}"
            pip install --upgrade pip -q
            pip install -e . -q
        fi
        $PYTHON "$PROJECT_DIR/installer/cli_installer.py" --use-case "$shortcut"
        exit $?
    fi
done

HAS_DISPLAY=false
if [ -n "$DISPLAY" ] || [ -n "$WAYLAND_DISPLAY" ]; then
    HAS_DISPLAY=true
fi
# macOS always has a display (Aqua)
if [ "$(uname)" = "Darwin" ]; then
    HAS_DISPLAY=true
fi

# Check for --no-gui / --cli flag
if [[ "$*" == *"--no-gui"* ]] || [[ "$*" == *"--cli"* ]]; then
    HAS_DISPLAY=false
fi

if [ "$HAS_DISPLAY" = true ]; then
    # Try GUI installer
    echo -e "\n  ${CYAN}Launching GUI installer...${NC}"
    echo -e "  ${BOLD}(Use ./install.sh --cli for terminal mode)${NC}"
    echo ""

    # Check for tkinter
    if $PYTHON -c "import tkinter" 2>/dev/null; then
        $PYTHON "$PROJECT_DIR/installer/setup_wizard.py"
    else
        echo -e "  ${YELLOW}⚠ tkinter not available, falling back to CLI${NC}"
        echo -e "  Install with: ${CYAN}sudo apt install python3-tk${NC}"
        echo ""
        $PYTHON "$PROJECT_DIR/installer/cli_installer.py"
    fi
else
    echo -e "\n  ${CYAN}Launching CLI installer...${NC}"
    echo -e "  ${BOLD}(No display detected or --cli flag used)${NC}"
    echo ""

    # Strip --cli and --no-gui from args before passing to Python installer
    PASS_ARGS=()
    for arg in "$@"; do
        if [ "$arg" != "--cli" ] && [ "$arg" != "--no-gui" ]; then
            PASS_ARGS+=("$arg")
        fi
    done
    $PYTHON "$PROJECT_DIR/installer/cli_installer.py" "${PASS_ARGS[@]}"
fi
