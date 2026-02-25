#!/usr/bin/env python3
# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Reflection CLI Installer
=============================

Interactive terminal installer for environments without a display.
Same steps as the GUI wizard, driven by prompts.

Usage:
    python3 installer/cli_installer.py
    python3 installer/cli_installer.py --use-case healthcare --provider ollama --auto
"""

import os
import sys
import argparse
import secrets
import shutil
import subprocess
import platform
from pathlib import Path

# ============================================================================
# Colors (ANSI)
# ============================================================================

class C:
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    @staticmethod
    def ok(msg):    print(f"  {C.GREEN}✓{C.RESET} {msg}")
    @staticmethod
    def warn(msg):  print(f"  {C.YELLOW}⚠{C.RESET} {msg}")
    @staticmethod
    def fail(msg):  print(f"  {C.RED}✗{C.RESET} {msg}")
    @staticmethod
    def step(msg):  print(f"\n{C.CYAN}→{C.RESET} {C.BOLD}{msg}{C.RESET}")
    @staticmethod
    def info(msg):  print(f"  {msg}")


def banner():
    print(f"""
{C.CYAN}╔══════════════════════════════════════════════════╗
║       🐍  Reflection Setup  🐍               ║
║    Enterprise Multi-Tenant AI Companion Platform      ║
╚══════════════════════════════════════════════════╝{C.RESET}
""")


def prompt_choice(question, options, default=None):
    """Prompt user to pick from numbered options."""
    print(f"\n{C.BOLD}{question}{C.RESET}")
    for i, (key, label) in enumerate(options, 1):
        marker = " (default)" if key == default else ""
        print(f"  {C.CYAN}{i}{C.RESET}) {label}{C.DIM}{marker}{C.RESET}")

    while True:
        raw = input(f"\nChoose [1-{len(options)}]: ").strip()
        if not raw and default:
            return default
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx][0]
        except ValueError:
            pass
        print(f"  {C.RED}Invalid choice{C.RESET}")


def prompt_string(question, default="", secret=False):
    """Prompt for a string value."""
    suffix = f" [{default}]" if default and not secret else ""
    prompt_text = f"{question}{suffix}: "
    if secret:
        import getpass
        val = getpass.getpass(prompt_text)
    else:
        val = input(prompt_text)
    return val.strip() or default


# ============================================================================
# Install Steps
# ============================================================================

def check_system():
    C.step("Checking system requirements")
    if sys.version_info < (3, 11):
        C.fail(f"Python 3.11+ required, found {sys.version}")
        sys.exit(1)
    C.ok(f"Python {sys.version_info.major}.{sys.version_info.minor}")
    C.ok(f"Platform: {platform.system()} {platform.machine()}")


def check_docker(interactive=True):
    C.step("Checking Docker")
    if not shutil.which("docker"):
        C.fail("Docker not found")
        print()
        print(f"  Docker is needed for PostgreSQL and Redis in production.")
        print(f"  Install it from: {C.CYAN}https://docker.com/products/docker-desktop{C.RESET}")
        print()
        print(f"  On Ubuntu/Debian, you can install it right now with:")
        print(f"    {C.CYAN}curl -fsSL https://get.docker.com | sh{C.RESET}")
        print(f"    {C.CYAN}sudo usermod -aG docker $USER{C.RESET}")
        print(f"    Then {C.BOLD}log out and back in{C.RESET}, and re-run this installer.")
        print()

        if interactive:
            choice = input(f"  Continue without Docker? (uses SQLite, fine for testing) [y/N]: ").strip().lower()
            if choice == "y":
                C.warn("Continuing without Docker — using SQLite (not for production)")
                return False
        
        print("  Re-run the installer after installing Docker.")
        sys.exit(1)

    r = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if r.returncode != 0:
        C.fail("Docker is installed but not running")
        print(f"  Start Docker Desktop, or run: {C.CYAN}sudo systemctl start docker{C.RESET}")
        
        if interactive:
            choice = input(f"\n  Continue without Docker? (uses SQLite, fine for testing) [y/N]: ").strip().lower()
            if choice == "y":
                C.warn("Continuing without Docker — using SQLite (not for production)")
                return False
        
        sys.exit(1)
    
    C.ok("Docker is running")
    return True


def start_services(project_dir):
    C.step("Starting PostgreSQL and Redis")
    compose = project_dir / "docker-compose.yml"
    if not compose.exists():
        C.warn("docker-compose.yml not found, skipping")
        return

    # Try docker compose (v2) then docker-compose (v1)
    for cmd in [["docker", "compose"], ["docker-compose"]]:
        r = subprocess.run(
            cmd + ["-f", str(compose), "up", "-d", "postgres", "redis"],
            capture_output=True, text=True, cwd=str(project_dir)
        )
        if r.returncode == 0:
            C.ok("PostgreSQL on :5432")
            C.ok("Redis on :6379")
            return

    C.fail(f"Failed to start services")
    print(f"  {r.stderr[:200]}")
    sys.exit(1)


def install_ollama():
    C.step("Installing Ollama")
    if shutil.which("ollama"):
        C.ok("Already installed")
        return

    if platform.system() == "Windows":
        C.warn("Download Ollama from: https://ollama.ai/download")
        sys.exit(1)

    r = subprocess.run(["bash", "-c", "curl -fsSL https://ollama.ai/install.sh | sh"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        C.fail(f"Install failed: {r.stderr[:200]}")
        sys.exit(1)
    C.ok("Ollama installed")


def pull_model(model):
    C.step(f"Pulling model: {model}")
    C.info("This may take a few minutes on first run...")
    r = subprocess.run(["ollama", "pull", model], timeout=600)
    if r.returncode != 0:
        C.fail(f"Failed to pull {model}")
        sys.exit(1)
    C.ok(f"{model} ready")


def generate_config(project_dir, config):
    C.step("Generating configuration")

    jwt_secret = secrets.token_urlsafe(64)

    try:
        from cryptography.fernet import Fernet
        enc_key = Fernet.generate_key().decode()
    except ImportError:
        enc_key = secrets.token_urlsafe(32)
        C.warn("cryptography not installed yet — encryption key is placeholder")

    use_case = config["use_case"]
    provider = config["provider"]

    has_docker = config.get("has_docker", True)

    lines = [
        "# Reflection Configuration",
        f"# Use case: {use_case}",
        "",
        f"ENVIRONMENT={'production' if has_docker else 'development'}",
    ]

    if has_docker:
        lines.append("DATABASE_URL=postgresql+asyncpg://familiar:familiar@localhost:5432/reflection")
        lines.append("REDIS_URL=redis://localhost:6379/0")
    else:
        lines.append("# SQLite (development mode — install Docker for production)")
        lines.append(f"DATABASE_URL=sqlite+aiosqlite:///{project_dir / 'data' / 'familiar.db'}")
        lines.append("# REDIS_URL not set — using in-memory fallback")

    lines.extend([
        "",
        f"LLM_DEFAULT_PROVIDER={provider}",
    ])

    if config.get("anthropic_key"):
        lines.append(f"LLM_ANTHROPIC_API_KEY={config['anthropic_key']}")
    if config.get("openai_key"):
        lines.append(f"LLM_OPENAI_API_KEY={config['openai_key']}")
    if provider == "ollama" or use_case in ("healthcare", "privacy"):
        lines.append("LLM_OLLAMA_URL=http://localhost:11434")
        lines.append(f"LLM_OLLAMA_MODEL={config.get('ollama_model', 'llama3.2')}")

    lines.extend([
        "",
        f"SECURITY_JWT_SECRET_KEY={jwt_secret}",
        f"SECURITY_MASTER_ENCRYPTION_KEY={enc_key}",
    ])

    if use_case == "healthcare":
        lines.extend([
            "",
            "# HIPAA Compliance",
            "HIPAA_COMPLIANT=true",
            f"PHI_PROVIDER_NAME=ollama",
            f"PHI_MODEL={config.get('ollama_model', 'llama3.2')}",
        ])
        if config.get("anthropic_key"):
            lines.append("GENERAL_PROVIDER_NAME=anthropic")
        elif config.get("openai_key"):
            lines.append("GENERAL_PROVIDER_NAME=openai")

    # Skill preset (maps use_case to TenantConfig.skill_preset)
    preset_map = {
        "nonprofit": "nonprofit",
        "healthcare": "healthcare",
        "enterprise": "enterprise",
        "general": "general",
        "privacy": "general",
    }
    skill_preset = preset_map.get(use_case, "general")
    if skill_preset != "general":
        lines.extend([
            "",
            f"# Skill Preset",
            f"SKILL_PRESET={skill_preset}",
        ])

    env_path = project_dir / ".env"
    env_path.write_text("\n".join(lines) + "\n")

    if platform.system() != "Windows":
        os.chmod(env_path, 0o600)

    C.ok(f"Written to {env_path}")


def install_deps(project_dir):
    C.step("Installing Python dependencies")
    venv_dir = project_dir / ".venv"

    if not venv_dir.exists():
        import venv
        venv.create(venv_dir, with_pip=True)
        C.ok("Created virtual environment")

    if platform.system() == "Windows":
        pip = str(venv_dir / "Scripts" / "pip.exe")
    else:
        pip = str(venv_dir / "bin" / "pip")

    subprocess.run([pip, "install", "--upgrade", "pip", "wheel"],
                   capture_output=True, check=True)
    subprocess.run([pip, "install", "-e", str(project_dir)],
                   capture_output=True, check=True, cwd=str(project_dir))
    C.ok("Dependencies installed")


def create_dirs(project_dir):
    C.step("Creating data directories")
    data = project_dir / "data"
    for d in ["sessions", "memory", "audit", "backups", "logs"]:
        (data / d).mkdir(parents=True, exist_ok=True)
    C.ok(f"Data directory: {data}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Reflection CLI Installer")
    parser.add_argument("--use-case", choices=["nonprofit", "healthcare", "privacy", "enterprise", "general"])
    parser.add_argument("--provider", choices=["anthropic", "openai", "ollama"])
    parser.add_argument("--anthropic-key", default="")
    parser.add_argument("--openai-key", default="")
    parser.add_argument("--ollama-model", default="llama3.2")
    parser.add_argument("--auto", action="store_true", help="Non-interactive (requires --use-case and --provider)")
    parser.add_argument("--skip-docker", action="store_true", help="Skip Docker check and service startup")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama install and model pull")
    args = parser.parse_args()

    banner()
    check_system()

    project_dir = Path(__file__).resolve().parent.parent

    # ── Gather config ──────────────────────────────────────────────

    if args.auto:
        if not args.use_case or not args.provider:
            print(f"{C.RED}--auto requires --use-case and --provider{C.RESET}")
            sys.exit(1)
        config = {
            "use_case": args.use_case,
            "provider": args.provider,
            "anthropic_key": args.anthropic_key,
            "openai_key": args.openai_key,
            "ollama_model": args.ollama_model,
        }
    else:
        # Interactive prompts

        # Skip use-case prompt if pre-set via --use-case flag
        if args.use_case:
            use_case = args.use_case
            C.info(f"Use case: {C.BOLD}{use_case}{C.RESET}")
        else:
            use_case = prompt_choice("What's your use case?", [
                ("nonprofit",   "🏛️  Nonprofit / 501(c)(3) (donor CRM, grants, bookkeeping)"),
                ("healthcare",  "🏥  Healthcare / Medical (HIPAA compliant)"),
                ("privacy",     "🔒  Maximum Privacy (everything self-hosted)"),
                ("enterprise",  "🏢  Enterprise (full productivity suite)"),
                ("general",     "💼  General Business (API-powered)"),
            ], default="general")

        if use_case in ("healthcare", "privacy"):
            provider = "ollama"
            C.info(f"\n  {C.DIM}{'HIPAA' if use_case == 'healthcare' else 'Privacy'} mode → Ollama (self-hosted) selected automatically{C.RESET}")
        else:
            provider = prompt_choice("Choose your AI provider:", [
                ("anthropic", "Anthropic Claude (Recommended)"),
                ("openai",    "OpenAI GPT"),
                ("ollama",    "Ollama (self-hosted, free)"),
            ], default="anthropic")

        anthropic_key = ""
        openai_key = ""
        if provider == "anthropic" or (use_case == "healthcare" and provider != "ollama"):
            anthropic_key = prompt_string("Anthropic API key (get one at console.anthropic.com)", secret=True)
        if provider == "openai":
            openai_key = prompt_string("OpenAI API key (get one at platform.openai.com/api-keys)", secret=True)

        ollama_model = "llama3.2"
        if provider == "ollama" or use_case in ("healthcare", "privacy"):
            ollama_model = prompt_choice("Which Ollama model?", [
                ("qwen2.5:0.5b", "Qwen 0.5B — 400MB, Raspberry Pi friendly"),
                ("qwen2.5:3b",   "Qwen 3B — 2GB, good balance"),
                ("llama3.2",     "Llama 3.2 — 2GB, general purpose"),
                ("qwen2.5:7b",   "Qwen 7B — 5GB, best local quality"),
            ], default="llama3.2")

        # Healthcare hybrid: optionally add API provider for non-PHI
        if use_case == "healthcare" and not anthropic_key and not openai_key:
            add_api = input(f"\n  Add an API provider for non-PHI tasks? (faster/better quality) [y/N]: ").strip().lower()
            if add_api == "y":
                api_provider = prompt_choice("Which API provider for non-PHI?", [
                    ("anthropic", "Anthropic Claude"),
                    ("openai",    "OpenAI GPT"),
                ])
                if api_provider == "anthropic":
                    anthropic_key = prompt_string("Anthropic API key", secret=True)
                else:
                    openai_key = prompt_string("OpenAI API key", secret=True)

        config = {
            "use_case": use_case,
            "provider": provider,
            "anthropic_key": anthropic_key,
            "openai_key": openai_key,
            "ollama_model": ollama_model,
        }

    # ── Print summary ──────────────────────────────────────────────

    print(f"\n{C.BOLD}{'─' * 50}{C.RESET}")
    print(f"{C.BOLD}  Configuration Summary{C.RESET}")
    print(f"{'─' * 50}")
    print(f"  Use case:  {config['use_case']}")
    print(f"  Provider:  {config['provider']}")
    if config.get("ollama_model") and config["provider"] == "ollama":
        print(f"  Model:     {config['ollama_model']}")
    if config["use_case"] == "healthcare":
        print(f"  HIPAA:     {C.GREEN}enabled{C.RESET}")
    print(f"{'─' * 50}\n")

    if not args.auto:
        go = input("Proceed with installation? [Y/n]: ").strip().lower()
        if go == "n":
            print("Cancelled.")
            sys.exit(0)

    # ── Run install steps ──────────────────────────────────────────

    create_dirs(project_dir)

    has_docker = True
    if not args.skip_docker:
        has_docker = check_docker(interactive=not args.auto)
        if has_docker:
            start_services(project_dir)
    else:
        has_docker = False

    config["has_docker"] = has_docker

    needs_ollama = config["provider"] == "ollama" or config["use_case"] in ("healthcare", "privacy")
    if needs_ollama and not args.skip_ollama:
        install_ollama()
        pull_model(config["ollama_model"])

    generate_config(project_dir, config)
    install_deps(project_dir)

    # ── Done ───────────────────────────────────────────────────────

    if platform.system() == "Windows":
        activate = ".venv\\Scripts\\activate"
    else:
        activate = "source .venv/bin/activate"

    print(f"""
{C.GREEN}{'═' * 50}
  ✅  Reflection is installed and ready!
{'═' * 50}{C.RESET}

  Start the server:

    {C.CYAN}cd {project_dir}{C.RESET}
    {C.CYAN}{activate}{C.RESET}
    {C.CYAN}reflection serve{C.RESET}

  Then open: {C.CYAN}http://localhost:8000/docs{C.RESET}
""")

    if config["use_case"] == "healthcare":
        print(f"  {C.YELLOW}HIPAA mode is active.{C.RESET}")
        print(f"  PHI → Ollama ({config['ollama_model']})")
        if config.get("anthropic_key"):
            print(f"  Non-PHI → Anthropic Claude")
        elif config.get("openai_key"):
            print(f"  Non-PHI → OpenAI GPT")
        else:
            print(f"  All traffic → Ollama (self-hosted)")
        print()

    if config["use_case"] == "nonprofit":
        print(f"  {C.GREEN}Nonprofit mode is active.{C.RESET}")
        print(f"  Skills: donor CRM, grant tracker, bookkeeping,")
        print(f"          documents, reports, workflows, meetings")
        print()
        print(f"  Try:  {C.CYAN}\"Log a $500 gift from Jane Smith\"{C.RESET}")
        print(f"        {C.CYAN}\"Show upcoming grant deadlines\"{C.RESET}")
        print(f"        {C.CYAN}\"Prepare the board packet for next Tuesday\"{C.RESET}")
        print()

    if config["use_case"] == "enterprise":
        print(f"  {C.GREEN}Enterprise mode is active.{C.RESET}")
        print(f"  Full productivity suite: tasks, docs, reports, knowledge,")
        print(f"  workflows, audit, RBAC, search, notifications")
        print()

    if not config.get("has_docker", True):
        print(f"  {C.YELLOW}{'─' * 50}{C.RESET}")
        print(f"  {C.YELLOW}Running in development mode (SQLite, no Redis).{C.RESET}")
        print(f"  This is fine for testing and personal use.")
        print(f"  For production, install Docker and re-run the installer:")
        print(f"    {C.CYAN}curl -fsSL https://get.docker.com | sh{C.RESET}")
        print(f"    {C.CYAN}sudo usermod -aG docker $USER{C.RESET}")
        print(f"    Log out, log back in, then: {C.CYAN}./install.sh{C.RESET}")
        print()


if __name__ == "__main__":
    main()
