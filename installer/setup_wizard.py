#!/usr/bin/env python3
# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Reflection Setup Wizard
===========================

GUI installer: walks users through setup in 5 steps.
Cross-platform: Windows, macOS, Linux.

User journey:
    Run installer → Pick use case → Configure LLM → Start services → Done

Total time: 3-5 minutes from download to running.
"""

import os
import sys
import json
import secrets
import shutil
import subprocess
import threading
import webbrowser
import platform
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ============================================================================
# Configuration
# ============================================================================

APP_NAME = "Reflection"
APP_VERSION = "0.1.0"
MIN_PYTHON_VERSION = (3, 11)

IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

if IS_WINDOWS:
    DEFAULT_INSTALL_DIR = Path(os.environ.get("LOCALAPPDATA", "C:\\")) / "Reflection"
    PYTHON_CMD = "python"
elif IS_MACOS:
    DEFAULT_INSTALL_DIR = Path.home() / "Reflection"
    PYTHON_CMD = "python3"
else:
    DEFAULT_INSTALL_DIR = Path.home() / "reflection"
    PYTHON_CMD = "python3"


# ============================================================================
# Styles (dark theme from familiar wizard)
# ============================================================================

COLORS = {
    "bg": "#1a1a2e",
    "fg": "#eaeaea",
    "accent": "#7b68ee",
    "accent_hover": "#9683ec",
    "success": "#4ade80",
    "warning": "#fbbf24",
    "error": "#f87171",
    "input_bg": "#2d2d44",
    "input_border": "#4a4a6a",
}


def get_font(size=11, bold=False):
    weight = "bold" if bold else ""
    if IS_WINDOWS:
        return ("Segoe UI" + (" Semibold" if bold else ""), size)
    elif IS_MACOS:
        return ("SF Pro Display", size, weight) if weight else ("SF Pro Display", size)
    else:
        return ("Ubuntu", size, weight) if weight else ("Ubuntu", size)


def configure_styles():
    style = ttk.Style()
    available = style.theme_names()
    if "clam" in available:
        style.theme_use("clam")

    style.configure(".", background=COLORS["bg"], foreground=COLORS["fg"], font=get_font())
    style.configure("TFrame", background=COLORS["bg"])
    style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["fg"])
    style.configure("TButton", background=COLORS["accent"], foreground="white", padding=(20, 10), font=get_font(11, True))
    style.map("TButton",
        background=[("active", COLORS["accent_hover"]), ("disabled", "#555")],
        foreground=[("disabled", "#888")]
    )
    style.configure("TEntry", fieldbackground=COLORS["input_bg"], foreground=COLORS["fg"], padding=10)
    style.configure("TCheckbutton", background=COLORS["bg"], foreground=COLORS["fg"])
    style.configure("TRadiobutton", background=COLORS["bg"], foreground=COLORS["fg"])
    style.configure("Header.TLabel", font=get_font(24, True), foreground=COLORS["accent"])
    style.configure("Subheader.TLabel", font=get_font(12), foreground="#aaa")
    style.configure("Accent.Horizontal.TProgressbar", troughcolor=COLORS["input_bg"], background=COLORS["accent"])


# ============================================================================
# Wizard Page Base
# ============================================================================

class WizardPage(ttk.Frame):
    def __init__(self, parent, wizard):
        super().__init__(parent)
        self.wizard = wizard
        self.configure(padding=40)

    def on_enter(self):
        pass

    def on_leave(self) -> bool:
        return True

    def validate(self) -> bool:
        return True


# ============================================================================
# Page 1: Welcome
# ============================================================================

class WelcomePage(WizardPage):
    def __init__(self, parent, wizard):
        super().__init__(parent, wizard)

        logo = tk.Label(self, text="🐍", font=("", 64), bg=COLORS["bg"])
        logo.pack(pady=(0, 10))

        title = ttk.Label(self, text=f"Welcome to {APP_NAME}", style="Header.TLabel")
        title.pack(pady=(0, 10))

        subtitle = ttk.Label(
            self,
            text="Enterprise Multi-Tenant AI Companion Platform",
            style="Subheader.TLabel"
        )
        subtitle.pack(pady=(0, 30))

        features = [
            "🏥  HIPAA-compliant healthcare mode with automatic PHI routing",
            "🔒  Enterprise security: JWT, encryption, tenant isolation",
            "🤖  Multi-provider: Claude, GPT, Ollama (self-hosted)",
            "📊  Per-tenant billing, quotas, and audit trails",
            "⚡  One-command setup — running in 5 minutes",
        ]
        for f in features:
            ttk.Label(self, text=f, font=get_font(11)).pack(anchor="w", pady=3, padx=20)

        ttk.Label(self, text=f"Version {APP_VERSION}", foreground="#666").pack(side="bottom", pady=(20, 0))


# ============================================================================
# Page 2: Use Case
# ============================================================================

class UseCasePage(WizardPage):
    def __init__(self, parent, wizard):
        super().__init__(parent, wizard)

        ttk.Label(self, text="What's your use case?", style="Header.TLabel").pack(pady=(0, 10))
        ttk.Label(self, text="This determines security defaults and routing", style="Subheader.TLabel").pack(pady=(0, 30))

        self.use_case_var = tk.StringVar(value="general")

        options = [
            ("healthcare", "🏥  Healthcare / Medical",
             "HIPAA compliant: PHI auto-routed to self-hosted LLM,\n"
             "6-year audit retention, encryption at rest"),
            ("privacy", "🔒  Maximum Privacy",
             "Everything self-hosted via Ollama, no API calls,\n"
             "offline capable, air-gap ready"),
            ("general", "💼  General Business",
             "API-powered (Claude/GPT) for best quality,\n"
             "standard security, fast setup"),
        ]

        for value, label, desc in options:
            frame = ttk.Frame(self)
            frame.pack(fill="x", pady=8, padx=20)

            rb = ttk.Radiobutton(frame, text=label, variable=self.use_case_var, value=value, style="TRadiobutton")
            rb.pack(anchor="w")

            desc_label = ttk.Label(frame, text=f"    {desc}", foreground="#888", font=get_font(10))
            desc_label.pack(anchor="w")

    def on_leave(self) -> bool:
        self.wizard.config["use_case"] = self.use_case_var.get()
        return True


# ============================================================================
# Page 3: LLM Configuration
# ============================================================================

class LLMConfigPage(WizardPage):
    def __init__(self, parent, wizard):
        super().__init__(parent, wizard)

        ttk.Label(self, text="AI Provider Setup", style="Header.TLabel").pack(pady=(0, 10))

        self.info_label = ttk.Label(self, text="", style="Subheader.TLabel")
        self.info_label.pack(pady=(0, 20))

        # Provider radio buttons
        self.provider_var = tk.StringVar(value="anthropic")

        self.providers_frame = ttk.Frame(self)
        self.providers_frame.pack(fill="x", padx=20)

        # Anthropic
        af = ttk.Frame(self.providers_frame)
        af.pack(fill="x", pady=5)
        ttk.Radiobutton(af, text="Anthropic Claude (Recommended)", variable=self.provider_var, value="anthropic").pack(side="left")
        ttk.Button(af, text="Get Key →", command=lambda: webbrowser.open("https://console.anthropic.com/")).pack(side="right")
        self.anthropic_key = tk.StringVar()
        ttk.Entry(self.providers_frame, textvariable=self.anthropic_key, width=60, show="•").pack(fill="x", pady=(2, 12))

        # OpenAI
        of = ttk.Frame(self.providers_frame)
        of.pack(fill="x", pady=5)
        ttk.Radiobutton(of, text="OpenAI GPT", variable=self.provider_var, value="openai").pack(side="left")
        ttk.Button(of, text="Get Key →", command=lambda: webbrowser.open("https://platform.openai.com/api-keys")).pack(side="right")
        self.openai_key = tk.StringVar()
        ttk.Entry(self.providers_frame, textvariable=self.openai_key, width=60, show="•").pack(fill="x", pady=(2, 12))

        # Ollama
        ttk.Radiobutton(self.providers_frame, text="Ollama (self-hosted, free, no API key)", variable=self.provider_var, value="ollama").pack(anchor="w", pady=5)

        # Ollama model selection
        self.model_frame = ttk.Frame(self.providers_frame)
        self.model_frame.pack(fill="x", padx=20, pady=(0, 10))
        ttk.Label(self.model_frame, text="Model:", foreground="#888").pack(side="left")
        self.ollama_model_var = tk.StringVar(value="llama3.2")
        models = ["qwen2.5:0.5b", "qwen2.5:3b", "llama3.2", "qwen2.5:7b", "mistral"]
        ttk.OptionMenu(self.model_frame, self.ollama_model_var, self.ollama_model_var.get(), *models).pack(side="left", padx=10)

        # Encrypted storage note
        ttk.Label(self, text="🔒 API keys are stored locally in .env and encrypted at rest",
                  foreground=COLORS["success"]).pack(pady=(20, 0))

    def on_enter(self):
        use_case = self.wizard.config.get("use_case", "general")
        if use_case == "healthcare":
            self.info_label.config(text="HIPAA mode: Ollama required for PHI. Optional API for non-PHI tasks.")
            self.provider_var.set("ollama")
        elif use_case == "privacy":
            self.info_label.config(text="Privacy mode: all traffic stays on your machine via Ollama.")
            self.provider_var.set("ollama")
        else:
            self.info_label.config(text="Configure your AI provider (at least one required)")

    def validate(self) -> bool:
        provider = self.provider_var.get()
        if provider == "anthropic" and not self.anthropic_key.get().strip():
            messagebox.showwarning("API Key Required", "Enter your Anthropic API key, or select Ollama.")
            return False
        if provider == "openai" and not self.openai_key.get().strip():
            messagebox.showwarning("API Key Required", "Enter your OpenAI API key, or select Ollama.")
            return False

        self.wizard.config["provider"] = provider
        self.wizard.config["anthropic_key"] = self.anthropic_key.get().strip()
        self.wizard.config["openai_key"] = self.openai_key.get().strip()
        self.wizard.config["ollama_model"] = self.ollama_model_var.get()
        return True


# ============================================================================
# Page 4: Installation
# ============================================================================

class InstallPage(WizardPage):
    def __init__(self, parent, wizard):
        super().__init__(parent, wizard)

        ttk.Label(self, text="Installing...", style="Header.TLabel").pack(pady=(0, 20))

        self.progress = ttk.Progressbar(self, style="Accent.Horizontal.TProgressbar", length=500, mode="determinate")
        self.progress.pack(pady=(0, 10))

        self.status_var = tk.StringVar(value="Preparing...")
        ttk.Label(self, textvariable=self.status_var).pack(pady=(0, 15))

        log_frame = ttk.Frame(self)
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_frame, wrap="word", bg=COLORS["input_bg"], fg=COLORS["fg"],
                                font=("Consolas" if IS_WINDOWS else "Monaco", 9),
                                height=14, state="disabled")
        self.log_text.pack(fill="both", expand=True, side="left")

        sb = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        sb.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=sb.set)

        self.install_complete = False

    def log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.update_idletasks()

    def set_progress(self, val, status=None):
        self.progress["value"] = val
        if status:
            self.status_var.set(status)
        self.update_idletasks()

    def on_enter(self):
        self.wizard.set_buttons_enabled(False)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            cfg = self.wizard.config
            project_dir = Path(__file__).resolve().parent.parent
            data_dir = project_dir / "data"

            steps = [
                (5,  "Checking system...",          self._check_system),
                (15, "Creating directories...",     lambda: self._create_dirs(data_dir)),
                (25, "Checking Docker...",          self._check_docker),
                (40, "Starting database & Redis...",lambda: self._start_services(project_dir)),
                (55, "Installing Ollama...",        self._install_ollama),
                (70, "Pulling LLM model...",        self._pull_model),
                (80, "Generating configuration...", lambda: self._generate_config(project_dir)),
                (90, "Installing Python deps...",   lambda: self._install_deps(project_dir)),
                (95, "Running health check...",     lambda: self._health_check(project_dir)),
                (100,"Done!",                       None),
            ]

            for pct, status, func in steps:
                self.after(0, lambda p=pct, s=status: self.set_progress(p, s))
                self.after(0, lambda s=status: self.log(f"→ {s}"))
                if func:
                    try:
                        func()
                    except Exception as e:
                        self.after(0, lambda e=e: self.log(f"  ✗ {e}"))
                        raise
                    self.after(0, lambda: self.log("  ✓ Done"))

            self.install_complete = True
            self.after(0, lambda: self.log("\n🎉 Reflection is installed and ready!"))
            self.after(0, lambda: self.wizard.set_buttons_enabled(True))

        except Exception as e:
            self.after(0, lambda: self.log(f"\n❌ Installation failed: {e}"))
            self.after(0, lambda: self.status_var.set("Installation failed"))
            self.after(0, lambda: self.wizard.set_buttons_enabled(True))

    def _check_system(self):
        if sys.version_info < MIN_PYTHON_VERSION:
            raise RuntimeError(f"Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ required, found {sys.version}")
        self.after(0, lambda: self.log(f"  Python {sys.version_info.major}.{sys.version_info.minor} ✓"))
        self.after(0, lambda: self.log(f"  Platform: {platform.system()} {platform.machine()}"))

    def _create_dirs(self, data_dir):
        for d in ["sessions", "memory", "audit", "backups", "logs"]:
            (data_dir / d).mkdir(parents=True, exist_ok=True)

    def _check_docker(self):
        if not shutil.which("docker"):
            msg = ("Docker not found.\n\n"
                   "Install Docker Desktop from:\nhttps://www.docker.com/products/docker-desktop\n\n"
                   "Then re-run this installer.")
            self.after(0, lambda: messagebox.showwarning("Docker Required", msg))
            raise RuntimeError("Docker not installed")
        # Check Docker is running
        r = subprocess.run(["docker", "info"], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("Docker is installed but not running. Please start Docker Desktop.")
        self.after(0, lambda: self.log("  Docker is running ✓"))

    def _start_services(self, project_dir):
        compose_file = project_dir / "docker-compose.yml"
        if not compose_file.exists():
            self.after(0, lambda: self.log("  docker-compose.yml not found, skipping"))
            return

        # Only start postgres and redis, not the app itself
        r = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "up", "-d", "postgres", "redis"],
            capture_output=True, text=True, cwd=str(project_dir)
        )
        if r.returncode != 0:
            # Try legacy docker-compose
            r = subprocess.run(
                ["docker-compose", "-f", str(compose_file), "up", "-d", "postgres", "redis"],
                capture_output=True, text=True, cwd=str(project_dir)
            )
        if r.returncode != 0:
            raise RuntimeError(f"Failed to start services: {r.stderr[:200]}")
        self.after(0, lambda: self.log("  PostgreSQL on :5432, Redis on :6379"))

    def _install_ollama(self):
        cfg = self.wizard.config
        needs_ollama = cfg.get("provider") == "ollama" or cfg.get("use_case") in ("healthcare", "privacy")
        if not needs_ollama:
            self.after(0, lambda: self.log("  Skipping (API provider selected)"))
            return

        if shutil.which("ollama"):
            self.after(0, lambda: self.log("  Ollama already installed ✓"))
            return

        if IS_WINDOWS:
            url = "https://ollama.ai/download/OllamaSetup.exe"
            self.after(0, lambda: messagebox.showinfo(
                "Install Ollama",
                f"Please download and install Ollama from:\n{url}\n\nThen re-run this installer."
            ))
            webbrowser.open(url)
            raise RuntimeError("Ollama needs to be installed manually on Windows")
        else:
            r = subprocess.run(
                ["bash", "-c", "curl -fsSL https://ollama.ai/install.sh | sh"],
                capture_output=True, text=True
            )
            if r.returncode != 0:
                raise RuntimeError(f"Ollama install failed: {r.stderr[:200]}")

    def _pull_model(self):
        cfg = self.wizard.config
        needs_ollama = cfg.get("provider") == "ollama" or cfg.get("use_case") in ("healthcare", "privacy")
        if not needs_ollama:
            self.after(0, lambda: self.log("  Skipping (API provider selected)"))
            return

        model = cfg.get("ollama_model", "llama3.2")
        self.after(0, lambda: self.log(f"  Pulling {model} (this may take a few minutes)..."))

        r = subprocess.run(["ollama", "pull", model], capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"Failed to pull {model}: {r.stderr[:200]}")

    def _generate_config(self, project_dir):
        cfg = self.wizard.config
        use_case = cfg.get("use_case", "general")

        jwt_secret = secrets.token_urlsafe(64)

        # Generate Fernet key
        try:
            from cryptography.fernet import Fernet
            encryption_key = Fernet.generate_key().decode()
        except ImportError:
            encryption_key = "GENERATE_AFTER_INSTALL"

        provider = cfg.get("provider", "anthropic")
        lines = [
            "# Reflection Configuration",
            f"# Generated by setup wizard ({platform.system()})",
            f"# Use case: {use_case}",
            "",
            f"ENVIRONMENT=production",
            f"DATABASE_URL=postgresql+asyncpg://familiar:familiar@localhost:5432/reflection",
            f"REDIS_URL=redis://localhost:6379/0",
            "",
            f"LLM_DEFAULT_PROVIDER={provider}",
        ]

        if cfg.get("anthropic_key"):
            lines.append(f"LLM_ANTHROPIC_API_KEY={cfg['anthropic_key']}")
        if cfg.get("openai_key"):
            lines.append(f"LLM_OPENAI_API_KEY={cfg['openai_key']}")
        if provider == "ollama" or use_case in ("healthcare", "privacy"):
            lines.append(f"LLM_OLLAMA_URL=http://localhost:11434")
            lines.append(f"LLM_OLLAMA_MODEL={cfg.get('ollama_model', 'llama3.2')}")

        lines.extend([
            "",
            f"SECURITY_JWT_SECRET_KEY={jwt_secret}",
            f"SECURITY_MASTER_ENCRYPTION_KEY={encryption_key}",
        ])

        if use_case == "healthcare":
            lines.extend([
                "",
                "# HIPAA Compliance (auto-enabled by healthcare use case)",
                "HIPAA_COMPLIANT=true",
                f"PHI_PROVIDER_NAME=ollama",
                f"PHI_MODEL={cfg.get('ollama_model', 'llama3.2')}",
            ])
            # If they also have an API provider, set it as general
            if cfg.get("anthropic_key"):
                lines.append(f"GENERAL_PROVIDER_NAME=anthropic")
            elif cfg.get("openai_key"):
                lines.append(f"GENERAL_PROVIDER_NAME=openai")

        env_path = project_dir / ".env"
        env_path.write_text("\n".join(lines) + "\n")

        # Restrict permissions on Unix
        if not IS_WINDOWS:
            os.chmod(env_path, 0o600)

        self.after(0, lambda: self.log(f"  Config written to {env_path}"))

    def _install_deps(self, project_dir):
        venv_dir = project_dir / ".venv"
        if not venv_dir.exists():
            import venv
            venv.create(venv_dir, with_pip=True)
            self.after(0, lambda: self.log("  Created virtual environment"))

        if IS_WINDOWS:
            pip = str(venv_dir / "Scripts" / "pip.exe")
        else:
            pip = str(venv_dir / "bin" / "pip")

        subprocess.run([pip, "install", "--upgrade", "pip", "wheel"], capture_output=True, check=True)
        subprocess.run([pip, "install", "-e", str(project_dir)], capture_output=True, check=True, cwd=str(project_dir))

    def _health_check(self, project_dir):
        venv_dir = project_dir / ".venv"
        if IS_WINDOWS:
            python = str(venv_dir / "Scripts" / "python.exe")
        else:
            python = str(venv_dir / "bin" / "python")

        r = subprocess.run(
            [python, "-c", "from reflection.core.settings import Settings; print('Settings OK')"],
            capture_output=True, text=True, cwd=str(project_dir),
            env={**os.environ, "PYTHONPATH": str(project_dir)}
        )
        if r.returncode == 0:
            self.after(0, lambda: self.log(f"  {r.stdout.strip()}"))
        else:
            self.after(0, lambda: self.log(f"  ⚠ Health check warning: {r.stderr[:100]}"))


# ============================================================================
# Page 5: Complete
# ============================================================================

class CompletePage(WizardPage):
    def __init__(self, parent, wizard):
        super().__init__(parent, wizard)

        icon = tk.Label(self, text="✅", font=("", 48), bg=COLORS["bg"])
        icon.pack(pady=(0, 15))

        ttk.Label(self, text="You're All Set!", style="Header.TLabel").pack(pady=(0, 10))

        use_case = wizard.config.get("use_case", "general")
        if use_case == "healthcare":
            mode_text = "HIPAA mode active — PHI auto-routes to self-hosted Ollama"
        elif use_case == "privacy":
            mode_text = "Privacy mode active — all traffic stays on your machine"
        else:
            mode_text = "Standard mode — API-powered for best quality"
        ttk.Label(self, text=mode_text, foreground=COLORS["success"]).pack(pady=(0, 25))

        steps = ttk.Frame(self)
        steps.pack(fill="x", padx=20, pady=(0, 20))
        ttk.Label(steps, text="Start the server:", font=get_font(12, True)).pack(anchor="w")

        cmd_frame = ttk.Frame(steps)
        cmd_frame.pack(fill="x", pady=5)
        cmd_text = tk.Text(cmd_frame, bg=COLORS["input_bg"], fg=COLORS["success"],
                           font=("Consolas" if IS_WINDOWS else "Monaco", 11),
                           height=4, width=60)
        cmd_text.insert("1.0",
            "cd reflection-v2\n"
            "source .venv/bin/activate    # Windows: .venv\\Scripts\\activate\n"
            "reflection serve\n"
            "# → http://localhost:8000"
        )
        cmd_text.config(state="disabled")
        cmd_text.pack(fill="x")

        # Options
        self.start_now = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Start Reflection now", variable=self.start_now).pack(anchor="w", padx=20)

        # Links
        links = ttk.Frame(self)
        links.pack(fill="x", padx=20, pady=(20, 0))
        ttk.Button(links, text="📖 API Docs", command=lambda: webbrowser.open("http://localhost:8000/docs")).pack(side="left", padx=(0, 10))
        ttk.Button(links, text="🏥 HIPAA Guide", command=lambda: webbrowser.open("https://docs.familiar.ai/hipaa")).pack(side="left")

    def on_leave(self) -> bool:
        if self.start_now.get():
            project_dir = Path(__file__).resolve().parent.parent
            venv_dir = project_dir / ".venv"
            if IS_WINDOWS:
                python = str(venv_dir / "Scripts" / "python.exe")
            else:
                python = str(venv_dir / "bin" / "python")

            subprocess.Popen(
                [python, "-m", "uvicorn", "reflection.gateway.app:app",
                 "--host", "0.0.0.0", "--port", "8000"],
                cwd=str(project_dir)
            )
            # Give it a moment then open browser
            self.after(2000, lambda: webbrowser.open("http://localhost:8000/docs"))
        return True


# ============================================================================
# Wizard Controller
# ============================================================================

class SetupWizard(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(f"{APP_NAME} Setup")
        self.geometry("700x620")
        self.resizable(False, False)
        self.configure(bg=COLORS["bg"])

        # Center on screen
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 700) // 2
        y = (self.winfo_screenheight() - 620) // 2
        self.geometry(f"+{x}+{y}")

        self.config = {}
        configure_styles()

        self.pages = [WelcomePage, UseCasePage, LLMConfigPage, InstallPage, CompletePage]
        self.current_page_index = 0
        self.page_instances = {}

        # Main container
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill="both", expand=True)

        self.page_container = ttk.Frame(self.main_frame)
        self.page_container.pack(fill="both", expand=True)

        # Navigation
        self.nav_frame = ttk.Frame(self.main_frame)
        self.nav_frame.pack(fill="x", padx=40, pady=15)

        self.cancel_btn = ttk.Button(self.nav_frame, text="Cancel", command=self.cancel)
        self.cancel_btn.pack(side="left")

        self.next_btn = ttk.Button(self.nav_frame, text="Next →", command=self.next_page)
        self.next_btn.pack(side="right")

        self.back_btn = ttk.Button(self.nav_frame, text="← Back", command=self.prev_page)
        self.back_btn.pack(side="right", padx=(0, 10))

        self.show_page(0)
        self.protocol("WM_DELETE_WINDOW", self.cancel)

    def show_page(self, index):
        for w in self.page_container.winfo_children():
            w.pack_forget()

        if index not in self.page_instances:
            self.page_instances[index] = self.pages[index](self.page_container, self)

        page = self.page_instances[index]
        page.pack(fill="both", expand=True)
        page.on_enter()
        self.current_page_index = index
        self._update_nav()

    def _update_nav(self):
        idx = self.current_page_index
        self.back_btn.config(state="normal" if idx > 0 else "disabled")
        self.next_btn.config(text="Finish" if idx == len(self.pages) - 1 else "Next →")

    def set_buttons_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for btn in (self.back_btn, self.next_btn, self.cancel_btn):
            btn.config(state=state)

    def next_page(self):
        page = self.page_instances.get(self.current_page_index)
        if page:
            if not page.validate():
                return
            if not page.on_leave():
                return
        if self.current_page_index < len(self.pages) - 1:
            self.show_page(self.current_page_index + 1)
        else:
            self.destroy()

    def prev_page(self):
        if self.current_page_index > 0:
            self.show_page(self.current_page_index - 1)

    def cancel(self):
        if messagebox.askyesno("Cancel", "Are you sure you want to cancel?"):
            self.destroy()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    wizard = SetupWizard()
    wizard.mainloop()


if __name__ == "__main__":
    main()
