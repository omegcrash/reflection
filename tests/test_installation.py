"""
Test Suite: Installation Scripts
================================

Verifies that install scripts are syntactically valid and contain
the expected features.
"""

import os
import subprocess
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestRunSh:

    @pytest.fixture
    def run_sh(self):
        return PROJECT_ROOT / "run.sh"

    def test_exists_and_executable(self, run_sh):
        assert run_sh.exists()
        assert os.access(run_sh, os.X_OK)

    def test_syntax_valid(self, run_sh):
        result = subprocess.run(["bash", "-n", str(run_sh)], capture_output=True)
        assert result.returncode == 0, f"Syntax error: {result.stderr.decode()}"

    def test_contains_provider_cascade(self, run_sh):
        content = run_sh.read_text()
        assert "ANTHROPIC_API_KEY" in content
        assert "OPENAI_API_KEY" in content
        assert "ollama" in content

    def test_contains_database_fallback(self, run_sh):
        content = run_sh.read_text()
        assert "sqlite" in content.lower()
        assert "postgresql" in content.lower()

    def test_contains_version(self, run_sh):
        content = run_sh.read_text()
        assert "2.0.0" in content

    def test_python_311_minimum(self, run_sh):
        content = run_sh.read_text()
        assert "3.11" in content or "python3.11" in content


class TestInstallSh:

    @pytest.fixture
    def install_sh(self):
        return PROJECT_ROOT / "install.sh"

    def test_exists_and_executable(self, install_sh):
        assert install_sh.exists()
        assert os.access(install_sh, os.X_OK)

    def test_syntax_valid(self, install_sh):
        result = subprocess.run(["bash", "-n", str(install_sh)], capture_output=True)
        assert result.returncode == 0, f"Syntax error: {result.stderr.decode()}"

    def test_nonprofit_shortcut(self, install_sh):
        content = install_sh.read_text()
        assert "--nonprofit" in content

    def test_healthcare_shortcut(self, install_sh):
        content = install_sh.read_text()
        assert "--healthcare" in content

    def test_enterprise_shortcut(self, install_sh):
        content = install_sh.read_text()
        assert "--enterprise" in content

    def test_subcommands(self, install_sh):
        content = install_sh.read_text()
        for cmd in ["setup", "dev", "test", "docker", "shell"]:
            assert cmd in content


class TestCLIInstaller:

    @pytest.fixture
    def cli_installer(self):
        return PROJECT_ROOT / "installer" / "cli_installer.py"

    def test_syntax_valid(self, cli_installer):
        result = subprocess.run(
            ["python3", "-m", "py_compile", str(cli_installer)],
            capture_output=True
        )
        assert result.returncode == 0

    def test_has_nonprofit_use_case(self, cli_installer):
        content = cli_installer.read_text()
        assert '"nonprofit"' in content

    def test_has_enterprise_use_case(self, cli_installer):
        content = cli_installer.read_text()
        assert '"enterprise"' in content

    def test_writes_skill_preset(self, cli_installer):
        content = cli_installer.read_text()
        assert "SKILL_PRESET" in content

    def test_help_output(self, cli_installer):
        result = subprocess.run(
            ["python3", str(cli_installer), "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "nonprofit" in result.stdout
        assert "healthcare" in result.stdout
        assert "enterprise" in result.stdout
