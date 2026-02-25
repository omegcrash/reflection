"""
Reflection Integration Test Suite — Shared Fixtures
"""

import os
import sys
import shutil
import tempfile
import pytest
from pathlib import Path
from uuid import uuid4

# Ensure the project is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set a test home directory so we don't pollute the real filesystem
_TEST_HOME = None


@pytest.fixture(autouse=True)
def isolate_filesystem(tmp_path, monkeypatch):
    """
    Every test gets a clean, isolated filesystem.
    Redirects ~/.familiar to a temp directory.
    """
    global _TEST_HOME
    _TEST_HOME = tmp_path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Reset paths module to use the new home
    from familiar.core import paths
    paths.HOME_DIR = tmp_path
    paths.AGENT_DIR = tmp_path / ".familiar"
    paths.DATA_DIR = paths.AGENT_DIR / "data"
    paths.MEMORY_FILE = paths.DATA_DIR / "memory.json"
    paths.HISTORY_FILE = paths.DATA_DIR / "history.json"
    paths.TASKS_FILE = paths.DATA_DIR / "scheduled_tasks.json"
    paths.SKILLS_DIR = paths.AGENT_DIR / "skills"
    paths.LOGS_DIR = paths.AGENT_DIR / "logs"
    paths._tenant_id = None

    # Create base dirs
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths.SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    paths.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    yield tmp_path


@pytest.fixture
def tenant_id():
    """Generate a unique tenant ID."""
    return uuid4()


@pytest.fixture
def second_tenant_id():
    """Generate a second tenant ID for isolation tests."""
    return uuid4()
