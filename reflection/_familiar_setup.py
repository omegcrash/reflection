# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Familiar Integration

Verifies that the Familiar package (>= 1.4.0) is installed and
importable.  Familiar is now an external dependency declared in
pyproject.toml — no sys.path manipulation required.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

_MIN_FAMILIAR_VERSION = "1.4.0"

try:
    import familiar

    FAMILIAR_AVAILABLE = True
    FAMILIAR_VERSION = getattr(familiar, "__version__", None)

    # Fall back to installed package metadata when the module attribute
    # is missing (e.g. when Familiar is installed via pip).
    if FAMILIAR_VERSION is None:
        try:
            FAMILIAR_VERSION = _pkg_version("familiar")
        except PackageNotFoundError:
            FAMILIAR_VERSION = "unknown"

except ImportError:
    FAMILIAR_AVAILABLE = False
    FAMILIAR_VERSION = None

__all__ = ["FAMILIAR_AVAILABLE", "FAMILIAR_VERSION", "_MIN_FAMILIAR_VERSION"]
