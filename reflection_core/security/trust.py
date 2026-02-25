# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Trust and Capability Model

Re-exports the canonical security enums from Familiar so that both
Familiar and Reflection share a single source of truth.

Falls back to a local definition when Familiar is not installed
(e.g. during standalone reflection_core tests).
"""

try:
    # Canonical source — Familiar v1.4.0+
    from familiar.core.security import (
        TRUST_CAPABILITIES as _FAMILIAR_TRUST_CAPS,
    )
    from familiar.core.security import (  # type: ignore[import-untyped]
        Capability,
        TrustLevel,
    )

    # Expose Familiar's mapping under the name this package has always used
    DEFAULT_CAPABILITIES = _FAMILIAR_TRUST_CAPS

except ImportError:
    # Fallback: standalone definitions (used in dev/testing without Familiar)
    from enum import StrEnum

    class TrustLevel(StrEnum):  # type: ignore[no-redef]
        STRANGER = "stranger"
        KNOWN = "known"
        TRUSTED = "trusted"
        OWNER = "owner"

        def __ge__(self, other: "TrustLevel") -> bool:
            order = [TrustLevel.STRANGER, TrustLevel.KNOWN, TrustLevel.TRUSTED, TrustLevel.OWNER]
            return order.index(self) >= order.index(other)

        def __gt__(self, other: "TrustLevel") -> bool:
            order = [TrustLevel.STRANGER, TrustLevel.KNOWN, TrustLevel.TRUSTED, TrustLevel.OWNER]
            return order.index(self) > order.index(other)

        def __le__(self, other: "TrustLevel") -> bool:
            return not self > other

        def __lt__(self, other: "TrustLevel") -> bool:
            return not self >= other

    class Capability(StrEnum):  # type: ignore[no-redef]
        READ_TIME = "read:time"
        READ_WEATHER = "read:weather"
        READ_CALENDAR = "read:calendar"
        READ_EMAIL = "read:email"
        READ_FILES = "read:files"
        READ_DATABASE = "read:database"
        READ_MEMORY = "read:memory"
        READ_WEB = "read:web"
        WRITE_CALENDAR = "write:calendar"
        WRITE_EMAIL = "write:email"
        WRITE_FILES = "write:files"
        WRITE_DATABASE = "write:database"
        WRITE_MEMORY = "write:memory"
        EXECUTE_SHELL = "execute:shell"
        EXECUTE_CODE = "execute:code"
        EXECUTE_HTTP = "execute:http"
        EXECUTE_BROWSER = "execute:browser"
        ADMIN_USERS = "admin:users"
        ADMIN_CONFIG = "admin:config"
        ADMIN_SKILLS = "admin:skills"
        ADMIN_AUDIT = "admin:audit"

    DEFAULT_CAPABILITIES: dict[TrustLevel, frozenset[Capability]] = {
        TrustLevel.STRANGER: frozenset({Capability.READ_TIME, Capability.READ_WEATHER}),
        TrustLevel.KNOWN: frozenset(
            {
                Capability.READ_TIME,
                Capability.READ_WEATHER,
                Capability.READ_WEB,
                Capability.READ_MEMORY,
                Capability.WRITE_MEMORY,
            }
        ),
        TrustLevel.TRUSTED: frozenset(
            {
                Capability.READ_TIME,
                Capability.READ_WEATHER,
                Capability.READ_CALENDAR,
                Capability.READ_EMAIL,
                Capability.READ_FILES,
                Capability.READ_MEMORY,
                Capability.READ_WEB,
                Capability.WRITE_CALENDAR,
                Capability.WRITE_EMAIL,
                Capability.WRITE_FILES,
                Capability.WRITE_MEMORY,
                Capability.EXECUTE_HTTP,
            }
        ),
        TrustLevel.OWNER: frozenset(set(Capability)),
    }


def get_capabilities_for_trust(trust_level: "TrustLevel") -> "frozenset[Capability]":
    """Get default capabilities for a trust level."""
    return DEFAULT_CAPABILITIES.get(trust_level, frozenset())


def check_capability(
    required: "Capability",
    user_capabilities: "set[Capability]",
    user_trust: "TrustLevel",
) -> bool:
    """Check if a user has a required capability."""
    if required in user_capabilities:
        return True
    default_caps = get_capabilities_for_trust(user_trust)
    return required in default_caps


__all__ = [
    "TrustLevel",
    "Capability",
    "DEFAULT_CAPABILITIES",
    "get_capabilities_for_trust",
    "check_capability",
]
