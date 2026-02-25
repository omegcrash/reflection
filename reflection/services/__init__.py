# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Enterprise Services

Services layer connecting:
- Tenant wrappers (which wrap Familiar)
- Database persistence
- Billing/usage tracking
- Admin operations
"""

from .orchestrator import (
    FAMILIAR_AVAILABLE,
    AgentOrchestrator,
    OrchestratorEvent,
    OrchestratorResult,
    create_orchestrator,
)

__all__ = [
    "AgentOrchestrator",
    "OrchestratorEvent",
    "OrchestratorResult",
    "create_orchestrator",
    "FAMILIAR_AVAILABLE",
]
