# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Intelligent LLM Routing

Automatically routes messages to appropriate LLM providers based on content:
- PHI/PII → Self-hosted (HIPAA compliant)
- General tasks → API (faster, better quality)
"""

from .phi_detector import PHIDetector, PIIDetector
from .smart_router import RoutingDecision, SmartLLMRouter, TaskType

__all__ = [
    "PHIDetector",
    "PIIDetector",
    "SmartLLMRouter",
    "RoutingDecision",
    "TaskType",
]
