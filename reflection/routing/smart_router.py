# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Smart LLM Router

Automatically routes messages to appropriate LLM providers based on content:
- PHI/PII/Sensitive → Self-hosted (HIPAA compliant)
- General tasks → API (faster, better quality)

Supports both automatic detection and manual tagging.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

# Import PHI/PII detectors
try:
    from .phi_detector import PHIDetector, PIIDetector
except ImportError:
    from phi_detector import PHIDetector, PIIDetector

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Classification of task types."""

    PHI = "phi"  # Contains Protected Health Information
    PII = "pii"  # Contains Personally Identifiable Information
    SENSITIVE = "sensitive"  # Confidential business data
    GENERAL = "general"  # Non-sensitive general tasks
    CODE = "code"  # Code generation/review
    ANALYSIS = "analysis"  # Data analysis (check for PII)


@dataclass
class RoutingDecision:
    """Result of routing decision."""

    provider: str  # "ollama", "openai", "anthropic", etc.
    model: str  # Model to use
    reason: str  # Why this routing was chosen
    task_type: TaskType  # Classified task type
    confidence: float  # Confidence in classification (0-1)
    phi_detected: bool  # Was PHI detected?
    pii_detected: bool  # Was PII detected?


class SmartLLMRouter:
    """
    Intelligent LLM routing based on content analysis.

    Routes to:
    - Self-hosted (Ollama) for PHI/PII/sensitive data
    - API services for general tasks (faster, better quality)

    For HIPAA tenants, this is configured automatically from the
    tenant's hipaa_compliant setting — no manual routing config needed.

    Supports TWO modes:
    1. Automatic PHI detection (default, easy)
    2. Manual PHI tagging (explicit, user-controlled)
    """

    # Self-hosted provider names that are HIPAA-safe
    SELF_HOSTED_PROVIDERS = frozenset({"ollama", "llama.cpp", "local"})

    def __init__(
        self,
        phi_provider: dict[str, Any],
        general_provider: dict[str, Any] | None = None,
        enable_phi_detection: bool = True,
        enable_pii_detection: bool = True,
        allow_manual_tagging: bool = True,
    ):
        """
        Initialize router.

        Args:
            phi_provider: Provider config for PHI (must be self-hosted)
            general_provider: Provider config for general tasks (can be API)
            enable_phi_detection: Enable automatic PHI detection
            enable_pii_detection: Enable automatic PII detection
            allow_manual_tagging: Allow users to manually tag PHI with metadata
        """
        self.phi_provider = phi_provider
        self.general_provider = general_provider or phi_provider

        self.phi_detector = PHIDetector() if enable_phi_detection else None
        self.pii_detector = PIIDetector() if enable_pii_detection else None
        self.allow_manual_tagging = allow_manual_tagging

        # Validate phi_provider is self-hosted
        if phi_provider["provider"] not in self.SELF_HOSTED_PROVIDERS:
            logger.warning(
                f"PHI provider '{phi_provider['provider']}' is not self-hosted. "
                f"This may violate HIPAA compliance!"
            )

    @classmethod
    def from_tenant_config(
        cls,
        tenant_config: dict[str, Any],
        platform_llm_settings: Any | None = None,
    ) -> Optional["SmartLLMRouter"]:
        """
        Create a router from tenant configuration settings.

        Returns None if the tenant is not HIPAA-compliant (no routing needed).

        This is the preferred construction path — it reads hipaa_compliant
        and the provider settings directly from the tenant config, so
        healthcare orgs don't need to manually build routing dicts.

        Args:
            tenant_config: Tenant configuration dict (from TenantConfig.to_dict())
            platform_llm_settings: Platform LLM settings for defaults

        Returns:
            SmartLLMRouter if HIPAA tenant, None otherwise

        Example:
            router = SmartLLMRouter.from_tenant_config(tenant.config.to_dict())
            if router:
                decision = router.route("Patient MRN: 12345")
        """
        if not tenant_config.get("hipaa_compliant", False):
            return None

        # Resolve PHI provider
        phi_provider_name = tenant_config.get("phi_provider_name", "ollama")
        phi_model = (
            tenant_config.get("phi_model")
            or tenant_config.get("ollama_model")
            or (platform_llm_settings.ollama_model if platform_llm_settings else "llama3.2")
        )
        phi_provider = {"provider": phi_provider_name, "model": phi_model}

        # Resolve general provider (optional)
        general_provider = None
        general_provider_name = tenant_config.get("general_provider_name")
        if general_provider_name:
            general_model = (
                tenant_config.get("general_model")
                or tenant_config.get("default_model")
                or (platform_llm_settings.default_model if platform_llm_settings else None)
            )
            if general_model:
                general_provider = {"provider": general_provider_name, "model": general_model}

        return cls(
            phi_provider=phi_provider,
            general_provider=general_provider,
            enable_phi_detection=True,
            enable_pii_detection=True,
            allow_manual_tagging=True,
        )

    def route(
        self,
        message: str,
        context: dict | None = None,
        manual_phi_tag: bool | None = None,
    ) -> RoutingDecision:
        """
        Route message to appropriate LLM provider.

        Args:
            message: User message to analyze
            context: Optional context (conversation history, metadata)
            manual_phi_tag: Manual PHI override
                - True: Force PHI routing (user tagged as PHI)
                - False: Force non-PHI routing (user confirmed no PHI)
                - None: Use automatic detection (default)

        Returns:
            RoutingDecision with provider and reasoning
        """

        # Step 0: Check for manual PHI tagging (takes precedence)
        if manual_phi_tag is not None:
            if manual_phi_tag is True:
                logger.info("Manual PHI tag: routing to self-hosted")
                return RoutingDecision(
                    provider=self.phi_provider["provider"],
                    model=self.phi_provider["model"],
                    reason="Manually tagged as PHI by user",
                    task_type=TaskType.PHI,
                    confidence=1.0,
                    phi_detected=True,
                    pii_detected=False,
                )
            elif manual_phi_tag is False:
                logger.info("Manual non-PHI tag: bypassing automatic detection")
                # Skip automatic detection, go straight to general routing
                task_type = self._classify_task(message)
                return RoutingDecision(
                    provider=self.general_provider["provider"],
                    model=self.general_provider["model"],
                    reason=f"User confirmed no PHI - {task_type.value} task",
                    task_type=task_type,
                    confidence=1.0,
                    phi_detected=False,
                    pii_detected=False,
                )

        # Step 1: Check for PHI (automatic detection)
        phi_detected = False
        phi_types = []

        if self.phi_detector:
            phi_detected, phi_types = self.phi_detector.contains_phi(message)

        if phi_detected:
            logger.warning(f"PHI detected: {phi_types}")
            return RoutingDecision(
                provider=self.phi_provider["provider"],
                model=self.phi_provider["model"],
                reason=f"PHI detected ({', '.join(phi_types)}) - routing to self-hosted",
                task_type=TaskType.PHI,
                confidence=1.0,
                phi_detected=True,
                pii_detected=False,
            )

        # Step 2: Check for PII
        pii_detected = False
        pii_types = []

        if self.pii_detector:
            pii_detected, pii_types = self.pii_detector.contains_pii(message)

        if pii_detected:
            logger.info(f"PII detected: {pii_types}")
            return RoutingDecision(
                provider=self.phi_provider["provider"],
                model=self.phi_provider["model"],
                reason=f"PII detected ({', '.join(pii_types)}) - routing to self-hosted",
                task_type=TaskType.PII,
                confidence=0.9,
                phi_detected=False,
                pii_detected=True,
            )

        # Step 3: Check for sensitive keywords
        sensitive_keywords = [
            "confidential",
            "proprietary",
            "secret",
            "private",
            "password",
            "api key",
            "token",
            "credential",
        ]

        if any(keyword in message.lower() for keyword in sensitive_keywords):
            logger.info("Sensitive keywords detected")
            return RoutingDecision(
                provider=self.phi_provider["provider"],
                model=self.phi_provider["model"],
                reason="Sensitive keywords detected - routing to self-hosted",
                task_type=TaskType.SENSITIVE,
                confidence=0.8,
                phi_detected=False,
                pii_detected=False,
            )

        # Step 4: Classify task type
        task_type = self._classify_task(message)

        # Step 5: Route to general provider
        logger.info(f"Routing to general provider for {task_type.value} task")
        return RoutingDecision(
            provider=self.general_provider["provider"],
            model=self.general_provider["model"],
            reason=f"General {task_type.value} task - using API for better performance",
            task_type=task_type,
            confidence=0.7,
            phi_detected=False,
            pii_detected=False,
        )

    def _classify_task(self, message: str) -> TaskType:
        """Classify the task type based on message content."""

        message_lower = message.lower()

        # Code generation/review
        code_keywords = ["code", "function", "script", "program", "debug", "syntax", "api"]
        if any(keyword in message_lower for keyword in code_keywords):
            return TaskType.CODE

        # Data analysis
        analysis_keywords = ["analyze", "analysis", "data", "report", "summary", "trend"]
        if any(keyword in message_lower for keyword in analysis_keywords):
            return TaskType.ANALYSIS

        return TaskType.GENERAL

    def get_provider_config(self, decision: RoutingDecision) -> dict[str, Any]:
        """Get full provider configuration for routing decision."""

        if decision.provider == self.phi_provider["provider"]:
            return self.phi_provider
        else:
            return self.general_provider
