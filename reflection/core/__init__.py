# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Core Module

Enterprise-specific components that extend Familiar:
- Settings: Enterprise configuration
- Orchestrator: Database-backed agent loop (sync path)
- AsyncOrchestrator: True async with dual-path routing (Phase 5)
- Memory: Enterprise memory service with summarization and semantic search
- Async Providers: SDK-based async LLM providers
- UsageCalculator: Unified token counting for billing accuracy (v1.3.0)
- UsageAlerts: Budget monitoring and alerting (v1.3.0)
- CircuitBreaker: Resilience pattern for LLM providers (Phase 3)

All imports are lazy to avoid pulling in heavy dependencies (sqlalchemy,
httpx, etc.) when only lightweight submodules like settings are needed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .async_orchestrator import (
        AsyncOrchestrator as AsyncOrchestrator,
    )
    from .async_orchestrator import (
        AsyncOrchestratorEvent as AsyncOrchestratorEvent,
    )
    from .async_orchestrator import (
        AsyncOrchestratorResult as AsyncOrchestratorResult,
    )
    from .circuit_breaker import (
        CircuitBreaker as CircuitBreaker,
    )
    from .circuit_breaker import (
        CircuitBreakerConfig as CircuitBreakerConfig,
    )
    from .circuit_breaker import (
        CircuitBreakerRegistry as CircuitBreakerRegistry,
    )
    from .circuit_breaker import (
        CircuitBreakerStats as CircuitBreakerStats,
    )
    from .circuit_breaker import (
        CircuitOpenError as CircuitOpenError,
    )
    from .circuit_breaker import (
        CircuitState as CircuitState,
    )
    from .circuit_breaker import (
        get_circuit_breaker_registry as get_circuit_breaker_registry,
    )
    from .circuit_breaker import (
        get_provider_circuit_breaker as get_provider_circuit_breaker,
    )
    from .memory import (
        ContextWindow as ContextWindow,
    )
    from .memory import (
        ConversationSummary as ConversationSummary,
    )
    from .memory import (
        MemoryEntry as MemoryEntry,
    )
    from .memory import (
        MemoryService as MemoryService,
    )
    from .orchestrator import (
        FAMILIAR_AVAILABLE as FAMILIAR_AVAILABLE,
    )
    from .orchestrator import (
        AgentOrchestrator as AgentOrchestrator,
    )
    from .orchestrator import (
        OrchestratorEvent as OrchestratorEvent,
    )
    from .orchestrator import (
        OrchestratorResult as OrchestratorResult,
    )
    from .providers_async import (
        AsyncAnthropicProviderSDK as AsyncAnthropicProviderSDK,
    )
    from .providers_async import (
        AsyncLLMProviderSDK as AsyncLLMProviderSDK,
    )
    from .providers_async import (
        AsyncOpenAIProviderSDK as AsyncOpenAIProviderSDK,
    )
    from .providers_async import (
        AsyncProviderPool as AsyncProviderPool,
    )
    from .providers_async import (
        create_async_provider as create_async_provider,
    )
    from .providers_async import (
        get_async_provider_pool as get_async_provider_pool,
    )
    from .providers_async import (
        shutdown_async_providers as shutdown_async_providers,
    )
    from .settings import Settings as Settings
    from .settings import get_settings as get_settings
    from .usage_alerts import (
        AlertSeverity as AlertSeverity,
    )
    from .usage_alerts import (
        AlertThresholds as AlertThresholds,
    )
    from .usage_alerts import (
        AlertType as AlertType,
    )
    from .usage_alerts import (
        CallbackNotificationHandler as CallbackNotificationHandler,
    )
    from .usage_alerts import (
        LogNotificationHandler as LogNotificationHandler,
    )
    from .usage_alerts import (
        NotificationHandler as NotificationHandler,
    )
    from .usage_alerts import (
        UsageAlert as UsageAlert,
    )
    from .usage_alerts import (
        UsageAlertService as UsageAlertService,
    )
    from .usage_alerts import (
        WebhookNotificationHandler as WebhookNotificationHandler,
    )
    from .usage_alerts import (
        get_usage_alert_service as get_usage_alert_service,
    )
    from .usage_alerts import (
        init_usage_alert_service as init_usage_alert_service,
    )
    from .usage_alerts import (
        reset_usage_alert_service as reset_usage_alert_service,
    )
    from .usage_calculator import (
        AggregatedUsage as AggregatedUsage,
    )
    from .usage_calculator import (
        UsageCalculator as UsageCalculator,
    )
    from .usage_calculator import (
        UsageRecord as UsageRecord,
    )
    from .usage_calculator import (
        get_usage_calculator as get_usage_calculator,
    )
    from .usage_calculator import (
        reset_usage_calculator as reset_usage_calculator,
    )

# Map attribute names to (module, name) for lazy loading
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Settings
    "Settings": (".settings", "Settings"),
    "get_settings": (".settings", "get_settings"),
    # Orchestrator (sync path)
    "AgentOrchestrator": (".orchestrator", "AgentOrchestrator"),
    "OrchestratorEvent": (".orchestrator", "OrchestratorEvent"),
    "OrchestratorResult": (".orchestrator", "OrchestratorResult"),
    "FAMILIAR_AVAILABLE": (".orchestrator", "FAMILIAR_AVAILABLE"),
    # AsyncOrchestrator (Phase 5 dual-path)
    "AsyncOrchestrator": (".async_orchestrator", "AsyncOrchestrator"),
    "AsyncOrchestratorEvent": (".async_orchestrator", "AsyncOrchestratorEvent"),
    "AsyncOrchestratorResult": (".async_orchestrator", "AsyncOrchestratorResult"),
    # Memory
    "MemoryService": (".memory", "MemoryService"),
    "MemoryEntry": (".memory", "MemoryEntry"),
    "ContextWindow": (".memory", "ContextWindow"),
    "ConversationSummary": (".memory", "ConversationSummary"),
    # Async Providers (Phase 5)
    "AsyncLLMProviderSDK": (".providers_async", "AsyncLLMProviderSDK"),
    "AsyncAnthropicProviderSDK": (".providers_async", "AsyncAnthropicProviderSDK"),
    "AsyncOpenAIProviderSDK": (".providers_async", "AsyncOpenAIProviderSDK"),
    "AsyncProviderPool": (".providers_async", "AsyncProviderPool"),
    "get_async_provider_pool": (".providers_async", "get_async_provider_pool"),
    "shutdown_async_providers": (".providers_async", "shutdown_async_providers"),
    "create_async_provider": (".providers_async", "create_async_provider"),
    # Usage Calculator (v1.3.0)
    "UsageCalculator": (".usage_calculator", "UsageCalculator"),
    "UsageRecord": (".usage_calculator", "UsageRecord"),
    "AggregatedUsage": (".usage_calculator", "AggregatedUsage"),
    "get_usage_calculator": (".usage_calculator", "get_usage_calculator"),
    "reset_usage_calculator": (".usage_calculator", "reset_usage_calculator"),
    # Usage Alerts (v1.3.0)
    "UsageAlertService": (".usage_alerts", "UsageAlertService"),
    "UsageAlert": (".usage_alerts", "UsageAlert"),
    "AlertType": (".usage_alerts", "AlertType"),
    "AlertSeverity": (".usage_alerts", "AlertSeverity"),
    "AlertThresholds": (".usage_alerts", "AlertThresholds"),
    "NotificationHandler": (".usage_alerts", "NotificationHandler"),
    "WebhookNotificationHandler": (".usage_alerts", "WebhookNotificationHandler"),
    "LogNotificationHandler": (".usage_alerts", "LogNotificationHandler"),
    "CallbackNotificationHandler": (".usage_alerts", "CallbackNotificationHandler"),
    "get_usage_alert_service": (".usage_alerts", "get_usage_alert_service"),
    "init_usage_alert_service": (".usage_alerts", "init_usage_alert_service"),
    "reset_usage_alert_service": (".usage_alerts", "reset_usage_alert_service"),
    # Circuit Breaker (Phase 3)
    "CircuitBreaker": (".circuit_breaker", "CircuitBreaker"),
    "CircuitBreakerConfig": (".circuit_breaker", "CircuitBreakerConfig"),
    "CircuitBreakerStats": (".circuit_breaker", "CircuitBreakerStats"),
    "CircuitState": (".circuit_breaker", "CircuitState"),
    "CircuitOpenError": (".circuit_breaker", "CircuitOpenError"),
    "CircuitBreakerRegistry": (".circuit_breaker", "CircuitBreakerRegistry"),
    "get_circuit_breaker_registry": (".circuit_breaker", "get_circuit_breaker_registry"),
    "get_provider_circuit_breaker": (".circuit_breaker", "get_provider_circuit_breaker"),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        # Cache on the module to avoid repeated lookups
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
