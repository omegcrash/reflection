# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Tenant Agent

Multi-tenant wrapper around Familiar's core Agent.
Adds:
- Tenant context and isolation
- Per-tenant configuration
- Usage tracking for billing
- Database persistence
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

# Import from Familiar core
from familiar.core.agent import Agent
from familiar.core.config import Config, load_config
from familiar.core.paths import set_tenant_data_root
from familiar.core.providers import StreamEvent

from ..core.tokens import get_token_counter

# Smart routing for HIPAA compliance
from ..routing import SmartLLMRouter

# Reflection additions
from ..tenants.context import get_current_tenant_id, set_current_tenant
from ..tenants.quotas import check_quota

logger = logging.getLogger(__name__)


class TenantAgent(Agent):
    """
    Multi-tenant agent extending Familiar's core Agent.

    Adds tenant isolation while reusing all of Familiar's:
    - LLM providers
    - Tool registry
    - Memory system
    - Security model
    - Observability

    Usage:
        agent = TenantAgent(tenant_id=uuid, config=config)
        response = agent.chat("Hello!", user_id="user123")

        # With tenant context manager
        with agent.tenant_context():
            response = agent.chat("Hello!")
    """

    def __init__(
        self,
        tenant_id: UUID,
        config: Config | None = None,
        tenant_config: dict[str, Any] | None = None,
        security_mode=None,
        db_session=None,
    ):
        self.tenant_id = tenant_id
        self.tenant_config = tenant_config or {}

        # Scope all data paths to this tenant's directory
        # Must happen BEFORE parent Agent.__init__() which loads skills
        set_tenant_data_root(str(tenant_id))

        # Apply tenant-specific config overrides
        config = self._apply_tenant_config(config)

        # Initialize parent Agent (Familiar v1.4.0 signature)
        super().__init__(config=config, security_mode=security_mode)

        # Tenant-specific tracking
        self._usage_callback = None

        # Wire TenantMemory for database-backed tenant isolation
        if db_session is not None:
            from .memory import TenantMemory

            self.memory = TenantMemory(
                tenant_id=tenant_id,
                session=db_session,
            )

        # Wire TenantToolRegistry for per-tenant tool isolation
        from .tools import get_tenant_tool_registry

        self._tenant_tools = get_tenant_tool_registry(tenant_id)

        # Apply skill preset filtering if configured
        if hasattr(self, "_allowed_skills"):
            for tool_name in list(self._tenant_tools.get_all().keys()):
                if tool_name not in self._allowed_skills:
                    self._tenant_tools.disable_tool(tool_name)

        # Initialize smart routing for HIPAA compliance
        self._router = None
        self._init_smart_routing()

        logger.info(f"TenantAgent initialized for tenant {tenant_id}")

    def _apply_tenant_config(self, config: Config | None) -> Config:
        """Apply tenant-specific configuration overrides."""
        config = config or load_config()

        # Override with tenant settings
        if "agent_name" in self.tenant_config:
            config.agent.name = self.tenant_config["agent_name"]

        if "persona" in self.tenant_config:
            config.agent.persona = self.tenant_config["persona"]

        if "default_provider" in self.tenant_config:
            config.llm.default_provider = self.tenant_config["default_provider"]

        if "default_model" in self.tenant_config:
            config.llm.default_model = self.tenant_config["default_model"]

        if "max_tokens" in self.tenant_config:
            config.llm.max_tokens = self.tenant_config["max_tokens"]

        # Apply skill preset (nonprofit, healthcare, enterprise)
        if "skill_preset" in self.tenant_config:
            from ..tenants.models import TenantConfig as TenantCfg

            tc = TenantCfg(skill_preset=self.tenant_config["skill_preset"])

            # Set persona from preset if not explicitly overridden
            if "persona" not in self.tenant_config:
                config.agent.persona = tc.default_agent_persona

            # Store allowed skills for filtering during skill loading
            allowed = tc.get_allowed_skills()
            if allowed is not None:
                self._allowed_skills = allowed
                logger.info(
                    f"Skill preset '{self.tenant_config['skill_preset']}' "
                    f"activated {len(allowed)} skills for tenant {self.tenant_id}"
                )

            # Phase 1: Constitutional personality overlay
            # Injects role-specific context on top of the constitutional base
            try:
                from familiar.core.constitution import get_constitutional_preset

                preset_name = self.tenant_config["skill_preset"]
                self._persona_overlay = get_constitutional_preset(preset_name)
                if self._persona_overlay:
                    logger.info(
                        f"Constitutional preset '{preset_name}' "
                        f"activated for tenant {self.tenant_id}"
                    )
            except ImportError:
                logger.debug("Constitution module not available for tenant preset")
                self._persona_overlay = ""

        return config

    def _init_smart_routing(self):
        """
        Initialize smart routing based on tenant compliance settings.

        When hipaa_compliant is True in tenant config, routing is
        automatically configured — no manual routing dict needed.

        Non-HIPAA tenants skip routing entirely (all traffic goes to
        the tenant's default provider).
        """
        hipaa_compliant = self.tenant_config.get("hipaa_compliant", False)

        if not hipaa_compliant:
            # Non-HIPAA tenants don't need PHI routing
            logger.debug(f"Tenant {self.tenant_id} is not HIPAA — smart routing disabled")
            return

        # Build provider configs from tenant settings
        from ..core.settings import get_settings

        platform_settings = get_settings()

        # PHI provider: must be self-hosted
        phi_provider_name = self.tenant_config.get("phi_provider_name", "ollama")
        phi_model = (
            self.tenant_config.get("phi_model")
            or self.tenant_config.get("ollama_model")
            or platform_settings.llm.ollama_model
        )
        phi_provider = {
            "provider": phi_provider_name,
            "model": phi_model,
        }

        # General provider: optional, for non-PHI traffic
        # If not configured, all traffic stays on the self-hosted provider
        general_provider_name = self.tenant_config.get("general_provider_name")
        if general_provider_name:
            general_model = (
                self.tenant_config.get("general_model")
                or self.tenant_config.get("default_model")
                or platform_settings.llm.default_model
            )
            general_provider = {
                "provider": general_provider_name,
                "model": general_model,
            }
        else:
            general_provider = None  # SmartLLMRouter falls back to phi_provider

        # Initialize router
        try:
            self._router = SmartLLMRouter(
                phi_provider=phi_provider,
                general_provider=general_provider,
                enable_phi_detection=True,
                enable_pii_detection=True,
                allow_manual_tagging=True,
            )
            logger.info(
                f"HIPAA routing enabled for tenant {self.tenant_id}: "
                f"PHI → {phi_provider_name}/{phi_model}"
                f"{f', General → {general_provider_name}/{general_model}' if general_provider_name else ', all traffic self-hosted'}"
            )
        except Exception as e:
            logger.error(
                f"CRITICAL: Failed to initialize HIPAA routing for tenant "
                f"{self.tenant_id}: {e}. All traffic will use default provider — "
                f"this may violate HIPAA compliance!"
            )
            self._router = None

    @contextmanager
    def tenant_context(self):
        """Context manager for tenant isolation."""
        previous = get_current_tenant_id()
        set_current_tenant(self.tenant_id)
        try:
            yield
        finally:
            set_current_tenant(previous)

    def set_usage_callback(self, callback):
        """Set callback for usage tracking (for billing)."""
        self._usage_callback = callback

    def _track_usage(
        self,
        user_id: Any,
        channel: str,
        input_tokens: int,
        output_tokens: int,
        model: str,
        latency_ms: int,
    ):
        """Track usage for billing."""
        if self._usage_callback:
            self._usage_callback(
                tenant_id=self.tenant_id,
                user_id=user_id,
                channel=channel,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                latency_ms=latency_ms,
            )

    @contextmanager
    def _routed_config(self, provider: str, model: str):
        """Temporarily override the agent's provider/model for smart routing.

        Familiar v1.4.0's ``Agent.chat()`` does not accept provider/model
        kwargs.  Instead we reconfigure ``self.config`` for the duration
        of the call and restore afterwards.
        """
        prev_provider = self.config.llm.default_provider
        prev_model = self.config.llm.default_model
        try:
            self.config.llm.default_provider = provider
            self.config.llm.default_model = model
            yield
        finally:
            self.config.llm.default_provider = prev_provider
            self.config.llm.default_model = prev_model

    def chat(
        self,
        message: str,
        user_id: Any = "default",
        channel: str = "default",
        include_history: bool = True,
        contains_phi: bool | None = None,
    ) -> str:
        """Process a message with tenant context and smart routing.

        Wraps parent chat() with:
        - Tenant isolation
        - Quota checking
        - Smart routing (PHI detection + routing)
        - Usage tracking

        Args:
            message: User message
            user_id: User identifier
            channel: Communication channel
            include_history: Include conversation history
            contains_phi: Manual PHI override (optional)
                - True: Force PHI routing (self-hosted)
                - False: Force non-PHI routing (API if available)
                - None: Use automatic detection (default)

        Returns:
            Response string
        """
        # Check quotas before processing
        check_quota(self.tenant_id, "requests_per_day")
        check_quota(self.tenant_id, "tokens_per_day")

        # Smart routing (if enabled)
        routing_metadata = {}
        routed_provider = None
        routed_model = None
        if self._router:
            try:
                decision = self._router.route(
                    message=message,
                    manual_phi_tag=contains_phi,
                )

                logger.info(
                    "Routing decision for tenant %s: %s/%s - %s",
                    self.tenant_id,
                    decision.provider,
                    decision.model,
                    decision.reason,
                )

                if contains_phi is not None:
                    logger.info(
                        "Manual PHI tag: %s (user %s)",
                        contains_phi,
                        "confirmed PHI" if contains_phi else "confirmed no PHI",
                    )

                routed_provider = decision.provider
                routed_model = decision.model

                routing_metadata = {
                    "provider": decision.provider,
                    "model": decision.model,
                    "reason": decision.reason,
                    "task_type": decision.task_type.value,
                    "phi_detected": decision.phi_detected,
                    "pii_detected": decision.pii_detected,
                    "manual_tag": contains_phi,
                    "confidence": decision.confidence,
                }

            except Exception as e:
                logger.error("Smart routing failed: %s", e)

        start_time = datetime.now(UTC)

        # Run within tenant context, optionally with routed config
        with self.tenant_context():
            if routed_provider and routed_model:
                with self._routed_config(routed_provider, routed_model):
                    response = super().chat(
                        message=message,
                        user_id=user_id,
                        channel=channel,
                        include_history=include_history,
                    )
            else:
                response = super().chat(
                    message=message,
                    user_id=user_id,
                    channel=channel,
                    include_history=include_history,
                )

        # Track usage
        latency_ms = int((datetime.now(UTC) - start_time).total_seconds() * 1000)

        from ..core.settings import get_settings

        model_name = routing_metadata.get("model") or self.tenant_config.get(
            "default_model",
            get_settings().llm.default_model,
        )
        token_counter = get_token_counter()
        input_tokens = token_counter.count_tokens(message, model_name).count
        output_tokens = token_counter.count_tokens(response, model_name).count

        self._track_usage(
            user_id=user_id,
            channel=channel,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model_name,
            latency_ms=latency_ms,
        )

        if routing_metadata:
            logger.info(
                "Message processed - Routing: %s, PHI: %s, Manual tag: %s",
                routing_metadata["provider"],
                routing_metadata["phi_detected"],
                routing_metadata.get("manual_tag"),
            )

        return response

    def stream(
        self,
        message: str,
        user_id: Any = "default",
        channel: str = "default",
        include_history: bool = True,
    ) -> Generator[StreamEvent, None, str]:
        """Stream response with tenant context.

        Delegates to Familiar v1.4.0's ``Agent.chat_stream()`` while
        keeping the public name ``stream()`` for Reflection callers.
        """
        check_quota(self.tenant_id, "requests_per_day")

        with self.tenant_context():
            yield from super().chat_stream(
                message=message,
                user_id=user_id,
                channel=channel,
                include_history=include_history,
            )

    def get_tenant_memory_key(self, key: str) -> str:
        """Get tenant-scoped memory key."""
        return f"tenant:{self.tenant_id}:{key}"

    def remember(self, key: str, value: str, **kwargs):
        """Store tenant-scoped memory."""
        if self.memory:
            scoped_key = self.get_tenant_memory_key(key)
            self.memory.remember(scoped_key, value, **kwargs)

    def recall(self, key: str) -> str | None:
        """Retrieve tenant-scoped memory."""
        if self.memory:
            scoped_key = self.get_tenant_memory_key(key)
            return self.memory.recall(scoped_key)
        return None


class TenantAgentPool:
    """
    Pool of tenant agents with O(1) LRU eviction.

    Uses OrderedDict for efficient access ordering.
    Thread-safe for concurrent access.

    Performance characteristics:
    - get_agent: O(1) average case
    - remove_agent: O(1)
    - eviction: O(1)

    Previous implementation used list for access_order which was O(n)
    for removal operations. This implementation uses OrderedDict.move_to_end()
    which is O(1).
    """

    def __init__(self, max_agents: int = 100):
        from collections import OrderedDict
        from threading import Lock

        self.max_agents = max_agents
        self._agents: OrderedDict[UUID, TenantAgent] = OrderedDict()
        self._lock = Lock()

        # Metrics for monitoring
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get_agent(
        self,
        tenant_id: UUID,
        config: Config | None = None,
        tenant_config: dict[str, Any] | None = None,
    ) -> TenantAgent:
        """
        Get or create agent for tenant with O(1) operations.

        Thread-safe implementation using Lock for concurrent access.
        Agent cleanup happens OUTSIDE the lock to prevent blocking
        other threads during potentially slow I/O operations.
        """
        agents_to_cleanup = []

        with self._lock:
            if tenant_id in self._agents:
                # Move to end (most recently used) - O(1)
                self._agents.move_to_end(tenant_id)
                self._hits += 1
                return self._agents[tenant_id]

            self._misses += 1

            # Evict oldest if at capacity - O(1)
            # Collect agents to cleanup but DON'T cleanup under lock
            while len(self._agents) >= self.max_agents:
                oldest_id, oldest_agent = self._agents.popitem(last=False)
                self._evictions += 1
                agents_to_cleanup.append((oldest_id, oldest_agent))
                logger.debug(f"Evicted agent for tenant {oldest_id}")

            # Create new agent
            agent = TenantAgent(
                tenant_id=tenant_id,
                config=config,
                tenant_config=tenant_config,
            )

            self._agents[tenant_id] = agent

        # Cleanup evicted agents OUTSIDE the lock
        # This prevents blocking other threads during slow cleanup I/O
        for evicted_id, evicted_agent in agents_to_cleanup:
            self._cleanup_agent_safe(evicted_agent, evicted_id)

        return agent

    def _cleanup_agent_safe(self, agent: TenantAgent, tenant_id: UUID) -> None:
        """
        Safely cleanup an evicted agent's resources.

        Called OUTSIDE the lock to prevent blocking other threads.
        """
        try:
            if hasattr(agent, "cleanup"):
                agent.cleanup()
        except Exception as e:
            logger.warning(f"Agent cleanup failed for tenant {tenant_id}: {e}")

    def remove_agent(self, tenant_id: UUID) -> bool:
        """Remove agent from pool. Returns True if agent was found."""
        agent_to_cleanup = None

        with self._lock:
            if tenant_id in self._agents:
                agent_to_cleanup = self._agents.pop(tenant_id)

        # Cleanup outside lock to prevent blocking
        if agent_to_cleanup:
            self._cleanup_agent_safe(agent_to_cleanup, tenant_id)
            return True
        return False

    def clear(self):
        """Clear all agents."""
        agents_to_cleanup = []

        with self._lock:
            # Collect all agents to cleanup
            agents_to_cleanup = list(self._agents.items())
            self._agents.clear()

        # Cleanup outside lock to prevent blocking
        for tenant_id, agent in agents_to_cleanup:
            self._cleanup_agent_safe(agent, tenant_id)

    def get_metrics(self) -> dict[str, Any]:
        """Get pool metrics for monitoring."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "size": len(self._agents),
                "max_size": self.max_agents,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
            }

    def __len__(self) -> int:
        return len(self._agents)


# Global agent pool
_agent_pool: TenantAgentPool | None = None


def get_agent_pool() -> TenantAgentPool:
    """Get global agent pool."""
    global _agent_pool
    if _agent_pool is None:
        _agent_pool = TenantAgentPool()
    return _agent_pool


def get_tenant_agent(
    tenant_id: UUID,
    config: Config | None = None,
    tenant_config: dict[str, Any] | None = None,
) -> TenantAgent:
    """Get agent for tenant from pool."""
    pool = get_agent_pool()
    return pool.get_agent(tenant_id, config, tenant_config)


__all__ = [
    "TenantAgent",
    "TenantAgentPool",
    "get_agent_pool",
    "get_tenant_agent",
]
