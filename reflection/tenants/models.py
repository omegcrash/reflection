# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Tenant Model and Management

The tenant is the fundamental unit of isolation in Reflection.
Each tenant represents an organization with:
- Isolated data (memory, history, files)
- Independent configuration
- Separate quotas and billing
- Dedicated encryption keys

Hierarchy:
    Platform
    └── Tenant (Organization)
        ├── Users
        ├── Agents
        ├── Skills
        ├── Memory
        └── Channels
"""

import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ..core.settings import DEFAULT_ANTHROPIC_MODEL

logger = logging.getLogger(__name__)


def utcnow() -> datetime:
    """Get current UTC time."""
    return datetime.now(UTC)


class TenantStatus(StrEnum):
    """Tenant lifecycle status."""

    PROVISIONING = "provisioning"  # Being set up
    ACTIVE = "active"  # Normal operation
    SUSPENDED = "suspended"  # Billing/policy issue
    DEACTIVATING = "deactivating"  # Cleanup in progress
    DELETED = "deleted"  # Soft deleted


class TenantTier(StrEnum):
    """Tenant subscription tier."""

    FREE = "free"  # Limited quotas, community support
    PRO = "professional"  # Standard quotas, email support
    PROFESSIONAL = "professional"  # Alias for PRO
    ENTERPRISE = "enterprise"  # Custom quotas, dedicated support


@dataclass
class TenantQuotas:
    """
    Resource quotas for a tenant.

    Enforced at runtime by the quota manager.
    Can be customized per tier or per tenant.
    """

    # Concurrency
    max_concurrent_requests: int = 10
    max_concurrent_agents: int = 5

    # Rate limits
    max_requests_per_minute: int = 100
    max_requests_per_hour: int = 1000
    max_requests_per_day: int = 10000

    # Token limits
    max_tokens_per_request: int = 100_000
    max_tokens_per_day: int = 1_000_000
    max_tokens_per_month: int = 20_000_000

    # Tool limits
    max_tool_executions_per_hour: int = 500
    max_shell_executions_per_hour: int = 50
    max_http_requests_per_hour: int = 200

    # Storage limits
    max_memory_entries: int = 10_000
    max_memory_size_mb: int = 100
    max_file_storage_mb: int = 1000
    max_conversation_history_days: int = 90

    # Feature limits
    max_users: int = 10
    max_skills: int = 20
    max_channels: int = 5
    max_api_keys: int = 10

    @classmethod
    def for_tier(cls, tier: TenantTier) -> "TenantQuotas":
        """Get default quotas for a tier."""
        if tier == TenantTier.FREE:
            return cls(
                max_concurrent_requests=5,
                max_users=5,
                max_tokens_per_day=100_000,
                max_memory_entries=1000,
            )
        elif tier == TenantTier.PROFESSIONAL:
            return cls(
                max_concurrent_requests=25,
                max_users=50,
                max_tokens_per_day=5_000_000,
                max_memory_entries=50_000,
            )
        else:  # ENTERPRISE
            return cls(
                max_concurrent_requests=100,
                max_users=500,
                max_tokens_per_day=50_000_000,
                max_memory_entries=500_000,
                max_conversation_history_days=365,
            )


@dataclass
class TenantConfig:
    """
    Tenant-specific configuration.

    Overrides platform defaults for this tenant.
    """

    # Provider settings
    default_provider: str = "anthropic"
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL
    openai_model: str = "gpt-4o"
    ollama_model: str | None = None
    ollama_endpoint: str | None = None

    # Security settings
    security_mode: str = "balanced"  # paranoid, balanced, permissive
    default_trust_level: str = "stranger"  # For new users
    require_mfa: bool = False
    allowed_ip_ranges: list[str] = field(default_factory=list)

    # HIPAA Compliance
    # When True, automatically enables:
    #   - PHI/PII detection on all messages
    #   - Routing of sensitive content to self-hosted provider
    #   - HIPAA audit logging
    #   - Encryption at rest for conversations and memory
    #   - Extended audit log retention (minimum 6 years per HIPAA)
    hipaa_compliant: bool = False

    # Self-hosted provider for PHI routing (required when hipaa_compliant=True)
    # If not set, falls back to platform-level ollama settings
    phi_provider_name: str = "ollama"
    phi_model: str | None = None  # Falls back to ollama_model or platform default

    # General provider for non-PHI when in HIPAA mode (optional)
    # If not set, all traffic goes to the self-hosted provider
    general_provider_name: str | None = None  # e.g. "anthropic", "openai"
    general_model: str | None = None

    # Feature flags
    enable_shell_execution: bool = False
    enable_browser_automation: bool = False
    enable_file_write: bool = True
    enable_http_requests: bool = True
    enable_memory: bool = True
    enable_skills: bool = True

    # Skill preset: activates a curated bundle of skills and persona
    # "general" (default), "nonprofit", "healthcare", "enterprise"
    skill_preset: str = "general"

    # Agent behavior
    max_agent_iterations: int = 15
    default_agent_persona: str = "a helpful AI assistant"

    # Data retention
    audit_log_retention_days: int = 90
    conversation_retention_days: int = 90

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "default_provider": self.default_provider,
            "anthropic_model": self.anthropic_model,
            "openai_model": self.openai_model,
            "ollama_model": self.ollama_model,
            "ollama_endpoint": self.ollama_endpoint,
            "security_mode": self.security_mode,
            "default_trust_level": self.default_trust_level,
            "require_mfa": self.require_mfa,
            "allowed_ip_ranges": self.allowed_ip_ranges,
            "hipaa_compliant": self.hipaa_compliant,
            "phi_provider_name": self.phi_provider_name,
            "phi_model": self.phi_model,
            "general_provider_name": self.general_provider_name,
            "general_model": self.general_model,
            "enable_shell_execution": self.enable_shell_execution,
            "enable_browser_automation": self.enable_browser_automation,
            "enable_file_write": self.enable_file_write,
            "enable_http_requests": self.enable_http_requests,
            "enable_memory": self.enable_memory,
            "enable_skills": self.enable_skills,
            "skill_preset": self.skill_preset,
            "max_agent_iterations": self.max_agent_iterations,
            "default_agent_persona": self.default_agent_persona,
            "audit_log_retention_days": self.audit_log_retention_days,
            "conversation_retention_days": self.conversation_retention_days,
        }

    def validate_hipaa(self) -> list[str]:
        """
        Validate HIPAA compliance requirements are met.

        Returns list of validation errors (empty = valid).
        Called automatically when hipaa_compliant is True.
        """
        errors = []

        if not self.hipaa_compliant:
            return errors

        # PHI provider must be self-hosted
        if self.phi_provider_name not in ("ollama", "llama.cpp", "local"):
            errors.append(
                f"HIPAA requires a self-hosted PHI provider, got '{self.phi_provider_name}'. "
                f"Use 'ollama', 'llama.cpp', or 'local'."
            )

        # Security mode must be paranoid or balanced
        if self.security_mode == "permissive":
            errors.append(
                "HIPAA compliance is incompatible with 'permissive' security mode. "
                "Use 'paranoid' or 'balanced'."
            )

        # Audit logs must be retained for 6 years (HIPAA minimum)
        if self.audit_log_retention_days < 2190:  # ~6 years
            errors.append(
                f"HIPAA requires minimum 6-year audit log retention "
                f"(2190 days), got {self.audit_log_retention_days}."
            )

        # Memory encryption should be enabled
        # (This is a warning — enforced at the platform level)

        return errors

    def __post_init__(self):
        """Validate after initialization."""
        # Apply skill preset (before HIPAA overrides)
        if self.skill_preset != "general":
            self._apply_preset()

        if self.hipaa_compliant:
            # Auto-apply HIPAA-required settings
            if self.audit_log_retention_days < 2190:
                logger.info(
                    f"HIPAA mode: raising audit_log_retention_days from "
                    f"{self.audit_log_retention_days} to 2190 (6-year HIPAA minimum)"
                )
                self.audit_log_retention_days = 2190

            if self.security_mode == "permissive":
                logger.warning("HIPAA mode: overriding 'permissive' security_mode to 'balanced'")
                self.security_mode = "balanced"

    # Skill presets: curated skill bundles with appropriate personas
    # Phase 1: Constitutional personality replaces flat persona strings.
    # The base personality comes from familiar.core.constitution (Articles 0-7).
    # These presets provide role-specific overlays on top of the constitutional base.
    SKILL_PRESETS = {
        "general": {
            "skills": [],  # All skills enabled, no filtering
            "persona": "a helpful AI assistant",
            "constitutional_preset": "general",
        },
        "nonprofit": {
            "skills": [
                "nonprofit",
                "bookkeeping",
                "contacts",
                "documents",
                "reports",
                "workflows",
                "meetings",
                "tasks",
                "calendar",
                "email",
                "knowledge_base",
                "websearch",
                "smart_search",
                "filereader",
            ],
            "persona": (
                "an executive assistant for a 501(c)(3) nonprofit organization. "
                "You help with donor management, grant tracking, bookkeeping, "
                "board packet preparation, and day-to-day operations. You understand "
                "nonprofit compliance, fund accounting, and IRS reporting requirements."
            ),
            "constitutional_preset": "nonprofit",
        },
        "healthcare": {
            "skills": [
                "tasks",
                "calendar",
                "contacts",
                "documents",
                "reports",
                "meetings",
                "knowledge_base",
                "phi_detection",
                "audit",
                "encryption",
                "rbac",
                "user_management",
                "email",
            ],
            "persona": (
                "a HIPAA-compliant healthcare assistant. You help with scheduling, "
                "documentation, and administrative tasks while maintaining strict "
                "patient data privacy. You never include PHI in responses routed "
                "to external providers."
            ),
            "constitutional_preset": "healthcare",
        },
        "enterprise": {
            "skills": [
                "tasks",
                "calendar",
                "contacts",
                "documents",
                "reports",
                "workflows",
                "meetings",
                "knowledge_base",
                "websearch",
                "smart_search",
                "filereader",
                "email",
                "audit",
                "rbac",
                "user_management",
                "sessions",
                "encryption",
                "notifications",
            ],
            "persona": (
                "an enterprise AI assistant for team productivity. You help with "
                "task management, document creation, meeting scheduling, knowledge "
                "management, and workflow automation."
            ),
            "constitutional_preset": "enterprise",
        },
    }

    def _apply_preset(self):
        """Apply skill preset configuration."""
        preset = self.SKILL_PRESETS.get(self.skill_preset)
        if not preset:
            logger.warning(
                f"Unknown skill_preset '{self.skill_preset}', "
                f"valid options: {list(self.SKILL_PRESETS.keys())}"
            )
            return

        # Only override persona if it's still the default
        if self.default_agent_persona == "a helpful AI assistant":
            self.default_agent_persona = preset["persona"]

        logger.info(f"Applied skill preset: {self.skill_preset}")

    def get_allowed_skills(self) -> list[str] | None:
        """
        Get the list of allowed skills for this tenant.

        Returns None if all skills are allowed (general preset),
        or a list of skill names for filtered presets.
        """
        preset = self.SKILL_PRESETS.get(self.skill_preset, {})
        skills = preset.get("skills", [])
        return skills if skills else None  # Empty list = all skills allowed

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TenantConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Tenant:
    """
    A tenant (organization) in Reflection.

    Attributes:
        id: Unique identifier (UUID)
        name: Human-readable organization name
        slug: URL-safe identifier (used in subdomains)
        tier: Subscription tier
        status: Lifecycle status
        config: Tenant-specific configuration
        quotas: Resource quotas
        encryption_key_id: Reference to tenant's encryption key in KMS
        created_at: Creation timestamp
        updated_at: Last update timestamp
        metadata: Additional key-value data
    """

    id: str
    name: str
    slug: str
    tier: TenantTier = TenantTier.FREE
    status: TenantStatus = TenantStatus.PROVISIONING
    config: TenantConfig = field(default_factory=TenantConfig)
    quotas: TenantQuotas = field(default_factory=TenantQuotas)
    encryption_key_id: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Apply tier-based quotas if not customized
        if self.quotas == TenantQuotas():
            self.quotas = TenantQuotas.for_tier(self.tier)

    @classmethod
    def create(
        cls,
        name: str,
        slug: str,
        tier: TenantTier = TenantTier.FREE,
        config: TenantConfig | None = None,
        quotas: TenantQuotas | None = None,
    ) -> "Tenant":
        """Factory method to create a new tenant."""
        tenant_id = f"tenant_{uuid.uuid4().hex[:16]}"

        return cls(
            id=tenant_id,
            name=name,
            slug=slug.lower().replace(" ", "-"),
            tier=tier,
            status=TenantStatus.PROVISIONING,
            config=config or TenantConfig(),
            quotas=quotas or TenantQuotas.for_tier(tier),
        )

    def activate(self) -> None:
        """Mark tenant as active after provisioning."""
        if self.status != TenantStatus.PROVISIONING:
            raise ValueError(f"Cannot activate tenant in status: {self.status}")
        self.status = TenantStatus.ACTIVE
        self.updated_at = utcnow()

    def suspend(self, reason: str = "") -> None:
        """Suspend tenant (e.g., billing issue)."""
        self.status = TenantStatus.SUSPENDED
        self.metadata["suspension_reason"] = reason
        self.metadata["suspended_at"] = utcnow().isoformat()
        self.updated_at = utcnow()

    def reactivate(self) -> None:
        """Reactivate a suspended tenant."""
        if self.status != TenantStatus.SUSPENDED:
            raise ValueError(f"Cannot reactivate tenant in status: {self.status}")
        self.status = TenantStatus.ACTIVE
        self.metadata.pop("suspension_reason", None)
        self.metadata.pop("suspended_at", None)
        self.updated_at = utcnow()

    def delete(self) -> None:
        """Soft delete tenant."""
        self.status = TenantStatus.DEACTIVATING
        self.metadata["deletion_requested_at"] = utcnow().isoformat()
        self.updated_at = utcnow()

    @property
    def is_active(self) -> bool:
        """Check if tenant is operational."""
        return self.status == TenantStatus.ACTIVE

    @property
    def domain_prefix(self) -> str:
        """Get subdomain prefix for this tenant."""
        return self.slug

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "tier": self.tier.value,
            "status": self.status.value,
            "config": self.config.to_dict(),
            "quotas": dict(self.quotas.__dict__.items()),
            "encryption_key_id": self.encryption_key_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tenant":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            slug=data["slug"],
            tier=TenantTier(data["tier"]),
            status=TenantStatus(data["status"]),
            config=TenantConfig.from_dict(data.get("config", {})),
            quotas=TenantQuotas(**data.get("quotas", {})),
            encryption_key_id=data.get("encryption_key_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TenantAPIKey:
    """
    API key for tenant access.

    Keys are scoped to a tenant and optionally to specific permissions.
    """

    id: str
    tenant_id: str
    name: str  # Human-readable name
    key_hash: str  # SHA-256 hash of the key (never store plaintext)
    key_prefix: str  # First 8 chars for identification (e.g., "me_live_")
    permissions: set[str] = field(default_factory=lambda: {"chat", "read"})
    rate_limit_override: int | None = None  # Requests per minute
    expires_at: datetime | None = None
    created_at: datetime = field(default_factory=utcnow)
    last_used_at: datetime | None = None
    is_active: bool = True

    @classmethod
    def generate(
        cls,
        tenant_id: str,
        name: str,
        permissions: set[str] | None = None,
        expires_at: datetime | None = None,
    ) -> tuple["TenantAPIKey", str]:
        """
        Generate a new API key.

        Returns:
            (TenantAPIKey, plaintext_key)

        The plaintext key is only returned once and should be shown to the user.
        """
        import hashlib

        # Generate key: me_live_<32 random chars>
        key_id = f"key_{uuid.uuid4().hex[:12]}"
        random_part = secrets.token_urlsafe(32)
        plaintext_key = f"me_live_{random_part}"

        # Hash for storage
        key_hash = hashlib.sha256(plaintext_key.encode()).hexdigest()

        api_key = cls(
            id=key_id,
            tenant_id=tenant_id,
            name=name,
            key_hash=key_hash,
            key_prefix=plaintext_key[:12],
            permissions=permissions or {"chat", "read"},
            expires_at=expires_at,
        )

        return api_key, plaintext_key

    def verify(self, plaintext_key: str) -> bool:
        """Verify a plaintext key against this key's hash."""
        import hashlib

        provided_hash = hashlib.sha256(plaintext_key.encode()).hexdigest()
        return secrets.compare_digest(provided_hash, self.key_hash)

    @property
    def is_valid(self) -> bool:
        """Check if key is valid (active and not expired)."""
        return self.is_active and not (self.expires_at and self.expires_at < utcnow())

    def has_permission(self, permission: str) -> bool:
        """Check if key has a specific permission."""
        return permission in self.permissions or "admin" in self.permissions


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "Tenant",
    "TenantStatus",
    "TenantTier",
    "TenantConfig",
    "TenantQuotas",
    "TenantAPIKey",
]
