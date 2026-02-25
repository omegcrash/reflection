# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Application Settings

Configuration management using pydantic-settings.
Supports environment variables and .env files.
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Central model constant — change here to update everywhere
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"


class DatabaseSettings(BaseSettings):
    """PostgreSQL database configuration."""

    model_config = SettingsConfigDict(env_prefix="DATABASE_")

    url: str = Field(
        default="postgresql+asyncpg://localhost:5432/reflection",
        description="PostgreSQL connection URL",
    )
    pool_size: int = Field(default=10, ge=1, le=100)
    max_overflow: int = Field(default=20, ge=0, le=100)
    echo: bool = Field(default=False, description="Echo SQL queries")


class RedisSettings(BaseSettings):
    """Redis configuration."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    max_connections: int = Field(default=50, ge=1, le=500)


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    default_provider: str = Field(default="anthropic")
    default_model: str = Field(
        default=DEFAULT_ANTHROPIC_MODEL,
        description="Default model for agents (centralized, overridable per-tenant)",
    )

    # Anthropic
    anthropic_api_key: str | None = Field(default=None)
    anthropic_model: str = Field(default=DEFAULT_ANTHROPIC_MODEL)

    # OpenAI
    openai_api_key: str | None = Field(default=None)
    openai_model: str = Field(default="gpt-4o")

    # Ollama
    ollama_url: str | None = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2")

    # Limits
    max_tokens_per_request: int = Field(default=100_000)
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    request_timeout: int = Field(default=300, description="Seconds")


class SecuritySettings(BaseSettings):
    """Security configuration."""

    model_config = SettingsConfigDict(env_prefix="SECURITY_")

    # Master encryption key (encrypts tenant keys)
    master_encryption_key: str | None = Field(
        default=None, description="Base64-encoded 32-byte key"
    )

    # JWT
    jwt_secret_key: str = Field(
        default="CHANGE_ME_IN_PRODUCTION", description="Secret key for JWT signing"
    )
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiry_hours: int = Field(default=24)
    jwt_access_token_expire_minutes: int = Field(default=60)
    jwt_refresh_token_expire_days: int = Field(default=7)

    # API Keys
    api_key_prefix: str = Field(default="me_live_")

    # CORS
    cors_origins: list[str] = Field(default=["http://localhost:3000"])

    # Rate Limiting
    auth_rate_limit_per_minute: int = Field(
        default=5, description="Max login attempts per IP per minute"
    )
    auth_lockout_threshold: int = Field(default=5, description="Failed attempts before lockout")
    auth_lockout_duration_minutes: int = Field(
        default=15, description="Lockout duration in minutes"
    )

    @field_validator("master_encryption_key")
    @classmethod
    def validate_encryption_key(cls, v):
        if v and len(v) < 32:
            raise ValueError("Encryption key must be at least 32 characters")
        return v

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret(cls, v: str, info) -> str:
        """
        Validate JWT secret meets security requirements.

        CRITICAL: This prevents deployment with insecure defaults.
        """
        # List of known insecure default values
        insecure_values = {
            "change_me_in_production",
            "change-me-in-production",
            "change-me",
            "changeme",
            "secret",
            "jwt-secret",
            "jwt_secret",
            "your-secret-key",
            "your_secret_key",
            "supersecret",
            "password",
            "123456",
            "development",
            "dev-secret",
        }

        # Check for insecure values (case-insensitive)
        if v.lower().replace("-", "_") in insecure_values:
            raise ValueError(
                "\n" + "=" * 60 + "\n"
                "SECURITY ERROR: JWT_SECRET_KEY is set to an insecure default!\n"
                "=" * 60 + "\n\n"
                "You must set SECURITY_JWT_SECRET_KEY to a secure random value.\n\n"
                "Generate a secure key with:\n"
                '  python -c "import secrets; print(secrets.token_urlsafe(64))"\n\n'
                "Then set it in your environment:\n"
                "  export SECURITY_JWT_SECRET_KEY='your-generated-key'\n\n"
                "Or in your .env file:\n"
                "  SECURITY_JWT_SECRET_KEY=your-generated-key\n" + "=" * 60
            )

        # Check minimum length
        if len(v) < 32:
            raise ValueError(
                f"JWT_SECRET_KEY must be at least 32 characters (got {len(v)}). "
                'Generate a secure key with: python -c "import secrets; print(secrets.token_urlsafe(64))"'
            )

        # Check entropy (basic check for non-trivial strings)
        unique_chars = len(set(v))
        if unique_chars < 10:
            raise ValueError(
                "JWT_SECRET_KEY appears to have low entropy (too many repeated characters). "
                "Use a cryptographically secure random string."
            )

        return v


class ObservabilitySettings(BaseSettings):
    """Observability configuration."""

    model_config = SettingsConfigDict(env_prefix="OTEL_")

    enabled: bool = Field(default=True)
    service_name: str = Field(default="reflection")
    exporter_otlp_endpoint: str | None = Field(default=None)

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")  # json or console


class Settings(BaseSettings):
    """
    Main application settings.

    Usage:
        settings = get_settings()
        print(settings.database.url)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="Reflection")
    app_version: str = Field(default="2.0.0")
    debug: bool = Field(default=False)
    environment: str = Field(default="development")  # development, staging, production

    # Development Mode Settings
    auto_create_tenants: bool = Field(
        default=False,
        description="Allow automatic tenant creation via X-Tenant-ID header (dev mode only)",
    )
    allow_x_tenant_id_auth: bool = Field(
        default=False, description="Allow X-Tenant-ID header authentication (NEVER in production)"
    )

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)

    # Data Export
    export_directory: str | None = Field(
        default=None, description="Directory for data exports. If not set, uses secure tempdir."
    )

    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def is_staging(self) -> bool:
        return self.environment == "staging"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Safety check: never allow dev features in production
        if self.is_production:
            if self.auto_create_tenants:
                raise ValueError(
                    "AUTO_CREATE_TENANTS cannot be enabled in production environment. "
                    "This setting is only for development/testing."
                )
            if self.allow_x_tenant_id_auth:
                raise ValueError(
                    "ALLOW_X_TENANT_ID_AUTH cannot be enabled in production environment. "
                    "Use JWT or API key authentication instead."
                )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Settings are loaded once and cached for the application lifetime.
    """
    return Settings()


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "Settings",
    "DatabaseSettings",
    "RedisSettings",
    "LLMSettings",
    "SecuritySettings",
    "ObservabilitySettings",
    "get_settings",
]
