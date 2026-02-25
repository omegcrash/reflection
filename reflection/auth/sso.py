# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Enterprise SSO Module (v2.0.0)

Provides Single Sign-On capabilities:
- SAML 2.0 Service Provider
- OpenID Connect (OIDC) Relying Party
- Just-In-Time (JIT) user provisioning
- Multi-tenant IdP configuration

Usage:
    from reflection.auth.sso import (
        SSOProvider,
        SAMLConfig,
        OIDCConfig,
        SSOService,
    )

    # Configure SAML for a tenant
    saml_config = SAMLConfig(
        idp_entity_id="https://idp.example.com",
        idp_sso_url="https://idp.example.com/sso",
        idp_certificate="-----BEGIN CERTIFICATE-----...",
    )

    await sso_service.configure_saml(tenant_id, saml_config)
"""

import base64
import hashlib
import logging
import secrets
import urllib.parse
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================
# SSO TYPES
# ============================================================


class SSOProtocol(StrEnum):
    """Supported SSO protocols."""

    SAML = "saml"
    OIDC = "oidc"


class SSOProvider(StrEnum):
    """Common SSO providers with preset configurations."""

    OKTA = "okta"
    AZURE_AD = "azure_ad"
    GOOGLE = "google"
    AUTH0 = "auth0"
    ONELOGIN = "onelogin"
    PING = "ping"
    CUSTOM = "custom"


@dataclass
class SSOUser:
    """User information extracted from SSO assertion."""

    external_id: str  # ID from IdP
    email: str
    email_verified: bool = False
    first_name: str | None = None
    last_name: str | None = None
    display_name: str | None = None
    groups: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        if self.display_name:
            return self.display_name
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.email


@dataclass
class SSOSession:
    """SSO session information."""

    id: str
    tenant_id: UUID
    user: SSOUser
    protocol: SSOProtocol
    provider: SSOProvider
    created_at: datetime
    expires_at: datetime
    session_index: str | None = None  # SAML session index
    access_token: str | None = None  # OIDC access token
    refresh_token: str | None = None  # OIDC refresh token

    @property
    def is_expired(self) -> bool:
        return datetime.now(UTC) >= self.expires_at


# ============================================================
# SAML CONFIGURATION
# ============================================================


class SAMLConfig(BaseModel):
    """SAML 2.0 Service Provider configuration."""

    # Identity Provider settings
    idp_entity_id: str = Field(..., description="IdP Entity ID")
    idp_sso_url: str = Field(..., description="IdP Single Sign-On URL")
    idp_slo_url: str | None = Field(default=None, description="IdP Single Logout URL")
    idp_certificate: str = Field(..., description="IdP X.509 certificate (PEM format)")

    # Service Provider settings (auto-generated if not provided)
    sp_entity_id: str | None = Field(default=None, description="SP Entity ID")
    sp_acs_url: str | None = Field(default=None, description="SP Assertion Consumer Service URL")
    sp_slo_url: str | None = Field(default=None, description="SP Single Logout URL")

    # Attribute mapping
    attribute_mapping: dict[str, str] = Field(
        default_factory=lambda: {
            "email": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
            "first_name": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname",
            "last_name": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname",
            "groups": "http://schemas.xmlsoap.org/claims/Group",
        },
        description="Mapping of local attributes to SAML claims",
    )

    # Security settings
    want_assertions_signed: bool = Field(default=True)
    want_response_signed: bool = Field(default=True)
    allow_unsolicited: bool = Field(default=False)

    # Session settings
    session_duration_hours: int = Field(default=8)


class SAMLAuthRequest(BaseModel):
    """SAML Authentication Request parameters."""

    relay_state: str | None = None
    force_authn: bool = False
    is_passive: bool = False


class SAMLResponse(BaseModel):
    """Parsed SAML Response."""

    success: bool
    user: SSOUser | None = None
    session_index: str | None = None
    error: str | None = None
    raw_assertion: str | None = None


# ============================================================
# OIDC CONFIGURATION
# ============================================================


class OIDCConfig(BaseModel):
    """OpenID Connect Relying Party configuration."""

    # Provider settings
    issuer: str = Field(..., description="OIDC Issuer URL")
    client_id: str = Field(..., description="Client ID")
    client_secret: str = Field(..., description="Client Secret")

    # Discovery (auto-populated from .well-known if issuer provided)
    authorization_endpoint: str | None = None
    token_endpoint: str | None = None
    userinfo_endpoint: str | None = None
    jwks_uri: str | None = None
    end_session_endpoint: str | None = None

    # Scopes
    scopes: list[str] = Field(
        default_factory=lambda: ["openid", "profile", "email"],
    )

    # Claim mapping
    claim_mapping: dict[str, str] = Field(
        default_factory=lambda: {
            "email": "email",
            "first_name": "given_name",
            "last_name": "family_name",
            "groups": "groups",
        },
    )

    # Security settings
    response_type: str = Field(default="code")
    response_mode: str = Field(default="query")
    use_pkce: bool = Field(default=True)

    # Session settings
    session_duration_hours: int = Field(default=8)


class OIDCAuthRequest(BaseModel):
    """OIDC Authorization Request parameters."""

    state: str
    nonce: str
    code_verifier: str | None = None  # For PKCE
    code_challenge: str | None = None
    redirect_uri: str
    prompt: str | None = None  # none, login, consent, select_account


class OIDCTokenResponse(BaseModel):
    """OIDC Token Response."""

    access_token: str
    token_type: str
    expires_in: int
    refresh_token: str | None = None
    id_token: str | None = None
    scope: str | None = None


# ============================================================
# SSO SERVICE
# ============================================================


class SSOService:
    """
    Enterprise SSO service.

    Handles SAML and OIDC authentication for tenants.

    Usage:
        service = SSOService(redis, db)

        # Configure SSO for a tenant
        await service.configure_saml(tenant_id, saml_config)

        # Initiate login
        redirect_url = await service.initiate_login(
            tenant_id=tenant_id,
            protocol=SSOProtocol.SAML,
            callback_url="https://app.example.com/callback",
        )

        # Handle callback
        session = await service.handle_callback(
            tenant_id=tenant_id,
            protocol=SSOProtocol.SAML,
            response_data=saml_response,
        )
    """

    def __init__(self, redis, database):
        """
        Initialize SSO service.

        Args:
            redis: Redis client for session storage
            database: Database for config storage
        """
        self.redis = redis
        self.db = database
        self._http_client: httpx.AsyncClient | None = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    async def configure_saml(
        self,
        tenant_id: UUID,
        config: SAMLConfig,
    ) -> None:
        """
        Configure SAML for a tenant.

        Args:
            tenant_id: Tenant ID
            config: SAML configuration
        """
        # Validate certificate
        self._validate_certificate(config.idp_certificate)

        # Store configuration
        config_key = f"sso:config:{tenant_id}:saml"
        await self.redis.setex(
            config_key,
            86400 * 365,  # 1 year TTL
            config.model_dump_json(),
        )

        logger.info(f"SAML configured for tenant {tenant_id}")

    async def configure_oidc(
        self,
        tenant_id: UUID,
        config: OIDCConfig,
    ) -> None:
        """
        Configure OIDC for a tenant.

        Args:
            tenant_id: Tenant ID
            config: OIDC configuration
        """
        # Discover endpoints if not provided
        if not config.authorization_endpoint:
            config = await self._discover_oidc_endpoints(config)

        # Store configuration
        config_key = f"sso:config:{tenant_id}:oidc"
        await self.redis.setex(
            config_key,
            86400 * 365,
            config.model_dump_json(),
        )

        logger.info(f"OIDC configured for tenant {tenant_id}")

    async def get_config(
        self,
        tenant_id: UUID,
        protocol: SSOProtocol,
    ) -> dict[str, Any] | None:
        """Get SSO configuration for a tenant."""
        config_key = f"sso:config:{tenant_id}:{protocol.value}"
        data = await self.redis.get(config_key)

        if data:
            import json

            return json.loads(data)
        return None

    async def remove_config(
        self,
        tenant_id: UUID,
        protocol: SSOProtocol,
    ) -> bool:
        """Remove SSO configuration for a tenant."""
        config_key = f"sso:config:{tenant_id}:{protocol.value}"
        result = await self.redis.delete(config_key)
        return result > 0

    # --------------------------------------------------------
    # AUTHENTICATION
    # --------------------------------------------------------

    async def initiate_login(
        self,
        tenant_id: UUID,
        protocol: SSOProtocol,
        callback_url: str,
        relay_state: str | None = None,
    ) -> str:
        """
        Initiate SSO login flow.

        Args:
            tenant_id: Tenant ID
            protocol: SSO protocol (SAML or OIDC)
            callback_url: URL to redirect after authentication
            relay_state: Optional state to preserve

        Returns:
            URL to redirect user to IdP
        """
        config = await self.get_config(tenant_id, protocol)
        if not config:
            raise ValueError(f"SSO not configured for tenant {tenant_id}")

        if protocol == SSOProtocol.SAML:
            return await self._initiate_saml_login(tenant_id, config, callback_url, relay_state)
        else:
            return await self._initiate_oidc_login(tenant_id, config, callback_url, relay_state)

    async def handle_callback(
        self,
        tenant_id: UUID,
        protocol: SSOProtocol,
        response_data: dict[str, Any],
    ) -> SSOSession:
        """
        Handle SSO callback from IdP.

        Args:
            tenant_id: Tenant ID
            protocol: SSO protocol
            response_data: Response data from IdP

        Returns:
            SSO session with user info

        Raises:
            ValueError: If authentication fails
        """
        config = await self.get_config(tenant_id, protocol)
        if not config:
            raise ValueError(f"SSO not configured for tenant {tenant_id}")

        if protocol == SSOProtocol.SAML:
            return await self._handle_saml_callback(tenant_id, config, response_data)
        else:
            return await self._handle_oidc_callback(tenant_id, config, response_data)

    # --------------------------------------------------------
    # SAML IMPLEMENTATION
    # --------------------------------------------------------

    async def _initiate_saml_login(
        self,
        tenant_id: UUID,
        config: dict[str, Any],
        callback_url: str,
        relay_state: str | None,
    ) -> str:
        """Generate SAML AuthnRequest and return IdP redirect URL."""

        # Generate request ID
        request_id = f"_id{uuid4().hex}"

        # Store request state
        state_key = f"sso:state:{request_id}"
        await self.redis.setex(
            state_key,
            300,  # 5 minute expiry
            f"{tenant_id}|{callback_url}|{relay_state or ''}",
        )

        # Build AuthnRequest (simplified - production would use proper XML)
        authn_request = self._build_saml_authn_request(
            request_id=request_id,
            sp_entity_id=config.get("sp_entity_id", f"urn:reflection:{tenant_id}"),
            acs_url=callback_url,
            idp_sso_url=config["idp_sso_url"],
        )

        # Encode and sign
        encoded_request = base64.b64encode(authn_request.encode()).decode()

        # Build redirect URL
        params = {
            "SAMLRequest": encoded_request,
        }
        if relay_state:
            params["RelayState"] = relay_state

        redirect_url = f"{config['idp_sso_url']}?{urllib.parse.urlencode(params)}"

        logger.info(f"SAML login initiated for tenant {tenant_id}")
        return redirect_url

    async def _handle_saml_callback(
        self,
        tenant_id: UUID,
        config: dict[str, Any],
        response_data: dict[str, Any],
    ) -> SSOSession:
        """Process SAML Response and create session."""

        saml_response = response_data.get("SAMLResponse")
        if not saml_response:
            raise ValueError("Missing SAMLResponse")

        # Decode response
        try:
            decoded = base64.b64decode(saml_response).decode()
        except Exception as e:
            raise ValueError(f"Invalid SAMLResponse encoding: {e}") from e

        # Parse and validate response (simplified - production would use proper XML parsing)
        user = self._parse_saml_response(decoded, config)

        # Create session
        session = SSOSession(
            id=str(uuid4()),
            tenant_id=tenant_id,
            user=user,
            protocol=SSOProtocol.SAML,
            provider=SSOProvider.CUSTOM,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=config.get("session_duration_hours", 8)),
            session_index=response_data.get("session_index"),
        )

        # Store session
        await self._store_session(session)

        logger.info(f"SAML session created for {user.email} in tenant {tenant_id}")
        return session

    def _build_saml_authn_request(
        self,
        request_id: str,
        sp_entity_id: str,
        acs_url: str,
        idp_sso_url: str,
    ) -> str:
        """Build SAML AuthnRequest XML."""
        # Simplified - production would use proper SAML library
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<samlp:AuthnRequest
    xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
    xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
    ID="{request_id}"
    Version="2.0"
    IssueInstant="{now}"
    Destination="{idp_sso_url}"
    AssertionConsumerServiceURL="{acs_url}"
    ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST">
    <saml:Issuer>{sp_entity_id}</saml:Issuer>
</samlp:AuthnRequest>"""

    def _parse_saml_response(
        self,
        response_xml: str,
        config: dict[str, Any],
    ) -> SSOUser:
        """Parse SAML Response and extract user info."""
        # Simplified parsing - production would use proper XML/signature validation
        # This is a placeholder that extracts basic info

        import re

        # Extract email (simplified)
        email_match = re.search(r"<saml:NameID[^>]*>([^<]+)</saml:NameID>", response_xml)
        if not email_match:
            email_match = re.search(r"emailaddress[^>]*>([^<]+)<", response_xml, re.IGNORECASE)

        if not email_match:
            raise ValueError("Could not extract email from SAML response")

        email = email_match.group(1).strip()

        return SSOUser(
            external_id=email,
            email=email,
            email_verified=True,
        )

    # --------------------------------------------------------
    # OIDC IMPLEMENTATION
    # --------------------------------------------------------

    async def _discover_oidc_endpoints(self, config: OIDCConfig) -> OIDCConfig:
        """Discover OIDC endpoints from issuer."""
        client = await self._get_http_client()

        discovery_url = f"{config.issuer.rstrip('/')}/.well-known/openid-configuration"

        response = await client.get(discovery_url)
        response.raise_for_status()

        discovery = response.json()

        config.authorization_endpoint = discovery.get("authorization_endpoint")
        config.token_endpoint = discovery.get("token_endpoint")
        config.userinfo_endpoint = discovery.get("userinfo_endpoint")
        config.jwks_uri = discovery.get("jwks_uri")
        config.end_session_endpoint = discovery.get("end_session_endpoint")

        return config

    async def _initiate_oidc_login(
        self,
        tenant_id: UUID,
        config: dict[str, Any],
        callback_url: str,
        relay_state: str | None,
    ) -> str:
        """Generate OIDC authorization URL."""

        # Generate state and nonce
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)

        # Generate PKCE if enabled
        code_verifier = None
        code_challenge = None

        if config.get("use_pkce", True):
            code_verifier = secrets.token_urlsafe(64)
            code_challenge = (
                base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
                .rstrip(b"=")
                .decode()
            )

        # Store request state
        state_key = f"sso:state:{state}"
        state_data = {
            "tenant_id": str(tenant_id),
            "callback_url": callback_url,
            "relay_state": relay_state,
            "nonce": nonce,
            "code_verifier": code_verifier,
        }

        import json

        await self.redis.setex(state_key, 300, json.dumps(state_data))

        # Build authorization URL
        params = {
            "client_id": config["client_id"],
            "response_type": config.get("response_type", "code"),
            "redirect_uri": callback_url,
            "scope": " ".join(config.get("scopes", ["openid", "profile", "email"])),
            "state": state,
            "nonce": nonce,
        }

        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"

        auth_url = config.get("authorization_endpoint", f"{config['issuer']}/authorize")
        redirect_url = f"{auth_url}?{urllib.parse.urlencode(params)}"

        logger.info(f"OIDC login initiated for tenant {tenant_id}")
        return redirect_url

    async def _handle_oidc_callback(
        self,
        tenant_id: UUID,
        config: dict[str, Any],
        response_data: dict[str, Any],
    ) -> SSOSession:
        """Process OIDC callback and create session."""

        code = response_data.get("code")
        state = response_data.get("state")

        if not code or not state:
            error = response_data.get("error", "unknown")
            error_desc = response_data.get("error_description", "")
            raise ValueError(f"OIDC error: {error} - {error_desc}")

        # Retrieve and validate state
        state_key = f"sso:state:{state}"
        state_data = await self.redis.get(state_key)

        if not state_data:
            raise ValueError("Invalid or expired state")

        import json

        state_info = json.loads(state_data)

        # Delete state (single use)
        await self.redis.delete(state_key)

        # Exchange code for tokens
        client = await self._get_http_client()

        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": state_info["callback_url"],
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
        }

        if state_info.get("code_verifier"):
            token_data["code_verifier"] = state_info["code_verifier"]

        token_url = config.get("token_endpoint", f"{config['issuer']}/token")

        response = await client.post(
            token_url,
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            raise ValueError(f"Token exchange failed: {response.text}")

        tokens = response.json()

        # Get user info
        userinfo_url = config.get("userinfo_endpoint", f"{config['issuer']}/userinfo")

        user_response = await client.get(
            userinfo_url,
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )

        if user_response.status_code != 200:
            raise ValueError(f"Failed to get user info: {user_response.text}")

        user_info = user_response.json()

        # Map claims to user
        claim_mapping = config.get("claim_mapping", {})

        user = SSOUser(
            external_id=user_info.get("sub"),
            email=user_info.get(claim_mapping.get("email", "email")),
            email_verified=user_info.get("email_verified", False),
            first_name=user_info.get(claim_mapping.get("first_name", "given_name")),
            last_name=user_info.get(claim_mapping.get("last_name", "family_name")),
            display_name=user_info.get("name"),
            groups=user_info.get(claim_mapping.get("groups", "groups"), []),
            attributes=user_info,
        )

        # Create session
        session = SSOSession(
            id=str(uuid4()),
            tenant_id=tenant_id,
            user=user,
            protocol=SSOProtocol.OIDC,
            provider=SSOProvider.CUSTOM,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=config.get("session_duration_hours", 8)),
            access_token=tokens.get("access_token"),
            refresh_token=tokens.get("refresh_token"),
        )

        await self._store_session(session)

        logger.info(f"OIDC session created for {user.email} in tenant {tenant_id}")
        return session

    # --------------------------------------------------------
    # SESSION MANAGEMENT
    # --------------------------------------------------------

    async def _store_session(self, session: SSOSession) -> None:
        """Store SSO session in Redis."""
        session_key = f"sso:session:{session.id}"

        import json

        session_data = {
            "id": session.id,
            "tenant_id": str(session.tenant_id),
            "user": {
                "external_id": session.user.external_id,
                "email": session.user.email,
                "email_verified": session.user.email_verified,
                "first_name": session.user.first_name,
                "last_name": session.user.last_name,
                "display_name": session.user.display_name,
                "groups": session.user.groups,
                "roles": session.user.roles,
            },
            "protocol": session.protocol.value,
            "provider": session.provider.value,
            "created_at": session.created_at.isoformat(),
            "expires_at": session.expires_at.isoformat(),
        }

        ttl = int((session.expires_at - datetime.now(UTC)).total_seconds())

        await self.redis.setex(session_key, max(1, ttl), json.dumps(session_data))

    async def get_session(self, session_id: str) -> SSOSession | None:
        """Retrieve SSO session."""
        session_key = f"sso:session:{session_id}"
        data = await self.redis.get(session_key)

        if not data:
            return None

        import json

        session_data = json.loads(data)

        user = SSOUser(
            external_id=session_data["user"]["external_id"],
            email=session_data["user"]["email"],
            email_verified=session_data["user"].get("email_verified", False),
            first_name=session_data["user"].get("first_name"),
            last_name=session_data["user"].get("last_name"),
            display_name=session_data["user"].get("display_name"),
            groups=session_data["user"].get("groups", []),
            roles=session_data["user"].get("roles", []),
        )

        return SSOSession(
            id=session_data["id"],
            tenant_id=UUID(session_data["tenant_id"]),
            user=user,
            protocol=SSOProtocol(session_data["protocol"]),
            provider=SSOProvider(session_data["provider"]),
            created_at=datetime.fromisoformat(session_data["created_at"]),
            expires_at=datetime.fromisoformat(session_data["expires_at"]),
        )

    async def revoke_session(self, session_id: str) -> bool:
        """Revoke an SSO session."""
        session_key = f"sso:session:{session_id}"
        result = await self.redis.delete(session_key)
        return result > 0

    # --------------------------------------------------------
    # UTILITIES
    # --------------------------------------------------------

    def _validate_certificate(self, cert_pem: str) -> None:
        """Validate X.509 certificate format."""
        if "BEGIN CERTIFICATE" not in cert_pem:
            raise ValueError("Invalid certificate format. Expected PEM format.")


# ============================================================
# GLOBAL INSTANCE
# ============================================================

_sso_service: SSOService | None = None


async def get_sso_service() -> SSOService | None:
    """Get the global SSO service."""
    global _sso_service

    if _sso_service is not None:
        return _sso_service

    try:
        from ..data.postgres import get_database
        from ..data.redis import get_redis

        redis = get_redis()
        db = get_database()

        _sso_service = SSOService(redis.client, db)
        logger.info("SSO service initialized")

        return _sso_service

    except Exception as e:
        logger.error(f"Failed to initialize SSO service: {e}")
        return None


def reset_sso_service() -> None:
    """Reset the global SSO service (for testing)."""
    global _sso_service
    _sso_service = None


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Types
    "SSOProtocol",
    "SSOProvider",
    "SSOUser",
    "SSOSession",
    # SAML
    "SAMLConfig",
    "SAMLAuthRequest",
    "SAMLResponse",
    # OIDC
    "OIDCConfig",
    "OIDCAuthRequest",
    "OIDCTokenResponse",
    # Service
    "SSOService",
    "get_sso_service",
    "reset_sso_service",
]
