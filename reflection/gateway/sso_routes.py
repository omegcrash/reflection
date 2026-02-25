# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
SSO API Routes (v2.0.0)

REST API for Enterprise Single Sign-On:
- Configure SAML/OIDC for tenants
- Initiate SSO login
- Handle SSO callbacks
- Manage SSO sessions
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from ..auth.sso import (
    OIDCConfig,
    SAMLConfig,
    SSOProtocol,
    SSOService,
    SSOUser,
    get_sso_service,
)
from ..core.settings import get_settings
from .auth import TokenPayload, get_current_user

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/sso", tags=["SSO"])


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================


class SAMLConfigRequest(BaseModel):
    """Request to configure SAML for a tenant."""

    idp_entity_id: str = Field(..., description="Identity Provider Entity ID")
    idp_sso_url: str = Field(..., description="IdP Single Sign-On URL")
    idp_slo_url: str | None = Field(default=None, description="IdP Single Logout URL")
    idp_certificate: str = Field(..., description="IdP X.509 certificate (PEM)")
    attribute_mapping: dict[str, str] | None = Field(default=None)
    session_duration_hours: int = Field(default=8, ge=1, le=24)


class OIDCConfigRequest(BaseModel):
    """Request to configure OIDC for a tenant."""

    issuer: str = Field(..., description="OIDC Issuer URL")
    client_id: str = Field(..., description="Client ID")
    client_secret: str = Field(..., description="Client Secret")
    scopes: list[str] = Field(default=["openid", "profile", "email"])
    claim_mapping: dict[str, str] | None = Field(default=None)
    use_pkce: bool = Field(default=True)
    session_duration_hours: int = Field(default=8, ge=1, le=24)


class SSOConfigResponse(BaseModel):
    """SSO configuration response."""

    protocol: str
    configured: bool
    idp_entity_id: str | None = None
    issuer: str | None = None


class SSOLoginRequest(BaseModel):
    """Request to initiate SSO login."""

    callback_url: str = Field(..., description="URL to redirect after authentication")
    relay_state: str | None = Field(default=None, description="State to preserve")


class SSOLoginResponse(BaseModel):
    """SSO login initiation response."""

    redirect_url: str


class SSOUserResponse(BaseModel):
    """SSO user information."""

    external_id: str
    email: str
    email_verified: bool
    first_name: str | None = None
    last_name: str | None = None
    display_name: str | None = None
    groups: list[str] = []

    @classmethod
    def from_user(cls, user: SSOUser) -> "SSOUserResponse":
        return cls(
            external_id=user.external_id,
            email=user.email,
            email_verified=user.email_verified,
            first_name=user.first_name,
            last_name=user.last_name,
            display_name=user.display_name,
            groups=user.groups,
        )


class SSOSessionResponse(BaseModel):
    """SSO session response."""

    session_id: str
    user: SSOUserResponse
    protocol: str
    provider: str
    created_at: str
    expires_at: str


class SSOCallbackResponse(BaseModel):
    """SSO callback result."""

    success: bool
    session: SSOSessionResponse | None = None
    error: str | None = None
    jwt_token: str | None = None
    refresh_token: str | None = None


# ============================================================
# DEPENDENCIES
# ============================================================


async def get_initialized_sso_service() -> SSOService:
    """Get SSO service."""
    service = await get_sso_service()
    if service is None:
        raise HTTPException(status_code=503, detail="SSO service not available")
    return service


async def require_admin(current_user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
    """Require admin role for SSO configuration."""
    # Simplified - enhance with proper role checking
    return current_user


# ============================================================
# CONFIGURATION ROUTES
# ============================================================


@router.post("/config/saml", response_model=SSOConfigResponse)
async def configure_saml(
    request: SAMLConfigRequest,
    current_user: TokenPayload = Depends(require_admin),
    sso_service: SSOService = Depends(get_initialized_sso_service),
):
    """
    Configure SAML 2.0 for a tenant.

    Requires admin privileges.

    Example IdP certificate (PEM format):
    ```
    -----BEGIN CERTIFICATE-----
    MIICpDCCAYwCCQDU+pQ4...
    -----END CERTIFICATE-----
    ```
    """
    tenant_id = current_user.tenant_id

    try:
        config = SAMLConfig(
            idp_entity_id=request.idp_entity_id,
            idp_sso_url=request.idp_sso_url,
            idp_slo_url=request.idp_slo_url,
            idp_certificate=request.idp_certificate,
            session_duration_hours=request.session_duration_hours,
        )

        if request.attribute_mapping:
            config.attribute_mapping = request.attribute_mapping

        await sso_service.configure_saml(tenant_id, config)

        return SSOConfigResponse(
            protocol="saml",
            configured=True,
            idp_entity_id=request.idp_entity_id,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/config/oidc", response_model=SSOConfigResponse)
async def configure_oidc(
    request: OIDCConfigRequest,
    current_user: TokenPayload = Depends(require_admin),
    sso_service: SSOService = Depends(get_initialized_sso_service),
):
    """
    Configure OpenID Connect for a tenant.

    Requires admin privileges.

    The issuer URL will be used to discover endpoints via
    `{issuer}/.well-known/openid-configuration`.
    """
    tenant_id = current_user.tenant_id

    try:
        config = OIDCConfig(
            issuer=request.issuer,
            client_id=request.client_id,
            client_secret=request.client_secret,
            scopes=request.scopes,
            use_pkce=request.use_pkce,
            session_duration_hours=request.session_duration_hours,
        )

        if request.claim_mapping:
            config.claim_mapping = request.claim_mapping

        await sso_service.configure_oidc(tenant_id, config)

        return SSOConfigResponse(
            protocol="oidc",
            configured=True,
            issuer=request.issuer,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/config/{protocol}", response_model=SSOConfigResponse)
async def get_sso_config(
    protocol: str,
    current_user: TokenPayload = Depends(require_admin),
    sso_service: SSOService = Depends(get_initialized_sso_service),
):
    """
    Get SSO configuration for a tenant.

    Protocol must be 'saml' or 'oidc'.
    """
    tenant_id = current_user.tenant_id

    try:
        sso_protocol = SSOProtocol(protocol.lower())
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid protocol: {protocol}. Use 'saml' or 'oidc'."
        ) from e

    config = await sso_service.get_config(tenant_id, sso_protocol)

    if not config:
        return SSOConfigResponse(
            protocol=protocol,
            configured=False,
        )

    if sso_protocol == SSOProtocol.SAML:
        return SSOConfigResponse(
            protocol=protocol,
            configured=True,
            idp_entity_id=config.get("idp_entity_id"),
        )
    else:
        return SSOConfigResponse(
            protocol=protocol,
            configured=True,
            issuer=config.get("issuer"),
        )


@router.delete("/config/{protocol}")
async def remove_sso_config(
    protocol: str,
    current_user: TokenPayload = Depends(require_admin),
    sso_service: SSOService = Depends(get_initialized_sso_service),
):
    """
    Remove SSO configuration for a tenant.
    """
    tenant_id = current_user.tenant_id

    try:
        sso_protocol = SSOProtocol(protocol.lower())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid protocol: {protocol}") from e

    removed = await sso_service.remove_config(tenant_id, sso_protocol)

    if not removed:
        raise HTTPException(status_code=404, detail="SSO not configured")

    return {"status": "removed", "protocol": protocol}


# ============================================================
# LOGIN ROUTES
# ============================================================


@router.post("/login/{protocol}", response_model=SSOLoginResponse)
async def initiate_sso_login(
    protocol: str,
    request: SSOLoginRequest,
    tenant_id: UUID = Query(..., description="Tenant ID"),
    sso_service: SSOService = Depends(get_initialized_sso_service),
):
    """
    Initiate SSO login flow.

    Returns a redirect URL to the Identity Provider.
    The user should be redirected to this URL to complete authentication.
    """
    try:
        sso_protocol = SSOProtocol(protocol.lower())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid protocol: {protocol}") from e

    try:
        redirect_url = await sso_service.initiate_login(
            tenant_id=tenant_id,
            protocol=sso_protocol,
            callback_url=request.callback_url,
            relay_state=request.relay_state,
        )

        return SSOLoginResponse(redirect_url=redirect_url)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/login/{protocol}/redirect")
async def sso_login_redirect(
    protocol: str,
    tenant_id: UUID = Query(...),
    callback_url: str = Query(...),
    relay_state: str | None = Query(default=None),
    sso_service: SSOService = Depends(get_initialized_sso_service),
):
    """
    Convenience endpoint that redirects directly to IdP.

    Use this for browser-based SSO flows.
    """
    try:
        sso_protocol = SSOProtocol(protocol.lower())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid protocol: {protocol}") from e

    try:
        redirect_url = await sso_service.initiate_login(
            tenant_id=tenant_id,
            protocol=sso_protocol,
            callback_url=callback_url,
            relay_state=relay_state,
        )

        return RedirectResponse(url=redirect_url, status_code=302)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# ============================================================
# CALLBACK ROUTES
# ============================================================


@router.post("/callback/saml", response_model=SSOCallbackResponse)
async def saml_callback(
    request: Request,
    tenant_id: UUID = Query(...),
    sso_service: SSOService = Depends(get_initialized_sso_service),
):
    """
    Handle SAML assertion callback.

    The IdP posts the SAML Response to this endpoint.
    """
    try:
        # Get form data
        form_data = await request.form()

        response_data = {
            "SAMLResponse": form_data.get("SAMLResponse"),
            "RelayState": form_data.get("RelayState"),
        }

        if not response_data["SAMLResponse"]:
            raise ValueError("Missing SAMLResponse")

        session = await sso_service.handle_callback(
            tenant_id=tenant_id,
            protocol=SSOProtocol.SAML,
            response_data=response_data,
        )

        # Generate JWT tokens for the user
        from .auth import get_jwt_service

        jwt_service = get_jwt_service()

        tokens = jwt_service.create_tokens(
            tenant_id=str(tenant_id),
            user_id=session.user.external_id,
            email=session.user.email,
        )

        return SSOCallbackResponse(
            success=True,
            session=SSOSessionResponse(
                session_id=session.id,
                user=SSOUserResponse.from_user(session.user),
                protocol=session.protocol.value,
                provider=session.provider.value,
                created_at=session.created_at.isoformat(),
                expires_at=session.expires_at.isoformat(),
            ),
            jwt_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
        )

    except ValueError as e:
        return SSOCallbackResponse(
            success=False,
            error=str(e),
        )


@router.get("/callback/oidc", response_model=SSOCallbackResponse)
async def oidc_callback(
    request: Request,
    tenant_id: UUID = Query(...),
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
    error_description: str | None = Query(default=None),
    sso_service: SSOService = Depends(get_initialized_sso_service),
):
    """
    Handle OIDC authorization callback.

    The IdP redirects to this endpoint with an authorization code.
    """
    if error:
        return SSOCallbackResponse(
            success=False,
            error=f"{error}: {error_description or ''}",
        )

    try:
        response_data = {
            "code": code,
            "state": state,
        }

        session = await sso_service.handle_callback(
            tenant_id=tenant_id,
            protocol=SSOProtocol.OIDC,
            response_data=response_data,
        )

        # Generate JWT tokens
        from .auth import get_jwt_service

        jwt_service = get_jwt_service()

        tokens = jwt_service.create_tokens(
            tenant_id=str(tenant_id),
            user_id=session.user.external_id,
            email=session.user.email,
        )

        return SSOCallbackResponse(
            success=True,
            session=SSOSessionResponse(
                session_id=session.id,
                user=SSOUserResponse.from_user(session.user),
                protocol=session.protocol.value,
                provider=session.provider.value,
                created_at=session.created_at.isoformat(),
                expires_at=session.expires_at.isoformat(),
            ),
            jwt_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
        )

    except ValueError as e:
        return SSOCallbackResponse(
            success=False,
            error=str(e),
        )


# ============================================================
# SESSION ROUTES
# ============================================================


@router.get("/session/{session_id}")
async def get_sso_session(
    session_id: str,
    current_user: TokenPayload = Depends(get_current_user),
    sso_service: SSOService = Depends(get_initialized_sso_service),
):
    """
    Get SSO session information.
    """
    session = await sso_service.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Verify tenant
    if session.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Session not found")

    return SSOSessionResponse(
        session_id=session.id,
        user=SSOUserResponse.from_user(session.user),
        protocol=session.protocol.value,
        provider=session.provider.value,
        created_at=session.created_at.isoformat(),
        expires_at=session.expires_at.isoformat(),
    )


@router.delete("/session/{session_id}")
async def revoke_sso_session(
    session_id: str,
    current_user: TokenPayload = Depends(get_current_user),
    sso_service: SSOService = Depends(get_initialized_sso_service),
):
    """
    Revoke an SSO session (logout).
    """
    session = await sso_service.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Session not found")

    await sso_service.revoke_session(session_id)

    return {"status": "revoked", "session_id": session_id}
