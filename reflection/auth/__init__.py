# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

from .sso import (
    OIDCAuthRequest,
    # OIDC
    OIDCConfig,
    OIDCTokenResponse,
    SAMLAuthRequest,
    # SAML
    SAMLConfig,
    SAMLResponse,
    # Types
    SSOProtocol,
    SSOProvider,
    # Service
    SSOService,
    SSOSession,
    SSOUser,
    get_sso_service,
    reset_sso_service,
)

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
