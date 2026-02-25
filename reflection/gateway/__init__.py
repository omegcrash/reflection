# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

from .app import app, create_app
from .auth import (
    AuthTokens,
    JWTService,
    PasswordService,
    TokenExpiredError,
    TokenInvalidError,
    TokenPayload,
    TokenRevokedError,
    TokenType,
    get_jwt_service,
    get_password_service,
)
from .request_context import (
    RequestContext,
    RequestContextFilter,
    RequestContextMiddleware,
    get_request_context,
    get_request_context_dependency,
    get_request_id,
    get_request_id_dependency,
    set_context_extra,
    set_tenant_context,
)
from .token_store import (
    SessionInfo,
    TokenReuseError,
    TokenStore,
    get_token_store,
)

__all__ = [
    # App
    "app",
    "create_app",
    # Auth
    "PasswordService",
    "JWTService",
    "TokenType",
    "TokenPayload",
    "AuthTokens",
    "TokenExpiredError",
    "TokenInvalidError",
    "TokenRevokedError",
    "get_password_service",
    "get_jwt_service",
    # Token Store (v1.2.0)
    "TokenStore",
    "SessionInfo",
    "TokenReuseError",
    "get_token_store",
    # Request Context (v1.4.0)
    "RequestContext",
    "RequestContextMiddleware",
    "RequestContextFilter",
    "get_request_id",
    "get_request_context",
    "set_tenant_context",
    "set_context_extra",
    "get_request_id_dependency",
    "get_request_context_dependency",
]
