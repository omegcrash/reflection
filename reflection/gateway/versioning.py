# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
API Versioning Infrastructure (v2.0.0)

Provides versioned API routing with:
- Multiple API versions (/api/v1/*, /api/v2/*)
- Version negotiation via headers
- Deprecation warnings
- Version-specific feature flags

Usage:
    from reflection.gateway.versioning import (
        APIVersion,
        VersionedAPIRouter,
        get_api_version,
    )

    # Create versioned router
    router = VersionedAPIRouter(prefix="/chat")

    @router.post("", versions=["v1", "v2"])
    async def chat(request: ChatRequest):
        ...

    @router.post("/stream", versions=["v2"])  # v2 only
    async def stream(request: ChatRequest):
        ...
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from functools import wraps
from typing import Any, Optional, TypeVar

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ============================================================
# VERSION DEFINITIONS
# ============================================================


class APIVersion(StrEnum):
    """Supported API versions."""

    V1 = "v1"
    V2 = "v2"

    @classmethod
    def latest(cls) -> "APIVersion":
        return cls.V2

    @classmethod
    def from_string(cls, version: str) -> Optional["APIVersion"]:
        """Parse version string."""
        version = version.lower().strip()
        if not version.startswith("v"):
            version = f"v{version}"

        try:
            return cls(version)
        except ValueError:
            return None


@dataclass
class VersionInfo:
    """Information about an API version."""

    version: APIVersion
    release_date: date
    status: str  # stable, deprecated, sunset
    sunset_date: date | None = None
    deprecation_message: str | None = None
    features: set[str] = field(default_factory=set)


# Version registry
VERSION_INFO: dict[APIVersion, VersionInfo] = {
    APIVersion.V1: VersionInfo(
        version=APIVersion.V1,
        release_date=date(2026, 2, 1),
        status="deprecated",
        sunset_date=date(2026, 12, 1),
        deprecation_message="API v1 is deprecated. Please migrate to v2 by December 2026.",
        features={"chat", "agents", "auth", "quotas"},
    ),
    APIVersion.V2: VersionInfo(
        version=APIVersion.V2,
        release_date=date(2026, 6, 1),
        status="stable",
        features={"chat", "agents", "auth", "quotas", "streaming", "jobs", "lifecycle", "sso"},
    ),
}


# ============================================================
# VERSION CONTEXT
# ============================================================


@dataclass
class VersionContext:
    """Request-scoped version context."""

    version: APIVersion
    info: VersionInfo
    negotiated: bool = False  # True if version was negotiated via header

    @property
    def is_deprecated(self) -> bool:
        return self.info.status == "deprecated"

    @property
    def is_sunset(self) -> bool:
        if self.info.sunset_date:
            return date.today() >= self.info.sunset_date
        return False

    def has_feature(self, feature: str) -> bool:
        return feature in self.info.features


# ============================================================
# VERSION NEGOTIATION
# ============================================================


def get_api_version(
    request: Request,
    accept_version: str | None = Header(
        default=None,
        alias="Accept-Version",
        description="Requested API version (e.g., 'v2')",
    ),
    x_api_version: str | None = Header(
        default=None,
        alias="X-API-Version",
        description="Requested API version (alternative header)",
    ),
) -> VersionContext:
    """
    FastAPI dependency to get API version from request.

    Version is determined by:
    1. URL path prefix (/api/v1/*, /api/v2/*)
    2. Accept-Version header
    3. X-API-Version header
    4. Default to latest

    Returns VersionContext with version info.
    """
    # Try to get from URL path
    path = request.url.path
    version_from_path = None

    for v in APIVersion:
        if f"/api/{v.value}/" in path or path.endswith(f"/api/{v.value}"):
            version_from_path = v
            break

    # Try headers
    header_version = accept_version or x_api_version
    version_from_header = None

    if header_version:
        version_from_header = APIVersion.from_string(header_version)

    # Determine final version
    if version_from_path:
        version = version_from_path
        negotiated = False
    elif version_from_header:
        version = version_from_header
        negotiated = True
    else:
        version = APIVersion.latest()
        negotiated = False

    # Get version info
    info = VERSION_INFO.get(version)
    if not info:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown API version: {version.value}",
        )

    # Check if version is sunset
    if info.status == "sunset" or (info.sunset_date and date.today() >= info.sunset_date):
        raise HTTPException(
            status_code=410,
            detail=f"API {version.value} has been sunset. Please upgrade to {APIVersion.latest().value}.",
        )

    return VersionContext(
        version=version,
        info=info,
        negotiated=negotiated,
    )


# ============================================================
# DEPRECATION MIDDLEWARE
# ============================================================


async def add_deprecation_headers(
    request: Request,
    call_next,
) -> Response:
    """
    Middleware to add deprecation headers to responses.
    """
    response = await call_next(request)

    # Get version from request state if available
    version_ctx = getattr(request.state, "api_version", None)

    if version_ctx and version_ctx.is_deprecated:
        info = version_ctx.info

        # Add deprecation headers (RFC 8594)
        response.headers["Deprecation"] = info.release_date.isoformat()

        if info.sunset_date:
            response.headers["Sunset"] = info.sunset_date.isoformat()

        if info.deprecation_message:
            response.headers["X-Deprecation-Notice"] = info.deprecation_message

        # Link to migration docs
        response.headers["Link"] = '<https://docs.example.com/api/migration>; rel="deprecation"'

    return response


# ============================================================
# VERSIONED ROUTER
# ============================================================


class VersionedAPIRouter(APIRouter):
    """
    API Router with version support.

    Routes can specify which versions they support:

        router = VersionedAPIRouter()

        @router.get("/endpoint", versions=["v1", "v2"])
        async def endpoint():
            ...

        @router.get("/new-feature", versions=["v2"])
        async def new_feature():
            ...
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._version_routes: dict[str, set[APIVersion]] = {}

    def api_route(
        self,
        path: str,
        versions: list[str] | None = None,
        **kwargs,
    ):
        """
        Decorator for versioned routes.

        Args:
            path: Route path
            versions: List of supported versions (e.g., ["v1", "v2"])
            **kwargs: Standard FastAPI route kwargs
        """
        # Default to all versions
        if versions is None:
            supported = set(APIVersion)
        else:
            supported = {APIVersion.from_string(v) for v in versions if APIVersion.from_string(v)}

        def decorator(func: F) -> F:
            # Store version info
            route_key = f"{kwargs.get('methods', ['GET'])[0]}:{path}"
            self._version_routes[route_key] = supported

            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Get version from request
                request = kwargs.get("request")
                if request:
                    version_ctx = getattr(request.state, "api_version", None)
                    if version_ctx and version_ctx.version not in supported:
                        raise HTTPException(
                            status_code=404,
                            detail=f"Endpoint not available in API {version_ctx.version.value}",
                        )

                return await func(*args, **kwargs)

            # Register route
            return super(VersionedAPIRouter, self).api_route(path, **kwargs)(wrapper)

        return decorator

    def get_version_routes(self, version: APIVersion) -> list[str]:
        """Get routes available for a specific version."""
        return [route for route, versions in self._version_routes.items() if version in versions]


# ============================================================
# VERSION-SPECIFIC BEHAVIOR
# ============================================================


def require_version(min_version: str):
    """
    Decorator to require minimum API version.

    Usage:
        @require_version("v2")
        async def new_endpoint():
            ...
    """
    min_ver = APIVersion.from_string(min_version)

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, version: VersionContext = Depends(get_api_version), **kwargs):
            if version.version.value < min_ver.value:
                raise HTTPException(
                    status_code=404,
                    detail=f"This endpoint requires API {min_ver.value} or higher",
                )
            return await func(*args, version=version, **kwargs)

        return wrapper

    return decorator


def deprecated_in(version: str, message: str | None = None):
    """
    Mark endpoint as deprecated in a specific version.

    Usage:
        @deprecated_in("v2", "Use /api/v2/chat/stream instead")
        async def old_endpoint():
            ...
    """
    dep_ver = APIVersion.from_string(version)

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, version: VersionContext = Depends(get_api_version), **kwargs):
            if version.version == dep_ver:
                logger.warning(
                    f"Deprecated endpoint called: {func.__name__} (deprecated in {version.value})"
                )
            return await func(*args, version=version, **kwargs)

        return wrapper

    return decorator


# ============================================================
# RESPONSE MODELS
# ============================================================


class VersionInfoResponse(BaseModel):
    """API version information response."""

    current_version: str
    latest_version: str
    supported_versions: list[str]
    deprecated_versions: list[str]


class VersionDetailResponse(BaseModel):
    """Detailed version information."""

    version: str
    status: str
    release_date: str
    sunset_date: str | None = None
    deprecation_message: str | None = None
    features: list[str]


# ============================================================
# VERSION INFO ENDPOINT
# ============================================================

version_router = APIRouter(tags=["Version"])


@version_router.get("/api/versions", response_model=VersionInfoResponse)
async def list_versions():
    """
    List all API versions and their status.
    """
    supported = []
    deprecated = []

    for v, info in VERSION_INFO.items():
        if info.status == "stable":
            supported.append(v.value)
        elif info.status == "deprecated":
            deprecated.append(v.value)

    return VersionInfoResponse(
        current_version=APIVersion.latest().value,
        latest_version=APIVersion.latest().value,
        supported_versions=supported + deprecated,
        deprecated_versions=deprecated,
    )


@version_router.get("/api/versions/{version}", response_model=VersionDetailResponse)
async def get_version_info(version: str):
    """
    Get detailed information about a specific API version.
    """
    api_version = APIVersion.from_string(version)

    if not api_version or api_version not in VERSION_INFO:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown API version: {version}",
        )

    info = VERSION_INFO[api_version]

    return VersionDetailResponse(
        version=info.version.value,
        status=info.status,
        release_date=info.release_date.isoformat(),
        sunset_date=info.sunset_date.isoformat() if info.sunset_date else None,
        deprecation_message=info.deprecation_message,
        features=sorted(info.features),
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Types
    "APIVersion",
    "VersionInfo",
    "VersionContext",
    # Dependencies
    "get_api_version",
    # Router
    "VersionedAPIRouter",
    "version_router",
    # Decorators
    "require_version",
    "deprecated_in",
    # Middleware
    "add_deprecation_headers",
    # Registry
    "VERSION_INFO",
]
