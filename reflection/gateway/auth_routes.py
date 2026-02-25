# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Authentication Routes

Endpoints for:
- User registration
- Login (password-based)
- Token refresh
- Password reset
- Email verification
"""

import logging
from datetime import datetime
from uuid import UUID

import jwt
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..data.models import TenantModel, TenantUserModel
from ..data.postgres import get_db_session
from ..data.repositories import TenantRepository, TenantUserRepository
from .auth import (
    JWTService,
    PasswordService,
    TokenExpiredError,
    TokenInvalidError,
    TokenType,
    get_jwt_service,
    get_password_service,
)
from .rate_limit import (
    get_client_ip,
    get_login_rate_limiter,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================


class RegisterRequest(BaseModel):
    """User registration request."""

    email: EmailStr
    password: str = Field(..., min_length=8)
    display_name: str | None = None

    # For first user, can create tenant
    tenant_name: str | None = None
    tenant_slug: str | None = Field(
        default=None, pattern=r"^[a-z0-9-]+$", min_length=1, max_length=63
    )


class LoginRequest(BaseModel):
    """Login request."""

    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response."""

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    """Token refresh request."""

    refresh_token: str


class PasswordResetRequest(BaseModel):
    """Request password reset email."""

    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Confirm password reset with token."""

    token: str
    new_password: str = Field(..., min_length=8)


class ChangePasswordRequest(BaseModel):
    """Change password (authenticated)."""

    current_password: str
    new_password: str = Field(..., min_length=8)


class UserProfileResponse(BaseModel):
    """User profile response."""

    id: str
    email: str
    display_name: str | None
    role: str
    trust_level: str
    tenant_id: str
    tenant_name: str
    email_verified: bool
    mfa_enabled: bool
    created_at: datetime


# ============================================================
# DEPENDENCIES
# ============================================================


async def get_current_user(
    authorization: str | None = Header(None),
    session: AsyncSession = Depends(get_db_session),
    jwt_service: JWTService = Depends(get_jwt_service),
) -> tuple[TenantUserModel, TenantModel]:
    """
    Dependency to get current authenticated user from JWT.

    Returns:
        (user, tenant) tuple
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization[7:]

    try:
        payload = jwt_service.decode_token(token, TokenType.ACCESS)
    except TokenExpiredError as e:
        raise HTTPException(
            status_code=401,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except TokenInvalidError as e:
        raise HTTPException(
            status_code=401,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    # Get user and tenant
    user_repo = TenantUserRepository(session)
    tenant_repo = TenantRepository(session)

    try:
        user_id = UUID(payload.sub)
        tenant_id = UUID(payload.tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=401, detail="Invalid token payload") from e

    user = await user_repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="User is deactivated")

    tenant = await tenant_repo.get_by_id(tenant_id)
    if not tenant:
        raise HTTPException(status_code=401, detail="Tenant not found")

    if tenant.status != "active":
        raise HTTPException(status_code=403, detail=f"Tenant is {tenant.status}")

    return user, tenant


async def get_optional_user(
    authorization: str | None = Header(None),
    session: AsyncSession = Depends(get_db_session),
    jwt_service: JWTService = Depends(get_jwt_service),
) -> tuple[TenantUserModel, TenantModel] | None:
    """
    Dependency to optionally get current user.

    Returns None if no valid token provided.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None

    try:
        return await get_current_user(authorization, session, jwt_service)
    except HTTPException:
        return None


# ============================================================
# ROUTES
# ============================================================


@router.post("/register", response_model=TokenResponse)
async def register(
    request: RegisterRequest,
    x_tenant_id: str | None = Header(None, alias="X-Tenant-ID"),
    session: AsyncSession = Depends(get_db_session),
    password_service: PasswordService = Depends(get_password_service),
    jwt_service: JWTService = Depends(get_jwt_service),
):
    """
    Register a new user.

    If X-Tenant-ID is provided, adds user to existing tenant.
    If tenant_name and tenant_slug are provided, creates new tenant.
    """
    tenant_repo = TenantRepository(session)
    user_repo = TenantUserRepository(session)

    # Validate password strength
    is_valid, error = password_service.validate_password_strength(request.password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)

    # Determine tenant
    tenant = None
    is_owner = False

    if x_tenant_id:
        # Join existing tenant
        tenant = await tenant_repo.get_by_slug(x_tenant_id)
        if not tenant:
            raise HTTPException(status_code=404, detail="Tenant not found")
        if tenant.status != "active":
            raise HTTPException(status_code=403, detail=f"Tenant is {tenant.status}")

    elif request.tenant_name and request.tenant_slug:
        # Create new tenant
        existing = await tenant_repo.get_by_slug(request.tenant_slug)
        if existing:
            raise HTTPException(status_code=409, detail="Tenant slug already exists")

        tenant = await tenant_repo.create(
            name=request.tenant_name,
            slug=request.tenant_slug,
            tier="free",
            status="active",
            config={},
            quotas={"max_users": 5, "max_agents": 2},  # Free tier limits
        )
        is_owner = True

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide X-Tenant-ID to join existing tenant, or tenant_name and tenant_slug to create new",
        )

    # Check if user exists
    existing_user = await user_repo.get_by_email(tenant.id, request.email)
    if existing_user:
        raise HTTPException(status_code=409, detail="User with this email already exists")

    # Check user quota
    user_count = await user_repo.count_by_tenant(tenant.id)
    max_users = tenant.quotas.get("max_users", 100)
    if user_count >= max_users:
        raise HTTPException(status_code=403, detail=f"Tenant user limit reached ({max_users})")

    # Hash password
    password_hash = password_service.hash_password(request.password)

    # Create user
    user = await user_repo.create(
        tenant_id=tenant.id,
        email=request.email,
        password_hash=password_hash,
        display_name=request.display_name,
        role="owner" if is_owner else "member",
        trust_level="owner" if is_owner else "known",
    )

    await session.commit()

    logger.info(f"User registered: {user.email} in tenant {tenant.slug}")

    # Generate tokens
    tokens = jwt_service.create_tokens(
        user_id=str(user.id),
        tenant_id=str(tenant.id),
        email=user.email,
        role=user.role,
        trust_level=user.trust_level,
    )

    return TokenResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        token_type=tokens.token_type,
        expires_in=tokens.expires_in,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    credentials: LoginRequest,
    x_tenant_id: str = Header(..., alias="X-Tenant-ID"),
    session: AsyncSession = Depends(get_db_session),
    password_service: PasswordService = Depends(get_password_service),
    jwt_service: JWTService = Depends(get_jwt_service),
):
    """
    Login with email and password.

    Requires X-Tenant-ID header to identify the tenant.

    Rate limited to prevent brute force attacks:
    - Per IP: 10 attempts per 5 minutes
    - Per email: 5 attempts per 5 minutes
    - Lockout: 15 minutes after threshold
    """
    # Get client IP for rate limiting
    ip_address = get_client_ip(request)
    email = credentials.email.lower().strip()

    # Check rate limit BEFORE any database queries (prevents timing attacks)
    rate_limiter = await get_login_rate_limiter()
    if rate_limiter:
        rate_result = await rate_limiter.check_login_allowed(ip_address, email)

        if not rate_result.allowed:
            logger.warning(
                f"Login rate limited: ip={ip_address}, email={email[:3]}***, "
                f"retry_after={rate_result.retry_after_seconds}s"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many login attempts. Please try again later.",
                headers={"Retry-After": str(rate_result.retry_after_seconds)},
            )

    tenant_repo = TenantRepository(session)
    user_repo = TenantUserRepository(session)

    # Get tenant
    tenant = await tenant_repo.get_by_slug(x_tenant_id)
    if not tenant:
        # Record failure before returning error
        if rate_limiter:
            await rate_limiter.record_failure(ip_address, email)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if tenant.status != "active":
        raise HTTPException(status_code=403, detail=f"Tenant is {tenant.status}")

    # Get user
    user = await user_repo.get_by_email(tenant.id, email)
    if not user:
        # Record failure - use generic error to prevent user enumeration
        if rate_limiter:
            await rate_limiter.record_failure(ip_address, email)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user.is_active:
        # Record failure for deactivated accounts too
        if rate_limiter:
            await rate_limiter.record_failure(ip_address, email)
        raise HTTPException(status_code=403, detail="User is deactivated")

    if not user.password_hash:
        raise HTTPException(status_code=401, detail="Password not set. Use SSO or reset password.")

    # Verify password
    if not password_service.verify_password(credentials.password, user.password_hash):
        # Record failure BEFORE returning error
        if rate_limiter:
            await rate_limiter.record_failure(ip_address, email)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # SUCCESS - Clear rate limit counters
    if rate_limiter:
        await rate_limiter.record_success(ip_address, email)

    # Check if password needs rehash
    if password_service.needs_rehash(user.password_hash):
        new_hash = password_service.hash_password(credentials.password)
        await user_repo.set_password(user.id, new_hash)

    # Update last login
    await user_repo.update_last_login(user.id)
    await session.commit()

    logger.info(f"User logged in: {user.email} in tenant {tenant.slug} from {ip_address}")

    # Generate tokens with session tracking (v1.2.0)
    user_agent = request.headers.get("User-Agent")
    tokens = await jwt_service.create_tokens_with_session(
        user_id=str(user.id),
        tenant_id=str(tenant.id),
        email=user.email,
        role=user.role,
        trust_level=user.trust_level,
        ip_address=ip_address,
        user_agent=user_agent,
    )

    return TokenResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        token_type=tokens.token_type,
        expires_in=tokens.expires_in,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    http_request: Request,
    request: RefreshRequest,
    jwt_service: JWTService = Depends(get_jwt_service),
):
    """
    Refresh access token using refresh token.

    Returns new access and refresh tokens (token rotation).

    v1.2.0 Changes:
    - Refresh tokens are now single-use (rotation)
    - Old refresh token is revoked when new one is issued
    - Token reuse triggers automatic revocation of all sessions
    """
    from .auth import TokenRevokedError as AuthTokenRevokedError
    from .rate_limit import get_client_ip
    from .token_store import TokenReuseError

    ip_address = get_client_ip(http_request)
    user_agent = http_request.headers.get("User-Agent")

    try:
        tokens = await jwt_service.refresh_access_token_async(
            request.refresh_token,
            ip_address=ip_address,
            user_agent=user_agent,
        )
    except TokenExpiredError as e:
        raise HTTPException(status_code=401, detail="Refresh token has expired") from e
    except AuthTokenRevokedError as e:
        raise HTTPException(status_code=401, detail="Refresh token has been revoked") from e
    except TokenReuseError as e:
        # Token reuse detected - all sessions revoked
        raise HTTPException(
            status_code=401,
            detail=str(e),
            headers={"X-Session-Revoked": "true"},
        ) from e
    except TokenInvalidError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e

    return TokenResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        token_type=tokens.token_type,
        expires_in=tokens.expires_in,
    )


@router.get("/me", response_model=UserProfileResponse)
async def get_current_user_profile(
    auth: tuple = Depends(get_current_user),
):
    """Get current user's profile."""
    user, tenant = auth

    return UserProfileResponse(
        id=str(user.id),
        email=user.email,
        display_name=user.display_name,
        role=user.role,
        trust_level=user.trust_level,
        tenant_id=str(tenant.id),
        tenant_name=tenant.name,
        email_verified=user.email_verified,
        mfa_enabled=user.mfa_enabled,
        created_at=user.created_at,
    )


@router.post("/password/change")
async def change_password(
    request: ChangePasswordRequest,
    auth: tuple = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    password_service: PasswordService = Depends(get_password_service),
):
    """Change password (requires current password)."""
    user, tenant = auth
    user_repo = TenantUserRepository(session)

    # Verify current password
    if not user.password_hash:
        raise HTTPException(status_code=400, detail="No password set")

    if not password_service.verify_password(request.current_password, user.password_hash):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    # Validate new password
    is_valid, error = password_service.validate_password_strength(request.new_password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)

    # Hash and save new password
    new_hash = password_service.hash_password(request.new_password)
    await user_repo.set_password(user.id, new_hash)
    await session.commit()

    logger.info(f"Password changed for user: {user.email}")

    return {"status": "success", "message": "Password changed successfully"}


@router.post("/password/reset/request")
async def request_password_reset(
    request: PasswordResetRequest,
    x_tenant_id: str = Header(..., alias="X-Tenant-ID"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    session: AsyncSession = Depends(get_db_session),
    jwt_service: JWTService = Depends(get_jwt_service),
):
    """
    Request a password reset email.

    Always returns success to prevent email enumeration.
    """
    tenant_repo = TenantRepository(session)
    user_repo = TenantUserRepository(session)

    tenant = await tenant_repo.get_by_slug(x_tenant_id)
    if tenant:
        user = await user_repo.get_by_email(tenant.id, request.email)
        if user and user.is_active:
            # Generate reset token
            reset_token = jwt_service.create_password_reset_token(
                user_id=str(user.id),
                tenant_id=str(tenant.id),
            )

            # TODO: Send email in background
            # background_tasks.add_task(send_password_reset_email, user.email, reset_token)

            logger.info(f"Password reset requested for: {user.email}")

            # For development, log the token
            logger.debug(f"Reset token (dev): {reset_token}")

    # Always return success
    return {
        "status": "success",
        "message": "If an account exists with this email, a reset link will be sent",
    }


@router.post("/password/reset/confirm")
async def confirm_password_reset(
    request: PasswordResetConfirm,
    session: AsyncSession = Depends(get_db_session),
    password_service: PasswordService = Depends(get_password_service),
    jwt_service: JWTService = Depends(get_jwt_service),
):
    """
    Confirm password reset with token.
    """
    try:
        payload = jwt_service.decode_token(request.token, TokenType.PASSWORD_RESET)
    except TokenExpiredError as e:
        raise HTTPException(status_code=400, detail="Reset token has expired") from e
    except TokenInvalidError as e:
        raise HTTPException(status_code=400, detail="Invalid reset token") from e

    # Validate new password
    is_valid, error = password_service.validate_password_strength(request.new_password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)

    # Get user
    user_repo = TenantUserRepository(session)

    try:
        user_id = UUID(payload.sub)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid token") from e

    user = await user_repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    # Hash and save new password
    new_hash = password_service.hash_password(request.new_password)
    await user_repo.set_password(user.id, new_hash)
    await session.commit()

    logger.info(f"Password reset completed for user: {user.email}")

    return {"status": "success", "message": "Password has been reset"}


@router.post("/email/verify/request")
async def request_email_verification(
    auth: tuple = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    jwt_service: JWTService = Depends(get_jwt_service),
):
    """
    Request an email verification link.
    """
    user, tenant = auth

    if user.email_verified:
        return {"status": "success", "message": "Email is already verified"}

    # Generate verification token
    verify_token = jwt_service.create_email_verification_token(
        user_id=str(user.id),
        tenant_id=str(tenant.id),
        email=user.email,
    )

    # TODO: Send email in background
    # background_tasks.add_task(send_verification_email, user.email, verify_token)

    logger.info(f"Email verification requested for: {user.email}")
    logger.debug(f"Verification token (dev): {verify_token}")

    return {"status": "success", "message": "Verification email sent"}


@router.post("/email/verify/confirm")
async def confirm_email_verification(
    token: str,
    session: AsyncSession = Depends(get_db_session),
    jwt_service: JWTService = Depends(get_jwt_service),
):
    """
    Confirm email verification with token.
    """
    try:
        payload = jwt_service.decode_token(token, TokenType.EMAIL_VERIFICATION)
    except TokenExpiredError as e:
        raise HTTPException(status_code=400, detail="Verification token has expired") from e
    except TokenInvalidError as e:
        raise HTTPException(status_code=400, detail="Invalid verification token") from e

    # Get user
    user_repo = TenantUserRepository(session)

    try:
        user_id = UUID(payload.sub)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid token") from e

    user = await user_repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    # Verify the email matches
    if payload.email != user.email:
        raise HTTPException(status_code=400, detail="Email mismatch")

    # Mark email as verified
    await user_repo.verify_email(user.id)
    await session.commit()

    logger.info(f"Email verified for user: {user.email}")

    return {"status": "success", "message": "Email has been verified"}


# ============================================================
# SESSION MANAGEMENT (v1.2.0)
# ============================================================


class SessionResponse(BaseModel):
    """Session information response."""

    session_id: str
    created_at: datetime
    last_used_at: datetime
    expires_at: datetime
    ip_address: str | None
    user_agent: str | None
    device_name: str | None
    is_current: bool


class SessionListResponse(BaseModel):
    """List of sessions response."""

    sessions: list[SessionResponse]
    total: int


class LogoutResponse(BaseModel):
    """Logout response."""

    status: str
    message: str
    sessions_revoked: int = 0


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    authorization: str = Header(...),
    jwt_service: JWTService = Depends(get_jwt_service),
):
    """
    List all active sessions for the current user.

    Returns a list of active sessions including device info and last activity.
    The current session is marked with is_current=True.

    v1.2.0: Session management endpoints.
    """
    from .auth import TokenRevokedError as AuthTokenRevokedError
    from .token_store import get_token_store

    # Validate token
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization[7:]
    token_store = await get_token_store()

    try:
        payload = await jwt_service.decode_token_async(
            token,
            TokenType.ACCESS,
            check_revocation=True,
            token_store=token_store,
        )
    except TokenExpiredError as e:
        raise HTTPException(status_code=401, detail="Token has expired") from e
    except (TokenInvalidError, AuthTokenRevokedError) as e:
        raise HTTPException(status_code=401, detail="Invalid or revoked token") from e

    current_jti = payload.jti

    # Get all sessions for this user
    sessions = await token_store.get_user_sessions(payload.sub)

    session_list = []
    for sess in sessions:
        session_list.append(
            SessionResponse(
                session_id=sess.jti,
                created_at=sess.created_at,
                last_used_at=sess.last_used_at,
                expires_at=sess.expires_at,
                ip_address=sess.ip_address,
                user_agent=sess.user_agent,
                device_name=sess.device_name,
                is_current=(sess.jti == current_jti),
            )
        )

    return SessionListResponse(
        sessions=session_list,
        total=len(session_list),
    )


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    authorization: str = Header(...),
    jwt_service: JWTService = Depends(get_jwt_service),
):
    """
    Logout the current session.

    Revokes the current access and refresh tokens. The user will need
    to login again to get new tokens.

    v1.2.0: Session management endpoints.
    """
    from .token_store import get_token_store

    # Validate token
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization[7:]
    token_store = await get_token_store()

    try:
        # Don't check revocation - allow logout even with revoked token
        payload = jwt_service.decode_token(token, TokenType.ACCESS)
    except TokenExpiredError:
        # Allow logout with expired token
        try:
            payload = jwt.decode(
                token,
                jwt_service.secret_key,
                algorithms=[jwt_service.algorithm],
                options={"verify_exp": False},
            )
            jti = payload.get("jti")
            if jti:
                # Still try to revoke
                await token_store.revoke_session(jti, reason="logout")
        except Exception:
            pass
        return LogoutResponse(
            status="success",
            message="Session ended (token was already expired)",
            sessions_revoked=1,
        )
    except TokenInvalidError as e:
        raise HTTPException(status_code=401, detail="Invalid token") from e

    # Revoke the session
    if payload.jti:
        await token_store.revoke_session(payload.jti, reason="logout")

    logger.info(
        f"User logged out: {payload.sub[:8]}... jti={payload.jti[:8] if payload.jti else 'none'}..."
    )

    return LogoutResponse(
        status="success",
        message="Successfully logged out",
        sessions_revoked=1,
    )


@router.post("/logout-all", response_model=LogoutResponse)
async def logout_all(
    authorization: str = Header(...),
    keep_current: bool = False,
    jwt_service: JWTService = Depends(get_jwt_service),
):
    """
    Logout all sessions for the current user.

    Revokes all active sessions, forcing re-authentication on all devices.
    Use this if you suspect your account has been compromised.

    Query Parameters:
        keep_current: If True, keeps the current session active (default: False)

    v1.2.0: Session management endpoints.
    """
    from .auth import TokenRevokedError as AuthTokenRevokedError
    from .token_store import get_token_store

    # Validate token
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization[7:]
    token_store = await get_token_store()

    try:
        payload = await jwt_service.decode_token_async(
            token,
            TokenType.ACCESS,
            check_revocation=True,
            token_store=token_store,
        )
    except TokenExpiredError as e:
        raise HTTPException(status_code=401, detail="Token has expired") from e
    except (TokenInvalidError, AuthTokenRevokedError) as e:
        raise HTTPException(status_code=401, detail="Invalid or revoked token") from e

    # Revoke all sessions
    exclude_jti = payload.jti if keep_current else None
    revoked_count = await token_store.revoke_all_user_sessions(
        payload.sub,
        reason="logout_all",
        exclude_jti=exclude_jti,
    )

    logger.info(
        f"User logged out of all sessions: {payload.sub[:8]}... "
        f"revoked={revoked_count} keep_current={keep_current}"
    )

    if keep_current:
        message = f"Logged out of {revoked_count} other session(s). Current session kept active."
    else:
        message = f"Logged out of all {revoked_count} session(s)."

    return LogoutResponse(
        status="success",
        message=message,
        sessions_revoked=revoked_count,
    )


@router.delete("/sessions/{session_id}", response_model=LogoutResponse)
async def revoke_session(
    session_id: str,
    authorization: str = Header(...),
    jwt_service: JWTService = Depends(get_jwt_service),
):
    """
    Revoke a specific session by ID.

    Use this to remotely log out a specific device.

    Path Parameters:
        session_id: The session ID (JTI) to revoke

    v1.2.0: Session management endpoints.
    """
    from .auth import TokenRevokedError as AuthTokenRevokedError
    from .token_store import get_token_store

    # Validate token
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization[7:]
    token_store = await get_token_store()

    try:
        payload = await jwt_service.decode_token_async(
            token,
            TokenType.ACCESS,
            check_revocation=True,
            token_store=token_store,
        )
    except TokenExpiredError as e:
        raise HTTPException(status_code=401, detail="Token has expired") from e
    except (TokenInvalidError, AuthTokenRevokedError) as e:
        raise HTTPException(status_code=401, detail="Invalid or revoked token") from e

    # Verify the session belongs to this user
    target_session = await token_store.get_session(session_id)
    if not target_session:
        raise HTTPException(status_code=404, detail="Session not found")

    if target_session.user_id != payload.sub:
        raise HTTPException(status_code=403, detail="Cannot revoke another user's session")

    # Revoke the session
    await token_store.revoke_session(session_id, reason="remote_logout")

    is_current = session_id == payload.jti

    logger.info(
        f"Session revoked: jti={session_id[:8]}... by user={payload.sub[:8]}... "
        f"is_current={is_current}"
    )

    return LogoutResponse(
        status="success",
        message="Session revoked" + (" (you have been logged out)" if is_current else ""),
        sessions_revoked=1,
    )
