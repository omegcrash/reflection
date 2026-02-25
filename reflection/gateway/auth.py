# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Authentication Service

Provides:
- Password hashing (bcrypt)
- JWT token generation and validation
- Refresh token flow
- Session management
"""

import logging
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Optional

import jwt
from fastapi import Header, HTTPException, status
from passlib.context import CryptContext

from ..core.settings import SecuritySettings, get_settings

if TYPE_CHECKING:
    from .token_store import TokenStore

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,  # OWASP recommended minimum
)


class TokenType(StrEnum):
    """Types of JWT tokens."""

    ACCESS = "access"
    REFRESH = "refresh"
    PASSWORD_RESET = "password_reset"
    EMAIL_VERIFICATION = "email_verification"


@dataclass
class TokenPayload:
    """Decoded JWT token payload."""

    sub: str  # Subject (user ID)
    tenant_id: str
    token_type: TokenType
    exp: datetime
    iat: datetime
    jti: str | None = None  # JWT ID for revocation

    # Optional claims
    email: str | None = None
    role: str | None = None
    trust_level: str | None = None
    permissions: list | None = None


@dataclass
class AuthTokens:
    """Pair of access and refresh tokens."""

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int = 0  # Seconds until access token expires


class AuthenticationError(Exception):
    """Authentication failed."""

    pass


class TokenExpiredError(AuthenticationError):
    """Token has expired."""

    pass


class TokenInvalidError(AuthenticationError):
    """Token is invalid."""

    pass


class TokenRevokedError(AuthenticationError):
    """Token has been revoked."""

    pass


class PasswordService:
    """
    Password hashing and verification.

    Uses bcrypt with 12 rounds (OWASP recommended).
    """

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password for storage.

        Args:
            password: Plaintext password

        Returns:
            Bcrypt hash string
        """
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            password: Plaintext password to verify
            password_hash: Stored bcrypt hash

        Returns:
            True if password matches
        """
        try:
            return pwd_context.verify(password, password_hash)
        except Exception:
            return False

    @staticmethod
    def needs_rehash(password_hash: str) -> bool:
        """
        Check if a password hash needs to be rehashed.

        This happens when bcrypt rounds are increased.
        """
        return pwd_context.needs_update(password_hash)

    @staticmethod
    def generate_reset_token() -> str:
        """Generate a secure password reset token."""
        return secrets.token_urlsafe(32)

    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, str | None]:
        """
        Validate password meets security requirements.

        Requirements:
        - Minimum 8 characters
        - At least one uppercase
        - At least one lowercase
        - At least one digit
        - At least one special character

        Returns:
            (is_valid, error_message)
        """
        if len(password) < 8:
            return False, "Password must be at least 8 characters"

        if not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"

        if not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"

        if not any(c.isdigit() for c in password):
            return False, "Password must contain at least one digit"

        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        if not any(c in special_chars for c in password):
            return False, "Password must contain at least one special character"

        return True, None


class JWTService:
    """
    JWT token generation and validation.

    Usage:
        jwt_service = JWTService(settings)

        # Generate tokens
        tokens = jwt_service.create_tokens(user_id, tenant_id)

        # Validate token
        payload = jwt_service.decode_token(tokens.access_token)
    """

    # Known insecure default values that must be rejected
    INSECURE_SECRETS = frozenset(
        {
            "your-secret-key",
            "changeme",
            "secret",
            "password",
            "jwt-secret",
            "supersecret",
            "development",
            "dev-secret",
            "test-secret",
            "example-secret",
            "placeholder",
            "fixme",
            "change-me",
            "replace-me",
            "your_secret_key",
            "mysecret",
        }
    )

    def __init__(self, settings: SecuritySettings | None = None):
        self.settings = settings or get_settings().security
        self.algorithm = self.settings.jwt_algorithm
        self.secret_key = self.settings.jwt_secret_key
        self.access_token_expire = timedelta(hours=1)
        self.refresh_token_expire = timedelta(days=7)

        # Validate secret key at initialization
        self._validate_secret_key()

    def _validate_secret_key(self) -> None:
        """
        Validate JWT secret key meets security requirements.

        Requirements:
        - Must be present (not None or empty)
        - Must be at least 32 characters
        - Must not be a known insecure default

        Raises:
            ValueError: If secret key fails validation
        """
        if not self.secret_key:
            raise ValueError(
                "JWT_SECRET_KEY environment variable is required. "
                'Generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )

        if len(self.secret_key) < 32:
            raise ValueError(
                f"JWT_SECRET_KEY must be at least 32 characters for security "
                f"(current length: {len(self.secret_key)}). "
                'Generate a secure key with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )

        # Check against known insecure defaults (case-insensitive)
        if self.secret_key.lower() in self.INSECURE_SECRETS:
            raise ValueError(
                f"JWT_SECRET_KEY is using an insecure default value '{self.secret_key[:8]}...'. "
                "This is a critical security risk. "
                'Generate a secure key with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )

        # Warn about low entropy (optional but helpful)
        unique_chars = len(set(self.secret_key))
        if unique_chars < 10:
            logger.warning(
                f"JWT_SECRET_KEY has low entropy ({unique_chars} unique characters). "
                "Consider using a more random key for production."
            )

    def create_access_token(
        self,
        user_id: str,
        tenant_id: str,
        email: str | None = None,
        role: str | None = None,
        trust_level: str | None = None,
        permissions: list | None = None,
        expires_delta: timedelta | None = None,
    ) -> str:
        """
        Create an access token.

        Access tokens are short-lived (1 hour default).
        """
        now = datetime.now(UTC)
        expire = now + (expires_delta or self.access_token_expire)

        payload = {
            "sub": user_id,
            "tenant_id": tenant_id,
            "type": TokenType.ACCESS.value,
            "iat": now,
            "exp": expire,
            "jti": secrets.token_hex(16),
        }

        if email:
            payload["email"] = email
        if role:
            payload["role"] = role
        if trust_level:
            payload["trust_level"] = trust_level
        if permissions:
            payload["permissions"] = permissions

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(
        self,
        user_id: str,
        tenant_id: str,
        expires_delta: timedelta | None = None,
    ) -> str:
        """
        Create a refresh token.

        Refresh tokens are longer-lived (7 days default).
        """
        now = datetime.now(UTC)
        expire = now + (expires_delta or self.refresh_token_expire)

        payload = {
            "sub": user_id,
            "tenant_id": tenant_id,
            "type": TokenType.REFRESH.value,
            "iat": now,
            "exp": expire,
            "jti": secrets.token_hex(16),
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_tokens(
        self,
        user_id: str,
        tenant_id: str,
        email: str | None = None,
        role: str | None = None,
        trust_level: str | None = None,
        permissions: list | None = None,
    ) -> AuthTokens:
        """
        Create both access and refresh tokens.
        """
        access_token = self.create_access_token(
            user_id=user_id,
            tenant_id=tenant_id,
            email=email,
            role=role,
            trust_level=trust_level,
            permissions=permissions,
        )

        refresh_token = self.create_refresh_token(
            user_id=user_id,
            tenant_id=tenant_id,
        )

        return AuthTokens(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=int(self.access_token_expire.total_seconds()),
        )

    def create_password_reset_token(
        self,
        user_id: str,
        tenant_id: str,
        expires_delta: timedelta = timedelta(hours=1),
    ) -> str:
        """Create a password reset token (1 hour expiry)."""
        now = datetime.now(UTC)
        expire = now + expires_delta

        payload = {
            "sub": user_id,
            "tenant_id": tenant_id,
            "type": TokenType.PASSWORD_RESET.value,
            "iat": now,
            "exp": expire,
            "jti": secrets.token_hex(16),
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_email_verification_token(
        self,
        user_id: str,
        tenant_id: str,
        email: str,
        expires_delta: timedelta = timedelta(days=1),
    ) -> str:
        """Create an email verification token (24 hour expiry)."""
        now = datetime.now(UTC)
        expire = now + expires_delta

        payload = {
            "sub": user_id,
            "tenant_id": tenant_id,
            "email": email,
            "type": TokenType.EMAIL_VERIFICATION.value,
            "iat": now,
            "exp": expire,
            "jti": secrets.token_hex(16),
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def decode_token(
        self,
        token: str,
        expected_type: TokenType | None = None,
    ) -> TokenPayload:
        """
        Decode and validate a JWT token.

        Args:
            token: JWT string
            expected_type: If provided, validates token type

        Returns:
            TokenPayload with decoded claims

        Raises:
            TokenExpiredError: Token has expired
            TokenInvalidError: Token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
            )
        except jwt.ExpiredSignatureError as e:
            raise TokenExpiredError("Token has expired") from e
        except jwt.InvalidTokenError as e:
            raise TokenInvalidError(f"Invalid token: {e}") from e

        # Validate token type
        token_type = TokenType(payload.get("type", "access"))
        if expected_type and token_type != expected_type:
            raise TokenInvalidError(f"Expected {expected_type.value} token, got {token_type.value}")

        return TokenPayload(
            sub=payload["sub"],
            tenant_id=payload["tenant_id"],
            token_type=token_type,
            exp=datetime.fromtimestamp(payload["exp"], tz=UTC),
            iat=datetime.fromtimestamp(payload["iat"], tz=UTC),
            jti=payload.get("jti"),
            email=payload.get("email"),
            role=payload.get("role"),
            trust_level=payload.get("trust_level"),
            permissions=payload.get("permissions"),
        )

    def refresh_access_token(self, refresh_token: str) -> AuthTokens:
        """
        Use a refresh token to get new access and refresh tokens.

        Note: This creates a new refresh token (rotation).

        DEPRECATED: Use refresh_access_token_async for revocation support.
        """
        payload = self.decode_token(refresh_token, TokenType.REFRESH)

        return self.create_tokens(
            user_id=payload.sub,
            tenant_id=payload.tenant_id,
        )

    async def decode_token_async(
        self,
        token: str,
        expected_type: TokenType | None = None,
        check_revocation: bool = True,
        token_store: Optional["TokenStore"] = None,
    ) -> TokenPayload:
        """
        Decode and validate a JWT token with revocation checking.

        This is the async version that integrates with TokenStore for
        revocation checking. Use this for all production token validation.

        Args:
            token: JWT string
            expected_type: If provided, validates token type
            check_revocation: Whether to check token revocation (default: True)
            token_store: TokenStore instance (fetched automatically if None)

        Returns:
            TokenPayload with decoded claims

        Raises:
            TokenExpiredError: Token has expired
            TokenInvalidError: Token is invalid
            TokenRevokedError: Token has been revoked
        """
        # First, do the synchronous decode
        payload = self.decode_token(token, expected_type)

        # Then check revocation asynchronously
        if check_revocation and payload.jti:
            if token_store is None:
                from .token_store import get_token_store

                token_store = await get_token_store()

            if await token_store.is_revoked(payload.jti):
                reason = await token_store.get_revocation_reason(payload.jti)
                logger.warning(f"Revoked token used: jti={payload.jti[:8]}... reason={reason}")
                raise TokenRevokedError("Token has been revoked")

            # Update session activity (non-blocking)
            try:
                await token_store.update_session_activity(payload.jti)
            except Exception as e:
                logger.debug(f"Failed to update session activity: {e}")

        return payload

    async def refresh_access_token_async(
        self,
        refresh_token: str,
        token_store: Optional["TokenStore"] = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> AuthTokens:
        """
        Use a refresh token to get new tokens with proper rotation.

        This method:
        1. Validates the refresh token
        2. Checks if it has been revoked
        3. Marks the old refresh token as used (prevents reuse)
        4. Creates new access and refresh tokens
        5. Registers the new session

        Token reuse detection: If a refresh token that has already been
        used is presented again, ALL sessions for the user are revoked
        as a security measure.

        Args:
            refresh_token: Current refresh token
            token_store: TokenStore instance (fetched automatically if None)
            ip_address: Client IP for session tracking
            user_agent: Client user agent for session tracking

        Returns:
            New AuthTokens (access + refresh)

        Raises:
            TokenExpiredError: Refresh token has expired
            TokenInvalidError: Refresh token is invalid
            TokenRevokedError: Refresh token has been revoked
            TokenReuseError: Refresh token has already been used (attack detected)
        """
        if token_store is None:
            from .token_store import get_token_store

            token_store = await get_token_store()

        # Decode and validate (includes revocation check)
        payload = await self.decode_token_async(
            refresh_token,
            TokenType.REFRESH,
            check_revocation=True,
            token_store=token_store,
        )

        old_jti = payload.jti

        # Mark the old refresh token as used
        # This is critical for detecting token reuse attacks
        if old_jti:
            was_valid = await token_store.mark_refresh_token_used(old_jti)
            if not was_valid:
                # Token was already used - potential attack!
                from .token_store import TokenReuseError

                logger.warning(
                    f"Refresh token reuse detected: jti={old_jti[:8]}... user={payload.sub[:8]}..."
                )
                # Revoke all sessions for this user
                await token_store.revoke_all_user_sessions(
                    payload.sub, reason="token_reuse_detected"
                )
                raise TokenReuseError(
                    "Refresh token has already been used. "
                    "All sessions have been revoked for security."
                )

            # Revoke the old refresh token
            await token_store.revoke_token(old_jti, payload.exp, reason="rotated")

        # Create new tokens
        new_tokens = self.create_tokens(
            user_id=payload.sub,
            tenant_id=payload.tenant_id,
        )

        # Extract new JTI from refresh token for session registration
        new_payload = self.decode_token(new_tokens.refresh_token, TokenType.REFRESH)

        # Register the new session
        await token_store.register_session(
            user_id=payload.sub,
            tenant_id=payload.tenant_id,
            jti=new_payload.jti,
            expires_at=new_payload.exp,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        # Register the refresh token for rotation tracking
        await token_store.register_refresh_token(
            jti=new_payload.jti,
            user_id=payload.sub,
            tenant_id=payload.tenant_id,
            expires_at=new_payload.exp,
            parent_jti=old_jti,
        )

        logger.info(
            f"Token rotated: old_jti={old_jti[:8] if old_jti else 'none'}... "
            f"new_jti={new_payload.jti[:8]}... user={payload.sub[:8]}..."
        )

        return new_tokens

    async def create_tokens_with_session(
        self,
        user_id: str,
        tenant_id: str,
        email: str | None = None,
        role: str | None = None,
        trust_level: str | None = None,
        permissions: list | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        token_store: Optional["TokenStore"] = None,
    ) -> AuthTokens:
        """
        Create tokens and register the session.

        Use this for initial login to set up proper session tracking.

        Args:
            user_id: User ID
            tenant_id: Tenant ID
            email: User email
            role: User role
            trust_level: Trust level
            permissions: List of permissions
            ip_address: Client IP address
            user_agent: Client user agent
            token_store: TokenStore instance (fetched automatically if None)

        Returns:
            AuthTokens with access and refresh tokens
        """
        if token_store is None:
            from .token_store import get_token_store

            token_store = await get_token_store()

        # Create tokens
        tokens = self.create_tokens(
            user_id=user_id,
            tenant_id=tenant_id,
            email=email,
            role=role,
            trust_level=trust_level,
            permissions=permissions,
        )

        # Extract JTI from refresh token
        payload = self.decode_token(tokens.refresh_token, TokenType.REFRESH)

        # Register session
        await token_store.register_session(
            user_id=user_id,
            tenant_id=tenant_id,
            jti=payload.jti,
            expires_at=payload.exp,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        # Register refresh token for rotation tracking
        await token_store.register_refresh_token(
            jti=payload.jti,
            user_id=user_id,
            tenant_id=tenant_id,
            expires_at=payload.exp,
        )

        logger.info(
            f"Session created: jti={payload.jti[:8]}... user={user_id[:8]}... ip={ip_address}"
        )

        return tokens


# ============================================================
# GLOBAL INSTANCES
# ============================================================

_password_service: PasswordService | None = None
_jwt_service: JWTService | None = None


def get_password_service() -> PasswordService:
    """Get the global password service."""
    global _password_service
    if _password_service is None:
        _password_service = PasswordService()
    return _password_service


def get_jwt_service() -> JWTService:
    """Get the global JWT service."""
    global _jwt_service
    if _jwt_service is None:
        _jwt_service = JWTService()
    return _jwt_service


# ============================================================
# FASTAPI DEPENDENCIES
# ============================================================


async def get_current_user(
    authorization: str | None = Header(None),
) -> TokenPayload:
    """FastAPI dependency: extract and validate JWT from Authorization header.

    Usage in routes:
        current_user: TokenPayload = Depends(get_current_user)
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract Bearer token
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Use: Bearer <token>",
        )

    token = parts[1]
    jwt_service = get_jwt_service()

    try:
        payload = jwt_service.decode_token(token)
        return payload
    except TokenExpiredError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        ) from e
    except (TokenInvalidError, TokenRevokedError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked token",
        ) from e


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "TokenType",
    "TokenPayload",
    "AuthTokens",
    "AuthenticationError",
    "TokenExpiredError",
    "TokenInvalidError",
    "PasswordService",
    "JWTService",
    "get_password_service",
    "get_jwt_service",
    "get_current_user",
]
