# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Token Store

Redis-backed token storage for:
- Token revocation (blacklisting)
- Session management
- Refresh token rotation tracking

v1.2.0 Authentication Enhancement - Token revocation and session management.
"""

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Information about an active session."""

    jti: str
    user_id: str
    tenant_id: str
    created_at: datetime
    last_used_at: datetime
    expires_at: datetime
    ip_address: str | None = None
    user_agent: str | None = None
    device_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "jti": self.jti,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "device_name": self.device_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionInfo":
        """Create from dictionary."""
        return cls(
            jti=data["jti"],
            user_id=data["user_id"],
            tenant_id=data["tenant_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_used_at=datetime.fromisoformat(data["last_used_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            device_name=data.get("device_name"),
        )

    def to_api_response(self) -> dict[str, Any]:
        """Convert to API response format."""
        return {
            "session_id": self.jti,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "device_name": self.device_name,
            "is_current": False,  # Set by caller
        }


class TokenStore:
    """
    Redis-backed token store for revocation and session management.

    Key Schema:
    - tokens:revoked:{jti} -> "1" (TTL = token expiry)
    - tokens:sessions:{user_id} -> Set of active JTIs
    - tokens:session:{jti} -> Session metadata JSON
    - tokens:refresh:{jti} -> Refresh token metadata (for rotation)
    - tokens:refresh_chain:{original_jti} -> Current JTI in rotation chain

    Usage:
        store = TokenStore(redis)

        # Register new session
        await store.register_session(user_id, jti, session_info)

        # Check if token is revoked
        if await store.is_revoked(jti):
            raise TokenRevokedError()

        # Revoke single token
        await store.revoke_token(jti, expires_at)

        # Revoke all user sessions (logout everywhere)
        await store.revoke_all_user_tokens(user_id)
    """

    def __init__(
        self,
        redis: Redis,
        key_prefix: str = "tokens",
        default_session_ttl: int = 7 * 24 * 3600,  # 7 days
    ):
        """
        Initialize token store.

        Args:
            redis: Redis connection
            key_prefix: Prefix for all Redis keys
            default_session_ttl: Default TTL for session data (seconds)
        """
        self.redis = redis
        self.prefix = key_prefix
        self.default_session_ttl = default_session_ttl

    def _key(self, *parts: str) -> str:
        """Build a Redis key from parts."""
        return ":".join([self.prefix] + list(parts))

    # =========================================================
    # TOKEN REVOCATION
    # =========================================================

    async def revoke_token(
        self,
        jti: str,
        expires_at: datetime,
        reason: str | None = None,
    ) -> bool:
        """
        Add a token to the revocation list.

        The revocation entry is stored with a TTL matching the token's
        expiry time, so it auto-cleans after the token would have
        expired anyway.

        Args:
            jti: JWT ID to revoke
            expires_at: When the token expires (for TTL calculation)
            reason: Optional reason for revocation (logged)

        Returns:
            True if revoked, False if already revoked
        """
        key = self._key("revoked", jti)

        # Calculate TTL (token expiry - now)
        now = datetime.now(UTC)
        ttl = int((expires_at - now).total_seconds())

        if ttl <= 0:
            # Token already expired, no need to store
            logger.debug(f"Token {jti[:8]}... already expired, skipping revocation storage")
            return True

        # Store revocation with TTL
        was_new = await self.redis.set(key, reason or "revoked", ex=ttl, nx=True)

        if was_new:
            logger.info(f"Token revoked: {jti[:8]}... reason={reason}")
        else:
            logger.debug(f"Token {jti[:8]}... was already revoked")

        return bool(was_new)

    async def is_revoked(self, jti: str) -> bool:
        """
        Check if a token has been revoked.

        Args:
            jti: JWT ID to check

        Returns:
            True if revoked, False if valid
        """
        key = self._key("revoked", jti)
        return await self.redis.exists(key) > 0

    async def get_revocation_reason(self, jti: str) -> str | None:
        """Get the reason a token was revoked."""
        key = self._key("revoked", jti)
        reason = await self.redis.get(key)
        return reason.decode() if reason else None

    # =========================================================
    # SESSION MANAGEMENT
    # =========================================================

    async def register_session(
        self,
        user_id: str,
        tenant_id: str,
        jti: str,
        expires_at: datetime,
        ip_address: str | None = None,
        user_agent: str | None = None,
        device_name: str | None = None,
    ) -> SessionInfo:
        """
        Register a new session (when tokens are created).

        Args:
            user_id: User ID
            tenant_id: Tenant ID
            jti: JWT ID (session identifier)
            expires_at: When the refresh token expires
            ip_address: Client IP address
            user_agent: Client user agent string
            device_name: Optional device name

        Returns:
            SessionInfo object
        """
        now = datetime.now(UTC)

        session = SessionInfo(
            jti=jti,
            user_id=user_id,
            tenant_id=tenant_id,
            created_at=now,
            last_used_at=now,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=self._truncate_user_agent(user_agent),
            device_name=device_name,
        )

        # Calculate TTL
        ttl = int((expires_at - now).total_seconds())
        if ttl <= 0:
            ttl = self.default_session_ttl

        # Store session data and add to user's session set
        session_key = self._key("session", jti)
        sessions_key = self._key("sessions", user_id)

        pipe = self.redis.pipeline()
        pipe.set(session_key, json.dumps(session.to_dict()), ex=ttl)
        pipe.sadd(sessions_key, jti)
        pipe.expire(sessions_key, self.default_session_ttl)
        await pipe.execute()

        logger.info(f"Session registered: user={user_id[:8]}... jti={jti[:8]}... ip={ip_address}")

        return session

    async def get_session(self, jti: str) -> SessionInfo | None:
        """Get session information by JTI."""
        key = self._key("session", jti)
        data = await self.redis.get(key)

        if not data:
            return None

        return SessionInfo.from_dict(json.loads(data))

    async def update_session_activity(self, jti: str) -> bool:
        """
        Update last_used_at timestamp for a session.

        Call this when a token is used successfully.

        Returns:
            True if session exists and was updated
        """
        session = await self.get_session(jti)
        if not session:
            return False

        session.last_used_at = datetime.now(UTC)

        # Calculate remaining TTL
        ttl = await self.redis.ttl(self._key("session", jti))
        if ttl <= 0:
            ttl = self.default_session_ttl

        await self.redis.set(
            self._key("session", jti),
            json.dumps(session.to_dict()),
            ex=ttl,
        )

        return True

    async def get_user_sessions(
        self,
        user_id: str,
        include_expired: bool = False,
    ) -> list[SessionInfo]:
        """
        Get all active sessions for a user.

        Args:
            user_id: User ID
            include_expired: Include sessions that may have expired

        Returns:
            List of SessionInfo objects
        """
        sessions_key = self._key("sessions", user_id)
        jtis = await self.redis.smembers(sessions_key)

        if not jtis:
            return []

        sessions = []
        expired_jtis = []
        now = datetime.now(UTC)

        for jti in jtis:
            jti_str = jti.decode() if isinstance(jti, bytes) else jti
            session = await self.get_session(jti_str)

            if session:
                if include_expired or session.expires_at > now:
                    sessions.append(session)
                else:
                    expired_jtis.append(jti_str)
            else:
                expired_jtis.append(jti_str)

        # Clean up expired JTIs from the set
        if expired_jtis:
            await self.redis.srem(sessions_key, *expired_jtis)

        # Sort by created_at descending (newest first)
        sessions.sort(key=lambda s: s.created_at, reverse=True)

        return sessions

    async def revoke_session(
        self,
        jti: str,
        reason: str = "logout",
    ) -> bool:
        """
        Revoke a single session.

        Args:
            jti: Session JTI to revoke
            reason: Reason for revocation

        Returns:
            True if session was found and revoked
        """
        session = await self.get_session(jti)
        if not session:
            return False

        # Add to revocation list
        await self.revoke_token(jti, session.expires_at, reason)

        # Remove session data
        session_key = self._key("session", jti)
        sessions_key = self._key("sessions", session.user_id)

        pipe = self.redis.pipeline()
        pipe.delete(session_key)
        pipe.srem(sessions_key, jti)
        await pipe.execute()

        logger.info(f"Session revoked: jti={jti[:8]}... reason={reason}")

        return True

    async def revoke_all_user_sessions(
        self,
        user_id: str,
        reason: str = "logout_all",
        exclude_jti: str | None = None,
    ) -> int:
        """
        Revoke all sessions for a user (logout everywhere).

        Args:
            user_id: User ID
            reason: Reason for revocation
            exclude_jti: Optional JTI to exclude (keep current session)

        Returns:
            Number of sessions revoked
        """
        sessions = await self.get_user_sessions(user_id, include_expired=True)

        if not sessions:
            return 0

        revoked_count = 0

        for session in sessions:
            if exclude_jti and session.jti == exclude_jti:
                continue

            await self.revoke_token(session.jti, session.expires_at, reason)
            await self.redis.delete(self._key("session", session.jti))
            revoked_count += 1

        # Clear the sessions set (or just the excluded one remains)
        sessions_key = self._key("sessions", user_id)
        if exclude_jti:
            # Remove all except the excluded one
            jtis_to_remove = [s.jti for s in sessions if s.jti != exclude_jti]
            if jtis_to_remove:
                await self.redis.srem(sessions_key, *jtis_to_remove)
        else:
            await self.redis.delete(sessions_key)

        logger.info(
            f"All sessions revoked for user {user_id[:8]}...: count={revoked_count} reason={reason}"
        )

        return revoked_count

    # =========================================================
    # REFRESH TOKEN ROTATION
    # =========================================================

    async def register_refresh_token(
        self,
        jti: str,
        user_id: str,
        tenant_id: str,
        expires_at: datetime,
        parent_jti: str | None = None,
    ) -> None:
        """
        Register a refresh token for rotation tracking.

        When a refresh token is used, we revoke it and issue a new one.
        This tracks the chain to detect token reuse attacks.

        Args:
            jti: New refresh token JTI
            user_id: User ID
            tenant_id: Tenant ID
            expires_at: Token expiry
            parent_jti: JTI of the token being rotated (if any)
        """
        now = datetime.now(UTC)
        ttl = int((expires_at - now).total_seconds())
        if ttl <= 0:
            ttl = self.default_session_ttl

        # Store refresh token metadata
        refresh_key = self._key("refresh", jti)
        metadata = {
            "jti": jti,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "created_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "parent_jti": parent_jti,
            "used": False,
        }

        await self.redis.set(refresh_key, json.dumps(metadata), ex=ttl)

        # If this is a rotation, update the chain
        if parent_jti:
            self._key("refresh_chain", parent_jti)
            # Check if parent still exists (detect reuse)
            parent_data = await self.redis.get(self._key("refresh", parent_jti))
            if parent_data:
                parent_meta = json.loads(parent_data)
                if parent_meta.get("used"):
                    # TOKEN REUSE DETECTED - revoke entire chain
                    logger.warning(
                        f"Refresh token reuse detected! parent={parent_jti[:8]}... "
                        f"Revoking all sessions for user {user_id[:8]}..."
                    )
                    await self.revoke_all_user_sessions(user_id, "token_reuse_detected")
                    raise TokenReuseError(
                        "Refresh token has already been used. "
                        "All sessions have been revoked for security."
                    )

                # Mark parent as used
                parent_meta["used"] = True
                parent_meta["rotated_to"] = jti
                await self.redis.set(
                    self._key("refresh", parent_jti),
                    json.dumps(parent_meta),
                    ex=ttl,
                )

    async def mark_refresh_token_used(self, jti: str) -> bool:
        """
        Mark a refresh token as used (before rotation).

        Returns:
            True if token was valid and marked, False if already used
        """
        refresh_key = self._key("refresh", jti)
        data = await self.redis.get(refresh_key)

        if not data:
            return False

        metadata = json.loads(data)

        if metadata.get("used"):
            return False

        metadata["used"] = True
        metadata["used_at"] = datetime.now(UTC).isoformat()

        # Keep same TTL
        ttl = await self.redis.ttl(refresh_key)
        if ttl <= 0:
            ttl = 3600  # 1 hour buffer

        await self.redis.set(refresh_key, json.dumps(metadata), ex=ttl)

        return True

    # =========================================================
    # UTILITIES
    # =========================================================

    @staticmethod
    def _truncate_user_agent(user_agent: str | None, max_length: int = 200) -> str | None:
        """Truncate user agent to reasonable length."""
        if not user_agent:
            return None
        return user_agent[:max_length] if len(user_agent) > max_length else user_agent

    async def cleanup_expired_sessions(self, user_id: str) -> int:
        """
        Clean up expired sessions for a user.

        Returns:
            Number of expired sessions removed
        """
        sessions = await self.get_user_sessions(user_id, include_expired=True)
        now = datetime.now(UTC)

        expired = [s for s in sessions if s.expires_at <= now]

        if not expired:
            return 0

        sessions_key = self._key("sessions", user_id)
        jtis = [s.jti for s in expired]

        pipe = self.redis.pipeline()
        for jti in jtis:
            pipe.delete(self._key("session", jti))
        pipe.srem(sessions_key, *jtis)
        await pipe.execute()

        return len(expired)

    async def get_session_count(self, user_id: str) -> int:
        """Get the number of active sessions for a user."""
        sessions = await self.get_user_sessions(user_id)
        return len(sessions)


class TokenReuseError(Exception):
    """Raised when refresh token reuse is detected."""

    pass


# ============================================================
# DEPENDENCY INJECTION
# ============================================================

_token_store: TokenStore | None = None


async def get_token_store() -> TokenStore:
    """
    Get the global token store instance.

    Usage:
        token_store = await get_token_store()
    """
    global _token_store

    if _token_store is None:
        from ..data.redis import get_redis

        redis_client = await get_redis()
        _token_store = TokenStore(redis_client)

    return _token_store


async def reset_token_store():
    """Reset the global token store (for testing)."""
    global _token_store
    _token_store = None


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "TokenStore",
    "SessionInfo",
    "TokenReuseError",
    "get_token_store",
    "reset_token_store",
]
