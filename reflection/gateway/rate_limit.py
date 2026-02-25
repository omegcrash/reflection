# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Rate Limiting Module

Provides Redis-backed rate limiting for:
- Login attempts (per IP and per email)
- API requests (per tenant)
- Password reset requests

Features:
- Sliding window algorithm
- Progressive lockout after failures
- Distributed across multiple instances
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import Request

logger = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int
    limit: int
    reset_at: datetime
    retry_after_seconds: int | None = None

    @property
    def is_locked_out(self) -> bool:
        """Check if this is a lockout (vs just rate limited)."""
        return self.retry_after_seconds is not None and self.retry_after_seconds > 60


class RateLimiter:
    """
    Redis-backed rate limiter using sliding window algorithm.

    Usage:
        limiter = RateLimiter(redis_client)

        result = await limiter.check(
            key="login:192.168.1.1",
            limit=5,
            window_seconds=60
        )

        if not result.allowed:
            raise HTTPException(
                status_code=429,
                detail="Too many attempts",
                headers={"Retry-After": str(result.retry_after_seconds)}
            )
    """

    def __init__(self, redis):
        """
        Initialize rate limiter.

        Args:
            redis: aioredis client instance
        """
        self.redis = redis

    async def check(self, key: str, limit: int, window_seconds: int = 60) -> RateLimitResult:
        """
        Check if request is within rate limit.

        Uses sliding window log algorithm for accuracy.

        Args:
            key: Unique identifier (e.g., "login:ip:192.168.1.1")
            limit: Maximum requests allowed in window
            window_seconds: Time window in seconds

        Returns:
            RateLimitResult with allowed status and metadata
        """
        now = time.time()
        window_start = now - window_seconds
        rate_key = f"ratelimit:{key}"

        pipe = self.redis.pipeline()

        # Remove entries older than window
        pipe.zremrangebyscore(rate_key, "-inf", window_start)

        # Count current entries in window
        pipe.zcard(rate_key)

        # Add current request with timestamp as score
        request_id = f"{now}:{id(self)}"
        pipe.zadd(rate_key, {request_id: now})

        # Set key expiry (cleanup)
        pipe.expire(rate_key, window_seconds + 10)

        results = await pipe.execute()
        current_count = results[1]

        reset_at = datetime.fromtimestamp(now + window_seconds, tz=UTC)

        if current_count >= limit:
            # Over limit - remove the entry we just added
            await self.redis.zrem(rate_key, request_id)

            # Calculate when oldest entry expires
            oldest = await self.redis.zrange(rate_key, 0, 0, withscores=True)
            if oldest:
                oldest_time = oldest[0][1]
                retry_after = int(oldest_time + window_seconds - now) + 1
            else:
                retry_after = window_seconds

            logger.warning(f"Rate limit exceeded for {key}: {current_count}/{limit}")

            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=limit,
                reset_at=reset_at,
                retry_after_seconds=max(1, retry_after),
            )

        return RateLimitResult(
            allowed=True,
            remaining=limit - current_count - 1,
            limit=limit,
            reset_at=reset_at,
        )

    async def reset(self, key: str):
        """Reset rate limit for a key (e.g., after successful login)."""
        await self.redis.delete(f"ratelimit:{key}")


class LoginRateLimiter:
    """
    Specialized rate limiter for login attempts.

    Features:
    - Per-IP rate limiting (more lenient for shared IPs)
    - Per-email rate limiting (protects individual accounts)
    - Progressive lockout after failures
    - Automatic reset after successful login

    Usage:
        limiter = LoginRateLimiter(redis_client)

        # Before login attempt
        result = await limiter.check_login_allowed(ip, email)
        if not result.allowed:
            raise HTTPException(429, detail="Too many attempts")

        # After failed login
        await limiter.record_failure(ip, email)

        # After successful login
        await limiter.record_success(ip, email)
    """

    def __init__(
        self,
        redis,
        max_attempts_per_ip: int = 10,
        max_attempts_per_email: int = 5,
        window_minutes: int = 5,
        lockout_minutes: int = 15,
    ):
        """
        Initialize login rate limiter.

        Args:
            redis: aioredis client instance
            max_attempts_per_ip: Max failures per IP before lockout
            max_attempts_per_email: Max failures per email before lockout
            window_minutes: Time window for counting failures
            lockout_minutes: How long to lock out after threshold
        """
        self.redis = redis
        self.base_limiter = RateLimiter(redis)
        self.max_attempts_per_ip = max_attempts_per_ip
        self.max_attempts_per_email = max_attempts_per_email
        self.window_seconds = window_minutes * 60
        self.lockout_seconds = lockout_minutes * 60

    def _hash_email(self, email: str) -> str:
        """Hash email for privacy in Redis keys."""
        return hashlib.sha256(email.lower().strip().encode()).hexdigest()[:16]

    async def check_login_allowed(
        self, ip_address: str, email: str | None = None
    ) -> RateLimitResult:
        """
        Check if a login attempt is allowed.

        Checks both IP lockout and email lockout.

        Args:
            ip_address: Client IP address
            email: Email being used (optional, for email-specific limiting)

        Returns:
            RateLimitResult indicating if login can proceed
        """
        now = time.time()

        # Check IP lockout
        ip_lockout_key = f"lockout:ip:{ip_address}"
        ip_lockout = await self.redis.get(ip_lockout_key)

        if ip_lockout:
            ttl = await self.redis.ttl(ip_lockout_key)
            logger.warning(f"IP {ip_address} is locked out for {ttl}s")
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=self.max_attempts_per_ip,
                reset_at=datetime.fromtimestamp(now + ttl, tz=UTC),
                retry_after_seconds=max(1, ttl),
            )

        # Check email lockout (if email provided)
        if email:
            email_hash = self._hash_email(email)
            email_lockout_key = f"lockout:email:{email_hash}"
            email_lockout = await self.redis.get(email_lockout_key)

            if email_lockout:
                ttl = await self.redis.ttl(email_lockout_key)
                logger.warning(f"Email {email[:3]}*** is locked out for {ttl}s")
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    limit=self.max_attempts_per_email,
                    reset_at=datetime.fromtimestamp(now + ttl, tz=UTC),
                    retry_after_seconds=max(1, ttl),
                )

        # Check IP rate limit (more lenient)
        ip_result = await self.base_limiter.check(
            key=f"login:ip:{ip_address}",
            limit=self.max_attempts_per_ip * 2,  # 2x before hard limit
            window_seconds=self.window_seconds,
        )

        if not ip_result.allowed:
            # Trigger IP lockout
            await self.redis.setex(ip_lockout_key, self.lockout_seconds, "1")
            logger.warning(f"IP {ip_address} locked out due to rate limit")
            ip_result.retry_after_seconds = self.lockout_seconds
            return ip_result

        return ip_result

    async def record_failure(self, ip_address: str, email: str | None = None):
        """
        Record a failed login attempt.

        Increments failure counters and may trigger lockout.

        Args:
            ip_address: Client IP address
            email: Email that was attempted (optional)
        """
        # Increment IP failure count
        ip_fail_key = f"login:fail:ip:{ip_address}"
        ip_failures = await self.redis.incr(ip_fail_key)
        await self.redis.expire(ip_fail_key, self.window_seconds)

        # Check for IP lockout threshold
        if ip_failures >= self.max_attempts_per_ip:
            await self.redis.setex(f"lockout:ip:{ip_address}", self.lockout_seconds, "1")
            logger.warning(f"IP {ip_address} locked out after {ip_failures} failures")

        # Increment email failure count (if provided)
        if email:
            email_hash = self._hash_email(email)
            email_fail_key = f"login:fail:email:{email_hash}"
            email_failures = await self.redis.incr(email_fail_key)
            await self.redis.expire(email_fail_key, self.window_seconds)

            # Check for email lockout threshold
            if email_failures >= self.max_attempts_per_email:
                await self.redis.setex(f"lockout:email:{email_hash}", self.lockout_seconds, "1")
                logger.warning(f"Email {email[:3]}*** locked out after {email_failures} failures")

    async def record_success(self, ip_address: str, email: str):
        """
        Record a successful login.

        Clears failure counters for the IP and email.

        Args:
            ip_address: Client IP address
            email: Email that successfully logged in
        """
        email_hash = self._hash_email(email)

        # Clear all failure-related keys
        await self.redis.delete(
            f"login:fail:ip:{ip_address}",
            f"login:fail:email:{email_hash}",
            f"ratelimit:login:ip:{ip_address}",
        )

        logger.debug(f"Login success counters cleared for {ip_address}")


def get_client_ip(request: Request) -> str:
    """
    Extract client IP from request.

    Respects X-Forwarded-For header for reverse proxy setups.

    Args:
        request: FastAPI request object

    Returns:
        Client IP address string
    """
    # Check X-Forwarded-For (set by reverse proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the first IP (original client)
        # Format: "client, proxy1, proxy2"
        return forwarded.split(",")[0].strip()

    # Check X-Real-IP (nginx convention)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Fall back to direct connection
    if request.client:
        return request.client.host

    return "unknown"


# ============================================================
# DEPENDENCY INJECTION
# ============================================================

_login_limiter: LoginRateLimiter | None = None


async def get_login_rate_limiter() -> LoginRateLimiter | None:
    """
    Get the global login rate limiter.

    Returns None if Redis is not available.
    """
    global _login_limiter

    if _login_limiter is not None:
        return _login_limiter

    try:
        from ..core.settings import get_settings
        from ..data.redis import get_redis

        settings = get_settings()
        redis_client = get_redis()

        # Check if connected
        try:
            redis = redis_client.client
        except RuntimeError:
            # Not connected yet - try to connect
            await redis_client.connect()
            redis = redis_client.client

        _login_limiter = LoginRateLimiter(
            redis=redis,
            max_attempts_per_ip=settings.security.auth_lockout_threshold * 2,
            max_attempts_per_email=settings.security.auth_lockout_threshold,
            window_minutes=5,
            lockout_minutes=settings.security.auth_lockout_duration_minutes,
        )

        logger.info(
            f"Login rate limiter initialized: "
            f"{settings.security.auth_lockout_threshold} attempts, "
            f"{settings.security.auth_lockout_duration_minutes}min lockout"
        )

        return _login_limiter

    except Exception as e:
        logger.error(f"Failed to initialize login rate limiter: {e}")
        return None


async def reset_login_limiter():
    """Reset the global login rate limiter (for testing)."""
    global _login_limiter
    _login_limiter = None


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "RateLimitResult",
    "RateLimiter",
    "LoginRateLimiter",
    "get_client_ip",
    "get_login_rate_limiter",
    "reset_login_limiter",
]
