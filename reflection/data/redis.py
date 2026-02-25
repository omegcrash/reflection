# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Redis connection management with in-memory fallback.

Production: connects to Redis via REDIS_URL.
Development: if redis package is unavailable or connection fails,
falls back to a simple in-memory dict that satisfies the same interface
for rate limiting, caching, and session storage.
"""

import contextlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

_redis_client: Any | None = None


class _InMemoryRedis:
    """Minimal Redis-compatible in-memory store for development.

    Supports the subset of commands used by Reflection:
    get, set, delete, exists, expire, incr, decr, keys, flushdb.
    """

    def __init__(self):
        self._store: dict[str, Any] = {}
        logger.info("Using in-memory Redis fallback (development mode)")

    async def get(self, key: str) -> bytes | None:
        val = self._store.get(key)
        if val is None:
            return None
        return val if isinstance(val, bytes) else str(val).encode()

    async def set(self, key: str, value: Any, ex: int | None = None, **kwargs) -> bool:
        self._store[key] = value if isinstance(value, bytes) else str(value).encode()
        return True

    async def setex(self, key: str, seconds: int, value: Any) -> bool:
        return await self.set(key, value, ex=seconds)

    async def delete(self, *keys: str) -> int:
        count = 0
        for k in keys:
            if k in self._store:
                del self._store[k]
                count += 1
        return count

    async def exists(self, *keys: str) -> int:
        return sum(1 for k in keys if k in self._store)

    async def expire(self, key: str, seconds: int) -> bool:
        return key in self._store

    async def incr(self, key: str) -> int:
        val = int(self._store.get(key, 0)) + 1
        self._store[key] = str(val).encode()
        return val

    async def decr(self, key: str) -> int:
        val = int(self._store.get(key, 0)) - 1
        self._store[key] = str(val).encode()
        return val

    async def keys(self, pattern: str = "*") -> list[bytes]:
        import fnmatch

        return [k.encode() for k in self._store if fnmatch.fnmatch(k, pattern)]

    async def flushdb(self) -> bool:
        self._store.clear()
        return True

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        self._store.clear()

    async def aclose(self) -> None:
        await self.close()


def _get_settings():
    from ..core.settings import get_settings

    return get_settings()


async def init_redis() -> None:
    """Connect to Redis, or fall back to in-memory store."""
    global _redis_client

    settings = _get_settings()
    url = settings.redis.url

    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(
            url,
            max_connections=settings.redis.max_connections,
            decode_responses=False,
        )
        await client.ping()
        _redis_client = client
        logger.info("Connected to Redis at %s", url.split("@")[-1] if "@" in url else url)
    except ImportError:
        logger.warning("redis package not installed — using in-memory fallback")
        _redis_client = _InMemoryRedis()
    except Exception as exc:
        logger.warning("Redis connection failed (%s) — using in-memory fallback", exc)
        _redis_client = _InMemoryRedis()


async def close_redis() -> None:
    """Close the Redis connection."""
    global _redis_client
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except AttributeError:
            with contextlib.suppress(Exception):
                await _redis_client.close()
        logger.info("Redis connection closed")
    _redis_client = None


def get_redis() -> Any:
    """Return the active Redis client (real or in-memory fallback).

    Raises RuntimeError if init_redis() hasn't been called.
    """
    if _redis_client is None:
        raise RuntimeError("Redis not initialized. Call init_redis() first.")
    return _redis_client
