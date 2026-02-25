# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Memory Service

Provides:
- Conversation summarization
- Context retrieval
- Long-term memory storage
- Semantic search (with embeddings)
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .tokens import get_token_counter

logger = logging.getLogger(__name__)


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class MemoryEntry:
    """A single memory entry."""

    id: str
    tenant_id: str
    user_id: str | None
    content: str
    summary: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    relevance_score: float = 0.0


@dataclass
class ConversationSummary:
    """Summary of a conversation."""

    conversation_id: str
    summary: str
    key_topics: list[str]
    key_entities: list[str]
    sentiment: str  # positive, negative, neutral
    action_items: list[str]
    message_count: int
    token_count: int
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ContextWindow:
    """Context to inject into a conversation."""

    recent_messages: list[dict[str, Any]]
    relevant_memories: list[MemoryEntry]
    conversation_summary: str | None
    system_context: str | None
    total_tokens: int = 0


# ============================================================
# MEMORY BACKEND INTERFACE
# ============================================================


class MemoryBackend(ABC):
    """Abstract interface for memory storage."""

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry."""
        pass

    @abstractmethod
    async def search(
        self,
        tenant_id: str,
        query: str,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search for relevant memories."""
        pass

    @abstractmethod
    async def get_recent(
        self,
        tenant_id: str,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Get recent memories."""
        pass

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        pass


class InMemoryBackend(MemoryBackend):
    """In-memory storage for development."""

    def __init__(self):
        self._memories: dict[str, MemoryEntry] = {}

    async def store(self, entry: MemoryEntry) -> None:
        self._memories[entry.id] = entry

    async def search(
        self,
        tenant_id: str,
        query: str,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        # Simple keyword search for dev
        query_lower = query.lower()
        results = []

        for entry in self._memories.values():
            if entry.tenant_id != tenant_id:
                continue
            if user_id and entry.user_id != user_id:
                continue

            # Score by keyword match
            content_lower = entry.content.lower()
            if query_lower in content_lower:
                entry.relevance_score = 1.0
                results.append(entry)
            elif any(word in content_lower for word in query_lower.split()):
                entry.relevance_score = 0.5
                results.append(entry)

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:limit]

    async def get_recent(
        self,
        tenant_id: str,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        results = [
            e
            for e in self._memories.values()
            if e.tenant_id == tenant_id and (user_id is None or e.user_id == user_id)
        ]
        results.sort(key=lambda x: x.created_at, reverse=True)
        return results[:limit]

    async def delete(self, entry_id: str) -> bool:
        if entry_id in self._memories:
            del self._memories[entry_id]
            return True
        return False


# ============================================================
# MEMORY SERVICE
# ============================================================


class MemoryService:
    """
    Service for managing conversation memory and context.

    Provides:
    - Conversation summarization
    - Context window building
    - Relevant memory retrieval
    - Long-term storage

    Usage:
        memory = MemoryService(backend, llm_provider)

        # Store a memory
        await memory.store_memory(
            tenant_id="...",
            content="User prefers dark mode",
            tags=["preference", "ui"]
        )

        # Build context for a conversation
        context = await memory.build_context(
            tenant_id="...",
            user_id="...",
            current_message="...",
            conversation_history=[...]
        )
    """

    def __init__(
        self,
        backend: MemoryBackend | None = None,
        llm_provider: Any | None = None,  # For summarization
        max_context_tokens: int = 4000,
    ):
        self.backend = backend or InMemoryBackend()
        self.llm_provider = llm_provider
        self.max_context_tokens = max_context_tokens

    async def store_memory(
        self,
        tenant_id: str,
        content: str,
        user_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Store a new memory entry."""
        import uuid

        entry = MemoryEntry(
            id=f"mem_{uuid.uuid4().hex[:16]}",
            tenant_id=tenant_id,
            user_id=user_id,
            content=content,
            tags=tags or [],
            metadata=metadata or {},
        )

        await self.backend.store(entry)
        logger.debug(f"Stored memory: {entry.id}")

        return entry

    async def search_memories(
        self,
        tenant_id: str,
        query: str,
        user_id: str | None = None,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """Search for relevant memories."""
        return await self.backend.search(
            tenant_id=tenant_id,
            query=query,
            user_id=user_id,
            limit=limit,
        )

    async def get_recent_memories(
        self,
        tenant_id: str,
        user_id: str | None = None,
        limit: int = 5,
    ) -> list[MemoryEntry]:
        """Get recent memories."""
        return await self.backend.get_recent(
            tenant_id=tenant_id,
            user_id=user_id,
            limit=limit,
        )

    async def summarize_conversation(
        self,
        messages: list[dict[str, Any]],
    ) -> ConversationSummary:
        """
        Generate a summary of a conversation.

        Uses LLM to extract:
        - Overall summary
        - Key topics
        - Named entities
        - Sentiment
        - Action items
        """
        if not self.llm_provider:
            # Return basic summary without LLM
            token_counter = get_token_counter()
            total_tokens = sum(
                token_counter.count_tokens(m.get("content", "")).count for m in messages
            )
            return ConversationSummary(
                conversation_id="",
                summary=f"Conversation with {len(messages)} messages",
                key_topics=[],
                key_entities=[],
                sentiment="neutral",
                action_items=[],
                message_count=len(messages),
                token_count=total_tokens,
            )

        # Build prompt for summarization
        conversation_text = "\n".join(
            [
                f"{m['role']}: {m['content'][:500]}"
                for m in messages[-20:]  # Last 20 messages
            ]
        )

        summary_prompt = f"""Analyze this conversation and provide a JSON response with:
- summary: A 2-3 sentence summary
- key_topics: List of main topics discussed
- key_entities: List of named entities (people, places, products)
- sentiment: Overall sentiment (positive, negative, neutral)
- action_items: List of any action items or todos mentioned

Conversation:
{conversation_text}

Respond with valid JSON only."""

        try:
            response = await self.llm_provider.chat(
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=500,
                temperature=0.3,
            )

            # Parse JSON response
            result = json.loads(response.text)

            return ConversationSummary(
                conversation_id="",
                summary=result.get("summary", ""),
                key_topics=result.get("key_topics", []),
                key_entities=result.get("key_entities", []),
                sentiment=result.get("sentiment", "neutral"),
                action_items=result.get("action_items", []),
                message_count=len(messages),
                token_count=response.usage.total_tokens,
            )

        except Exception as e:
            logger.warning(f"Failed to summarize conversation: {e}")
            return ConversationSummary(
                conversation_id="",
                summary=f"Conversation with {len(messages)} messages",
                key_topics=[],
                key_entities=[],
                sentiment="neutral",
                action_items=[],
                message_count=len(messages),
                token_count=0,
            )

    async def build_context(
        self,
        tenant_id: str,
        current_message: str,
        conversation_history: list[dict[str, Any]] | None = None,
        user_id: str | None = None,
        max_history_messages: int = 20,
        include_memories: bool = True,
    ) -> ContextWindow:
        """
        Build a context window for an LLM request.

        Combines:
        - Recent conversation history
        - Relevant memories
        - Conversation summary (if history is long)
        """
        conversation_history = conversation_history or []

        # Get recent messages
        recent_messages = conversation_history[-max_history_messages:]

        # Estimate tokens used by messages - use proper tokenizer
        token_counter = get_token_counter()
        message_tokens = sum(
            token_counter.count_tokens(m.get("content", "")).count for m in recent_messages
        )

        # Get relevant memories if space allows
        relevant_memories = []
        if include_memories and message_tokens < self.max_context_tokens * 0.7:
            memories = await self.search_memories(
                tenant_id=tenant_id,
                query=current_message,
                user_id=user_id,
                limit=5,
            )
            relevant_memories = memories

        # Summarize old history if conversation is long
        conversation_summary = None
        if len(conversation_history) > max_history_messages:
            old_messages = conversation_history[:-max_history_messages]
            summary = await self.summarize_conversation(old_messages)
            conversation_summary = summary.summary

        # Build system context from memories
        system_context = None
        if relevant_memories:
            memory_text = "\n".join([f"- {m.content}" for m in relevant_memories[:3]])
            system_context = f"Relevant context from previous interactions:\n{memory_text}"

        return ContextWindow(
            recent_messages=recent_messages,
            relevant_memories=relevant_memories,
            conversation_summary=conversation_summary,
            system_context=system_context,
            total_tokens=message_tokens,
        )

    async def extract_and_store_memories(
        self,
        tenant_id: str,
        messages: list[dict[str, Any]],
        user_id: str | None = None,
    ) -> list[MemoryEntry]:
        """
        Extract memorable information from a conversation and store it.

        Looks for:
        - User preferences
        - Important facts
        - Decisions made
        - Action items
        """
        if not self.llm_provider or len(messages) < 2:
            return []

        # Get conversation text
        conversation_text = "\n".join(
            [f"{m['role']}: {m['content'][:300]}" for m in messages[-10:]]
        )

        extraction_prompt = f"""Extract any important information worth remembering from this conversation.
Focus on:
- User preferences (likes/dislikes, settings)
- Important facts about the user
- Decisions or conclusions reached
- Commitments or action items

Return a JSON array of memories, each with:
- content: The information to remember (1-2 sentences)
- tags: Relevant tags

If nothing worth remembering, return an empty array.

Conversation:
{conversation_text}

Respond with valid JSON array only."""

        try:
            response = await self.llm_provider.chat(
                messages=[{"role": "user", "content": extraction_prompt}],
                max_tokens=500,
                temperature=0.3,
            )

            memories_data = json.loads(response.text)

            stored_memories = []
            for mem_data in memories_data[:5]:  # Max 5 memories per extraction
                if isinstance(mem_data, dict) and "content" in mem_data:
                    entry = await self.store_memory(
                        tenant_id=tenant_id,
                        user_id=user_id,
                        content=mem_data["content"],
                        tags=mem_data.get("tags", []),
                        metadata={"source": "auto_extraction"},
                    )
                    stored_memories.append(entry)

            return stored_memories

        except Exception as e:
            logger.warning(f"Failed to extract memories: {e}")
            return []


# ============================================================
# REDIS MEMORY BACKEND
# ============================================================


class RedisMemoryBackend(MemoryBackend):
    """
    Redis-based memory storage.

    Uses sorted sets for recency and hash maps for content.
    """

    def __init__(self, redis_client: Any):
        self.redis = redis_client

    def _key_prefix(self, tenant_id: str) -> str:
        return f"memory:{tenant_id}"

    async def store(self, entry: MemoryEntry) -> None:
        prefix = self._key_prefix(entry.tenant_id)

        # Store entry data
        await self.redis.hset(
            f"{prefix}:entries",
            entry.id,
            json.dumps(
                {
                    "id": entry.id,
                    "tenant_id": entry.tenant_id,
                    "user_id": entry.user_id,
                    "content": entry.content,
                    "summary": entry.summary,
                    "tags": entry.tags,
                    "metadata": entry.metadata,
                    "created_at": entry.created_at.isoformat(),
                }
            ),
        )

        # Add to recency index
        timestamp = entry.created_at.timestamp()
        await self.redis.zadd(f"{prefix}:recency", {entry.id: timestamp})

        # Add to user index if applicable
        if entry.user_id:
            await self.redis.zadd(f"{prefix}:user:{entry.user_id}", {entry.id: timestamp})

    async def search(
        self,
        tenant_id: str,
        query: str,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        # Simple implementation - get recent and filter
        # For production, use vector search (e.g., Redis Search)
        recent = await self.get_recent(tenant_id, user_id, limit=100)

        query_lower = query.lower()
        scored = []

        for entry in recent:
            content_lower = entry.content.lower()
            if query_lower in content_lower:
                entry.relevance_score = 1.0
                scored.append(entry)
            elif any(word in content_lower for word in query_lower.split()):
                entry.relevance_score = 0.5
                scored.append(entry)

        scored.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored[:limit]

    async def get_recent(
        self,
        tenant_id: str,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        prefix = self._key_prefix(tenant_id)

        # Get recent entry IDs
        if user_id:
            entry_ids = await self.redis.zrevrange(f"{prefix}:user:{user_id}", 0, limit - 1)
        else:
            entry_ids = await self.redis.zrevrange(f"{prefix}:recency", 0, limit - 1)

        if not entry_ids:
            return []

        # Fetch entry data
        entries = []
        for entry_id in entry_ids:
            data = await self.redis.hget(f"{prefix}:entries", entry_id)
            if data:
                entry_data = json.loads(data)
                entries.append(
                    MemoryEntry(
                        id=entry_data["id"],
                        tenant_id=entry_data["tenant_id"],
                        user_id=entry_data.get("user_id"),
                        content=entry_data["content"],
                        summary=entry_data.get("summary"),
                        tags=entry_data.get("tags", []),
                        metadata=entry_data.get("metadata", {}),
                        created_at=datetime.fromisoformat(entry_data["created_at"]),
                    )
                )

        return entries

    async def delete(self, entry_id: str) -> bool:
        # Would need tenant_id to properly delete
        # This is a simplified version
        return False


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "MemoryEntry",
    "ConversationSummary",
    "ContextWindow",
    "MemoryBackend",
    "InMemoryBackend",
    "RedisMemoryBackend",
    "MemoryService",
]
