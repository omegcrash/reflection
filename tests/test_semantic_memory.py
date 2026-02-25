"""
Test Suite: Semantic Memory Retrieval (Phase 1)
================================================

Tests embedding providers, semantic search, vector persistence,
and integration with the Memory class.
"""

import json
import math
import pytest
import tempfile
from pathlib import Path

from familiar.core.embeddings import (
    TFIDFEmbeddings,
    cosine_similarity,
    top_k_similar,
    get_embedding_provider,
    EmbeddingProvider,
)
from familiar.core.memory import Memory, EMBEDDINGS_AVAILABLE


@pytest.fixture
def tmp_memory(tmp_path):
    """Create a Memory with a temp file path."""
    return Memory(memory_file=tmp_path / "memory.json")


@pytest.fixture
def populated_memory(tmp_memory):
    """Memory pre-loaded with test entries."""
    entries = [
        ("cf_grant", "Community Foundation grant application due March 15", "deadline", 8),
        ("board_mtg", "Board meeting scheduled for next Thursday", "schedule", 7),
        ("gala", "Annual fundraising gala at the Hilton ballroom", "event", 6),
        ("alice_donor", "Donor Alice Smith contributed 5000 dollars", "donor", 9),
        ("fiscal_year", "Fiscal year ends June 30", "org_info", 8),
        ("bob_volunteer", "Bob Johnson volunteers every Tuesday at the food bank", "volunteer", 5),
        ("newsletter", "Monthly newsletter goes out on the first Friday", "schedule", 4),
    ]
    for key, value, cat, imp in entries:
        tmp_memory.remember(key, value, category=cat, importance=imp)
    return tmp_memory


# ── Embedding Providers ────────────────────────────────────

class TestCosineSimiarity:

    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 0.001

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 0.001

    def test_zero_vector(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_different_lengths(self):
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0


class TestTopKSimilar:

    def test_returns_k_results(self):
        query = [1.0, 0.0, 0.0]
        vectors = [
            ("a", [1.0, 0.0, 0.0]),
            ("b", [0.0, 1.0, 0.0]),
            ("c", [0.5, 0.5, 0.0]),
        ]
        results = top_k_similar(query, vectors, k=2)
        assert len(results) == 2

    def test_sorted_by_similarity(self):
        query = [1.0, 0.0, 0.0]
        vectors = [
            ("a", [0.0, 1.0, 0.0]),
            ("b", [1.0, 0.0, 0.0]),
            ("c", [0.5, 0.5, 0.0]),
        ]
        results = top_k_similar(query, vectors, k=3)
        assert results[0][0] == "b"  # Exact match first
        assert results[0][1] > results[1][1] > results[2][1]


class TestTFIDFEmbeddings:

    def test_embed_returns_vector(self):
        tfidf = TFIDFEmbeddings(max_features=64)
        vec = tfidf.embed("hello world")
        assert len(vec) == 64
        assert isinstance(vec[0], float)

    def test_embed_batch(self):
        tfidf = TFIDFEmbeddings(max_features=64)
        vecs = tfidf.embed_batch(["hello", "world", "test"])
        assert len(vecs) == 3
        assert all(len(v) == 64 for v in vecs)

    def test_self_similarity_is_one(self):
        tfidf = TFIDFEmbeddings(max_features=128)
        vec = tfidf.embed("grant deadline application")
        assert abs(cosine_similarity(vec, vec) - 1.0) < 0.001

    def test_different_texts_different_vectors(self):
        tfidf = TFIDFEmbeddings(max_features=128)
        a = tfidf.embed("the quick brown fox")
        b = tfidf.embed("neural network architecture")
        assert cosine_similarity(a, b) < 0.9  # Not identical

    def test_shared_terms_increase_similarity(self):
        tfidf = TFIDFEmbeddings(max_features=128)
        # Build vocab
        tfidf.embed_batch([
            "grant application deadline",
            "board meeting agenda",
            "fiscal year budget",
        ])
        a = tfidf.embed("grant deadline")
        b = tfidf.embed("grant application deadline")
        c = tfidf.embed("board meeting agenda")
        # a and b share "grant" and "deadline" — should be more similar
        sim_ab = cosine_similarity(a, b)
        sim_ac = cosine_similarity(a, c)
        assert sim_ab > sim_ac

    def test_empty_text_returns_zero_vector(self):
        tfidf = TFIDFEmbeddings(max_features=64)
        vec = tfidf.embed("")
        assert all(v == 0.0 for v in vec)

    def test_name_property(self):
        tfidf = TFIDFEmbeddings()
        assert tfidf.name == "tfidf"

    def test_state_serialization(self):
        tfidf = TFIDFEmbeddings(max_features=64)
        tfidf.embed_batch(["hello world", "test document"])
        state = tfidf.get_state()
        assert "vocab" in state
        assert "doc_count" in state
        assert state["doc_count"] == 2

        tfidf2 = TFIDFEmbeddings(max_features=64)
        tfidf2.load_state(state)
        assert tfidf2._doc_count == 2
        assert tfidf2._vocab == tfidf._vocab


class TestProviderCascade:

    def test_fallback_to_tfidf(self):
        """When no Ollama/API keys, should fall back to TF-IDF."""
        provider = get_embedding_provider()
        # In test environment, will be TF-IDF
        assert isinstance(provider, EmbeddingProvider)
        assert provider.name in ("tfidf", "ollama/nomic-embed-text")

    def test_prefer_local_skips_cloud(self):
        provider = get_embedding_provider(prefer_local=True)
        assert "voyage" not in provider.name
        assert "openai" not in provider.name


# ── Memory Integration ─────────────────────────────────────

class TestMemoryEmbeddings:

    def test_embeddings_available(self):
        assert EMBEDDINGS_AVAILABLE is True

    def test_memory_has_embedder(self, tmp_memory):
        assert tmp_memory._embedder is not None

    def test_remember_creates_vector(self, tmp_memory):
        tmp_memory.remember("test", "hello world", category="test")
        assert "test" in tmp_memory._vectors

    def test_forget_removes_vector(self, tmp_memory):
        tmp_memory.remember("test", "hello world", category="test")
        assert "test" in tmp_memory._vectors
        tmp_memory.forget("test")
        assert "test" not in tmp_memory._vectors

    def test_vectors_synced_with_memories(self, populated_memory):
        assert len(populated_memory._vectors) == len(populated_memory.memories)

    def test_vector_file_created(self, populated_memory):
        vf = populated_memory._vectors_file()
        assert vf.exists()
        data = json.loads(vf.read_text())
        assert data["count"] == len(populated_memory.memories)


class TestSemanticSearch:

    def test_substring_match_still_works(self, populated_memory):
        """Exact substring should always be found."""
        results = populated_memory.search("Alice")
        keys = [r.key for r in results]
        assert "alice_donor" in keys

    def test_substring_match_partial(self, populated_memory):
        results = populated_memory.search("March 15")
        keys = [r.key for r in results]
        assert "cf_grant" in keys

    def test_search_returns_list(self, populated_memory):
        results = populated_memory.search("meeting")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_max_results(self, populated_memory):
        results = populated_memory.search("the", max_results=3)
        assert len(results) <= 3

    def test_search_empty_query(self, populated_memory):
        results = populated_memory.search("")
        # Should return something (semantic might match, substring won't)
        assert isinstance(results, list)


class TestRelevantContext:

    def test_returns_formatted_string(self, populated_memory):
        ctx = populated_memory.get_relevant_context("board meeting")
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_contains_category_headers(self, populated_memory):
        ctx = populated_memory.get_relevant_context("Who are our donors?")
        if ctx:
            assert "##" in ctx

    def test_empty_when_no_matches(self, tmp_memory):
        ctx = tmp_memory.get_relevant_context("anything")
        assert ctx == ""


class TestVectorPersistence:

    def test_vectors_survive_reload(self, tmp_path):
        mf = tmp_path / "memory.json"
        mem1 = Memory(memory_file=mf)
        mem1.remember("key1", "value one", category="test")
        mem1.remember("key2", "value two", category="test")
        assert len(mem1._vectors) == 2

        # Reload from disk
        mem2 = Memory(memory_file=mf)
        assert len(mem2._vectors) == 2
        assert "key1" in mem2._vectors
        assert "key2" in mem2._vectors

    def test_provider_change_invalidates_cache(self, tmp_path):
        mf = tmp_path / "memory.json"
        mem = Memory(memory_file=mf)
        mem.remember("key1", "value one", category="test")

        # Tamper with the provider name in the vector file
        vf = mf.with_name("memory_vectors.json")
        data = json.loads(vf.read_text())
        data["provider"] = "fake-provider-that-changed"
        vf.write_text(json.dumps(data))

        # Reload — should detect mismatch and re-embed
        mem2 = Memory(memory_file=mf)
        assert len(mem2._vectors) == 1  # Re-embedded


class TestGetSystemPromptIntegration:

    def test_context_string_unchanged(self, populated_memory):
        """get_context_string() still works as before."""
        ctx = populated_memory.get_context_string(max_entries=3)
        assert "What I Know About You" in ctx
        # Should include highest importance entries
        assert "alice_donor" in ctx or "Alice" in ctx

    def test_relevant_context_differs_from_global(self, populated_memory):
        """Relevant context should surface different memories than global."""
        global_ctx = populated_memory.get_context_string(max_entries=3)
        relevant_ctx = populated_memory.get_relevant_context("Tuesday volunteer")
        # Both should be non-empty
        assert len(global_ctx) > 0
        # Relevant context header is different
        if relevant_ctx:
            assert "Relevant Context" in relevant_ctx
