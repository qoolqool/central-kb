"""Tests for app/embed.py."""
import pytest
import struct


def test_pack_unpack_vector_roundtrip():
    """Verify pack_vector output can be unpacked by unpack_vector."""
    from app.embed import pack_vector, unpack_vector

    original = [0.1, 0.2, 0.3, -0.5, 0.0, 1.0]
    blob = pack_vector(original)
    restored = unpack_vector(blob)

    assert len(original) == len(restored)
    for a, b in zip(original, restored):
        assert abs(a - b) < 1e-6


def test_cosine_similarity():
    """Verify cosine similarity computation."""
    from app.embed import cosine

    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert abs(cosine(a, b) - 0.0) < 1e-6

    c = [1.0, 2.0, 3.0]
    d = [2.0, 4.0, 6.0]
    assert abs(cosine(c, d) - 1.0) < 1e-6

    # Same vector
    assert abs(cosine(a, a) - 1.0) < 1e-6


def test_cosine_zero_vector():
    """Verify cosine handles zero vector (no division by zero)."""
    from app.embed import cosine
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_embed_text_default_dim():
    """Verify embed returns a 1024-dim vector (or None if server unavailable)."""
    from app.embed import embed_text

    result = embed_text("Hello world")
    # If embed server is not running, returns None
    if result is not None:
        assert len(result) == 1024
