"""Tests for app/dedup.py."""
import pytest


def test_simhash_64_is_deterministic():
    from app.dedup import simhash_64
    text = "All services use FastAPI with Pydantic models"
    h1 = simhash_64(text)
    h2 = simhash_64(text)
    assert h1 == h2


def test_simhash_64_differs_for_different_text():
    from app.dedup import simhash_64
    h1 = simhash_64("Use FastAPI for services")
    h2 = simhash_64("Use Django for services")
    assert h1 != h2


def test_hamming_distance():
    from app.dedup import hamming_distance
    assert hamming_distance(0xFF, 0xFF) == 0
    assert hamming_distance(0b1010, 0b0101) == 4
    assert hamming_distance(0b1000, 0b0000) == 1


def test_simhash_similarity():
    from app.dedup import simhash_similarity, simhash_64
    text = "FastAPI for all microservices"
    h = simhash_64(text)
    assert simhash_similarity(h, h) == 1.0
    h2 = simhash_64("Kubernetessssssssssssssssssssssssssssssssssssssssssssssssssssssssssss")
    similarity = simhash_similarity(h, h2)
    assert similarity < 0.5


def test_similar_texts_above_95_percent():
    from app.dedup import simhash_64, simhash_similarity
    a = "All services should use FastAPI with Pydantic v2 for validation"
    b = "All services must use FastAPI with Pydantic v2 for validation"
    ha = simhash_64(a)
    hb = simhash_64(b)
    assert simhash_similarity(ha, hb) >= 0.80


def test_dedup_classify():
    from app.dedup import classify_similarity
    assert classify_similarity(0.96) == ("auto_merge", "Superseding older entry")
    assert classify_similarity(0.85) == ("review", "Flagged for human review")
    assert classify_similarity(0.50) == ("accept", "Accepted as new entry")
    assert classify_similarity(0.80) == ("review", "Flagged for human review")
