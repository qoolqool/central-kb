"""Simhash-based dedup engine for Central KB."""
import hashlib
from typing import Tuple

SIMHASH_BITS = 64


def simhash_64(text: str) -> int:
    """Compute 64-bit simhash for dedup comparison."""
    features = text.lower().split()
    v = [0] * SIMHASH_BITS
    for feature in features:
        h = int(hashlib.md5(feature.encode()).hexdigest(), 16)
        for i in range(SIMHASH_BITS):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    fingerprint = 0
    for i in range(SIMHASH_BITS):
        if v[i] >= 0:
            fingerprint |= (1 << i)
    # Mask to 63 bits so it fits in SQLite's signed 64-bit INTEGER
    return fingerprint & ((1 << 63) - 1)


def hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two simhashes."""
    return bin(a ^ b).count("1")


def simhash_similarity(a: int, b: int) -> float:
    """Similarity between two simhashes (0.0–1.0)."""
    return 1.0 - hamming_distance(a, b) / SIMHASH_BITS


def classify_similarity(similarity: float) -> Tuple[str, str]:
    """Classify a similarity score into (action, reason).

    Returns:
        ("auto_merge", reason) — ≥ 95% similarity
        ("review", reason)     — 80–95% similarity
        ("accept", reason)     — < 80% similarity
    """
    if similarity >= 0.95:
        return ("auto_merge", "Superseding older entry")
    elif similarity >= 0.80:
        return ("review", "Flagged for human review")
    else:
        return ("accept", "Accepted as new entry")
