"""Hybrid search (cosine + FTS5) for Central KB.

Adapted from the local toy-rag's search-kb-memory.py patterns.
"""
import sqlite3
from typing import Iterator, Optional, Tuple

from app.embed import cosine, embed_text, unpack_vector, VEC_DIM


def _escape_fts(text: str) -> str:
    """Escape single quotes for FTS5 query safety."""
    return text.replace("'", "''")


def fts_search(
    conn: sqlite3.Connection,
    query: str,
    scope: Optional[str] = None,
    namespace: Optional[str] = None,
    limit: int = 10,
) -> Iterator[Tuple[str, float]]:
    """Search the FTS5 index, yielding (fqn, bm25_score) tuples."""
    safe_query = _escape_fts(query)
    conditions = []
    params = []

    if scope:
        conditions.append("scope = ?")
        params.append(scope)
    if namespace:
        conditions.append("namespace = ?")
        params.append(namespace)

    where = " AND ".join(conditions) if conditions else "1=1"

    sql = f"""
        SELECT fqn, bm25(fts_index) as bm25_score
        FROM fts_index
        WHERE fts_index MATCH ? AND {where}
        ORDER BY rank
        LIMIT ?
    """
    params = [safe_query] + params + [limit]

    for row in conn.execute(sql, params):
        fqn = row[0]
        raw_score = row[1]
        # BM25 is 0 = perfect match, negative = worse. Normalize to [0, 1].
        normalized = 1.0 / (1.0 + abs(raw_score)) if raw_score != 0 else 1.0
        yield (fqn, normalized)


def cosine_search(
    conn: sqlite3.Connection,
    query_vec: list[float],
    scope: Optional[str] = None,
    namespace: Optional[str] = None,
    limit: int = 10,
) -> Iterator[Tuple[str, float]]:
    """Search by cosine similarity against all entries, yielding (fqn, score)."""
    conditions = ["1=1"]
    params = []

    if scope:
        conditions.append("scope = ?")
        params.append(scope)
    if namespace:
        conditions.append("namespace = ?")
        params.append(namespace)

    sql = f"""
        SELECT fqn, vector, content
        FROM entries
        WHERE status = 'accepted' AND {' AND '.join(conditions)}
    """

    scored = []
    for row in conn.execute(sql, params):
        vec = unpack_vector(row[1])
        # Skip vectors with wrong dimensions to avoid cosine() errors
        if len(vec) != VEC_DIM:
            continue
        score = cosine(query_vec, vec)
        scored.append((score, row[0]))

    scored.sort(key=lambda x: x[0], reverse=True)
    for score, fqn in scored[:limit]:
        yield (fqn, score)


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    scope: Optional[str] = None,
    namespace: Optional[str] = None,
    limit: int = 10,
    alpha: float = 0.5,
) -> list[dict]:
    """Hybrid search combining FTS5 BM25 and cosine similarity.

    Alpha controls the blend: 1.0 = pure cosine, 0.0 = pure FTS.
    Results are sorted by blended score descending.
    """
    query_vec = embed_text(query)
    if query_vec is None:
        fts_results = dict(fts_search(conn, query, scope, namespace, limit))
        return [
            {"fqn": fqn, "score": score, "cosine_score": 0.0, "fts_score": score}
            for fqn, score in fts_results.items()
        ]

    fts_scores: dict[str, float] = dict(fts_search(conn, query, scope, namespace, limit))
    cosine_scores: dict[str, float] = dict(cosine_search(conn, query_vec, scope, namespace, limit))

    all_fqns = set(fts_scores.keys()) | set(cosine_scores.keys())

    results = []
    for fqn in all_fqns:
        fts_s = fts_scores.get(fqn, 0.0)
        cos_s = cosine_scores.get(fqn, 0.0)
        blended = alpha * cos_s + (1.0 - alpha) * fts_s
        results.append({
            "fqn": fqn,
            "score": blended,
            "cosine_score": cos_s,
            "fts_score": fts_s,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:limit]
