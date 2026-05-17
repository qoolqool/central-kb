"""Tests for app/promote.py."""
import sqlite3


def _seed_promotion_data(conn: sqlite3.Connection):
    from app.embed import pack_vector
    from app.db import init_schema
    init_schema(conn)

    vec = pack_vector([0.42] * 1024)

    entries = [
        ("x402-poc:pat:fastapi-sk", "patterns", "x402-poc", "fastapi-sk",
         "FastAPI service skeleton", "Use FastAPI with Pydantic v2, uvicorn",
         vec, 100, 1, "local:engineer-A"),
        ("project-B:pat:fastapi-pattern", "patterns", "project-B", "fastapi-pattern",
         "FastAPI pattern", "Use FastAPI with Pydantic v2",
         vec, 101, 1, "local:engineer-B"),
    ]
    for e in entries:
        conn.execute(
            "INSERT INTO entries (fqn, namespace, scope, key, title, content, "
            "vector, simhash, version, source, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'accepted')", e
        )
    conn.commit()


def test_scan_candidates_finds_promotion():
    from app.promote import scan_candidates
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    _seed_promotion_data(conn)
    candidates = list(scan_candidates(conn))
    assert len(candidates) >= 1
    c = candidates[0]
    assert c["project_count"] >= 2
    assert c["avg_similarity"] >= 0.85
    conn.close()


def test_scan_candidates_empty_when_only_one_project():
    from app.promote import scan_candidates
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    from app.db import init_schema
    init_schema(conn)
    from app.embed import pack_vector
    vec = pack_vector([0.5] * 1024)
    conn.execute(
        "INSERT INTO entries (fqn, namespace, scope, key, title, content, "
        "vector, simhash, version, source, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'accepted')",
        ("single:pat:only", "patterns", "single", "only",
         "Solo pattern", "Only in one project",
         vec, 99, 1, "local:test")
    )
    conn.commit()
    candidates = list(scan_candidates(conn))
    assert len(candidates) == 0
    conn.close()
