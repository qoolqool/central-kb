"""Tests for app/search.py."""
import pytest
import sqlite3


def _seed_test_data(conn: sqlite3.Connection):
    """Helper to insert test entries."""
    from app.embed import pack_vector
    vec = pack_vector([0.1] * 1024)
    entries = [
        ("test:dec:001", "decisions", "test", "001", "FastAPI stack",
         "All services use FastAPI with Pydantic", vec, 12345, 1, "test"),
        ("test:dec:002", "decisions", "test", "002", "Docker compose",
         "Services run in Docker containers with compose", vec, 12346, 2, "test"),
        ("test:pat:001", "patterns", "test", "001", "Health check pattern",
         "Use curl to check /health endpoint every 30s", vec, 12347, 3, "test"),
    ]
    for e in entries:
        conn.execute(
            "INSERT INTO entries (fqn, namespace, scope, key, title, content, "
            "vector, simhash, version, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            e
        )
    for e in entries:
        conn.execute(
            "INSERT INTO fts_index (fqn, scope, namespace, content) VALUES (?, ?, ?, ?)",
            (e[0], e[2], e[1], e[5])
        )
    conn.commit()


def test_fts_search_filters_by_scope_and_namespace(db_connection):
    from app.search import fts_search
    conn = db_connection
    _seed_test_data(conn)
    results = list(fts_search(conn, "FastAPI", scope="test", namespace="decisions", limit=10))
    assert len(results) == 1
    assert results[0][0] == "test:dec:001"


def test_fts_search_returns_all_when_no_filter(db_connection):
    from app.search import fts_search
    conn = db_connection
    _seed_test_data(conn)
    results = list(fts_search(conn, "FastAPI OR Docker OR health", limit=10))
    assert len(results) == 3


def test_escape_fts():
    from app.search import _escape_fts
    assert _escape_fts("hello") == "hello"
    assert _escape_fts("he'llo") == "he''llo"
    assert _escape_fts("") == ""


def test_hybrid_search_alpha_blending(db_connection):
    from app.search import hybrid_search
    conn = db_connection
    _seed_test_data(conn)
    results = hybrid_search(conn, "FastAPI", scope="test", namespace="decisions",
                           limit=5, alpha=1.0)
    assert len(results) >= 1
