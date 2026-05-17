"""Tests for app/db.py."""
import sqlite3
import pytest


def test_init_schema_creates_tables(db_connection):
    """Verify all core tables exist after init_schema."""
    conn = db_connection
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {row[0] for row in tables}

    assert "entries" in table_names
    assert "fts_index" in table_names
    assert "conflicts" in table_names
    assert "promotions" in table_names
    assert "meta" in table_names


def test_entries_table_columns(db_connection):
    """Verify entries table has expected columns."""
    conn = db_connection
    columns = conn.execute("PRAGMA table_info(entries)").fetchall()
    col_names = {row[1] for row in columns}

    for col in ("fqn", "namespace", "scope", "key", "title", "content",
                "metadata_json", "vector", "simhash", "version", "status", "source"):
        assert col in col_names, f"Missing column: {col}"


def test_entries_unique_constraint(db_connection):
    """Verify UNIQUE(scope, namespace, key) constraint."""
    conn = db_connection
    conn.execute(
        "INSERT INTO entries (fqn, namespace, scope, key, title, content, "
        "vector, simhash, version, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("test:dec:001", "decisions", "test", "001", "Title", "Content",
         b"0000", 12345, 1, "test")
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO entries (fqn, namespace, scope, key, title, content, "
            "vector, simhash, version, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("test:dec:001-dup", "decisions", "test", "001", "Title", "Content",
             b"0000", 12345, 1, "test")
        )


def test_fts_index_works(db_connection):
    """Verify FTS5 index is queryable."""
    conn = db_connection
    conn.execute(
        "INSERT INTO entries (fqn, namespace, scope, key, title, content, "
        "vector, simhash, version, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("test:dec:001", "decisions", "test", "001", "FastAPI stack",
         "All services use FastAPI with Pydantic", b"0000", 12345, 1, "test")
    )
    conn.execute(
        "INSERT INTO fts_index (fqn, scope, namespace, content) "
        "VALUES (?, ?, ?, ?)",
        ("test:dec:001", "test", "decisions", "All services use FastAPI with Pydantic")
    )
    results = conn.execute(
        "SELECT fqn FROM fts_index WHERE content MATCH ?", ("FastAPI",)
    ).fetchall()
    assert len(results) == 1
    assert results[0][0] == "test:dec:001"


def test_meta_table(db_connection):
    """Verify meta table stores key-value pairs."""
    conn = db_connection
    # current_version is set by init_schema — verify it's initialized
    result = conn.execute("SELECT value FROM meta WHERE key = ?", ("current_version",)).fetchone()
    assert result is not None
    assert result[0] == "0"

    # Insert and read a custom key-value pair
    conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)", ("last_sync", "2025-01-01"))
    result = conn.execute("SELECT value FROM meta WHERE key = ?", ("last_sync",)).fetchone()
    assert result is not None
    assert result[0] == "2025-01-01"
