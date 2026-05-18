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


def test_get_connection_sets_busy_timeout(tmp_db_path):
    """Verify get_connection sets busy_timeout >= 4000ms."""
    from app.db import get_connection
    conn = get_connection(tmp_db_path)
    try:
        row = conn.execute("PRAGMA busy_timeout").fetchone()
        timeout = row[0]
        assert timeout >= 4000, f"Expected busy_timeout >= 4000, got {timeout}"
    finally:
        conn.close()


from unittest.mock import MagicMock
from sqlite3 import OperationalError


def test_commit_with_retry_succeeds_on_first_attempt():
    """Verify commit_with_retry calls commit once on success."""
    from app.db import commit_with_retry
    mock_conn = MagicMock()
    commit_with_retry(mock_conn)
    mock_conn.commit.assert_called_once()


def test_commit_with_retry_retries_on_locked():
    """Verify commit_with_retry retries on OperationalError with 'locked'."""
    from app.db import commit_with_retry
    mock_conn = MagicMock()
    # Fail twice, succeed on third
    mock_conn.commit.side_effect = [
        OperationalError("database is locked"),
        OperationalError("database is locked"),
        None,
    ]
    commit_with_retry(mock_conn, max_retries=5, base_delay=0.01)
    assert mock_conn.commit.call_count == 3


def test_commit_with_retry_raises_after_exhaustion():
    """Verify commit_with_retry raises after max_retries exhausted."""
    from app.db import commit_with_retry
    mock_conn = MagicMock()
    mock_conn.commit.side_effect = OperationalError("database is locked")
    with pytest.raises(OperationalError):
        commit_with_retry(mock_conn, max_retries=3, base_delay=0.01)
    assert mock_conn.commit.call_count == 3


def test_commit_with_retry_passes_non_locked_errors():
    """Verify commit_with_retry does NOT retry non-locked errors."""
    from app.db import commit_with_retry
    mock_conn = MagicMock()
    mock_conn.commit.side_effect = OperationalError("no such table: foo")
    with pytest.raises(OperationalError):
        commit_with_retry(mock_conn, max_retries=3, base_delay=0.01)
    mock_conn.commit.assert_called_once()


def test_connection_pool_get_connection(tmp_db_path):
    """Verify get_connection returns a working connection with schema."""
    from app.db import ConnectionPool
    pool = ConnectionPool(tmp_db_path)
    with pool.get_connection() as conn:
        row = conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal" or row[0] == "WAL", f"Expected WAL, got {row[0]}"
        row = conn.execute("PRAGMA busy_timeout").fetchone()
        assert row[0] >= 4000, f"Expected busy_timeout >= 4000, got {row[0]}"
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'current_version'"
        ).fetchone()
        assert row is not None
        assert row[0] == "0"
    pool.close()


def test_connection_pool_connection_persistent(tmp_db_path):
    """Verify connection is reused (persistent across calls)."""
    from app.db import ConnectionPool
    pool = ConnectionPool(tmp_db_path)

    with pool.get_connection() as conn:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            ("test_key", "test_value")
        )
        conn.commit()

    with pool.get_connection() as conn:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'test_key'"
        ).fetchone()
        assert row is not None
        assert row[0] == "test_value"

    pool.close()


def test_connection_pool_serializes_access(tmp_db_path):
    """Verify get_connection serializes concurrent access."""
    import threading
    from app.db import ConnectionPool
    pool = ConnectionPool(tmp_db_path)

    results = []

    def worker_1():
        with pool.get_connection() as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                ("concurrent_a", "value_a")
            )
            conn.commit()
            results.append("a_done")

    def worker_2():
        with pool.get_connection() as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                ("concurrent_b", "value_b")
            )
            conn.commit()
            results.append("b_done")

    t1 = threading.Thread(target=worker_1)
    t2 = threading.Thread(target=worker_2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(results) == 2
    assert "a_done" in results
    assert "b_done" in results

    # Both values should be committed
    with pool.get_connection() as conn:
        for key in ("concurrent_a", "concurrent_b"):
            row = conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
            assert row is not None, f"Missing key: {key}"

    pool.close()
