"""Tests for scripts/seed.py."""
import os
import sqlite3
import tempfile


def test_collect_local_entries_with_embeddings_table():
    """Verify seed can read from a local KB with 'embeddings' table."""
    from app.embed import pack_vector
    from scripts.seed import collect_local_entries

    with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            namespace TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata_json TEXT DEFAULT '{}',
            vector BLOB NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO embeddings (key, namespace, content, metadata_json, vector) "
        "VALUES (?, ?, ?, ?, ?)",
        ("DEC-001", "decisions", "Use FastAPI for services",
         '{"title": "FastAPI stack", "status": "accepted"}',
         pack_vector([0.1] * 1024))
    )
    conn.commit()
    conn.close()

    entries = collect_local_entries(db_path)
    assert len(entries) == 1
    assert entries[0]["key"] == "DEC-001"
    assert entries[0]["namespace"] == "decisions"
    assert entries[0]["simhash"] > 0

    os.unlink(db_path)
