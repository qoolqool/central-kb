"""SQLite schema, connection management, and migrations for Central KB."""
import sqlite3
from typing import Optional


SCHEMA_SQL = """
-- Core entries table
CREATE TABLE IF NOT EXISTS entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fqn             TEXT NOT NULL UNIQUE,
    namespace       TEXT NOT NULL,
    scope           TEXT NOT NULL,
    key             TEXT NOT NULL,
    title           TEXT NOT NULL,
    content         TEXT NOT NULL,
    metadata_json   TEXT DEFAULT '{}',
    vector          BLOB NOT NULL,
    simhash         INTEGER NOT NULL,
    version         INTEGER NOT NULL,
    status          TEXT DEFAULT 'accepted',
    source          TEXT NOT NULL,
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(scope, namespace, key)
);

CREATE INDEX IF NOT EXISTS idx_entries_scope_ns ON entries(scope, namespace);
CREATE INDEX IF NOT EXISTS idx_entries_simhash ON entries(simhash);
CREATE INDEX IF NOT EXISTS idx_entries_version ON entries(version);

-- FTS5 mirror for BM25 keyword search
CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(
    fqn UNINDEXED,
    scope UNINDEXED,
    namespace UNINDEXED,
    content
);

-- Conflicts needing human review
CREATE TABLE IF NOT EXISTS conflicts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    existing_fqn     TEXT NOT NULL,
    proposed_fqn     TEXT NOT NULL,
    proposed_content TEXT NOT NULL,
    similarity       REAL,
    status           TEXT DEFAULT 'pending',
    resolution       TEXT,
    resolved_by      TEXT,
    resolved_at      TEXT,
    created_at       TEXT DEFAULT (datetime('now'))
);

-- Promotion candidates and verdicts
CREATE TABLE IF NOT EXISTS promotions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_fqn    TEXT NOT NULL,
    match_fqns       TEXT NOT NULL,
    avg_similarity   REAL NOT NULL,
    project_count    INTEGER NOT NULL,
    status           TEXT DEFAULT 'candidate',
    verdict_by       TEXT,
    verdict_at       TEXT,
    created_at       TEXT DEFAULT (datetime('now'))
);

-- Version cursor for pull synchronization
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes if they don't exist."""
    conn.executescript(SCHEMA_SQL)
    # Initialize version cursor if not set
    cur = conn.execute("SELECT value FROM meta WHERE key = 'current_version'")
    if cur.fetchone() is None:
        conn.execute("INSERT INTO meta (key, value) VALUES ('current_version', '0')")
    conn.commit()


def get_connection(db_path: str) -> sqlite3.Connection:
    """Create and return a connection to the central KB database."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn
