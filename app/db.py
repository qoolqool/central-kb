"""SQLite schema, connection management, and migrations for Central KB."""
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Generator, Optional


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
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_size_limit=0")
    conn.row_factory = sqlite3.Row
    return conn


def commit_with_retry(conn: sqlite3.Connection, max_retries: int = 3,
                       base_delay: float = 0.1) -> None:
    """Commit with exponential backoff retry for database-locked errors.

    Only retries when the error message contains 'locked' (SQLite's
    'database is locked' or 'database table is locked'). Other
    OperationalErrors (e.g. no such table) raise immediately.
    """
    for attempt in range(max_retries):
        try:
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            err_str = str(e)
            if "locked" not in err_str:
                raise  # non-lock errors are not retriable
            if attempt == max_retries - 1:
                raise  # last attempt, propagate error
            time.sleep(base_delay * (2 ** attempt))


class ConnectionPool:
    """SQLite connection pool for web server use.

    Single persistent connection for all operations (reads and writes).
    Serialized via threading.Lock. WAL checkpointed after writes to
    prevent file growth and avoid orphaned reader slots that cause
    persistent WAL locks.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    def _ensure_conn(self) -> None:
        """Create persistent connection and initialize schema if not already done."""
        if self._conn is None:
            self._conn = get_connection(self.db_path)
            init_schema(self._conn)

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield the persistent connection, serialized via Lock.

        Usage:
            with pool.get_connection() as conn:
                conn.execute(...)
                conn.commit()

        The Lock ensures only one thread uses the connection at a time.
        After the caller's commit, a WAL checkpoint is automatically
        attempted to keep the WAL file trimmed.
        """
        with self._lock:
            self._ensure_conn()
            yield self._conn
            # Attempt passive WAL checkpoint to keep WAL file trimmed.
            # This prevents the -wal/-shm files from growing stale and
            # eliminates orphaned reader slots that cause persistent locks.
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.OperationalError:
                pass  # non-critical; retries on next call

    def close(self) -> None:
        """Close the persistent connection."""
        if self._conn is not None:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.OperationalError:
                pass
            self._conn.close()
            self._conn = None
