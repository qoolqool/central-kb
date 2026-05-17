"""Shared fixtures for Central KB tests."""
import pytest
import sqlite3
from pathlib import Path


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_central.db")


@pytest.fixture
def db_connection(tmp_db_path: str):
    from app.db import init_schema
    conn = sqlite3.connect(tmp_db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    init_schema(conn)
    yield conn
    conn.close()
