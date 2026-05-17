"""Tests for app/drift.py."""
import sqlite3


def _seed_drift_data(conn: sqlite3.Connection):
    from app.embed import pack_vector
    from app.db import init_schema
    init_schema(conn)

    vec_a = pack_vector([0.1] * 1024)
    vec_b = pack_vector([0.9] * 1024)

    entries = [
        ("x402-poc:dec:001-dind", "decisions", "x402-poc", "001-dind",
         "DinD orchestration", "Use DinD with bootstrap.sh for container orchestration",
         vec_a, 100, 1, "local:engineer-A"),
        ("x402-poc:dec:002-jaeger", "decisions", "x402-poc", "002-jaeger",
         "Observability", "Use Jaeger v2 all-in-one for tracing",
         vec_b, 200, 2, "local:engineer-A"),
        ("project-B:dec:001-k8s", "decisions", "project-B", "001-k8s",
         "K8s orchestration", "Use Kubernetes with Helm charts for orchestration",
         vec_a, 300, 1, "local:engineer-B"),
    ]
    for e in entries:
        conn.execute(
            "INSERT INTO entries (fqn, namespace, scope, key, title, content, "
            "vector, simhash, version, source, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'accepted')", e
        )
    conn.commit()


def test_drift_detection_finds_divergence():
    from app.drift import detect_drift
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    _seed_drift_data(conn)
    drift = detect_drift(conn, "x402-poc")
    assert len(drift) > 0
    found = any("orchestrat" in d["your_entry"]["title"].lower() for d in drift)
    assert found
    conn.close()


def test_drift_no_false_positive_for_same_project():
    from app.drift import detect_drift
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    _seed_drift_data(conn)
    drift = detect_drift(conn, "nonexistent")
    assert len(drift) == 0
    conn.close()


def test_drift_accepts_empty_db():
    from app.drift import detect_drift
    from app.db import init_schema
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    init_schema(conn)
    drift = detect_drift(conn, "x402-poc")
    assert len(drift) == 0
    conn.close()
