"""Drift detection — compares a project's decisions against other projects."""
import sqlite3
from typing import Any

from app.embed import cosine, unpack_vector, VEC_DIM

DRIFT_SIMILARITY_THRESHOLD = 0.7


def detect_drift(conn: sqlite3.Connection, project: str) -> list[dict[str, Any]]:
    """Detect drift between a project's decisions and other projects.

    For each decision in the project, find decisions in other projects with
    similar topic (high cosine similarity) but different conclusions (different projects).
    """
    my_entries = conn.execute(
        "SELECT fqn, scope, namespace, key, title, content, vector FROM entries "
        "WHERE scope = ? AND status = 'accepted'",
        (project,)
    ).fetchall()

    if not my_entries:
        return []

    other_entries = conn.execute(
        "SELECT fqn, scope, namespace, key, title, content, vector FROM entries "
        "WHERE scope != ? AND status = 'accepted'",
        (project,)
    ).fetchall()

    if not other_entries:
        return []

    drift_items = []
    for my_row in my_entries:
        my_vec = unpack_vector(my_row[6])
        # Skip entries with wrong vector dimensions
        if len(my_vec) != VEC_DIM:
            continue
        my_title = my_row[4]
        my_content = my_row[5]

        for other_row in other_entries:
            other_vec = unpack_vector(other_row[6])
            # Skip entries with wrong vector dimensions
            if len(other_vec) != VEC_DIM:
                continue
            topic_sim = cosine(my_vec, other_vec)

            if topic_sim >= DRIFT_SIMILARITY_THRESHOLD:
                drift_items.append({
                    "your_entry": {
                        "fqn": my_row[0],
                        "title": my_title,
                        "conclusion": my_content[:200],
                    },
                    "conflicting_entry": {
                        "fqn": other_row[0],
                        "title": other_row[4],
                        "conclusion": other_row[5][:200],
                    },
                    "topic_similarity": round(topic_sim, 4),
                })

    return drift_items
