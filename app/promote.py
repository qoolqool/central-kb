"""Promotion candidate scanner — finds patterns to promote to global namespace."""
import json
import sqlite3
from typing import Any, Iterator

from app.embed import cosine, unpack_vector, VEC_DIM

PROMOTION_MIN_PROJECTS = 2
PROMOTION_MIN_SIMILARITY = 0.85


def scan_candidates(conn: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    """Scan for promotion candidates.

    Looks for entries that appear with ≥85% similarity across ≥2 projects.
    """
    entries = conn.execute(
        "SELECT fqn, scope, namespace, key, title, content, vector, simhash "
        "FROM entries WHERE status = 'accepted'"
    ).fetchall()

    if len(entries) < 2:
        return

    seen_pairs = set()
    for i, row_a in enumerate(entries):
        if row_a[1] == "global":
            continue
        vec_a = unpack_vector(row_a[6])
        # Skip entries with wrong vector dimensions
        if len(vec_a) != VEC_DIM:
            continue

        matches = []
        projects_seen = {row_a[1]}

        for j, row_b in enumerate(entries):
            if i == j or row_b[1] == "global":
                continue
            if row_b[2] != row_a[2]:
                continue

            pair_key = tuple(sorted([row_a[0], row_b[0]]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            vec_b = unpack_vector(row_b[6])
            # Skip entries with wrong vector dimensions
            if len(vec_b) != VEC_DIM:
                continue
            sim = cosine(vec_a, vec_b)

            if sim >= PROMOTION_MIN_SIMILARITY:
                matches.append(row_b[0])
                projects_seen.add(row_b[1])

        if len(projects_seen) >= PROMOTION_MIN_PROJECTS and matches:
            match_vecs = []
            for row in entries:
                if row[0] in matches:
                    v = unpack_vector(row[6])
                    if len(v) == VEC_DIM:
                        match_vecs.append(v)
            avg_sim = sum(
                cosine(vec_a, mv) for mv in match_vecs
            ) / len(match_vecs) if match_vecs else 0.0

            yield {
                "candidate_fqn": row_a[0],
                "match_fqns": json.dumps(matches),
                "avg_similarity": round(avg_sim, 4),
                "project_count": len(projects_seen),
            }
