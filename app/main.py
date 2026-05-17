"""FastAPI application for Central KB.

Endpoints:
    GET  /health   — health check
    POST /submit   — 3-phase ingest (dedup → conflict → publish)
    GET  /pull     — cursor-based pull of accepted entries
    GET  /search   — hybrid search (cosine + FTS5)
"""
import json
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from app.db import get_connection, init_schema
from app.dedup import simhash_64, classify_similarity, simhash_similarity
from app.embed import embed_text, pack_vector, VEC_DIM
from app.models import (
    Candidate,
    CandidateListResponse,
    Conflict,
    ConflictListResponse,
    ConflictResolveRequest,
    PromoteRequest,
    PullEntry,
    PullResponse,
    SearchResponse,
    SearchResult,
    SubmitRequest,
    SubmitResponse,
    SubmitDetail,
    EXPECTED_VEC_DIM,
)
from app.search import hybrid_search


def _make_fqn(scope: str, namespace: str, key: str) -> str:
    return f"{scope}:{namespace}:{key}"


def _resolve_db_url(db_url: Optional[str]) -> str:
    """Resolve the database URL, handling :memory: with shared cache."""
    if db_url is None:
        db_url = os.environ.get("CENTRAL_KB_DB_PATH", "/data/central-kb.sqlite3")
    return db_url


def _connect(db_url: str):
    """Create a connection, using shared cache for :memory: databases.
    Always initializes schema since in-memory databases are ephemeral.
    """
    if db_url == ":memory:":
        import sqlite3
        conn = sqlite3.connect("file::memory:?cache=shared", uri=True, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
    else:
        conn = get_connection(db_url)
    init_schema(conn)
    return conn


def create_app(db_url: Optional[str] = None) -> FastAPI:
    db_url = _resolve_db_url(db_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if db_url != ":memory:":
            os.makedirs(os.path.dirname(db_url), exist_ok=True)
        conn = _connect(db_url)
        init_schema(conn)
        conn.close()
        yield

    app = FastAPI(title="Central KB", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.post("/reset")
    async def reset_db():
        """Drop and recreate all tables. Clears all data including corrupt vectors."""
        conn = _connect(db_url)
        conn.executescript("""
            DROP TABLE IF EXISTS entries;
            DROP TABLE IF EXISTS fts_index;
            DROP TABLE IF EXISTS conflicts;
            DROP TABLE IF EXISTS promotions;
            DROP TABLE IF EXISTS meta;
        """)
        conn.commit()
        init_schema(conn)
        conn.close()
        return {"status": "ok", "message": "Database cleared and reinitialized."}

    @app.post("/submit", response_model=SubmitResponse)
    async def submit(req: SubmitRequest):
        conn = _connect(db_url)

        accepted = 0
        duplicates = 0
        conflicted = 0
        conflict_ids: list[int] = []
        details: list[SubmitDetail] = []

        for entry in req.entries:
            fqn = _make_fqn(req.project, entry.namespace, entry.key)
            content_text = f"{entry.title}\n{entry.content}"

            sh = entry.simhash if entry.simhash else simhash_64(content_text)

            vec = entry.vector
            if vec is None:
                vec = embed_text(content_text)
            if vec is None:
                details.append(SubmitDetail(fqn=fqn, status="error", version=None))
                continue
            # Validate vector dimension — must match bge-large-en-v1.5 (1024-dim)
            if len(vec) != EXPECTED_VEC_DIM:
                details.append(SubmitDetail(
                    fqn=fqn, status="error", version=None,
                ))
                continue
            vec_blob = pack_vector(vec)

            # Phase 1: Dedup — check for similar existing entries
            existing = conn.execute(
                "SELECT fqn, simhash, version, content FROM entries "
                "WHERE scope = ? AND namespace = ? AND status = 'accepted'",
                (req.project, entry.namespace)
            ).fetchall()

            dedup_hit = False
            for ex_row in existing:
                ex_simhash = ex_row[1]
                sim = simhash_similarity(sh, ex_simhash)
                action, reason = classify_similarity(sim)
                if action == "auto_merge":
                    conn.execute(
                        "UPDATE entries SET status = 'superseded' WHERE fqn = ?",
                        (ex_row[0],)
                    )
                    cur_version = conn.execute(
                        "SELECT value FROM meta WHERE key = 'current_version'"
                    ).fetchone()[0]
                    new_version = int(cur_version) + 1
                    conn.execute(
                        "INSERT INTO entries (fqn, namespace, scope, key, title, content,"
                        "metadata_json, vector, simhash, version, source, status) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'accepted')",
                        (fqn, entry.namespace, req.project, entry.key, entry.title,
                         entry.content, json.dumps(entry.metadata), vec_blob, sh,
                         new_version, req.source)
                    )
                    conn.execute(
                        "UPDATE meta SET value = ? WHERE key = 'current_version'",
                        (str(new_version),)
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO fts_index (fqn, scope, namespace, content) "
                        "VALUES (?, ?, ?, ?)",
                        (fqn, req.project, entry.namespace, entry.content)
                    )
                    conn.commit()

                    duplicates += 1
                    details.append(SubmitDetail(
                        fqn=fqn,
                        status="superseded_by",
                        version=new_version,
                        superseded_by=ex_row[0],
                    ))
                    dedup_hit = True
                    break
                elif action == "review":
                    conflicted += 1
                    cur = conn.execute(
                        "INSERT INTO conflicts (existing_fqn, proposed_fqn, proposed_content, similarity) "
                        "VALUES (?, ?, ?, ?)",
                        (ex_row[0], fqn, entry.content, sim)
                    )
                    conflict_ids.append(cur.lastrowid)
                    details.append(SubmitDetail(fqn=fqn, status="conflicted", conflict_id=cur.lastrowid))
                    dedup_hit = True
                    break

            if dedup_hit:
                continue

            # Phase 2: Check for conflict (same key, different content)
            existing_exact = conn.execute(
                "SELECT id FROM entries WHERE scope = ? AND namespace = ? AND key = ?",
                (req.project, entry.namespace, entry.key)
            ).fetchone()

            if existing_exact:
                conflicted += 1
                cur = conn.execute(
                    "INSERT INTO conflicts (existing_fqn, proposed_fqn, proposed_content, similarity) "
                    "VALUES (?, ?, ?, ?)",
                    (_make_fqn(req.project, entry.namespace, entry.key),
                     fqn, entry.content, 1.0)
                )
                conflict_ids.append(cur.lastrowid)
                details.append(SubmitDetail(fqn=fqn, status="conflicted", conflict_id=cur.lastrowid))
                continue

            # Phase 3: Publish
            cur_version = conn.execute(
                "SELECT value FROM meta WHERE key = 'current_version'"
            ).fetchone()[0]
            new_version = int(cur_version) + 1
            conn.execute(
                "INSERT INTO entries (fqn, namespace, scope, key, title, content,"
                "metadata_json, vector, simhash, version, source, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'accepted')",
                (fqn, entry.namespace, req.project, entry.key, entry.title,
                 entry.content, json.dumps(entry.metadata), vec_blob, sh,
                 new_version, req.source)
            )
            conn.execute(
                "UPDATE meta SET value = ? WHERE key = 'current_version'",
                (str(new_version),)
            )
            conn.execute(
                "INSERT OR REPLACE INTO fts_index (fqn, scope, namespace, content) "
                "VALUES (?, ?, ?, ?)",
                (fqn, req.project, entry.namespace, entry.content)
            )
            conn.commit()

            accepted += 1
            details.append(SubmitDetail(fqn=fqn, status="accepted", version=new_version))

        conn.close()
        return SubmitResponse(
            accepted=accepted,
            duplicates=duplicates,
            conflicted=conflicted,
            conflict_ids=conflict_ids,
            details=details,
        )

    @app.get("/pull", response_model=PullResponse)
    async def pull(
        project: str = Query(...),
        after_version: int = Query(0, alias="after_version"),
        scope: str = Query("own"),
    ):
        conn = _connect(db_url)
        scopes = [project]
        if "global" in scope:
            scopes.append("global")

        placeholders = ",".join("?" for _ in scopes)
        rows = conn.execute(
            f"SELECT fqn, namespace, scope, title, content, metadata_json, version, source "
            f"FROM entries WHERE scope IN ({placeholders}) AND version > ? AND status = 'accepted' "
            f"ORDER BY version ASC",
            (*scopes, after_version)
        ).fetchall()

        entries = []
        for row in rows:
            entries.append(PullEntry(
                fqn=row[0],
                namespace=row[1],
                scope=row[2],
                title=row[3],
                content=row[4],
                metadata=json.loads(row[5]) if isinstance(row[5], str) else {},
                version=row[6],
                source=row[7],
            ))

        next_cursor = after_version
        if entries:
            next_cursor = entries[-1].version

        conn.close()
        return PullResponse(entries=entries, next_cursor=next_cursor, drift_warnings=[])

    @app.get("/search", response_model=SearchResponse)
    async def search(
        q: str = Query(..., alias="q"),
        scope: Optional[str] = Query(None),
        namespace: Optional[str] = Query(None),
        mode: str = Query("hybrid"),
        alpha: float = Query(0.5, ge=0.0, le=1.0),
        limit: int = Query(10, le=50),
    ):
        conn = _connect(db_url)
        raw_results = hybrid_search(conn, q, scope=scope, namespace=namespace,
                                    limit=limit, alpha=alpha)

        results = []
        for r in raw_results:
            fqn = r["fqn"]
            row = conn.execute(
                "SELECT namespace, scope, title, content FROM entries WHERE fqn = ?",
                (fqn,)
            ).fetchone()
            if row:
                # row is sqlite3.Row — use named access
                results.append(SearchResult(
                    fqn=fqn,
                    scope=row["scope"],
                    namespace=row["namespace"],
                    title=row["title"],
                    content=row["content"],
                    score=r["score"],
                    cosine_score=r["cosine_score"],
                    fts_score=r["fts_score"],
                ))

        conn.close()
        return SearchResponse(query=q, results=results)

    @app.get("/drift")
    async def drift_report(project: str = Query(...)):
        from app.drift import detect_drift
        conn = _connect(db_url)
        items = detect_drift(conn, project)
        conn.close()
        return {"project": project, "drift_items": items}

    @app.get("/candidates", response_model=CandidateListResponse)
    async def list_candidates():
        from app.promote import scan_candidates
        conn = _connect(db_url)
        existing = {row[0] for row in
                     conn.execute("SELECT candidate_fqn FROM promotions WHERE status IN ('candidate', 'approved')")}

        candidates = []
        for cand in scan_candidates(conn):
            if cand["candidate_fqn"] not in existing:
                conn.execute(
                    "INSERT INTO promotions (candidate_fqn, match_fqns, avg_similarity, project_count, status) "
                    "VALUES (?, ?, ?, ?, 'candidate')",
                    (cand["candidate_fqn"], cand["match_fqns"],
                     cand["avg_similarity"], cand["project_count"])
                )
                cand_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                candidates.append({
                    "id": cand_id,
                    "candidate_fqn": cand["candidate_fqn"],
                    "match_fqns": json.loads(cand["match_fqns"]),
                    "avg_similarity": cand["avg_similarity"],
                    "project_count": cand["project_count"],
                    "status": "candidate",
                })
        conn.commit()
        conn.close()
        return CandidateListResponse(candidates=candidates)

    @app.post("/promote", response_model=None)
    async def promote(req: PromoteRequest):
        conn = _connect(db_url)
        row = conn.execute(
            "SELECT * FROM promotions WHERE id = ?", (req.candidate_id,)
        ).fetchone()
        if not row:
            conn.close()
            raise HTTPException(404, f"Promotion candidate #{req.candidate_id} not found")

        if req.action == "approve":
            entry_row = conn.execute(
                "SELECT * FROM entries WHERE fqn = ?", (row["candidate_fqn"],)
            ).fetchone()
            if entry_row:
                global_fqn = _make_fqn("global", entry_row["namespace"], entry_row["key"])
                existing_global = conn.execute(
                    "SELECT id FROM entries WHERE fqn = ?", (global_fqn,)
                ).fetchone()
                if not existing_global:
                    cur_version = conn.execute(
                        "SELECT value FROM meta WHERE key = 'current_version'"
                    ).fetchone()[0]
                    new_version = int(cur_version) + 1
                    conn.execute(
                        "INSERT INTO entries (fqn, namespace, scope, key, title, content, "
                        "metadata_json, vector, simhash, version, source, status) "
                        "VALUES (?, ?, 'global', ?, ?, ?, ?, ?, ?, ?, 'central:promotion', 'accepted')",
                        (global_fqn, entry_row["namespace"], entry_row["key"],
                         entry_row["title"], entry_row["content"],
                         entry_row["metadata_json"], entry_row["vector"],
                         entry_row["simhash"], new_version, req.verdict_by)
                    )
                    conn.execute(
                        "UPDATE meta SET value = ? WHERE key = 'current_version'",
                        (str(new_version),)
                    )
                    conn.execute(
                        "INSERT INTO fts_index (fqn, scope, namespace, content) VALUES (?, ?, ?, ?)",
                        (global_fqn, "global", entry_row["namespace"], entry_row["content"])
                    )

            conn.execute(
                "UPDATE promotions SET status = 'promoted', verdict_by = ?, verdict_at = datetime('now') "
                "WHERE id = ?",
                (req.verdict_by, req.candidate_id)
            )
        else:
            conn.execute(
                "UPDATE promotions SET status = 'rejected', verdict_by = ?, verdict_at = datetime('now') "
                "WHERE id = ?",
                (req.verdict_by, req.candidate_id)
            )

        conn.commit()
        conn.close()
        return {"status": "ok", "candidate_id": req.candidate_id, "action": req.action}

    @app.get("/conflicts", response_model=ConflictListResponse)
    async def list_conflicts():
        conn = _connect(db_url)
        rows = conn.execute(
            "SELECT id, existing_fqn, proposed_fqn, proposed_content, similarity, status, created_at "
            "FROM conflicts WHERE status = 'pending' ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        return ConflictListResponse(conflicts=[
            Conflict(
                id=r[0],
                existing_fqn=r[1],
                proposed_fqn=r[2],
                proposed_content=r[3],
                similarity=r[4],
                status=r[5],
                created_at=r[6],
            ) for r in rows
        ])

    @app.post("/conflicts/{conflict_id}/resolve")
    async def resolve_conflict(conflict_id: int, req: ConflictResolveRequest):
        conn = _connect(db_url)
        row = conn.execute(
            "SELECT * FROM conflicts WHERE id = ?", (conflict_id,)
        ).fetchone()
        if not row:
            conn.close()
            raise HTTPException(404, f"Conflict #{conflict_id} not found")

        if req.resolution == "accept_proposed":
            proposed_fqn = row[2]
            conn.execute(
                "UPDATE entries SET status = 'accepted' WHERE fqn = ?",
                (proposed_fqn,)
            )

        conn.execute(
            "UPDATE conflicts SET status = 'resolved', resolution = ?, resolved_by = ?, resolved_at = datetime('now') "
            "WHERE id = ?",
            (req.resolution, "api", conflict_id)
        )
        conn.commit()
        conn.close()
        return {"status": "ok", "conflict_id": conflict_id, "resolution": req.resolution}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
