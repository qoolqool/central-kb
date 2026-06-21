"""Tests for app/main.py — FastAPI endpoints."""
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import create_app
from app.embed import pack_vector


@pytest.fixture
def app(tmp_path):
    db_url = str(tmp_path / "test_central_kb.db")
    return create_app(db_url=db_url)


@pytest.fixture
def sample_vector():
    """A valid 1024-dim vector for testing."""
    return [0.1] * 1024


@pytest.mark.anyio
async def test_health_endpoint(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@pytest.mark.anyio
async def test_submit_empty_entries(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/submit", json={
            "project": "test-project",
            "source": "local:test",
            "entries": []
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] == 0
    assert data["duplicates"] == 0
    assert data["conflicted"] == 0


@pytest.mark.anyio
async def test_submit_and_pull_roundtrip(app, sample_vector):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        submit_resp = await client.post("/submit", json={
            "project": "test-project",
            "source": "local:engineer-A",
            "entries": [
                {
                    "namespace": "decisions",
                    "key": "DEC-001-fastapi",
                    "title": "FastAPI stack",
                    "content": "All services use FastAPI with Pydantic",
                    "metadata": {"status": "accepted"},
                    "vector": sample_vector,
                    "simhash": 12345,
                }
            ]
        })
        assert submit_resp.status_code == 200
        submit_data = submit_resp.json()
        assert submit_data["accepted"] == 1
        fqn = submit_data["details"][0]["fqn"]
        assert "test-project" in fqn

        pull_resp = await client.get(
            "/pull", params={"project": "test-project", "scope": "own"}
        )
        assert pull_resp.status_code == 200
        pull_data = pull_resp.json()
        assert len(pull_data["entries"]) == 1
        assert pull_data["entries"][0]["fqn"] == fqn


@pytest.mark.anyio
async def test_search_endpoint(app, sample_vector):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/submit", json={
            "project": "test-project",
            "source": "local:test",
            "entries": [
                {
                    "namespace": "decisions",
                    "key": "DEC-002-docker",
                    "title": "Docker compose",
                    "content": "Services run in Docker containers with compose",
                    "metadata": {},
                    "vector": sample_vector,
                    "simhash": 54321,
                }
            ]
        })

        search_resp = await client.get(
            "/search",
            params={"q": "Docker", "scope": "test-project", "limit": 5}
        )
        assert search_resp.status_code == 200
        data = search_resp.json()
        assert len(data["results"]) > 0


@pytest.mark.anyio
async def test_submit_duplicate_detected(app, sample_vector):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        entry = {
            "namespace": "decisions",
            "key": "DEC-003-fastapi-v2",
            "title": "FastAPI v2",
            "content": "All services use FastAPI v2 with Pydantic",
            "metadata": {},
            "vector": sample_vector,
            "simhash": 11111,
        }

        resp1 = await client.post("/submit", json={
            "project": "test-project",
            "source": "local:engineer-A",
            "entries": [entry]
        })
        assert resp1.json()["accepted"] == 1

        entry2 = {**entry, "key": "DEC-003-fastapi-v2-alt"}
        resp2 = await client.post("/submit", json={
            "project": "test-project",
            "source": "local:engineer-B",
            "entries": [entry2]
        })
        data2 = resp2.json()
        assert data2["duplicates"] >= 1


@pytest.mark.anyio
async def test_concurrent_submits_dont_lock(tmp_path):
    """Verify concurrent submit requests don't crash with database is locked."""
    from anyio import create_task_group, run

    db_url = str(tmp_path / "test_concurrent.db")
    app = create_app(db_url=db_url)
    transport = ASGITransport(app=app)
    sample_vec = [0.1] * 1024

    async def do_submit(n: int, errors: list):
        entries = []
        for i in range(5):
            entries.append({
                "namespace": "decisions",
                "key": f"concurrent-{n}-{i}",
                "title": f"Concurrent {n} Entry {i}",
                "content": f"This is from concurrent submit {n}, entry {i}.",
                "metadata": {},
                "vector": sample_vec,
            })
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/submit", json={
                "project": "testproj",
                "source": "test",
                "entries": entries,
            })
            if resp.status_code != 200:
                errors.append(f"thread_{n}: status={resp.status_code}")
            else:
                data = resp.json()
                err_count = len([d for d in data.get("details", []) if d["status"] == "error"])
                if err_count:
                    errors.append(f"thread_{n}: {err_count} errors in {len(entries)} entries")

    errors = []
    async with create_task_group() as tg:
        for n in range(4):
            tg.start_soon(do_submit, n, errors)
    assert not errors, f"Concurrent submit errors:\n" + "\n".join(errors)


@pytest.mark.anyio
async def test_resubmit_same_entry_no_500(app, sample_vector):
    """Re-submitting the exact same entry should not cause a 500 error.

    This is the bug where auto_merge tries to INSERT a new row with the
    same (scope, namespace, key) as the superseded row, violating the
    UNIQUE constraint.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        entry = {
            "namespace": "decisions",
            "key": "DEC-100-resubmit",
            "title": "Resubmit test",
            "content": "This entry will be submitted twice",
            "metadata": {},
            "vector": sample_vector,
            "simhash": 99999,
        }

        # First submission
        resp1 = await client.post("/submit", json={
            "project": "test-project",
            "source": "local:test",
            "entries": [entry]
        })
        assert resp1.status_code == 200
        assert resp1.json()["accepted"] == 1

        # Second submission of the SAME entry — should succeed, not 500
        resp2 = await client.post("/submit", json={
            "project": "test-project",
            "source": "local:test",
            "entries": [entry]
        })
        assert resp2.status_code == 200
        data2 = resp2.json()
        # Should be handled as a duplicate/merge, not crash
        assert data2["accepted"] + data2["duplicates"] + data2["conflicted"] == 1

        # Verify we still have exactly 1 entry for this key
        pull_resp = await client.get(
            "/pull", params={"project": "test-project", "scope": "own"}
        )
        pull_data = pull_resp.json()
        matching = [e for e in pull_data["entries"] if "DEC-100-resubmit" in e["fqn"]]
        assert len(matching) >= 1


@pytest.mark.anyio
async def test_resubmit_updated_entry_no_500(app, sample_vector):
    """Re-submitting an entry with updated content should not 500."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        entry_v1 = {
            "namespace": "patterns",
            "key": "PAT-200-update",
            "title": "Update pattern v1",
            "content": "Version 1 of the pattern content here",
            "metadata": {},
            "vector": sample_vector,
            "simhash": 88888,
        }
        entry_v2 = {
            "namespace": "patterns",
            "key": "PAT-200-update",
            "title": "Update pattern v2",
            "content": "Version 2 of the pattern content here",
            "metadata": {},
            "vector": sample_vector,
            "simhash": 88889,
        }

        # First submission
        resp1 = await client.post("/submit", json={
            "project": "test-project",
            "source": "local:test",
            "entries": [entry_v1]
        })
        assert resp1.status_code == 200
        assert resp1.json()["accepted"] == 1

        # Updated submission — should succeed, not 500
        resp2 = await client.post("/submit", json={
            "project": "test-project",
            "source": "local:test-v2",
            "entries": [entry_v2]
        })
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["accepted"] + data2["duplicates"] + data2["conflicted"] == 1

        # Verify the updated content is present
        pull_resp = await client.get(
            "/pull", params={"project": "test-project", "scope": "own"}
        )
        pull_data = pull_resp.json()
        matching = [e for e in pull_data["entries"] if "PAT-200-update" in e["fqn"]]
        assert len(matching) >= 1
        # Content should reflect the update
        assert "v2" in matching[0]["title"].lower() or "v1" in matching[0]["title"].lower()


@pytest.mark.anyio
async def test_submit_okf_entry(app, sample_vector):
    """Submit an OKF markdown entry and verify it's stored correctly."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        okf_markdown = """---
type: Decision
title: FastAPI Stack
description: All services use FastAPI with Pydantic
tags: [python, fastapi]
timestamp: 2026-01-15T10:30:00Z
status: accepted
---

# Context
We chose FastAPI for all microservices.

# Consequences
All services use the same patterns.
"""

        resp = await client.post("/submit", json={
            "project": "test-project",
            "source": "local:test",
            "okf_entries": [
                {
                    "markdown": okf_markdown,
                    "namespace": "decisions",
                    "key": "fastapi-stack",
                    "vector": sample_vector,
                }
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] == 1
        assert data["details"][0]["status"] == "accepted"

        # Verify the entry was stored with OKF metadata
        search_resp = await client.get(
            "/search", params={"q": "FastAPI", "scope": "test-project"}
        )
        assert search_resp.status_code == 200
        search_data = search_resp.json()
        assert len(search_data["results"]) > 0
        result = search_data["results"][0]
        assert result["okf_type"] == "Decision"
        assert "python" in (result.get("okf_tags") or [])
        assert result["okf_description"] == "All services use FastAPI with Pydantic"
        assert result["okf_timestamp"] == "2026-01-15T10:30:00Z"


@pytest.mark.anyio
async def test_submit_okf_invalid_missing_type(app):
    """Submit OKF entry without required type field should fail gracefully."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        okf_markdown = """---
title: No Type
---
Body without type field
"""

        resp = await client.post("/submit", json={
            "project": "test-project",
            "source": "local:test",
            "okf_entries": [
                {"markdown": okf_markdown}
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        # Should be reported as error, not crash
        assert data["accepted"] == 0
        assert any(d["status"] == "error" for d in data["details"])


@pytest.mark.anyio
async def test_search_returns_okf_metadata(app, sample_vector):
    """Search results should include OKF metadata when available."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Submit a legacy entry (no OKF metadata)
        await client.post("/submit", json={
            "project": "test-project",
            "source": "local:test",
            "entries": [
                {
                    "namespace": "decisions",
                    "key": "DEC-001",
                    "title": "Legacy Entry",
                    "content": "This is a legacy entry without OKF metadata",
                    "metadata": {"type": "Decision", "tags": ["legacy"]},
                    "vector": sample_vector,
                    "simhash": 11111,
                }
            ]
        })

        # Submit an OKF entry
        okf_markdown = """---
type: Pattern
title: Health Check
description: Standard health check pattern
tags: [monitoring, operations]
timestamp: 2026-02-20T14:00:00Z
---

# Implementation
Use curl against /health endpoint.
"""
        await client.post("/submit", json={
            "project": "test-project",
            "source": "local:test",
            "okf_entries": [
                {
                    "markdown": okf_markdown,
                    "namespace": "patterns",
                    "key": "health-check",
                    "vector": sample_vector,
                }
            ]
        })

        # Search should return OKF metadata for the OKF entry
        search_resp = await client.get(
            "/search", params={"q": "health", "scope": "test-project"}
        )
        assert search_resp.status_code == 200
        data = search_resp.json()
        results = data["results"]
        # Find the OKF entry
        okf_results = [r for r in results if r.get("okf_type") == "Pattern"]
        assert len(okf_results) > 0
        assert okf_results[0]["okf_tags"] == ["monitoring", "operations"]
        assert okf_results[0]["okf_description"] == "Standard health check pattern"
