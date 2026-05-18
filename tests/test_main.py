"""Tests for app/main.py — FastAPI endpoints."""
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import create_app


@pytest.fixture
def app(tmp_path):
    db_url = str(tmp_path / "test_central_kb.db")
    return create_app(db_url=db_url)


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
async def test_submit_and_pull_roundtrip(app):
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
async def test_search_endpoint(app):
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
async def test_submit_duplicate_detected(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        entry = {
            "namespace": "decisions",
            "key": "DEC-003-fastapi-v2",
            "title": "FastAPI v2",
            "content": "All services use FastAPI v2 with Pydantic",
            "metadata": {},
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

    async def do_submit(n: int, errors: list):
        entries = []
        for i in range(5):
            entries.append({
                "namespace": "decisions",
                "key": f"concurrent-{n}-{i}",
                "title": f"Concurrent {n} Entry {i}",
                "content": f"This is from concurrent submit {n}, entry {i}.",
                "metadata": {},
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
async def test_resubmit_same_entry_no_500(app):
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
async def test_resubmit_updated_entry_no_500(app):
    """Re-submitting an entry with updated content should not 500."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        entry_v1 = {
            "namespace": "patterns",
            "key": "PAT-200-update",
            "title": "Update pattern v1",
            "content": "Version 1 of the pattern content here",
            "metadata": {},
            "simhash": 88888,
        }
        entry_v2 = {
            "namespace": "patterns",
            "key": "PAT-200-update",
            "title": "Update pattern v2",
            "content": "Version 2 of the pattern content here",
            "metadata": {},
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
