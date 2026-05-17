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
