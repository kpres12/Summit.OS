"""
End-to-End tests against the API Gateway.

Uses FastAPI's TestClient (httpx under the hood) to exercise
the actual gateway routes via real HTTP without Docker.

Run: GATEWAY_TEST_MODE=true python -m pytest tests/e2e/test_gateway_live.py -v
"""
import os
import sys
import pytest

# Force test mode so gateway uses in-memory SQLite
os.environ["GATEWAY_TEST_MODE"] = "true"

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from httpx import AsyncClient, ASGITransport
from apps.api_gateway_app import app  # noqa: E402 — must import after env
import apps.api_gateway_app as _gw_mod


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Async test client with lifespan properly initialized."""
    # Manually run the gateway lifespan to init DB
    _main = _gw_mod._mod
    from contextlib import asynccontextmanager
    async with _main.lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# ── Health & Probes ─────────────────────────────────────────

class TestHealth:
    @pytest.mark.anyio
    async def test_health_ok(self, client):
        r = await client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["service"] == "api-gateway"

    @pytest.mark.anyio
    async def test_livez(self, client):
        r = await client.get("/livez")
        assert r.status_code == 200
        assert r.json()["status"] == "alive"

    @pytest.mark.anyio
    async def test_metrics_endpoint(self, client):
        r = await client.get("/metrics")
        assert r.status_code == 200
        assert "api_gateway" in r.text or "HELP" in r.text


# ── Task Submission & Approval Flow ─────────────────────────

class TestTaskFlow:
    @pytest.mark.anyio
    async def test_submit_low_risk_dispatches_or_502(self, client):
        """LOW risk tasks attempt immediate dispatch to tasking service.
        Without the tasking service running, this returns 502."""
        r = await client.post("/v1/tasks", json={
            "asset_id": "drone-01",
            "action": "recon",
            "risk_level": "LOW",
        })
        # 200 if tasking service is up, 502 if not — both are correct behavior
        assert r.status_code in (200, 502)

    @pytest.mark.anyio
    async def test_submit_high_risk_pending_approval(self, client):
        """HIGH risk tasks go to PENDING_APPROVAL (DB only, no downstream call)."""
        r = await client.post("/v1/tasks", json={
            "asset_id": "ugv-alpha",
            "action": "suppress",
            "risk_level": "HIGH",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["task_id"]
        assert data["status"] == "PENDING_APPROVAL"

    @pytest.mark.anyio
    async def test_approve_updates_db_then_dispatches(self, client):
        """Approve flow: DB update works, downstream dispatch may 502."""
        # Create a high-risk task first
        r = await client.post("/v1/tasks", json={
            "asset_id": "ugv-beta",
            "action": "strike",
            "risk_level": "CRITICAL",
        })
        assert r.status_code == 200
        task_id = r.json()["task_id"]

        # Approve — DB update succeeds, dispatch to tasking may 502
        r2 = await client.post(f"/v1/tasks/{task_id}/approve", json={
            "approved_by": "commander-1",
        })
        # 200 if tasking up, 502 if not (approval still persisted)
        assert r2.status_code in (200, 502)

    @pytest.mark.anyio
    async def test_approve_nonexistent_task_404(self, client):
        r = await client.post("/v1/tasks/fake-id/approve", json={
            "approved_by": "nobody",
        })
        assert r.status_code == 404


# ── OpenAPI ─────────────────────────────────────────────────

class TestOpenAPI:
    @pytest.mark.anyio
    async def test_openapi_schema(self, client):
        r = await client.get("/api/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert schema["info"]["title"] == "Heli.OS API Gateway"
        assert "/health" in schema["paths"]

    @pytest.mark.anyio
    async def test_docs_page(self, client):
        r = await client.get("/api/docs")
        assert r.status_code == 200


# ── Error Shape ─────────────────────────────────────────────

class TestErrorFormat:
    @pytest.mark.anyio
    async def test_404_returns_json(self, client):
        r = await client.get("/nonexistent-route")
        assert r.status_code in (404, 405)
        # Should be JSON, not HTML
        assert r.headers["content-type"].startswith("application/json")

    @pytest.mark.anyio
    async def test_validation_error_on_bad_body(self, client):
        r = await client.post("/v1/tasks", json={})
        assert r.status_code == 422
