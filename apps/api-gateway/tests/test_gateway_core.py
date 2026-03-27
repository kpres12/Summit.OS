"""API Gateway core functionality tests.

Tests run against the gateway directly (DB-backed, but proxied
calls to downstream services will fail gracefully in test env).
"""

import sys
import os

os.environ["GATEWAY_TEST_MODE"] = "true"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from main import app

import pytest


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --- Health / Probes ---


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["service"] == "api-gateway"


def test_livez(client):
    r = client.get("/livez")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "api_gateway_errors_total" in r.text


# --- Task Submission ---


def test_submit_low_risk_task(client):
    """Low risk tasks attempt immediate dispatch (will 502 since tasking isn't running)."""
    r = client.post(
        "/v1/tasks",
        json={
            "asset_id": "uav-001",
            "action": "patrol",
            "risk_level": "LOW",
        },
    )
    # 502 expected since TASKING_URL is unreachable in test
    assert r.status_code in (200, 502)


def test_submit_high_risk_task_requires_approval(client):
    """High risk tasks should be stored as PENDING_APPROVAL, no dispatch needed."""
    r = client.post(
        "/v1/tasks",
        json={
            "asset_id": "uav-002",
            "action": "deploy_payload",
            "risk_level": "HIGH",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "PENDING_APPROVAL"
    assert "task_id" in body


def test_submit_critical_risk_task_requires_approval(client):
    r = client.post(
        "/v1/tasks",
        json={
            "asset_id": "uav-003",
            "action": "emergency_drop",
            "risk_level": "CRITICAL",
        },
    )
    assert r.status_code == 200
    assert r.json()["status"] == "PENDING_APPROVAL"


# --- Pending Tasks ---


def test_list_pending_tasks(client):
    # Submit a high-risk task first
    client.post(
        "/v1/tasks",
        json={
            "asset_id": "uav-pending",
            "action": "high_risk_action",
            "risk_level": "HIGH",
        },
    )
    r = client.get("/v1/tasks/pending")
    assert r.status_code == 200
    body = r.json()
    assert "pending_tasks" in body
    assert isinstance(body["pending_tasks"], list)
    assert len(body["pending_tasks"]) >= 1


# --- Task Approval ---


def test_approve_task(client):
    """Approve a pending task."""
    # Submit high-risk
    submit_r = client.post(
        "/v1/tasks",
        json={
            "asset_id": "uav-approve-test",
            "action": "suppression",
            "risk_level": "HIGH",
        },
    )
    task_id = submit_r.json()["task_id"]

    # Approve it (dispatch will 502 since tasking isn't running, but approval record should update)
    r = client.post(
        f"/v1/tasks/{task_id}/approve",
        json={
            "approved_by": "test-supervisor",
        },
    )
    # 502 expected because tasking service isn't reachable
    assert r.status_code in (200, 502)


def test_approve_nonexistent_task(client):
    r = client.post(
        "/v1/tasks/nonexistent-task-id/approve",
        json={
            "approved_by": "test",
        },
    )
    assert r.status_code == 404


# --- Feature Flags ---


def test_feature_flags(client):
    r = client.get("/feature_flags")
    assert r.status_code == 200
    body = r.json()
    assert "features" in body


def test_feature_flags_with_domain(client):
    r = client.get("/feature_flags", params={"domain": "wildfire"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("domain") == "wildfire"


# --- Proxied endpoints (expect 502 since downstream services aren't running) ---


def test_worldstate_proxy(client):
    r = client.get("/v1/worldstate")
    # 502 expected since fabric isn't running
    assert r.status_code in (200, 502)


def test_alerts_proxy(client):
    r = client.get("/v1/alerts")
    assert r.status_code in (200, 502)


def test_observations_proxy(client):
    r = client.get("/v1/observations")
    assert r.status_code in (200, 502)


def test_advisories_proxy(client):
    r = client.get("/v1/advisories")
    assert r.status_code in (200, 502)
