"""
Tests for fabric service gap implementations.

Gap 6: Alert escalation service + ACK endpoint
Gap 7: GET /api/v1/elevation endpoint
"""

import os

os.environ["FABRIC_TEST_MODE"] = "true"

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── Gap 6: AlertEscalationService importable ─────────────────────────────────


def test_escalation_service_importable():
    from alert_escalation import AlertEscalationService

    svc = AlertEscalationService({})
    assert svc is not None


def test_escalation_config_defaults():
    from alert_escalation import EscalationConfig

    cfg = EscalationConfig()
    assert cfg.check_interval_s > 0
    assert cfg.unack_timeout_s > 0


def test_escalation_does_not_escalate_acknowledged():
    """Acknowledged alerts are skipped."""
    import asyncio
    from alert_escalation import AlertEscalationService

    store = {
        "alert-001": {
            "alert_id": "alert-001",
            "acknowledged": True,
            "ts_iso": "2020-01-01T00:00:00+00:00",
            "severity": "HIGH",
            "description": "test",
            "source": "test",
        }
    }
    svc = AlertEscalationService(store)
    asyncio.run(svc._check())
    # Should not be escalated
    assert "alert-001" not in svc._escalated


def test_escalation_escalates_stale_unacked():
    """Alert older than timeout and unacknowledged gets escalated."""
    import asyncio
    from alert_escalation import AlertEscalationService, EscalationConfig

    cfg = EscalationConfig(
        unack_timeout_s=0,  # escalate immediately
        webhook_url=None,  # no actual webhook
        smtp_host=None,  # no actual email
    )
    store = {
        "alert-002": {
            "alert_id": "alert-002",
            "acknowledged": False,
            "ts_iso": "2020-01-01T00:00:00+00:00",  # very old
            "severity": "CRITICAL",
            "description": "fire detected",
            "source": "cam-01",
        }
    }
    svc = AlertEscalationService(store, config=cfg)
    asyncio.run(svc._check())
    assert "alert-002" in svc._escalated
    assert store["alert-002"]["status"] == "escalated"


# ── Gap 6: ACK endpoint ───────────────────────────────────────────────────────


def test_ack_nonexistent_alert(client):
    """ACK on unknown alert returns 404."""
    r = client.post("/api/v1/alerts/nonexistent-alert-id/acknowledge")
    assert r.status_code == 404


def test_ack_known_alert(client):
    """ACK on alert in _escalation_alerts dict works."""
    from main import _escalation_alerts

    _escalation_alerts["test-alert-ack"] = {
        "alert_id": "test-alert-ack",
        "severity": "HIGH",
        "description": "test",
        "source": "test",
        "ts_iso": "2024-01-01T00:00:00+00:00",
        "acknowledged": False,
    }
    r = client.post("/api/v1/alerts/test-alert-ack/acknowledge")
    assert r.status_code == 200
    assert r.json()["status"] == "acknowledged"
    assert _escalation_alerts["test-alert-ack"]["acknowledged"] is True
    # Cleanup
    del _escalation_alerts["test-alert-ack"]


# ── Gap 7: Elevation endpoint ─────────────────────────────────────────────────


def test_elevation_endpoint_exists(client):
    """GET /api/v1/elevation returns 200 with elevation_m field."""
    r = client.get("/api/v1/elevation?lat=34.05&lon=-118.24")
    assert r.status_code == 200
    body = r.json()
    assert "elevation_m" in body
    assert isinstance(body["elevation_m"], (int, float))
    assert body["lat"] == pytest.approx(34.05)
    assert body["lon"] == pytest.approx(-118.24)


def test_elevation_endpoint_no_tiles():
    """When DEM_DIR has no tiles, returns 0.0 (fail-open)."""
    os.environ["DEM_DIR"] = "/nonexistent/dem"
    with TestClient(app) as c:
        r = c.get("/api/v1/elevation?lat=34.05&lon=-118.24")
    assert r.status_code == 200
    assert r.json()["elevation_m"] == 0.0
