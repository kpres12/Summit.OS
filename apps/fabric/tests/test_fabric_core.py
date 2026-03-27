"""Fabric service core functionality tests.

Uses FABRIC_TEST_MODE=true (SQLite backend, no MQTT/Redis).
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


# --- Health / Readiness ---


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["service"] == "fabric"


def test_livez(client):
    r = client.get("/livez")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


# --- Node Registration ---


def test_register_node(client):
    payload = {
        "id": "test-tower-001",
        "type": "TOWER",
        "pubkey": "pk_test",
        "fw_version": "2.0.0",
        "location": {"lat": 34.05, "lon": -118.24, "elev_m": 500},
        "capabilities": ["THERMAL", "EO"],
        "comm": ["LTE"],
    }
    r = client.post("/api/v1/nodes/register", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "accepted"
    assert "token" in body
    assert "mqtt_topics" in body
    assert "pub" in body["mqtt_topics"]
    assert "sub" in body["mqtt_topics"]


def test_get_registered_node(client):
    r = client.get("/api/v1/nodes/test-tower-001")
    assert r.status_code == 200
    node = r.json()
    assert node["id"] == "test-tower-001"
    assert node["type"] == "TOWER"


def test_get_nonexistent_node(client):
    r = client.get("/api/v1/nodes/does-not-exist")
    assert r.status_code == 404


def test_retire_node(client):
    # Register first
    client.post(
        "/api/v1/nodes/register",
        json={
            "id": "retire-me",
            "type": "DRONE",
            "capabilities": [],
            "comm": [],
        },
    )
    r = client.delete("/api/v1/nodes/retire-me")
    assert r.status_code == 200
    assert r.json()["status"] == "retired"


def test_refresh_token(client):
    r = client.post("/api/v1/nodes/test-tower-001/token")
    assert r.status_code == 200
    body = r.json()
    assert "token" in body
    assert body["expires_in"] == 600


# --- World State ---


def test_worldstate_returns_structure(client):
    r = client.get("/api/v1/worldstate")
    assert r.status_code == 200
    body = r.json()
    assert "devices" in body
    assert "alerts" in body
    assert "counts" in body
    assert "ts_iso" in body


def test_worldstate_org_filter(client):
    """Requesting with X-Org-ID header should not error (filtering is best-effort in test mode)."""
    r = client.get("/api/v1/worldstate", headers={"X-Org-ID": "org-test"})
    assert r.status_code == 200


# --- Coverage ---


def test_coverage_list_empty(client):
    r = client.get("/api/v1/coverage")
    assert r.status_code == 200
    assert "coverages" in r.json()


def test_coverage_union_empty(client):
    r = client.get("/api/v1/coverage/union")
    assert r.status_code == 200
    body = r.json()
    assert "union" in body
    assert "count" in body


# --- Geofences ---


def test_create_geofence(client):
    r = client.post(
        "/api/v1/geofences",
        json={
            "name": "test-zone",
            "props": {"type": "exclusion"},
        },
    )
    assert r.status_code == 200
    assert r.json()["status"] == "created"


def test_list_geofences(client):
    r = client.get("/api/v1/geofences")
    assert r.status_code == 200
    body = r.json()
    assert "geofences" in body
    assert isinstance(body["geofences"], list)


def test_geofence_contains_no_geo(client):
    """Without PostGIS, geofence_contains should return True (fallback)."""
    r = client.get("/api/v1/geofences/contains", params={"lat": 34.05, "lon": -118.24})
    assert r.status_code == 200
    assert r.json()["contains"] is True
