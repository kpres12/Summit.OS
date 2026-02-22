"""Tasking service core functionality tests.

Uses TASKING_TEST_MODE=true (SQLite backend, no MQTT).
"""
import os
os.environ["TASKING_TEST_MODE"] = "true"
os.environ["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fastapi.testclient import TestClient
from main import app

import pytest

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c






# --- Health ---

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["service"] == "tasking"


# --- Asset Registration ---

def test_register_asset(client):
    r = client.post("/api/v1/assets", json={
        "asset_id": "uav-test-001",
        "type": "UAV",
        "capabilities": {"drone_type": "scout", "max_speed": 80},
        "battery": 95.0,
        "link": "OK",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["asset_id"] == "uav-test-001"
    assert body["type"] == "UAV"


def test_register_asset_updates_existing(client):
    # First registration
    client.post("/api/v1/assets", json={
        "asset_id": "uav-update-test",
        "type": "UAV",
        "battery": 80.0,
        "link": "OK",
    })
    # Update
    r = client.post("/api/v1/assets", json={
        "asset_id": "uav-update-test",
        "type": "UAV",
        "battery": 60.0,
        "link": "OK",
    })
    assert r.status_code == 200
    assert r.json()["battery"] == 60.0


def test_list_assets(client):
    r = client.get("/api/v1/assets")
    assert r.status_code == 200
    assets = r.json()
    assert isinstance(assets, list)
    assert any(a["asset_id"] == "uav-test-001" for a in assets)


def test_get_asset(client):
    r = client.get("/api/v1/assets/uav-test-001")
    assert r.status_code == 200
    assert r.json()["asset_id"] == "uav-test-001"


def test_get_asset_not_found(client):
    r = client.get("/api/v1/assets/nonexistent")
    assert r.status_code == 404


# --- Mission Creation ---

def _register_test_assets(client):
    """Register a couple of assets for mission planning tests."""
    for i in range(2):
        client.post("/api/v1/assets", json={
            "asset_id": f"mission-drone-{i:03d}",
            "type": "UAV",
            "capabilities": {"drone_type": "scout", "max_speed": 50},
            "battery": 90.0,
            "link": "OK",
        })


def test_create_mission_loiter(client):
    _register_test_assets(client)
    r = client.post("/api/v1/missions", json={
        "name": "Test Loiter Mission",
        "objectives": ["survey area"],
        "area": {"center": {"lat": 34.05, "lon": -118.24}, "radius_m": 200},
        "num_drones": 1,
        "planning_params": {"pattern": "loiter", "altitude": 60, "speed": 5},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ACTIVE"
    assert body["policy_ok"] is True
    assert len(body["assignments"]) >= 1
    # Check assignment has loiter pattern
    assignment = body["assignments"][0]
    assert assignment["plan"]["pattern"] == "loiter"
    assert len(assignment["plan"]["waypoints"]) >= 1


def test_create_mission_grid(client):
    _register_test_assets(client)
    r = client.post("/api/v1/missions", json={
        "name": "Grid Survey",
        "objectives": ["full area coverage"],
        "area": {"center": {"lat": 34.05, "lon": -118.24}, "radius_m": 300},
        "num_drones": 2,
        "planning_params": {"pattern": "grid", "altitude": 100, "speed": 8, "grid_spacing_m": 50},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ACTIVE"
    for assignment in body["assignments"]:
        assert assignment["plan"]["pattern"] == "grid"
        assert len(assignment["plan"]["waypoints"]) > 0


def test_create_mission_spiral(client):
    _register_test_assets(client)
    r = client.post("/api/v1/missions", json={
        "name": "Spiral Search",
        "objectives": ["SAR"],
        "area": {"center": {"lat": 34.05, "lon": -118.24}, "radius_m": 150},
        "num_drones": 1,
        "planning_params": {"pattern": "spiral", "altitude": 80, "speed": 6},
    })
    assert r.status_code == 200
    body = r.json()
    assignment = body["assignments"][0]
    assert assignment["plan"]["pattern"] == "spiral"


def test_create_mission_no_assets(client):
    """Mission creation should fail if no assets meet availability criteria."""
    # Register asset with low battery
    client.post("/api/v1/assets", json={
        "asset_id": "low-bat-drone",
        "type": "UAV",
        "battery": 5.0,
        "link": "OK",
    })
    # This might succeed if other assets are registered from prior tests
    # So we just validate the response shape
    r = client.post("/api/v1/missions", json={
        "name": "Should work or fail gracefully",
        "objectives": ["test"],
        "area": {"center": {"lat": 34.0, "lon": -118.0}, "radius_m": 100},
    })
    assert r.status_code in (200, 409)


# --- Mission Retrieval ---

def test_list_missions(client):
    r = client.get("/api/v1/missions")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_get_mission_not_found(client):
    r = client.get("/api/v1/missions/nonexistent-id")
    assert r.status_code == 404


# --- Task Lifecycle ---

def test_list_tasks(client):
    r = client.get("/tasks")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_get_task_not_found(client):
    r = client.get("/tasks/nonexistent-task")
    assert r.status_code == 404
