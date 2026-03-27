"""
Tests for tasking service gap implementations.

Gap 3: ExecutionMonitor wiring (checked via import only — full test needs DB)
Gap 5: Mission replay router endpoints
Gap 7: DEM terrain wiring (smoke test import)
"""

import os

os.environ["TASKING_TEST_MODE"] = "true"
os.environ["PYTHONPATH"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── Gap 3: ExecutionMonitor importable ───────────────────────────────────────


def test_execution_monitor_importable():
    """ExecutionMonitor can be imported."""
    from execution_monitor import ExecutionMonitor

    assert ExecutionMonitor is not None


def test_execution_monitor_haversine():
    """_haversine_m returns reasonable distances."""
    from execution_monitor import _haversine_m

    # NYC to LA is ~3940 km
    dist = _haversine_m(40.7128, -74.0060, 34.0522, -118.2437)
    assert 3_900_000 < dist < 4_000_000


# ── Gap 5: Replay router ─────────────────────────────────────────────────────


def test_replay_timeline_404_for_unknown_mission(client):
    """Unknown mission returns 404."""
    r = client.get("/api/v1/missions/nonexistent-id/replay/timeline")
    assert r.status_code == 404


def test_replay_snapshot_404_for_unknown_mission(client):
    r = client.get("/api/v1/missions/nonexistent-id/replay/snapshot")
    assert r.status_code == 404


def test_replay_clear_nonexistent_mission(client):
    """DELETE on nonexistent replay is idempotent."""
    r = client.delete("/api/v1/missions/nonexistent-id/replay")
    assert r.status_code == 200
    assert r.json()["status"] == "cleared"


def test_replay_record_and_retrieve():
    """record_snapshot + GET timeline round-trip works."""
    from replay_router import record_snapshot, _replay_store

    mission_id = "test-mission-replay-001"
    # Clear any state
    _replay_store.pop(mission_id, None)

    snapshot = {
        "ts_iso": "2024-01-01T12:00:00+00:00",
        "mission_id": mission_id,
        "assignments": [
            {
                "asset_id": "uav-1",
                "lat": 34.0,
                "lon": -118.0,
                "status": "ACTIVE",
                "completed_seq": 0,
            }
        ],
        "events": [],
    }
    record_snapshot(mission_id, snapshot)
    assert len(_replay_store[mission_id]) == 1


def test_replay_timeline_via_api(client):
    """After recording a snapshot, timeline endpoint returns it."""
    from replay_router import record_snapshot, _replay_store

    mission_id = "test-mission-api-001"
    _replay_store.pop(mission_id, None)
    record_snapshot(
        mission_id,
        {
            "ts_iso": "2024-06-01T10:00:00+00:00",
            "mission_id": mission_id,
            "assignments": [],
            "events": [],
        },
    )
    r = client.get(f"/api/v1/missions/{mission_id}/replay/timeline")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["mission_id"] == mission_id


def test_replay_snapshot_by_index(client):
    """GET snapshot?index=0 returns the first snapshot."""
    from replay_router import record_snapshot, _replay_store

    mission_id = "test-mission-idx-001"
    _replay_store.pop(mission_id, None)
    record_snapshot(
        mission_id,
        {
            "ts_iso": "2024-06-01T10:00:00+00:00",
            "mission_id": mission_id,
            "assignments": [{"asset_id": "uav-2"}],
            "events": [],
        },
    )
    r = client.get(f"/api/v1/missions/{mission_id}/replay/snapshot?index=0")
    assert r.status_code == 200
    body = r.json()
    assert body["mission_id"] == mission_id


def test_replay_max_snapshots_cap():
    """Replay store respects REPLAY_MAX_SNAPSHOTS_PER_MISSION cap."""
    import replay_router

    original_cap = replay_router.REPLAY_MAX_PTS
    replay_router.REPLAY_MAX_PTS = 3
    mission_id = "test-mission-cap-001"
    replay_router._replay_store.pop(mission_id, None)
    for i in range(5):
        replay_router.record_snapshot(
            mission_id,
            {
                "ts_iso": f"2024-01-01T12:0{i}:00+00:00",
                "mission_id": mission_id,
            },
        )
    assert len(replay_router._replay_store[mission_id]) == 3
    replay_router.REPLAY_MAX_PTS = original_cap


# ── Gap 7: DEM import ────────────────────────────────────────────────────────


def test_dem_provider_importable():
    """DEMProvider can be imported from packages.geo.dem."""
    from packages.geo.dem import DEMProvider, get_provider

    assert DEMProvider is not None
    provider = DEMProvider()
    assert provider is not None


def test_dem_returns_zero_without_tiles():
    """DEMProvider returns 0.0 gracefully when no tiles are present."""
    from packages.geo.dem import DEMProvider

    dem = DEMProvider("/nonexistent/dem/dir")
    elev = dem.get_elevation(34.05, -118.24)
    assert elev == 0.0


def test_dem_haversine():
    """_haversine_m helper gives correct distance."""
    from packages.geo.dem import _haversine_m

    dist = _haversine_m(0, 0, 0, 1)
    # 1 degree lon at equator ≈ 111.32 km
    assert 111_000 < dist < 112_000


def test_dem_tile_key():
    """Tile key formatting is correct."""
    from packages.geo.dem import DEMProvider

    dem = DEMProvider()
    assert dem._tile_key(34.5, -118.3) == "N34W119"
    assert dem._tile_key(-10.2, 20.8) == "S11E020"


def test_dem_los_returns_true_without_tiles():
    """check_line_of_sight returns True (open) when no terrain data."""
    from packages.geo.dem import DEMProvider

    dem = DEMProvider("/nonexistent/dem/dir")
    result = dem.check_line_of_sight(34.0, -118.0, 100, 34.1, -118.1, 100)
    assert result is True


def test_dem_elevation_profile_length():
    """get_elevation_profile returns n_samples+1 points."""
    from packages.geo.dem import DEMProvider

    dem = DEMProvider("/nonexistent/dem/dir")
    profile = dem.get_elevation_profile(34.0, -118.0, 34.1, -118.1, n_samples=10)
    assert len(profile) == 11
    # First point distance is 0
    assert profile[0][0] == 0.0
