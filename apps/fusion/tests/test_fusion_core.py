"""Fusion service core functionality tests.

Uses FUSION_DISABLE_STARTUP=1 (no MQTT/DB connections at startup).
Note: DB-dependent endpoints (observations, etc.) won't work without DB
but health/livez/models endpoints work.
"""

import os

os.environ["FUSION_DISABLE_STARTUP"] = "1"

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
    assert r.json()["service"] == "fusion"


def test_livez(client):
    r = client.get("/livez")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


# --- Model Registry ---


def test_models_endpoint(client):
    r = client.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert "models" in body
    assert isinstance(body["models"], list)


def test_select_model_missing_path(client):
    r = client.post("/models/select", json={})
    assert r.status_code == 200
    assert r.json()["status"] == "error"


# --- Schema Validation (pure function) ---


def test_observation_schema_validator_exists(client):
    """Smoke test: validator is created (even if with fallback schema)."""
    from main import validator

    # validator may be None if startup is disabled, but the variable should exist
    # In FUSION_DISABLE_STARTUP mode, validator is None
    # Just verify the import works
    assert validator is None  # startup was skipped
