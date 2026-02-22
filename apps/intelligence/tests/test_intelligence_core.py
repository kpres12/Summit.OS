"""Intelligence service core functionality tests.

Uses INTELLIGENCE_TEST_MODE=true (SQLite backend, no Redis).
"""
import os
os.environ["INTELLIGENCE_TEST_MODE"] = "true"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from main import app, _calculate_risk_level, _generate_advisory_message, _extract_features

import pytest

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c






# --- Pure function tests (no DB needed) ---

def test_risk_level_critical(client):
    assert _calculate_risk_level({"confidence": 0.90}) == "CRITICAL"
    assert _calculate_risk_level({"confidence": 0.95}) == "CRITICAL"
    assert _calculate_risk_level({"confidence": 1.0}) == "CRITICAL"


def test_risk_level_high(client):
    assert _calculate_risk_level({"confidence": 0.70}) == "HIGH"
    assert _calculate_risk_level({"confidence": 0.80}) == "HIGH"


def test_risk_level_medium(client):
    assert _calculate_risk_level({"confidence": 0.50}) == "MEDIUM"
    assert _calculate_risk_level({"confidence": 0.65}) == "MEDIUM"


def test_risk_level_low(client):
    assert _calculate_risk_level({"confidence": 0.0}) == "LOW"
    assert _calculate_risk_level({"confidence": 0.30}) == "LOW"
    assert _calculate_risk_level({"confidence": 0.49}) == "LOW"


def test_risk_level_missing_confidence(client):
    assert _calculate_risk_level({}) == "LOW"


def test_advisory_message_with_location(client):
    msg = _generate_advisory_message(
        {"class": "smoke", "confidence": 0.85, "lat": 34.05, "lon": -118.24},
        "CRITICAL",
    )
    assert "CRITICAL risk" in msg
    assert "smoke" in msg
    assert "85%" in msg
    assert "34.0500" in msg


def test_advisory_message_without_location(client):
    msg = _generate_advisory_message(
        {"class": "leak", "confidence": 0.60},
        "MEDIUM",
    )
    assert "MEDIUM risk" in msg
    assert "leak" in msg
    assert "at (" not in msg


def test_advisory_message_unknown_class(client):
    msg = _generate_advisory_message({}, "LOW")
    assert "unknown" in msg


def test_extract_features_basic(client):
    features = _extract_features({
        "class": "smoke", "lat": 34.0, "lon": -118.0, "confidence": 0.9,
    })
    assert len(features) == 7
    assert features[0] == 34.0  # lat
    assert features[1] == -118.0  # lon
    assert features[2] == 0.9  # confidence


def test_extract_features_empty(client):
    features = _extract_features({})
    assert len(features) == 7
    assert features[0] == 0.0
    assert features[2] == 0.0


# --- API endpoint tests ---

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["service"] == "intelligence"


def test_livez(client):
    r = client.get("/livez")
    assert r.status_code == 200


def test_list_advisories_empty(client):
    r = client.get("/advisories")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_list_advisories_with_risk_filter(client):
    r = client.get("/advisories", params={"risk_level": "HIGH"})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_list_advisories_with_org_filter(client):
    r = client.get("/advisories", headers={"X-Org-ID": "org-test"})
    assert r.status_code == 200
