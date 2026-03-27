"""World state org filtering test — uses FABRIC_TEST_MODE (set by conftest)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_worldstate_org_filter(client):
    """Requesting with X-Org-ID should not error."""
    # No header => all
    r = client.get("/api/v1/worldstate")
    assert r.status_code == 200
    assert isinstance(r.json().get("devices", []), list)

    # With org header => filtered (best-effort in test mode)
    r2 = client.get("/api/v1/worldstate", headers={"X-Org-ID": "org1"})
    assert r2.status_code == 200
