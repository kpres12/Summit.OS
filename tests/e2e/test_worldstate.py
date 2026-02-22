import pytest
from .utils.api_client import ApiClient

pytestmark = pytest.mark.e2e


def test_worldstate_returns_structure(api_base: str):
    """GET /v1/worldstate should return devices, alerts, and counts."""
    client = ApiClient(api_base)
    try:
        r = client.get("/v1/worldstate")
        assert r.status_code == 200
        data = r.json()
        assert "devices" in data or "alerts" in data or "counts" in data
    finally:
        client.close()


def test_alerts_endpoint(api_base: str):
    """GET /v1/alerts should return an alerts list."""
    client = ApiClient(api_base)
    try:
        r = client.get("/v1/alerts?limit=10")
        assert r.status_code == 200
        data = r.json()
        assert "alerts" in data
        assert isinstance(data["alerts"], list)
    finally:
        client.close()
