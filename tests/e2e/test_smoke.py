import os
import pytest
from .utils.api_client import ApiClient

pytestmark = pytest.mark.e2e


def test_smoke_health(api_base: str):
    client = ApiClient(api_base)
    try:
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") in ("ok", "ready", "alive")
    finally:
        client.close()
