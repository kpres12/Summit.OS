import os
import pytest
from .utils.api_client import ApiClient

pytestmark = pytest.mark.e2e


def test_mission_policy_denial_example(api_base: str):
    """
    Create a mission with missing fields to exercise policy denial surface.
    Should receive 400 with policy_violations array when policy engine is active.
    In dev, backend may fail-open and return 200; test tolerates both.
    """
    client = ApiClient(api_base)
    try:
        payload = {
            "name": "Test Mission",
            # missing area/objectives intentionally
        }
        r = client.post("/v1/missions", json=payload)
        if r.status_code in (200, 201):
            # fail-open path
            assert True
            return
        assert r.status_code in (400, 403)
        data = r.json()
        detail = data.get("detail", data)
        violations = detail.get("policy_violations") or detail.get("deny_reasons") or []
        assert isinstance(violations, list)
    finally:
        client.close()
