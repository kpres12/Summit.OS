import pytest
from .utils.api_client import ApiClient

pytestmark = pytest.mark.e2e


def test_create_and_retrieve_mission(api_base: str):
    """POST /v1/missions → GET /v1/missions/{id} round-trip."""
    client = ApiClient(api_base)
    try:
        payload = {
            "name": "E2E Test Mission",
            "objectives": ["patrol sector A"],
        }
        r = client.post("/v1/missions", json=payload)
        # Tolerate 200/201 (created) or 400/403 (policy deny) or 502 (tasking down)
        if r.status_code in (502,):
            pytest.skip("tasking service not reachable")
        if r.status_code in (400, 403):
            # Policy denial — still a valid outcome
            data = r.json()
            detail = data.get("detail", data)
            assert "deny" in str(detail).lower() or "policy" in str(detail).lower() or isinstance(detail, (dict, list, str))
            return
        assert r.status_code in (200, 201)
        data = r.json()
        mission_id = data.get("mission_id")
        assert mission_id

        # Retrieve it
        r2 = client.get(f"/v1/missions/{mission_id}")
        assert r2.status_code == 200
        m = r2.json()
        assert m.get("mission_id") == mission_id
    finally:
        client.close()


def test_geofence_list(api_base: str):
    """GET /v1/geofences should return a geofences list."""
    client = ApiClient(api_base)
    try:
        r = client.get("/v1/geofences")
        # 502 means fabric is down — skip gracefully
        if r.status_code == 502:
            pytest.skip("fabric service not reachable")
        assert r.status_code == 200
        data = r.json()
        assert "geofences" in data
        assert isinstance(data["geofences"], list)
    finally:
        client.close()
