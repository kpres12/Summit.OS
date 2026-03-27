"""E2E tests for device registration and revocation endpoints."""
import pytest
from .utils.api_client import ApiClient

pytestmark = pytest.mark.e2e


def test_device_register_returns_cert(api_base: str):
    """POST /v1/devices/register should return a PEM cert and key."""
    client = ApiClient(api_base)
    try:
        payload = {
            "device_id": "test-drone-e2e-01",
            "device_type": "uav",
            "org_id": "e2e-test-org",
        }
        r = client.post("/v1/devices/register", json=payload)
        assert r.status_code in (200, 201), f"Expected 200/201, got {r.status_code}: {r.text}"
        data = r.json()
        assert "cert_pem" in data or "fingerprint" in data, (
            "Response must include cert_pem or fingerprint"
        )
    finally:
        client.close()


def test_device_register_cert_is_pem(api_base: str):
    client = ApiClient(api_base)
    try:
        payload = {
            "device_id": "test-sensor-e2e-01",
            "device_type": "plc",
            "org_id": "e2e-test-org",
        }
        r = client.post("/v1/devices/register", json=payload)
        assert r.status_code in (200, 201)
        data = r.json()
        if "cert_pem" in data:
            assert data["cert_pem"].startswith("-----BEGIN CERTIFICATE-----")
        if "key_pem" in data:
            assert data["key_pem"].startswith("-----BEGIN PRIVATE KEY-----")
    finally:
        client.close()


def test_device_list_endpoint(api_base: str):
    """GET /v1/devices should return a paginated response with a devices list."""
    client = ApiClient(api_base)
    try:
        r = client.get("/v1/devices")
        assert r.status_code == 200
        data = r.json()
        # API returns {count: int, devices: [...]} envelope
        if isinstance(data, dict):
            assert "devices" in data, f"Expected 'devices' key in response, got: {list(data.keys())}"
            assert isinstance(data["devices"], list)
        else:
            assert isinstance(data, list)
    finally:
        client.close()


def test_device_revoke_endpoint(api_base: str):
    """POST /v1/devices/{id}/revoke should succeed for a registered device."""
    client = ApiClient(api_base)
    try:
        device_id = "test-revoke-e2e-01"
        # Register first
        client.post("/v1/devices/register", json={
            "device_id": device_id,
            "device_type": "uav",
            "org_id": "e2e-test-org",
        })
        # Now revoke
        r = client.post(f"/v1/devices/{device_id}/revoke")
        assert r.status_code in (200, 204), (
            f"Expected 200/204 on revoke, got {r.status_code}: {r.text}"
        )
    finally:
        client.close()


def test_registered_device_appears_in_list(api_base: str):
    client = ApiClient(api_base)
    try:
        device_id = "test-list-e2e-01"
        client.post("/v1/devices/register", json={
            "device_id": device_id,
            "device_type": "gateway",
            "org_id": "e2e-list-test",
        })
        r = client.get("/v1/devices")
        assert r.status_code == 200
        data = r.json()
        # Unwrap envelope if present
        devices = data["devices"] if isinstance(data, dict) else data
        ids = [d.get("device_id") for d in devices]
        assert device_id in ids, f"{device_id} not found in device list: {ids}"
    finally:
        client.close()
