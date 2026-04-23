"""E2E tests for the /api/version endpoint and versioned response headers."""
import pytest
from .utils.api_client import ApiClient

pytestmark = pytest.mark.e2e


def test_api_version_endpoint_returns_200(api_base: str):
    client = ApiClient(api_base)
    try:
        r = client.get("/api/version")
        assert r.status_code == 200
    finally:
        client.close()


def test_api_version_response_structure(api_base: str):
    client = ApiClient(api_base)
    try:
        r = client.get("/api/version")
        assert r.status_code == 200
        data = r.json()
        assert "api_version" in data, "Missing 'api_version' field"
        assert "heli_os_version" in data, "Missing 'heli_os_version' field"
        assert "min_sdk_version" in data, "Missing 'min_sdk_version' field"
    finally:
        client.close()


def test_api_version_header_present(api_base: str):
    """All responses should carry X-Heli-API-Version header."""
    client = ApiClient(api_base)
    try:
        r = client.get("/health")
        assert "x-heli-api-version" in {k.lower() for k in r.headers}, (
            "X-Heli-API-Version header missing from response"
        )
    finally:
        client.close()


def test_os_version_header_present(api_base: str):
    """All responses should carry X-Heli-OS-Version header."""
    client = ApiClient(api_base)
    try:
        r = client.get("/health")
        assert "x-heli-os-version" in {k.lower() for k in r.headers}, (
            "X-Heli-OS-Version header missing from response"
        )
    finally:
        client.close()


def test_api_version_value_is_v1(api_base: str):
    """Current API version must be '1'."""
    client = ApiClient(api_base)
    try:
        r = client.get("/api/version")
        data = r.json()
        assert str(data["api_version"]) == "1", (
            f"Expected api_version='1', got {data['api_version']!r}"
        )
    finally:
        client.close()
