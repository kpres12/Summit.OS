# pytest fixtures for e2e tests
#
# GATEWAY_TEST_MODE=true (set by CI):
#   Mounts the gateway ASGI app via Starlette TestClient — no TCP socket needed.
#   The gateway uses SQLite in-memory (it checks GATEWAY_TEST_MODE itself).
#
# Without GATEWAY_TEST_MODE (local dev with live stack):
#   Uses SUMMIT_API env var (default http://localhost:8000).

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

GATEWAY_TEST_MODE = os.environ.get("GATEWAY_TEST_MODE", "false").lower() == "true"


@pytest.fixture(scope="session")
def api_base() -> str:
    if GATEWAY_TEST_MODE:
        return "http://testserver"
    return os.environ.get("SUMMIT_API", "http://localhost:8000")


@pytest.fixture(scope="session", autouse=True)
def patch_api_client_for_test_mode():
    """
    When GATEWAY_TEST_MODE=true, replace ApiClient's httpx.Client with a
    Starlette TestClient backed by the gateway ASGI app.

    TestClient handles async↔sync bridging internally, so sync test code
    works without any changes.
    """
    if not GATEWAY_TEST_MODE:
        yield
        return

    from starlette.testclient import TestClient
    from tests.e2e.utils import api_client as _ac_module

    # Add gateway directory so its local imports resolve
    gw_path = os.path.join(_REPO_ROOT, "apps", "api-gateway")
    if gw_path not in sys.path:
        sys.path.insert(0, gw_path)
    import main as _gw

    _test_client = TestClient(_gw.app, raise_server_exceptions=True)

    original_init  = _ac_module.ApiClient.__init__
    original_close = _ac_module.ApiClient.close

    def _patched_init(self, base_url: str) -> None:
        self.base = "http://testserver"
        self._client = _test_client

    def _patched_close(self) -> None:
        # TestClient is session-scoped — don't close between individual tests.
        pass

    _ac_module.ApiClient.__init__ = _patched_init
    _ac_module.ApiClient.close    = _patched_close

    with _test_client:
        yield

    _ac_module.ApiClient.__init__ = original_init
    _ac_module.ApiClient.close    = original_close


@pytest.fixture(scope="session")
def require_stack():
    """Skip if the live stack is unreachable (only relevant outside test mode)."""
    if GATEWAY_TEST_MODE:
        yield
        return

    import httpx
    base = os.environ.get("SUMMIT_API", "http://localhost:8000")
    try:
        httpx.get(f"{base}/health", timeout=3.0)
    except Exception:
        pytest.skip(f"Live stack not reachable at {base} — skipping")
    yield
