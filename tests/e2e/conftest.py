# pytest fixtures for e2e tests (no auto-env startup)
import os
import pytest

@pytest.fixture(scope="session")
def api_base() -> str:
    base = os.environ.get("SUMMIT_API", "http://localhost:8000")
    return base

@pytest.fixture(scope="session")
def require_stack():
    """Skip tests if STACK_REQUIRED is set and stack not reachable."""
    # no-op placeholder for future health checks
    yield
