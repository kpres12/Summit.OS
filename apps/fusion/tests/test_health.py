import os
from fastapi.testclient import TestClient

# Disable fusion startup side-effects (MQTT/DB) during tests
os.environ["FUSION_DISABLE_STARTUP"] = "1"

from main import app  # noqa: E402


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "fusion"
