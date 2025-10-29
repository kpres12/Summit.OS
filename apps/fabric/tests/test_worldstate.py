from fastapi.testclient import TestClient
from main import app

def test_worldstate_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("service") == "fabric"
