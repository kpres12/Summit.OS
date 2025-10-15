import os
os.environ["FABRIC_TEST_MODE"] = "true"

from fastapi.testclient import TestClient
import sys
import os as _os
sys.path.append(_os.path.dirname(_os.path.dirname(__file__)))
from main import app

client = TestClient(app)


def test_register_and_get_node():
    payload = {
        "id": "tower-042",
        "type": "TOWER",
        "pubkey": "BASE64...",
        "fw_version": "1.2.3",
        "location": {"lat": 34.123, "lon": -117.456, "elev_m": 1820},
        "capabilities": ["THERMAL", "EO"],
        "comm": ["LTE"]
    }
    r = client.post("/api/v1/nodes/register", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "accepted"
    assert "token" in body

    r2 = client.get("/api/v1/nodes/tower-042")
    assert r2.status_code == 200
    node = r2.json()
    assert node["id"] == "tower-042"
    assert node["status"] in ("ONLINE", "OFFLINE", "STALE", "RETIRED")


def test_coverage_union_empty_ok():
    r = client.get("/api/v1/coverage/union")
    assert r.status_code == 200
    body = r.json()
    assert "union" in body
    assert "count" in body


def test_refresh_token():
    r = client.post("/api/v1/nodes/tower-042/token")
    assert r.status_code == 200
    body = r.json()
    assert "token" in body and isinstance(body["token"], str)
    assert body["expires_in"] == 600
