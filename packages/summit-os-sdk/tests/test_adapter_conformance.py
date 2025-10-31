import json
from typing import Any, Dict

import pytest

from summit_os.bridges.adapter_base import BaseAdapter, AdapterConfig


class _StubMQTT:
    def __init__(self):
        self.published = []

    def publish(self, topic: str, payload: str, qos: int = 0):
        self.published.append((topic, json.loads(payload), qos))


def test_base_publish_serializes_json(monkeypatch):
    cfg = AdapterConfig(device_id="dev-1")
    adapter = BaseAdapter(cfg)
    stub = _StubMQTT()
    adapter._mqtt = stub  # type: ignore

    payload = {"device_id": "dev-1", "location": {"lat": 1, "lon": 2}}
    import asyncio
    asyncio.get_event_loop().run_until_complete(adapter.publish("telemetry/dev-1", payload))

    assert stub.published and stub.published[0][0] == "telemetry/dev-1"
    assert stub.published[0][1]["location"]["lat"] == 1


def test_register_payload(monkeypatch):
    cfg = AdapterConfig(device_id="dev-2")
    adapter = BaseAdapter(cfg)

    class _Resp:
        def __init__(self):
            self._json = {"status": "accepted", "token": "abc"}

        def raise_for_status(self):
            return None

        def json(self) -> Dict[str, Any]:
            return self._json

    def _fake_post(url, json=None, headers=None, timeout=5):  # type: ignore
        assert url.endswith("/api/v1/nodes/register")
        assert json["id"] == "dev-2" and json["type"] == "GENERIC"
        return _Resp()

    monkeypatch.setattr("requests.post", _fake_post)
    res = adapter.register_with_gateway("http://localhost:8000")
    assert res["status"] == "accepted"
