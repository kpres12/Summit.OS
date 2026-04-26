"""
EngagementMQTTBridge unit tests.

Verifies the bridge's track-handling logic without needing a live MQTT
broker — uses a fake client that captures publish() calls and feeds
on_message() with synthetic payloads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def gate():
    from packages.c2_intel.engagement_authorization import EngagementAuthorizationGate
    return EngagementAuthorizationGate.for_testing()


@pytest.fixture
def bridge(gate):
    from packages.c2_intel.engagement_mqtt_bridge import EngagementMQTTBridge
    return EngagementMQTTBridge(gate=gate, min_track_confidence=0.7)


def _fire(bridge, payload: dict) -> MagicMock:
    """Simulate an inbound MQTT message and return the fake client."""
    client = MagicMock()
    msg = MagicMock()
    msg.topic = "summit/engagement/track_confirmed"
    msg.payload = json.dumps(payload).encode("utf-8")
    bridge._on_message(client, None, msg)
    return client


def _published_topics(client) -> list[str]:
    return [c.args[0] for c in client.publish.call_args_list]


def _published_payload(client, topic: str) -> dict:
    for call in client.publish.call_args_list:
        if call.args[0] == topic:
            return json.loads(call.args[1])
    raise AssertionError(f"no publish to {topic}; saw {_published_topics(client)}")


def test_kinetic_track_opens_case(bridge):
    client = _fire(bridge, {
        "track_id": "T-1", "entity_id": "E-1",
        "classification": "rotary_uas", "confidence": 0.92,
        "sensors": ["radar"],
        "last_position": {"lat": 34.5, "lon": -118.0, "alt_m": 200},
    })
    assert "summit/engagement/case_opened" in _published_topics(client)
    payload = _published_payload(client, "summit/engagement/case_opened")
    assert payload["track_id"] == "T-1"
    assert payload["classification"] == "rotary_uas"
    assert "case_id" in payload


def test_low_confidence_rejected(bridge):
    client = _fire(bridge, {
        "track_id": "T-2", "entity_id": "E-2",
        "classification": "rotary_uas", "confidence": 0.3,
        "last_position": {"lat": 0, "lon": 0, "alt_m": 0},
    })
    assert "summit/engagement/case_rejected" in _published_topics(client)
    payload = _published_payload(client, "summit/engagement/case_rejected")
    assert "confidence" in payload["reason"]


def test_non_kinetic_class_rejected(bridge):
    client = _fire(bridge, {
        "track_id": "T-3", "entity_id": "E-3",
        "classification": "weather_balloon", "confidence": 0.95,
        "last_position": {"lat": 0, "lon": 0, "alt_m": 0},
    })
    assert "summit/engagement/case_rejected" in _published_topics(client)
    payload = _published_payload(client, "summit/engagement/case_rejected")
    assert "kinetic set" in payload["reason"]


def test_missing_required_field_rejected(bridge):
    client = _fire(bridge, {"track_id": "T-4", "confidence": 0.9})
    assert "summit/engagement/case_rejected" in _published_topics(client)


def test_classification_normalized_to_lowercase(bridge):
    client = _fire(bridge, {
        "track_id": "T-5", "entity_id": "E-5",
        "classification": "ROTARY_UAS", "confidence": 0.88,
        "last_position": {"lat": 0, "lon": 0, "alt_m": 0},
    })
    assert "summit/engagement/case_opened" in _published_topics(client)


def test_custom_kinetic_classes(gate):
    from packages.c2_intel.engagement_mqtt_bridge import EngagementMQTTBridge
    bridge = EngagementMQTTBridge(gate=gate, kinetic_classes={"my_threat"})
    # default class should now be rejected
    client = _fire(bridge, {
        "track_id": "T-6", "entity_id": "E-6",
        "classification": "rotary_uas", "confidence": 0.9,
        "last_position": {"lat": 0, "lon": 0, "alt_m": 0},
    })
    assert "summit/engagement/case_rejected" in _published_topics(client)
    # custom class should be accepted
    client2 = _fire(bridge, {
        "track_id": "T-7", "entity_id": "E-7",
        "classification": "my_threat", "confidence": 0.9,
        "last_position": {"lat": 0, "lon": 0, "alt_m": 0},
    })
    assert "summit/engagement/case_opened" in _published_topics(client2)


def test_malformed_payload_does_not_raise(bridge):
    """Bad JSON should be logged but never crash the bridge."""
    client = MagicMock()
    msg = MagicMock()
    msg.topic = "summit/engagement/track_confirmed"
    msg.payload = b"not-json-at-all"
    # Should not raise
    bridge._on_message(client, None, msg)
