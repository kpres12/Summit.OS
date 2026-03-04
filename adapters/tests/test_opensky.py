"""Tests for OpenSky Network adapter."""
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from opensky.adapter import OpenSkyAdapter


# Sample OpenSky API response (single state vector)
SAMPLE_STATE_VECTOR = [
    "abc123",           # 0: icao24
    "UAL123  ",         # 1: callsign (padded)
    "United States",    # 2: origin_country
    1700000000,         # 3: time_position
    1700000005,         # 4: last_contact
    -122.375,           # 5: longitude
    37.619,             # 6: latitude
    10668.0,            # 7: baro_altitude (meters)
    False,              # 8: on_ground
    257.0,              # 9: velocity (m/s)
    180.5,              # 10: true_track (degrees)
    -2.5,               # 11: vertical_rate
    None,               # 12: sensors
    10972.0,            # 13: geo_altitude
    "1200",             # 14: squawk
    False,              # 15: spi
    0,                  # 16: position_source
]

SAMPLE_API_RESPONSE = {
    "time": 1700000005,
    "states": [SAMPLE_STATE_VECTOR],
}


class TestOpenSkyEntityConversion:
    """Test state vector → entity dict conversion."""

    def test_basic_conversion(self):
        entity = OpenSkyAdapter._state_vector_to_entity(SAMPLE_STATE_VECTOR, "2024-01-01T00:00:00Z")
        assert entity is not None
        assert entity["entity_id"] == "opensky-abc123"
        assert entity["entity_type"] == "TRACK"
        assert entity["domain"] == "AERIAL"
        assert entity["class_label"] == "aircraft"
        assert entity["name"] == "UAL123"  # stripped
        assert entity["state"] == "ACTIVE"

    def test_kinematics(self):
        entity = OpenSkyAdapter._state_vector_to_entity(SAMPLE_STATE_VECTOR, "2024-01-01T00:00:00Z")
        k = entity["kinematics"]
        assert k["position"]["latitude"] == 37.619
        assert k["position"]["longitude"] == -122.375
        assert k["position"]["altitude_msl"] == 10668.0
        assert k["heading_deg"] == 180.5
        assert k["speed_mps"] == 257.0
        assert k["climb_rate"] == -2.5

    def test_aerial_data(self):
        entity = OpenSkyAdapter._state_vector_to_entity(SAMPLE_STATE_VECTOR, "2024-01-01T00:00:00Z")
        a = entity["aerial"]
        assert a["altitude_msl"] == 10972.0
        assert a["airspeed_mps"] == 257.0
        assert a["flight_mode"] == "AIRBORNE"

    def test_metadata(self):
        entity = OpenSkyAdapter._state_vector_to_entity(SAMPLE_STATE_VECTOR, "2024-01-01T00:00:00Z")
        m = entity["metadata"]
        assert m["icao24"] == "abc123"
        assert m["callsign"] == "UAL123"
        assert m["origin_country"] == "United States"
        assert m["squawk"] == "1200"
        assert m["source"] == "opensky"

    def test_on_ground_aircraft(self):
        sv = list(SAMPLE_STATE_VECTOR)
        sv[8] = True  # on_ground
        entity = OpenSkyAdapter._state_vector_to_entity(sv, "2024-01-01T00:00:00Z")
        assert entity["aerial"]["flight_mode"] == "GROUND"
        assert entity["metadata"]["on_ground"] == "true"

    def test_missing_position_returns_none(self):
        sv = list(SAMPLE_STATE_VECTOR)
        sv[6] = None  # latitude
        entity = OpenSkyAdapter._state_vector_to_entity(sv, "2024-01-01T00:00:00Z")
        assert entity is None

    def test_missing_icao_returns_none(self):
        sv = list(SAMPLE_STATE_VECTOR)
        sv[0] = None
        entity = OpenSkyAdapter._state_vector_to_entity(sv, "2024-01-01T00:00:00Z")
        assert entity is None

    def test_null_velocity_defaults_to_zero(self):
        sv = list(SAMPLE_STATE_VECTOR)
        sv[9] = None  # velocity
        entity = OpenSkyAdapter._state_vector_to_entity(sv, "2024-01-01T00:00:00Z")
        assert entity["kinematics"]["speed_mps"] == 0

    def test_ttl_set(self):
        entity = OpenSkyAdapter._state_vector_to_entity(SAMPLE_STATE_VECTOR, "2024-01-01T00:00:00Z")
        assert entity["ttl_seconds"] == 60

    def test_no_callsign_falls_back_to_icao(self):
        sv = list(SAMPLE_STATE_VECTOR)
        sv[1] = "        "  # empty callsign
        entity = OpenSkyAdapter._state_vector_to_entity(sv, "2024-01-01T00:00:00Z")
        assert entity["name"] == "ABC123"


class TestOpenSkyAdapter:
    """Test adapter polling and MQTT publish."""

    def test_bbox_parsing(self):
        mqtt = MagicMock()
        adapter = OpenSkyAdapter(mqtt_client=mqtt, bbox="25.0,-130.0,50.0,-60.0")
        assert adapter._bbox == (25.0, -130.0, 50.0, -60.0)

    def test_bbox_empty(self):
        mqtt = MagicMock()
        adapter = OpenSkyAdapter(mqtt_client=mqtt, bbox="")
        assert adapter._bbox is None

    @pytest.mark.asyncio
    async def test_poll_publishes_entities(self):
        mqtt = MagicMock()
        adapter = OpenSkyAdapter(mqtt_client=mqtt)

        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_API_RESPONSE
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        await adapter._poll(mock_client)

        # Should have published one entity
        assert mqtt.publish.call_count == 1
        topic = mqtt.publish.call_args[0][0]
        payload = json.loads(mqtt.publish.call_args[0][1])
        assert topic == "entities/opensky-abc123/update"
        assert payload["entity_type"] == "TRACK"
        assert payload["domain"] == "AERIAL"

    @pytest.mark.asyncio
    async def test_poll_handles_empty_response(self):
        mqtt = MagicMock()
        adapter = OpenSkyAdapter(mqtt_client=mqtt)

        mock_response = MagicMock()
        mock_response.json.return_value = {"time": 0, "states": None}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        await adapter._poll(mock_client)
        assert mqtt.publish.call_count == 0
