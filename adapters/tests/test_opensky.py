"""Tests for OpenSky Network adapter."""
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


def _make_adapter(**kwargs):
    """Create an OpenSkyAdapter without connecting to MQTT."""
    return OpenSkyAdapter(**kwargs)


class TestOpenSkyEntityConversion:
    """Test state vector → entity dict conversion."""

    def test_basic_conversion(self):
        adapter = _make_adapter()
        entity = adapter._state_vector_to_entity(SAMPLE_STATE_VECTOR)
        assert entity is not None
        assert entity["entity_id"] == "opensky-abc123"
        assert entity["entity_type"] == "TRACK"
        assert entity["domain"] == "AERIAL"
        assert entity["class_label"] == "aircraft"
        assert entity["name"] == "UAL123"  # stripped
        assert entity["state"] == "ACTIVE"

    def test_kinematics(self):
        adapter = _make_adapter()
        entity = adapter._state_vector_to_entity(SAMPLE_STATE_VECTOR)
        k = entity["kinematics"]
        assert k["position"]["latitude"] == 37.619
        assert k["position"]["longitude"] == -122.375
        assert k["position"]["altitude_msl"] == 10668.0  # baro_alt
        assert k["heading_deg"] == 180.5
        assert k["speed_mps"] == 257.0
        assert k["climb_rate"] == -2.5

    def test_aerial_data(self):
        adapter = _make_adapter()
        entity = adapter._state_vector_to_entity(SAMPLE_STATE_VECTOR)
        a = entity["aerial"]
        assert a["altitude_msl"] == 10668.0  # uses baro_alt via .at()
        assert a["airspeed_mps"] == 257.0
        assert a["flight_mode"] == "AIRBORNE"

    def test_metadata(self):
        adapter = _make_adapter()
        entity = adapter._state_vector_to_entity(SAMPLE_STATE_VECTOR)
        m = entity["metadata"]
        assert m["icao24"] == "abc123"
        assert m["callsign"] == "UAL123"
        assert m["origin_country"] == "United States"
        assert m["squawk"] == "1200"
        # source is in provenance, not metadata
        assert entity["provenance"]["source_id"] == "opensky"

    def test_on_ground_aircraft(self):
        adapter = _make_adapter()
        sv = list(SAMPLE_STATE_VECTOR)
        sv[8] = True  # on_ground
        entity = adapter._state_vector_to_entity(sv)
        assert entity["aerial"]["flight_mode"] == "GROUND"
        assert entity["metadata"]["on_ground"] == "true"

    def test_missing_position_returns_none(self):
        adapter = _make_adapter()
        sv = list(SAMPLE_STATE_VECTOR)
        sv[6] = None  # latitude
        entity = adapter._state_vector_to_entity(sv)
        assert entity is None

    def test_missing_icao_returns_none(self):
        adapter = _make_adapter()
        sv = list(SAMPLE_STATE_VECTOR)
        sv[0] = None
        entity = adapter._state_vector_to_entity(sv)
        assert entity is None

    def test_null_velocity_defaults_to_zero(self):
        adapter = _make_adapter()
        sv = list(SAMPLE_STATE_VECTOR)
        sv[9] = None  # velocity
        entity = adapter._state_vector_to_entity(sv)
        assert entity["kinematics"]["speed_mps"] == 0

    def test_ttl_set(self):
        adapter = _make_adapter()
        entity = adapter._state_vector_to_entity(SAMPLE_STATE_VECTOR)
        assert entity["ttl_seconds"] == 60

    def test_no_callsign_falls_back_to_icao(self):
        adapter = _make_adapter()
        sv = list(SAMPLE_STATE_VECTOR)
        sv[1] = "        "  # empty callsign
        entity = adapter._state_vector_to_entity(sv)
        assert entity["name"] == "ABC123"


class TestOpenSkyAdapter:
    """Test adapter polling and publish."""

    def test_bbox_parsing(self):
        adapter = _make_adapter(bbox="25.0,-130.0,50.0,-60.0")
        assert adapter._bbox == (25.0, -130.0, 50.0, -60.0)

    def test_bbox_empty(self):
        adapter = _make_adapter(bbox="")
        assert adapter._bbox is None

    @pytest.mark.asyncio
    async def test_poll_publishes_entities(self):
        adapter = _make_adapter()

        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_API_RESPONSE
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch.object(adapter, "publish") as mock_pub:
            await adapter._poll(mock_client)

        assert mock_pub.call_count == 1
        entity = mock_pub.call_args[0][0]
        assert entity["entity_id"] == "opensky-abc123"
        assert entity["entity_type"] == "TRACK"
        assert entity["domain"] == "AERIAL"

    @pytest.mark.asyncio
    async def test_poll_handles_empty_response(self):
        adapter = _make_adapter()

        mock_response = MagicMock()
        mock_response.json.return_value = {"time": 0, "states": None}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch.object(adapter, "publish") as mock_pub:
            await adapter._poll(mock_client)

        assert mock_pub.call_count == 0
