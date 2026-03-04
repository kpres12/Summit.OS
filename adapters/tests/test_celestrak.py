"""Tests for CelesTrak satellite adapter."""
import json
import math
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from celestrak.adapter import CelesTrakAdapter, _ecef_to_lla, _velocity_magnitude

# Sample TLE data (ISS)
SAMPLE_TLE_TEXT = """ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993
2 25544  51.6400 200.0000 0007000  50.0000 310.0000 15.49560000400000
STARLINK-1007
1 44713U 19074A   24001.50000000  .00001234  00000-0  12345-4 0  9999
2 44713  53.0000 100.0000 0001000  90.0000 270.0000 15.06000000100000
"""


class TestECEFConversion:
    """Test coordinate conversion utilities."""

    def test_ecef_to_lla_equator(self):
        """Point on equator at prime meridian."""
        lat, lon, alt = _ecef_to_lla(6771.0, 0.0, 0.0)
        assert abs(lat) < 0.1  # near equator
        assert abs(lon) < 0.1  # near prime meridian
        assert alt > 0  # above surface

    def test_ecef_to_lla_north_pole(self):
        """Point near north pole."""
        lat, lon, alt = _ecef_to_lla(0.0, 0.0, 6771.0)
        assert lat > 89.0  # near 90°

    def test_velocity_magnitude(self):
        v = _velocity_magnitude(5.0, 5.0, 5.0)
        expected = math.sqrt(75.0) * 1000  # km/s → m/s
        assert abs(v - expected) < 0.01


class TestCelesTrakTLEParsing:
    """Test TLE fetch and parse logic."""

    @pytest.mark.asyncio
    async def test_fetch_tles_parses_correctly(self):
        mqtt = MagicMock()
        adapter = CelesTrakAdapter(mqtt_client=mqtt, max_sats=10)

        mock_response = MagicMock()
        mock_response.text = SAMPLE_TLE_TEXT
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        await adapter._fetch_tles(mock_client)

        assert len(adapter._satellites) == 2
        assert adapter._satellites[0]["name"] == "ISS (ZARYA)"
        assert adapter._satellites[0]["norad_id"] == "25544"
        assert adapter._satellites[1]["name"] == "STARLINK-1007"
        assert adapter._satellites[1]["norad_id"] == "44713"

    @pytest.mark.asyncio
    async def test_fetch_tles_respects_max_sats(self):
        mqtt = MagicMock()
        adapter = CelesTrakAdapter(mqtt_client=mqtt, max_sats=1)

        mock_response = MagicMock()
        mock_response.text = SAMPLE_TLE_TEXT
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        await adapter._fetch_tles(mock_client)
        assert len(adapter._satellites) == 1


class TestCelesTrakPropagation:
    """Test SGP4 propagation and MQTT publish."""

    @pytest.mark.asyncio
    async def test_propagate_publishes_entities(self):
        mqtt = MagicMock()
        adapter = CelesTrakAdapter(mqtt_client=mqtt, max_sats=10)

        # Load TLEs
        mock_response = MagicMock()
        mock_response.text = SAMPLE_TLE_TEXT
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        await adapter._fetch_tles(mock_client)

        # Propagate
        adapter._propagate_and_publish()

        # Should publish entities for each satellite
        assert mqtt.publish.call_count >= 1
        topic = mqtt.publish.call_args_list[0][0][0]
        payload = json.loads(mqtt.publish.call_args_list[0][0][1])

        assert "celestrak-25544" in topic
        assert payload["entity_type"] == "TRACK"
        assert payload["domain"] == "AERIAL"
        assert payload["class_label"] == "satellite"
        assert payload["name"] == "ISS (ZARYA)"

        # Check position is reasonable
        pos = payload["kinematics"]["position"]
        assert -90 <= pos["latitude"] <= 90
        assert -180 <= pos["longitude"] <= 180
        assert pos["altitude_msl"] > 100000  # ISS > 100km

        # Check speed is orbital (~7.5 km/s = ~7500 m/s)
        speed = payload["kinematics"]["speed_mps"]
        assert 5000 < speed < 10000

    def test_entity_payload_has_required_fields(self):
        """Verify the MQTT payload matches what Fabric's _handle_entity_update expects."""
        mqtt = MagicMock()
        adapter = CelesTrakAdapter(mqtt_client=mqtt, max_sats=10)

        # Manually load a satellite
        try:
            from sgp4.api import Satrec
            line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993"
            line2 = "2 25544  51.6400 200.0000 0007000  50.0000 310.0000 15.49560000400000"
            satrec = Satrec.twoline2rv(line1, line2)
            adapter._satellites = [{
                "name": "ISS",
                "norad_id": "25544",
                "satrec": satrec,
                "line1": line1,
                "line2": line2,
            }]
        except ImportError:
            pytest.skip("sgp4 not installed")

        adapter._propagate_and_publish()

        payload = json.loads(mqtt.publish.call_args[0][1])
        # Required by _handle_entity_update in fabric/main.py
        assert "entity_id" in payload
        assert "id" in payload
        assert "entity_type" in payload
        assert "domain" in payload
        assert "kinematics" in payload
        assert "position" in payload["kinematics"]
        assert "latitude" in payload["kinematics"]["position"]
        assert "longitude" in payload["kinematics"]["position"]
