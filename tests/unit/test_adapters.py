"""
Unit tests for Summit.OS adapter simulation mode.

Tests exercise entity-building logic and manifest validation without
requiring hardware, live network connections, or hardware-specific
Python libraries (pymodbus, asyncua, pymavlink, sgp4).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "adapters"))


# ── Manifest validation ───────────────────────────────────────────────────────

class TestAdapterManifests:
    """Each adapter must have a valid MANIFEST that passes validation."""

    def test_opensky_manifest_valid(self):
        from opensky.adapter import OpenSkyAdapter
        errors = OpenSkyAdapter.MANIFEST.validate()
        assert errors == [], f"OpenSky manifest errors: {errors}"

    def test_celestrak_manifest_valid(self):
        from celestrak.adapter import CelesTrakAdapter
        errors = CelesTrakAdapter.MANIFEST.validate()
        assert errors == [], f"CelesTrak manifest errors: {errors}"

    def test_modbus_manifest_valid(self):
        from modbus.adapter import ModbusAdapter
        errors = ModbusAdapter.MANIFEST.validate()
        assert errors == [], f"Modbus manifest errors: {errors}"

    def test_opcua_manifest_valid(self):
        from opcua.adapter import OPCUAAdapter
        errors = OPCUAAdapter.MANIFEST.validate()
        assert errors == [], f"OPC-UA manifest errors: {errors}"

    def test_mavlink_manifest_valid(self):
        from mavlink.adapter import MAVLinkAdapter
        errors = MAVLinkAdapter.MANIFEST.validate()
        assert errors == [], f"MAVLink manifest errors: {errors}"

    def test_opensky_manifest_name(self):
        from opensky.adapter import OpenSkyAdapter
        assert OpenSkyAdapter.MANIFEST.name == "opensky"

    def test_celestrak_manifest_name(self):
        from celestrak.adapter import CelesTrakAdapter
        assert CelesTrakAdapter.MANIFEST.name == "celestrak"

    def test_modbus_manifest_name(self):
        from modbus.adapter import ModbusAdapter
        assert ModbusAdapter.MANIFEST.name == "modbus"

    def test_mavlink_requires_approval(self):
        """MAVLink has WRITE capability — must require approval."""
        from mavlink.adapter import MAVLinkAdapter
        assert MAVLinkAdapter.MANIFEST.requires_approval() is True

    def test_opensky_no_approval_required(self):
        """OpenSky is read-only — must not require approval."""
        from opensky.adapter import OpenSkyAdapter
        assert OpenSkyAdapter.MANIFEST.requires_approval() is False

    def test_celestrak_no_approval_required(self):
        from celestrak.adapter import CelesTrakAdapter
        assert CelesTrakAdapter.MANIFEST.requires_approval() is False


# ── OpenSky entity building ───────────────────────────────────────────────────
# Tests _state_vector_to_entity() with a mock ADS-B state vector.
# No network connection required.

class TestOpenSkyEntityBuilding:

    # ADS-B state vector: [icao24, callsign, origin, time, lastContact,
    #                      lon, lat, baroAlt, onGround, velocity,
    #                      trueTrack, vertRate, sensors, geoAlt, squawk, spi, ...]
    SAMPLE_SV = [
        "a1b2c3", "UAL123  ", "United States",
        1700000000, 1700000000,
        -118.25, 34.05, 9144.0, False,
        230.0, 270.0, 0.5,
        None, 9200.0, "1200", False, 0
    ]

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.setenv("OPENSKY_ENABLED", "true")
        from opensky.adapter import OpenSkyAdapter
        return OpenSkyAdapter()

    def test_entity_id_uses_icao24(self, adapter):
        entity = adapter._state_vector_to_entity(self.SAMPLE_SV)
        assert entity is not None
        assert "a1b2c3" in entity["entity_id"]

    def test_entity_type_is_track(self, adapter):
        entity = adapter._state_vector_to_entity(self.SAMPLE_SV)
        assert entity["entity_type"] == "TRACK"

    def test_entity_domain_is_aerial(self, adapter):
        entity = adapter._state_vector_to_entity(self.SAMPLE_SV)
        assert entity["domain"] == "AERIAL"

    def test_position_populated(self, adapter):
        entity = adapter._state_vector_to_entity(self.SAMPLE_SV)
        pos = entity["kinematics"]["position"]
        assert pos["latitude"] == pytest.approx(34.05)
        assert pos["longitude"] == pytest.approx(-118.25)

    def test_null_position_returns_none(self, adapter):
        sv = list(self.SAMPLE_SV)
        sv[6] = None  # lat = None
        entity = adapter._state_vector_to_entity(sv)
        assert entity is None

    def test_provenance_source_type(self, adapter):
        entity = adapter._state_vector_to_entity(self.SAMPLE_SV)
        assert entity["provenance"]["source_type"] == "adsb"

    def test_aerial_telemetry_flight_mode_airborne(self, adapter):
        entity = adapter._state_vector_to_entity(self.SAMPLE_SV)
        assert entity["aerial"]["flight_mode"] == "AIRBORNE"

    def test_aerial_telemetry_flight_mode_ground(self, adapter):
        sv = list(self.SAMPLE_SV)
        sv[8] = True  # on_ground = True
        entity = adapter._state_vector_to_entity(sv)
        assert entity["aerial"]["flight_mode"] == "GROUND"


# ── Modbus entity building ────────────────────────────────────────────────────

class TestModbusEntityBuilding:

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.setenv("MODBUS_ENABLED", "true")
        from modbus.adapter import ModbusAdapter
        return ModbusAdapter()

    @pytest.mark.asyncio
    async def test_poll_simulated_publishes_entities(self, adapter, monkeypatch):
        published = []

        def _fake_publish(entity, **kwargs):
            published.append(entity)

        adapter.publish = _fake_publish
        await adapter._poll_simulated()
        assert len(published) > 0

    @pytest.mark.asyncio
    async def test_simulated_entity_type_is_asset(self, adapter, monkeypatch):
        published = []
        adapter.publish = lambda e, **kw: published.append(e)
        await adapter._poll_simulated()
        assert all(e["entity_type"] == "ASSET" for e in published)

    @pytest.mark.asyncio
    async def test_simulated_entity_has_value_in_metadata(self, adapter):
        published = []
        adapter.publish = lambda e, **kw: published.append(e)
        await adapter._poll_simulated()
        entity = published[0]
        assert "value" in entity["metadata"]

    @pytest.mark.asyncio
    async def test_simulated_count_matches_register_map(self, adapter):
        published = []
        adapter.publish = lambda e, **kw: published.append(e)
        await adapter._poll_simulated()
        assert len(published) == len(adapter.registers)

    def test_reg_to_entity_warns_above_threshold(self, adapter):
        reg = {
            "address": 40001, "name": "Pressure", "unit": "PSI",
            "scale": 1.0, "offset": 0.0, "domain": "GROUND",
            "class_label": "pressure_sensor",
            "warn_above": 800.0, "critical_above": 950.0,
        }
        entity = adapter._reg_to_entity(reg, 850.0)
        assert entity["state"] == "WARNING"

    def test_reg_to_entity_critical_above_threshold(self, adapter):
        reg = {
            "address": 40001, "name": "Pressure", "unit": "PSI",
            "scale": 1.0, "offset": 0.0, "domain": "GROUND",
            "class_label": "pressure_sensor",
            "warn_above": 800.0, "critical_above": 950.0,
        }
        entity = adapter._reg_to_entity(reg, 970.0)
        assert entity["state"] == "CRITICAL"

    def test_reg_to_entity_normal_is_active(self, adapter):
        reg = {
            "address": 40001, "name": "Pressure", "unit": "PSI",
            "scale": 1.0, "offset": 0.0, "domain": "GROUND",
            "class_label": "pressure_sensor",
            "warn_above": 800.0, "critical_above": 950.0,
        }
        entity = adapter._reg_to_entity(reg, 200.0)
        assert entity["state"] == "ACTIVE"


# ── OPC-UA entity building ────────────────────────────────────────────────────

class TestOPCUAEntityBuilding:

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.setenv("OPCUA_ENABLED", "true")
        from opcua.adapter import OPCUAAdapter
        return OPCUAAdapter()

    def test_node_to_entity_basic(self, adapter):
        node_def = {
            "node_id": "ns=2;i=1001",
            "name": "Tank_Level",
            "unit": "liters",
            "class_label": "level_sensor",
        }
        entity = adapter._node_to_entity(node_def, 450.0)
        assert entity is not None
        assert entity["entity_type"] == "ASSET"

    def test_node_to_entity_domain_is_ground(self, adapter):
        node_def = {"node_id": "ns=2;i=1001", "name": "Temp", "unit": "degC",
                    "class_label": "temp_sensor"}
        entity = adapter._node_to_entity(node_def, 22.5)
        assert entity["domain"] == "GROUND"

    def test_node_to_entity_value_in_metadata(self, adapter):
        node_def = {"node_id": "ns=2;i=1002", "name": "Flow", "unit": "L/min",
                    "class_label": "flow_sensor"}
        entity = adapter._node_to_entity(node_def, 12.3)
        assert entity["metadata"]["value"] == "12.3"
        assert entity["metadata"]["unit"] == "L/min"

    def test_node_to_entity_provenance_source_type(self, adapter):
        node_def = {"node_id": "ns=2;i=1003", "name": "Pressure", "unit": "bar",
                    "class_label": "pressure_sensor"}
        entity = adapter._node_to_entity(node_def, 5.0)
        assert entity["provenance"]["source_type"] == "opcua"

    def test_node_to_entity_warn_above_threshold(self, adapter):
        node_def = {
            "node_id": "ns=2;i=1004", "name": "Temp", "unit": "degC",
            "class_label": "temp_sensor", "warn_above": 80.0,
        }
        entity = adapter._node_to_entity(node_def, 85.0)
        assert entity["state"] == "WARNING"

    def test_node_to_entity_non_numeric_value(self, adapter):
        """Non-numeric values should default to 0.0 without crashing."""
        node_def = {"node_id": "ns=2;i=1005", "name": "Status", "unit": "",
                    "class_label": "status"}
        entity = adapter._node_to_entity(node_def, "RUNNING")
        assert entity is not None
        assert entity["metadata"]["value"] == "0.0"


# ── MAVLink entity building ───────────────────────────────────────────────────

class TestMAVLinkEntityBuilding:

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.setenv("MAVLINK_ENABLED", "true")
        from mavlink.adapter import MAVLinkAdapter
        return MAVLinkAdapter()

    def _make_telem(self, **overrides):
        """Build a minimal fake telemetry object."""
        class T:
            lat = 34.05
            lon = -118.25
            alt = 120.0
            relative_alt = 115.0
            heading = 90.0
            groundspeed = 15.0
            airspeed = 16.0
            climb_rate = 0.5
            battery_voltage = 22.2
            battery_remaining = 85
            gps_fix_type = 3
            satellites_visible = 12
            flight_mode = "GUIDED"
            armed = True

        t = T()
        for k, v in overrides.items():
            setattr(t, k, v)
        return t

    @pytest.fixture
    def vehicle_config(self):
        return {
            "vehicle_id": "drone-alpha",
            "name": "Recon Alpha",
            "connection": "tcp:127.0.0.1:5760",
        }

    def test_entity_type_is_track(self, adapter, vehicle_config):
        telem = self._make_telem()
        entity = adapter._telemetry_to_entity(telem, vehicle_config)
        assert entity["entity_type"] == "TRACK"

    def test_entity_domain_is_aerial(self, adapter, vehicle_config):
        telem = self._make_telem()
        entity = adapter._telemetry_to_entity(telem, vehicle_config)
        assert entity["domain"] == "AERIAL"

    def test_entity_position_correct(self, adapter, vehicle_config):
        telem = self._make_telem()
        entity = adapter._telemetry_to_entity(telem, vehicle_config)
        pos = entity["kinematics"]["position"]
        assert pos["latitude"] == pytest.approx(34.05)
        assert pos["longitude"] == pytest.approx(-118.25)

    def test_aerial_telemetry_present(self, adapter, vehicle_config):
        telem = self._make_telem()
        entity = adapter._telemetry_to_entity(telem, vehicle_config)
        assert "aerial" in entity
        assert "flight_mode" in entity["aerial"]
        assert "battery_pct" in entity["aerial"]

    def test_active_state_when_armed(self, adapter, vehicle_config):
        telem = self._make_telem(armed=True, battery_remaining=85)
        entity = adapter._telemetry_to_entity(telem, vehicle_config)
        assert entity["state"] == "ACTIVE"

    def test_critical_state_when_battery_low(self, adapter, vehicle_config):
        telem = self._make_telem(armed=True, battery_remaining=5)
        entity = adapter._telemetry_to_entity(telem, vehicle_config)
        assert entity["state"] == "CRITICAL"

    def test_warning_state_when_battery_marginal(self, adapter, vehicle_config):
        telem = self._make_telem(armed=True, battery_remaining=15)
        entity = adapter._telemetry_to_entity(telem, vehicle_config)
        assert entity["state"] == "WARNING"

    def test_provenance_source_type(self, adapter, vehicle_config):
        telem = self._make_telem()
        entity = adapter._telemetry_to_entity(telem, vehicle_config)
        assert entity["provenance"]["source_type"] == "mavlink"
