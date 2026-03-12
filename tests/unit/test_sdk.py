"""Unit tests for the Summit.OS Adapter SDK."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))

from sdk.manifest import AdapterManifest, Protocol, Capability
from sdk.entity import EntityBuilder


# ── AdapterManifest ──────────────────────────────────────────────────────────

class TestAdapterManifest:

    def test_valid_manifest_passes_validation(self):
        m = AdapterManifest(
            name="test-adapter",
            version="1.0.0",
            protocol=Protocol.MODBUS,
            capabilities=[Capability.READ],
            entity_types=["ASSET"],
        )
        assert m.validate() == []

    def test_invalid_name_fails(self):
        m = AdapterManifest(
            name="bad name!",
            version="1.0.0",
            protocol=Protocol.MODBUS,
            capabilities=[Capability.READ],
            entity_types=["ASSET"],
        )
        errors = m.validate()
        assert any("name" in e for e in errors)

    def test_invalid_version_fails(self):
        m = AdapterManifest(
            name="ok-name",
            version="1.0",  # not semver
            protocol=Protocol.MODBUS,
            capabilities=[Capability.READ],
            entity_types=["ASSET"],
        )
        errors = m.validate()
        assert any("version" in e for e in errors)

    def test_write_without_read_fails(self):
        m = AdapterManifest(
            name="bad-caps",
            version="1.0.0",
            protocol=Protocol.MODBUS,
            capabilities=[Capability.WRITE],
            entity_types=["ASSET"],
        )
        errors = m.validate()
        assert any("WRITE" in e for e in errors)

    def test_requires_approval_when_write(self):
        m = AdapterManifest(
            name="rw-adapter",
            version="1.0.0",
            protocol=Protocol.MAVLINK,
            capabilities=[Capability.READ, Capability.WRITE],
            entity_types=["TRACK"],
        )
        assert m.requires_approval() is True

    def test_no_approval_required_for_read_only(self):
        m = AdapterManifest(
            name="ro-adapter",
            version="1.0.0",
            protocol=Protocol.ADSB,
            capabilities=[Capability.READ],
            entity_types=["TRACK"],
        )
        assert m.requires_approval() is False

    def test_to_dict_contains_required_keys(self):
        m = AdapterManifest(
            name="dict-test",
            version="2.1.0",
            protocol=Protocol.OPCUA,
            capabilities=[Capability.READ, Capability.SUBSCRIBE],
            entity_types=["ASSET"],
            description="Test",
        )
        d = m.to_dict()
        for key in ("name", "version", "protocol", "capabilities", "entity_types",
                    "requires_approval"):
            assert key in d, f"Missing key: {key}"

    def test_empty_capabilities_fails(self):
        m = AdapterManifest(
            name="empty-caps",
            version="1.0.0",
            protocol=Protocol.CUSTOM,
            capabilities=[],
            entity_types=["ASSET"],
        )
        errors = m.validate()
        assert any("capabilities" in e for e in errors)


# ── EntityBuilder ────────────────────────────────────────────────────────────

class TestEntityBuilder:

    def test_basic_asset_entity(self):
        e = EntityBuilder("test-001", "Test Sensor").asset().ground().build()
        assert e["entity_id"] == "test-001"
        assert e["name"] == "Test Sensor"
        assert e["entity_type"] == "ASSET"
        assert e["domain"] == "GROUND"
        assert e["state"] == "ACTIVE"

    def test_track_entity_aerial(self):
        e = EntityBuilder("drone-01", "Recon Alpha").track().aerial().build()
        assert e["entity_type"] == "TRACK"
        assert e["domain"] == "AERIAL"

    def test_position_set_correctly(self):
        e = EntityBuilder("pos-test", "Test").asset().at(34.05, -118.25, 100.0).build()
        pos = e["kinematics"]["position"]
        assert pos["latitude"] == pytest.approx(34.05)
        assert pos["longitude"] == pytest.approx(-118.25)
        assert pos["altitude_msl"] == pytest.approx(100.0)

    def test_value_stored_in_metadata(self):
        e = EntityBuilder("val-test", "Pressure").asset().value(255.5, "PSI").build()
        assert e["metadata"]["value"] == "255.5"
        assert e["metadata"]["unit"] == "PSI"

    def test_warn_above_sets_warning_state(self):
        e = (EntityBuilder("warn-test", "Sensor")
             .asset()
             .value(850.0, "PSI")
             .warn_above(800.0)
             .build())
        assert e["state"] == "WARNING"

    def test_critical_above_sets_critical_state(self):
        e = (EntityBuilder("crit-test", "Sensor")
             .asset()
             .value(960.0, "PSI")
             .warn_above(800.0)
             .critical_above(950.0)
             .build())
        assert e["state"] == "CRITICAL"

    def test_critical_below_sets_critical_state(self):
        e = (EntityBuilder("crit-below", "Level")
             .asset()
             .value(50.0, "liters")
             .critical_below(100.0)
             .build())
        assert e["state"] == "CRITICAL"

    def test_value_within_range_stays_active(self):
        e = (EntityBuilder("ok-sensor", "Temp")
             .asset()
             .value(25.0, "degC")
             .warn_above(45.0)
             .critical_above(60.0)
             .build())
        assert e["state"] == "ACTIVE"

    def test_provenance_source_set(self):
        e = (EntityBuilder("prov-test", "Test")
             .asset()
             .source("modbus", "pump-01")
             .org("acme")
             .build())
        assert e["provenance"]["source_type"] == "modbus"
        assert e["provenance"]["source_id"] == "pump-01"
        assert e["provenance"]["org_id"] == "acme"

    def test_ttl_set(self):
        e = EntityBuilder("ttl-test", "T").asset().ttl(30).build()
        assert e["ttl_seconds"] == 30

    def test_meta_dict(self):
        e = (EntityBuilder("meta-test", "T")
             .asset()
             .meta_dict({"foo": "bar", "baz": "qux"})
             .build())
        assert e["metadata"]["foo"] == "bar"
        assert e["metadata"]["baz"] == "qux"

    def test_aerial_telemetry_included(self):
        e = (EntityBuilder("aerial-test", "Drone")
             .track()
             .aerial()
             .aerial_telemetry(flight_mode="GUIDED", battery_pct=85.0)
             .build())
        assert "aerial" in e
        assert e["aerial"]["flight_mode"] == "GUIDED"
        assert e["aerial"]["battery_pct"] == pytest.approx(85.0)

    def test_required_schema_fields_present(self):
        e = EntityBuilder("schema-test", "T").asset().build()
        required = ["entity_id", "entity_type", "domain", "state", "name",
                    "kinematics", "provenance", "metadata", "ttl_seconds", "ts"]
        for field in required:
            assert field in e, f"Missing required field: {field}"

    def test_ts_is_iso_string(self):
        e = EntityBuilder("ts-test", "T").asset().build()
        from datetime import datetime
        # Should be parseable as ISO datetime
        datetime.fromisoformat(e["ts"].replace("Z", "+00:00"))
