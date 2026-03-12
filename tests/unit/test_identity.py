"""Unit tests for DeviceCA and DeviceRegistry."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))


# ── DeviceCA ─────────────────────────────────────────────────────────────────

class TestDeviceCA:
    """Skip gracefully if 'cryptography' library is not installed."""

    @pytest.fixture
    def ca(self, tmp_path):
        from identity.ca import DeviceCA
        return DeviceCA(
            ca_cert_path=str(tmp_path / "ca.crt"),
            ca_key_path=str(tmp_path / "ca.key"),
        )

    @pytest.fixture
    def initialized_ca(self, ca, event_loop):
        """Return a CA that has already been initialized."""
        try:
            import cryptography  # noqa: F401
        except ImportError:
            pytest.skip("cryptography not installed")
        event_loop.run_until_complete(ca.initialize())
        return ca

    @pytest.fixture
    def async_initialized_ca(self, ca):
        return ca

    @pytest.mark.asyncio
    async def test_initialize_creates_ca_cert(self, ca, tmp_path):
        try:
            import cryptography  # noqa: F401
        except ImportError:
            pytest.skip("cryptography not installed")
        await ca.initialize()
        assert os.path.exists(str(tmp_path / "ca.crt"))
        assert os.path.exists(str(tmp_path / "ca.key"))

    @pytest.mark.asyncio
    async def test_initialize_twice_loads_existing(self, ca, tmp_path):
        try:
            import cryptography  # noqa: F401
        except ImportError:
            pytest.skip("cryptography not installed")
        await ca.initialize()
        mtime_cert = os.path.getmtime(str(tmp_path / "ca.crt"))
        await ca.initialize()
        # File should not be regenerated
        assert os.path.getmtime(str(tmp_path / "ca.crt")) == mtime_cert

    @pytest.mark.asyncio
    async def test_issue_cert_returns_device_cert(self, ca):
        try:
            import cryptography  # noqa: F401
        except ImportError:
            pytest.skip("cryptography not installed")
        await ca.initialize()
        cert = await ca.issue_device_cert("drone-01", device_type="uav", org_id="acme")
        assert cert is not None
        assert cert.device_id == "drone-01"
        assert cert.device_type == "uav"
        assert cert.org_id == "acme"
        assert cert.cert_pem.startswith("-----BEGIN CERTIFICATE-----")
        assert cert.key_pem.startswith("-----BEGIN PRIVATE KEY-----")

    @pytest.mark.asyncio
    async def test_fingerprint_is_64_hex_chars(self, ca):
        try:
            import cryptography  # noqa: F401
        except ImportError:
            pytest.skip("cryptography not installed")
        await ca.initialize()
        cert = await ca.issue_device_cert("sensor-01")
        assert cert is not None
        assert len(cert.fingerprint) == 64
        assert all(c in "0123456789abcdef" for c in cert.fingerprint)

    @pytest.mark.asyncio
    async def test_verify_cert_returns_device_info(self, ca):
        try:
            import cryptography  # noqa: F401
        except ImportError:
            pytest.skip("cryptography not installed")
        await ca.initialize()
        issued = await ca.issue_device_cert("pump-01", device_type="plc", org_id="acme")
        assert issued is not None
        info = ca.verify_cert(issued.cert_pem)
        assert info is not None
        assert info["device_id"] == "pump-01"
        assert info["device_type"] == "plc"

    @pytest.mark.asyncio
    async def test_verify_invalid_cert_returns_none(self, ca):
        try:
            import cryptography  # noqa: F401
        except ImportError:
            pytest.skip("cryptography not installed")
        await ca.initialize()
        result = ca.verify_cert("not a valid cert")
        assert result is None

    @pytest.mark.asyncio
    async def test_issue_before_initialize_returns_none(self, ca):
        """CA must be initialized before issuing certs."""
        result = await ca.issue_device_cert("orphan-device")
        assert result is None

    @pytest.mark.asyncio
    async def test_to_dict_contains_required_fields(self, ca):
        try:
            import cryptography  # noqa: F401
        except ImportError:
            pytest.skip("cryptography not installed")
        await ca.initialize()
        cert = await ca.issue_device_cert("camera-01", org_id="acme")
        assert cert is not None
        d = cert.to_dict()
        for key in ("device_id", "device_type", "fingerprint", "serial_number",
                    "not_before", "not_after", "org_id"):
            assert key in d, f"Missing field: {key}"

    @pytest.mark.asyncio
    async def test_two_devices_have_different_fingerprints(self, ca):
        try:
            import cryptography  # noqa: F401
        except ImportError:
            pytest.skip("cryptography not installed")
        await ca.initialize()
        cert_a = await ca.issue_device_cert("device-a")
        cert_b = await ca.issue_device_cert("device-b")
        assert cert_a is not None and cert_b is not None
        assert cert_a.fingerprint != cert_b.fingerprint


# ── DeviceRegistry ───────────────────────────────────────────────────────────

class TestDeviceRegistry:
    """Uses the in-memory fallback (no Postgres required)."""

    @pytest.fixture
    def registry(self):
        from identity.registry import DeviceRegistry
        # Pass a non-connectable URL; init will fall back to in-memory
        return DeviceRegistry(db_url="postgresql://does-not-exist/test")

    @pytest.mark.asyncio
    async def test_register_and_is_authorized(self, registry):
        await registry.initialize()
        ok = await registry.register(
            "drone-01", fingerprint="abc123", device_type="uav", org_id="acme"
        )
        assert ok is True
        assert await registry.is_authorized("drone-01", "abc123") is True

    @pytest.mark.asyncio
    async def test_wrong_fingerprint_not_authorized(self, registry):
        await registry.initialize()
        await registry.register("drone-02", fingerprint="correct-fp")
        assert await registry.is_authorized("drone-02", "wrong-fp") is False

    @pytest.mark.asyncio
    async def test_unregistered_device_not_authorized(self, registry):
        await registry.initialize()
        assert await registry.is_authorized("ghost-device", "any-fp") is False

    @pytest.mark.asyncio
    async def test_revoke_blocks_access(self, registry):
        await registry.initialize()
        await registry.register("drone-03", fingerprint="fp-xyz")
        assert await registry.is_authorized("drone-03", "fp-xyz") is True
        await registry.revoke("drone-03", reason="compromised")
        assert await registry.is_authorized("drone-03", "fp-xyz") is False

    @pytest.mark.asyncio
    async def test_get_device_returns_entry(self, registry):
        await registry.initialize()
        await registry.register("sensor-01", fingerprint="fp-abc", device_type="plc")
        entry = await registry.get_device("sensor-01")
        assert entry is not None
        assert entry["device_type"] == "plc"
        assert entry["revoked"] is False

    @pytest.mark.asyncio
    async def test_get_device_returns_none_for_unknown(self, registry):
        await registry.initialize()
        result = await registry.get_device("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_devices_returns_all(self, registry):
        await registry.initialize()
        await registry.register("dev-a", fingerprint="fp-a", org_id="acme")
        await registry.register("dev-b", fingerprint="fp-b", org_id="acme")
        await registry.register("dev-c", fingerprint="fp-c", org_id="other")
        devices = await registry.list_devices()
        assert len(devices) == 3

    @pytest.mark.asyncio
    async def test_list_devices_filters_by_org(self, registry):
        await registry.initialize()
        await registry.register("org1-dev-a", fingerprint="fp-1", org_id="org1")
        await registry.register("org1-dev-b", fingerprint="fp-2", org_id="org1")
        await registry.register("org2-dev-a", fingerprint="fp-3", org_id="org2")
        devices = await registry.list_devices(org_id="org1")
        assert len(devices) == 2
        assert all(d["org_id"] == "org1" for d in devices)

    @pytest.mark.asyncio
    async def test_re_register_updates_fingerprint(self, registry):
        await registry.initialize()
        await registry.register("updatable", fingerprint="old-fp")
        await registry.register("updatable", fingerprint="new-fp")
        assert await registry.is_authorized("updatable", "new-fp") is True
        assert await registry.is_authorized("updatable", "old-fp") is False
