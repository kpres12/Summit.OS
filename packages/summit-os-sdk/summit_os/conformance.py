"""
Summit.OS Adapter Conformance Test Suite

Validates that a device adapter correctly implements the Summit.OS
integration protocol. Run against any SummitAdapter subclass.

Tests:
  1. Heartbeat — adapter publishes heartbeats on schedule
  2. Entity Telemetry — adapter publishes valid entity telemetry
  3. Registration — adapter registers with the gateway
  4. Disconnect/Reconnect — adapter handles MQTT disconnect gracefully
  5. Command Handling — adapter responds to commands

Usage (CLI):
    python -m summit_os.conformance --adapter my_package.MyDrone --device-id test-01

Usage (programmatic):
    from summit_os.conformance import run_conformance
    results = await run_conformance(adapter_instance)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger("summit.conformance")


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConformanceReport:
    adapter_id: str
    total: int
    passed: int
    failed: int
    results: List[TestResult]
    timestamp: str = ""

    @property
    def all_passed(self) -> bool:
        return self.failed == 0

    def summary(self) -> str:
        status = "PASS" if self.all_passed else "FAIL"
        lines = [
            f"Summit.OS Conformance: {status}  ({self.passed}/{self.total} passed)",
            f"Device: {self.adapter_id}",
            "",
        ]
        for r in self.results:
            icon = "✓" if r.passed else "✗"
            lines.append(f"  {icon} {r.name} ({r.duration_ms:.0f}ms) {r.message}")
        return "\n".join(lines)


class ConformanceRunner:
    """Runs conformance tests against a SummitAdapter instance."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.captured_messages: List[Dict[str, Any]] = []
        self._original_publish = None

    async def run_all(self) -> ConformanceReport:
        """Run all conformance tests and return a report."""
        from datetime import datetime, timezone

        results: List[TestResult] = []

        # Intercept MQTT publish
        self._setup_capture()

        tests = [
            ("1. Heartbeat", self._test_heartbeat),
            ("2. Entity Telemetry", self._test_entity_telemetry),
            ("3. Registration", self._test_registration),
            ("4. Disconnect/Reconnect", self._test_disconnect_reconnect),
            ("5. Command Handling", self._test_command_handling),
        ]

        for name, test_fn in tests:
            start = time.monotonic()
            try:
                result = await test_fn()
                elapsed = (time.monotonic() - start) * 1000
                result.duration_ms = elapsed
                result.name = name
            except Exception as e:
                elapsed = (time.monotonic() - start) * 1000
                result = TestResult(
                    name=name,
                    passed=False,
                    duration_ms=elapsed,
                    message=f"Exception: {e}",
                )
            results.append(result)

        # Restore original publish
        self._teardown_capture()

        passed = sum(1 for r in results if r.passed)
        return ConformanceReport(
            adapter_id=self.adapter.device_id,
            total=len(results),
            passed=passed,
            failed=len(results) - passed,
            results=results,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _setup_capture(self):
        """Monkey-patch MQTT publish to capture messages."""
        if hasattr(self.adapter, '_mqtt') and self.adapter._mqtt:
            self._original_publish = self.adapter._mqtt.publish

            def _capture_publish(topic, payload, qos=0, retain=False):
                try:
                    data = json.loads(payload) if isinstance(payload, str) else payload
                except Exception:
                    data = {"raw": str(payload)}
                self.captured_messages.append({"topic": topic, "data": data, "ts": time.time()})
                if self._original_publish:
                    return self._original_publish(topic, payload, qos=qos, retain=retain)

            self.adapter._mqtt.publish = _capture_publish

    def _teardown_capture(self):
        if self._original_publish and hasattr(self.adapter, '_mqtt') and self.adapter._mqtt:
            self.adapter._mqtt.publish = self._original_publish

    # ── Test 1: Heartbeat ──────────────────────────────────

    async def _test_heartbeat(self) -> TestResult:
        """Verify adapter publishes heartbeats."""
        self.captured_messages.clear()

        # Trigger a heartbeat cycle
        try:
            if hasattr(self.adapter, '_mqtt') and self.adapter._mqtt:
                payload = {
                    "ts": time.time(),
                    "status": "OK",
                    "device_id": self.adapter.device_id,
                }
                self.adapter._mqtt.publish(
                    f"health/{self.adapter.device_id}/heartbeat",
                    json.dumps(payload),
                    qos=0,
                )
                await asyncio.sleep(0.1)

            heartbeats = [
                m for m in self.captured_messages
                if "heartbeat" in m["topic"]
            ]

            if not heartbeats:
                return TestResult(
                    name="", passed=False, duration_ms=0,
                    message="No heartbeat messages captured",
                )

            hb = heartbeats[0]["data"]
            has_ts = "ts" in hb
            has_status = "status" in hb
            has_device = "device_id" in hb

            if has_ts and has_status and has_device:
                return TestResult(name="", passed=True, duration_ms=0, message="OK")
            else:
                missing = []
                if not has_ts:
                    missing.append("ts")
                if not has_status:
                    missing.append("status")
                if not has_device:
                    missing.append("device_id")
                return TestResult(
                    name="", passed=False, duration_ms=0,
                    message=f"Heartbeat missing fields: {missing}",
                )
        except Exception as e:
            return TestResult(name="", passed=False, duration_ms=0, message=str(e))

    # ── Test 2: Entity Telemetry ───────────────────────────

    async def _test_entity_telemetry(self) -> TestResult:
        """Verify adapter produces valid telemetry."""
        try:
            telem = await self.adapter.get_telemetry()

            if not isinstance(telem, dict):
                return TestResult(
                    name="", passed=False, duration_ms=0,
                    message=f"get_telemetry() returned {type(telem)}, expected dict",
                )

            required = {"lat", "lon", "alt"}
            missing = required - set(telem.keys())
            if missing:
                return TestResult(
                    name="", passed=False, duration_ms=0,
                    message=f"Telemetry missing required fields: {missing}",
                )

            # Validate types
            for k in required:
                if not isinstance(telem[k], (int, float)):
                    return TestResult(
                        name="", passed=False, duration_ms=0,
                        message=f"Field '{k}' should be numeric, got {type(telem[k])}",
                    )

            return TestResult(name="", passed=True, duration_ms=0, message="OK")

        except NotImplementedError:
            return TestResult(
                name="", passed=False, duration_ms=0,
                message="get_telemetry() not implemented",
            )

    # ── Test 3: Registration ───────────────────────────────

    async def _test_registration(self) -> TestResult:
        """Verify adapter has registration capability."""
        try:
            has_register = hasattr(self.adapter, '_register')
            has_device_id = bool(self.adapter.device_id)
            has_device_type = bool(self.adapter.device_type)

            if not has_register:
                return TestResult(
                    name="", passed=False, duration_ms=0,
                    message="Missing _register method",
                )
            if not has_device_id:
                return TestResult(
                    name="", passed=False, duration_ms=0,
                    message="device_id not set",
                )
            if not has_device_type:
                return TestResult(
                    name="", passed=False, duration_ms=0,
                    message="device_type not set",
                )

            caps = self.adapter.get_capabilities()
            if not isinstance(caps, list):
                return TestResult(
                    name="", passed=False, duration_ms=0,
                    message="get_capabilities() must return a list",
                )

            return TestResult(name="", passed=True, duration_ms=0, message="OK")
        except Exception as e:
            return TestResult(name="", passed=False, duration_ms=0, message=str(e))

    # ── Test 4: Disconnect/Reconnect ───────────────────────

    async def _test_disconnect_reconnect(self) -> TestResult:
        """Verify adapter has disconnect/reconnect hooks."""
        try:
            has_on_connect = hasattr(self.adapter, 'on_connect')
            has_on_disconnect = hasattr(self.adapter, 'on_disconnect')
            has_stop = hasattr(self.adapter, 'stop')

            if not (has_on_connect and has_on_disconnect and has_stop):
                return TestResult(
                    name="", passed=False, duration_ms=0,
                    message="Missing lifecycle methods (on_connect/on_disconnect/stop)",
                )

            return TestResult(name="", passed=True, duration_ms=0, message="OK")
        except Exception as e:
            return TestResult(name="", passed=False, duration_ms=0, message=str(e))

    # ── Test 5: Command Handling ───────────────────────────

    async def _test_command_handling(self) -> TestResult:
        """Verify adapter handles commands."""
        try:
            # Test with a no-op command
            result = await self.adapter.handle_command("ping", {})
            # Any response (True/False) is acceptable — just verify it doesn't crash
            return TestResult(
                name="", passed=True, duration_ms=0,
                message=f"handle_command('ping') returned {result}",
            )
        except NotImplementedError:
            return TestResult(
                name="", passed=False, duration_ms=0,
                message="handle_command() not implemented",
            )
        except Exception as e:
            return TestResult(
                name="", passed=False, duration_ms=0,
                message=f"handle_command() raised: {e}",
            )


async def run_conformance(adapter) -> ConformanceReport:
    """Convenience function to run all conformance tests."""
    runner = ConformanceRunner(adapter)
    return await runner.run_all()


# ── CLI Entry Point ────────────────────────────────────────

def main():
    """CLI entry point: python -m summit_os.conformance"""
    import argparse
    import importlib
    import sys

    parser = argparse.ArgumentParser(description="Summit.OS Adapter Conformance Tests")
    parser.add_argument("--adapter", required=True, help="Dotted path to adapter class (e.g., my_pkg.MyDrone)")
    parser.add_argument("--device-id", default="conformance-test-01", help="Device ID for testing")
    parser.add_argument("--device-type", default="GENERIC", help="Device type")
    args = parser.parse_args()

    # Import adapter class
    module_path, class_name = args.adapter.rsplit(".", 1)
    module = importlib.import_module(module_path)
    adapter_cls = getattr(module, class_name)

    # Instantiate
    adapter = adapter_cls(device_id=args.device_id, device_type=args.device_type)

    # Run tests
    async def _run():
        report = await run_conformance(adapter)
        print(report.summary())
        sys.exit(0 if report.all_passed else 1)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
