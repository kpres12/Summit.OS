"""
Engagement gate throughput + latency benchmark.

Confirms the human-in-the-loop gate isn't a bottleneck under realistic
ACE / counter-UAS load. Measures:
  - open_case() throughput per second
  - PID -> ROE -> deconfliction -> options -> authorize wall time
  - Audit-sink write latency at sustained rate

Marked with pytest.mark.perf so the default test suite skips it; run
explicitly with:
  pytest tests/perf/test_engagement_throughput.py -m perf
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.perf


def _make_track(i: int):
    from packages.c2_intel.engagement_authorization import TrackEvidence
    return TrackEvidence(
        track_id=f"t-{i}", entity_id=f"e-{i}",
        classification="rotary_uas", confidence=0.92,
        sensors=["radar"], last_position={"lat": 34.5, "lon": -118.0, "alt_m": 200},
        last_seen=datetime.now(timezone.utc),
    )


def test_open_case_throughput():
    """Open 5,000 cases — gate should sustain > 1,000/s on a laptop."""
    from packages.c2_intel.engagement_authorization import EngagementAuthorizationGate
    gate = EngagementAuthorizationGate.for_testing()

    n = 5000
    start = time.perf_counter()
    for i in range(n):
        gate.open_case(_make_track(i))
    elapsed = time.perf_counter() - start
    rate = n / elapsed

    print(f"\n  open_case: {n} cases in {elapsed:.2f}s -> {rate:.0f}/s")
    # Reasonable floor for an in-process state machine on any modern hardware
    assert rate > 1000, f"open_case throughput {rate:.0f}/s below 1000/s floor"
    assert len(gate.list_cases()) == n


def test_full_workflow_latency():
    """End-to-end open -> authorize wall time should be sub-millisecond
    in-process (ignoring network / signature / disk audit costs)."""
    from packages.c2_intel.engagement_authorization import (
        DeconflictionContext, EngagementAuthorizationGate,
        OperatorAuthorization, OperatorDecision, PIDEvidence,
        ROEContext, WeaponOption,
    )

    gate = EngagementAuthorizationGate.for_testing()
    n = 500
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        case = gate.open_case(_make_track(i))
        gate.submit_pid(case.case_id, PIDEvidence(method="iff", confidence=0.9))
        gate.submit_roe(case.case_id, ROEContext(
            roe_id="R", permits_engagement_type=True,
            proportionality_passed=True, collateral_estimate="low"))
        gate.submit_deconfliction(case.case_id, DeconflictionContext(
            blue_force_clear=True, airspace_clear=True))
        gate.surface_options(case.case_id, [WeaponOption(
            option_id="o-1", weapon_asset_id="a", weapon_class="soft_kill",
            range_m=100, time_of_flight_s=1, pk_estimate=0.9,
            roe_compliant=True, deconfliction_ok=True, rationale="")])
        gate.authorize(case.case_id, OperatorAuthorization(
            decision=OperatorDecision.AUTHORIZE,
            operator_id="op", operator_role="mission_commander",
            rationale="x", selected_option="o-1", signature=b"sig"),
            engagement_class="counter_uas")
        times.append(time.perf_counter() - t0)

    times.sort()
    p50 = times[n // 2]
    p95 = times[int(n * 0.95)]
    p99 = times[int(n * 0.99)]
    print(f"\n  full workflow latency (n={n}):"
          f"  p50={p50*1000:.2f}ms  p95={p95*1000:.2f}ms  p99={p99*1000:.2f}ms")
    # Floor: sub-10ms p99 for the in-memory gate
    assert p99 < 0.01, f"p99 latency {p99*1000:.2f}ms above 10ms"


def test_audit_sink_throughput(tmp_path):
    """ChainedHMACAuditSink should sustain > 10,000 writes/s for the
    engagement event volume we expect under load."""
    from packages.c2_intel.engagement_wiring import ChainedHMACAuditSink

    sink = ChainedHMACAuditSink(tmp_path / "audit.jsonl",
                                hmac_key=b"\x42" * 32)
    n = 10000
    start = time.perf_counter()
    for i in range(n):
        sink({"transition": "OPEN", "case_id": f"c-{i}",
              "to_state": "detected", "payload": {"x": i}})
    elapsed = time.perf_counter() - start
    rate = n / elapsed
    print(f"\n  audit sink: {n} writes in {elapsed:.2f}s -> {rate:.0f}/s")
    assert rate > 10000, f"audit sink throughput {rate:.0f}/s below 10k/s"

    # Verify the chain is intact after the burst
    ok, written, bad = sink.verify_chain()
    assert ok is True
    assert written == n
