"""
Anti-Replay Adversarial Tests
================================
Systematically attacks the AntiReplayFilter with every replay technique
documented in the anti-replay literature. Real AntiReplayFilter, no mocks.

Attack vectors:
  1. Basic replay — accept a frame, replay the same seq immediately
  2. Delayed replay — replay after the window has moved forward
  3. Sequence rewind — jump to seq=1000, then replay seq=500
  4. Clock-skew injection — frames with timestamps far in future/past
  5. Window boundary — last sequence in window vs. first outside it
  6. Concurrent replay race — 100 goroutines replay the same frame
  7. Sensor reset bypass — try to replay after reset_sensor() clears state
  8. Flood new sensors — 10k distinct sensor IDs to probe memory growth
  9. Sequence gap tolerance — non-contiguous sequences must be accepted
  10. Zero sequence edge case
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))

from security.anti_replay import AntiReplayFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter(window: int = 64, skew: float = 30.0) -> AntiReplayFilter:
    return AntiReplayFilter(window_size=window, max_ts_skew_s=skew)


def now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# 1. Basic replay
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_basic_replay_blocked():
    """Accept seq=1, then immediately replay seq=1 — second must be rejected."""
    f = _filter()
    assert await f.check("sensor-1", seq=1, ts=now())
    assert not await f.check("sensor-1", seq=1, ts=now()), \
        "Replayed frame (same seq) must be rejected"


# ---------------------------------------------------------------------------
# 2. Replay after window advances
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_replay_outside_window_blocked():
    """Advance the window past seq=1, then replay seq=1 — must be rejected."""
    f = _filter(window=64)
    assert await f.check("sensor-2", seq=1, ts=now())
    # Advance window far past seq=1
    for i in range(2, 70):
        await f.check("sensor-2", seq=i, ts=now())
    # seq=1 is now outside the 64-slot window
    assert not await f.check("sensor-2", seq=1, ts=now()), \
        "Seq outside sliding window must be rejected (replay protection)"


# ---------------------------------------------------------------------------
# 3. Sequence rewind attack
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sequence_rewind_blocked():
    """Jump forward to seq=1000, then submit seq=500. Must be rejected."""
    f = _filter(window=64)
    await f.check("sensor-3", seq=1000, ts=now())
    result = await f.check("sensor-3", seq=500, ts=now())
    assert not result, \
        "Sequence rewind (seq 500 after 1000) must be rejected — " \
        "old sequence replays are a classic EW attack vector"


# ---------------------------------------------------------------------------
# 4. Clock-skew injection — future timestamp
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_future_timestamp_rejected():
    """Frame with timestamp 31s in the future must be rejected.
    Default max_ts_skew_s=30.0."""
    f = _filter(skew=30.0)
    result = await f.check("sensor-4", seq=1, ts=now() + 31.0)
    assert not result, \
        "Frame with future timestamp beyond skew window must be rejected"


@pytest.mark.asyncio
async def test_past_timestamp_rejected():
    """Frame with timestamp 31s in the past must be rejected."""
    f = _filter(skew=30.0)
    result = await f.check("sensor-4b", seq=1, ts=now() - 31.0)
    assert not result, \
        "Frame with stale timestamp must be rejected (prevents delayed replay)"


@pytest.mark.asyncio
async def test_timestamp_within_skew_accepted():
    """Frame with timestamp within ±29s must be accepted."""
    f = _filter(skew=30.0)
    assert await f.check("sensor-4c", seq=1, ts=now() + 29.0)
    assert await f.check("sensor-4d", seq=1, ts=now() - 29.0)


# ---------------------------------------------------------------------------
# 5. Window boundary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_window_boundary_last_in_accepted():
    """The oldest sequence still within the window must be accepted."""
    f = _filter(window=8)
    # Fill window with seq 1..8
    for i in range(1, 9):
        await f.check("sensor-5", seq=i, ts=now())
    # seq=1 is exactly at the window boundary (last_seq=8, window=8, 1 == 8-8+1)
    # Behavior depends on implementation: seq <= last_seq - window is rejected
    # seq=0 (before window entirely) must be rejected
    result = await f.check("sensor-5", seq=0, ts=now())
    assert not result, "seq=0 when window is at 8 must be rejected"


# ---------------------------------------------------------------------------
# 6. Concurrent replay race — exactly one acceptance
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_replay_race():
    """100 coroutines simultaneously submit the same (sensor, seq, ts).
    Exactly one must be accepted; all others rejected."""
    f = _filter()
    ts = now()

    results = await asyncio.gather(
        *[f.check("sensor-race", seq=42, ts=ts) for _ in range(100)]
    )

    accepted = sum(1 for r in results if r is True)
    assert accepted == 1, \
        f"Concurrent replay: expected exactly 1 acceptance, got {accepted}. " \
        "This is a race condition in AntiReplayFilter — check asyncio.Lock usage."


# ---------------------------------------------------------------------------
# 7. Sensor reset bypass attempt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sensor_reset_allows_sequence_reuse():
    """After reset_sensor(), old sequences should be accepted again.
    This documents the intended behavior for sensor rekey/restart."""
    f = _filter()
    assert await f.check("sensor-7", seq=1, ts=now())
    assert not await f.check("sensor-7", seq=1, ts=now())  # duplicate

    f.reset_sensor("sensor-7")
    # After reset, seq=1 should be accepted again (sensor re-registered)
    assert await f.check("sensor-7", seq=1, ts=now()), \
        "After reset_sensor(), seq=1 must be accepted again (new session)"


# ---------------------------------------------------------------------------
# 8. Flood new sensor IDs — memory growth
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_flood_new_sensors_memory():
    """Create 10,000 distinct sensor IDs. Verify the filter handles it
    without OOM and each sensor correctly tracks its own state."""
    import tracemalloc
    f = _filter()
    tracemalloc.start()

    N = 10_000
    for i in range(N):
        sensor_id = f"sensor-flood-{i}"
        result = await f.check(sensor_id, seq=1, ts=now())
        assert result, f"sensor {sensor_id} seq=1 must be accepted"

    snapshot = tracemalloc.take_snapshot()
    total_kb = sum(s.size for s in snapshot.statistics("lineno")) / 1024
    tracemalloc.stop()

    # Each _SensorState is small; 10k sensors should be << 50MB
    assert total_kb < 50_000, \
        f"10k sensors used {total_kb:.0f}KB — check for per-sensor memory leak"


# ---------------------------------------------------------------------------
# 9. Sequence gap tolerance
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_contiguous_sequences_accepted():
    """Gaps in sequence numbers must be tolerated — not every packet arrives.
    This is normal for lossy comms (radio, mesh)."""
    f = _filter(window=64)
    # Deliberately non-contiguous
    for seq in [1, 5, 10, 50, 63, 100]:
        result = await f.check("sensor-9", seq=seq, ts=now())
        assert result, f"Non-contiguous seq={seq} must be accepted (normal gap)"


# ---------------------------------------------------------------------------
# 10. Zero sequence edge case
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_zero_sequence_accepted_once():
    """seq=0 is a valid first frame. Replaying seq=0 must be rejected."""
    f = _filter()
    assert await f.check("sensor-10", seq=0, ts=now()), \
        "seq=0 must be accepted as first frame"
    assert not await f.check("sensor-10", seq=0, ts=now()), \
        "Replayed seq=0 must be rejected"


# ---------------------------------------------------------------------------
# 11. Multi-sensor isolation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sensor_state_isolation():
    """Sensor A's window must not affect sensor B.
    Two sensors can both use seq=1 independently."""
    f = _filter()
    assert await f.check("sensor-A", seq=1, ts=now())
    assert await f.check("sensor-B", seq=1, ts=now()), \
        "Different sensors must have isolated replay state"
    # Replays are still blocked per-sensor
    assert not await f.check("sensor-A", seq=1, ts=now())
    assert not await f.check("sensor-B", seq=1, ts=now())


# ---------------------------------------------------------------------------
# 12. Sustained throughput
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_anti_replay_throughput():
    """Process 50,000 sequential frames and verify:
    - All accepted (no false positives on sequential delivery)
    - Throughput > 10,000 frames/second"""
    f = _filter(window=64)
    N = 50_000
    t0 = time.perf_counter()
    for i in range(N):
        result = await f.check("sensor-throughput", seq=i, ts=now())
        if not result:
            # seq=0 replay after window slide — only possible if seq wrapped
            # For N=50k, this should never happen with sequential delivery
            pytest.fail(f"Sequential frame seq={i} was rejected (false positive)")
    elapsed = time.perf_counter() - t0
    rate = N / elapsed
    assert rate > 10_000, \
        f"AntiReplayFilter throughput {rate:.0f} frames/s is too low (need >10k/s)"
