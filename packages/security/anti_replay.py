"""
Anti-Replay Protection — Heli.OS Security

Sliding-window deduplication for sensor frames. Each frame carries:
  - sensor_id: str
  - seq: int (monotonically increasing per sensor)
  - ts: float (unix timestamp)

Window size: 64 sequences. Frames older than WINDOW or with replayed seq are rejected.
Thread-safe via asyncio.Lock (not threading.Lock — all callers are async).
"""

import asyncio
import logging
import time
from collections import deque
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class _SensorState:
    """Per-sensor sliding window state."""

    __slots__ = ("last_seq", "seen", "lock")

    def __init__(self, window_size: int):
        self.last_seq: Optional[int] = None
        self.seen: deque = deque(maxlen=window_size)
        self.lock = asyncio.Lock()


class AntiReplayFilter:
    """
    Sliding-window anti-replay filter for sensor frames.

    Rejects frames that are:
      - Too old: seq <= last_seq - window_size
      - Duplicate: seq already in the seen window
      - Timestamp-skewed: |ts - now| > max_ts_skew_s
    """

    def __init__(self, window_size: int = 64, max_ts_skew_s: float = 30.0):
        self._window_size = window_size
        self._max_ts_skew_s = max_ts_skew_s
        self._sensors: Dict[str, _SensorState] = {}
        self._global_lock = asyncio.Lock()

    async def _get_or_create(self, sensor_id: str) -> _SensorState:
        async with self._global_lock:
            if sensor_id not in self._sensors:
                self._sensors[sensor_id] = _SensorState(self._window_size)
            return self._sensors[sensor_id]

    async def check(self, sensor_id: str, seq: int, ts: float) -> bool:
        """
        Return True if the frame should be accepted, False if it should be rejected.

        Rejection criteria (any one sufficient):
          - Timestamp skew > max_ts_skew_s
          - seq is in the seen deque (duplicate)
          - seq <= last_seq - window_size (too old / outside window)
        """
        now = time.time()

        # 1. Timestamp skew check
        if abs(ts - now) > self._max_ts_skew_s:
            logger.warning(
                "AntiReplayFilter: rejected sensor_id=%s seq=%d — timestamp skew %.1fs",
                sensor_id,
                seq,
                abs(ts - now),
            )
            return False

        state = await self._get_or_create(sensor_id)

        async with state.lock:
            last = state.last_seq

            # 2. Too-old check (outside sliding window)
            if last is not None and seq <= last - self._window_size:
                logger.warning(
                    "AntiReplayFilter: rejected sensor_id=%s seq=%d — too old (last=%d window=%d)",
                    sensor_id,
                    seq,
                    last,
                    self._window_size,
                )
                return False

            # 3. Duplicate check
            if seq in state.seen:
                logger.warning(
                    "AntiReplayFilter: rejected sensor_id=%s seq=%d — duplicate",
                    sensor_id,
                    seq,
                )
                return False

            # Accept: update state
            state.seen.append(seq)
            if last is None or seq > last:
                state.last_seq = seq

        return True

    def reset_sensor(self, sensor_id: str) -> None:
        """Reset state for a given sensor (e.g. after rekey or sensor restart)."""
        self._sensors.pop(sensor_id, None)
        logger.info("AntiReplayFilter.reset_sensor: cleared state for sensor_id=%s", sensor_id)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_filter_instance: Optional[AntiReplayFilter] = None


def get_filter(window_size: int = 64, max_ts_skew_s: float = 30.0) -> AntiReplayFilter:
    """Return the module-level singleton AntiReplayFilter."""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = AntiReplayFilter(
            window_size=window_size, max_ts_skew_s=max_ts_skew_s
        )
    return _filter_instance
