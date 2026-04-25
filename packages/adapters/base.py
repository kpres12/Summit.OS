"""
Heli.OS Adapter Base Framework
=================================

Defines the abstract base class that all Heli.OS signal adapters must
implement.  The framework handles:

  - Lifecycle management (start / stop with clean task cancellation)
  - Exponential backoff reconnect (1s → 2s → 4s → … capped at 60s)
  - Health tracking (status, last_connected, last_observation, throughput)
  - Observations-per-minute calculated from a sliding 60-second window
  - MQTT publish to ``summit/observations/{adapter_type}``
  - Graceful degradation: if no MQTT client is provided, observations are
    logged at DEBUG level instead (useful for local development and testing)

Minimal adapter implementation
-------------------------------
Subclass ``BaseAdapter``, set the class-level ``adapter_type`` attribute,
and implement the three abstract methods::

    class MyAdapter(BaseAdapter):
        adapter_type = "my_source"

        async def connect(self) -> None:
            # open HTTP session, serial port, socket, etc.
            ...

        async def disconnect(self) -> None:
            # clean up resources
            ...

        async def stream_observations(self) -> AsyncIterator[dict]:
            while True:
                obs = await self._fetch_one()
                yield obs
                await asyncio.sleep(self.config.poll_interval_seconds)

The framework calls ``connect()``, then iterates ``stream_observations()``,
calling ``_record_observation()`` and ``_publish()`` on each yielded dict.
If the generator raises, the framework disconnects, waits with backoff, and
reconnects automatically.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from pydantic import BaseModel

logger = logging.getLogger("heli.adapters")


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------


class AdapterConfig(BaseModel):
    """
    Configuration for a single adapter instance.

    ``extra`` holds adapter-specific settings (e.g. bounding box for
    OpenSky, TLE group for CelesTrak, RTSP URL for a camera).  This keeps
    the base config schema stable as new adapter types are added.
    """

    adapter_id: str
    """Unique slug identifying this adapter instance, e.g. ``opensky-us-west``."""

    adapter_type: str
    """
    Adapter type key, must match ``BaseAdapter.adapter_type`` on the
    implementation class.  E.g. ``"opensky"``, ``"celestrak"``, ``"rtsp"``.
    """

    display_name: str
    """Human-readable name shown in the DEV console."""

    description: str = ""
    """Optional description for the adapter registry listing."""

    enabled: bool = True
    """When ``False`` the adapter is registered but not started."""

    poll_interval_seconds: float = 5.0
    """
    How often the adapter should poll its source (seconds).  Adapters that
    stream continuously can ignore this; poll-based adapters should honour it
    via ``asyncio.sleep(self.config.poll_interval_seconds)`` in their
    ``stream_observations()`` loop.
    """

    extra: dict = {}
    """
    Adapter-specific configuration dictionary.  Keys and structure are
    defined by each adapter implementation.
    """

    class Config:
        extra = "allow"


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------


class AdapterStatus:
    """String constants for adapter operational status."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    ERROR = "ERROR"
    DISABLED = "DISABLED"


# ---------------------------------------------------------------------------
# Health snapshot
# ---------------------------------------------------------------------------


@dataclass
class AdapterHealth:
    """
    Point-in-time health snapshot for a single adapter.

    Returned by ``BaseAdapter.health()`` and collected by the registry for
    the ``GET /adapters`` API endpoint.
    """

    status: str
    """Current ``AdapterStatus`` value."""

    last_connected: Optional[datetime]
    """UTC timestamp of the most recent successful ``connect()`` call."""

    last_observation: Optional[datetime]
    """UTC timestamp of the most recent observation yielded by the adapter."""

    observations_total: int
    """Total observations emitted since the adapter process started."""

    observations_per_minute: float
    """
    Smoothed throughput over a sliding 60-second window.  Zero when the
    adapter has not yet emitted any observations.
    """

    error_message: Optional[str]
    """Last error message; ``None`` when status is CONNECTED."""

    uptime_seconds: float
    """
    Seconds since the adapter was last successfully connected.
    Zero if never connected.
    """


# ---------------------------------------------------------------------------
# Observation schema (reference only — not enforced at runtime)
# ---------------------------------------------------------------------------

# Adapters must yield dicts conforming to the following shape.  Fields
# marked Optional may be omitted (``None``) when the source does not
# provide that data.
#
# {
#     "source_id":    str,          # unique ID for this observation
#     "adapter_id":   str,          # self.config.adapter_id
#     "adapter_type": str,          # self.adapter_type
#     "entity_id":    str,          # stable entity identifier
#     "callsign":     str | None,
#     "position": {
#         "lat":   float,
#         "lon":   float,
#         "alt_m": float | None,
#     } | None,
#     "velocity": {
#         "heading_deg":   float | None,
#         "speed_mps":     float | None,
#         "vertical_mps":  float | None,
#     } | None,
#     "entity_type":    str,        # "AIRCRAFT", "GROUND", "SENSOR", "CAMERA", …
#     "classification": str | None,
#     "metadata":       dict,       # adapter-specific extra fields
#     "ts_iso":         str,        # UTC ISO8601
# }


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------


class BaseAdapter(ABC):
    """
    Abstract base for all Heli.OS signal adapters.

    An adapter connects to ONE signal source and emits a stream of
    observation dicts that the Fusion service ingests.

    Subclasses must:
      1. Set the class attribute ``adapter_type`` to a unique type slug.
      2. Implement ``connect()``, ``disconnect()``, and
         ``stream_observations()``.

    The framework handles reconnect backoff, health tracking, metrics
    counters, and MQTT publishing.
    """

    #: Override in subclass with the adapter's type slug.
    adapter_type: str = "base"

    # Backoff parameters
    _BACKOFF_BASE: float = 1.0
    _BACKOFF_MAX: float = 60.0

    def __init__(
        self,
        config: AdapterConfig,
        mqtt_client=None,
    ) -> None:
        self.config = config
        self.mqtt = mqtt_client

        self._status: str = AdapterStatus.DISCONNECTED
        self._obs_count: int = 0
        self._obs_window: deque[float] = deque(maxlen=120)  # timestamps (up to 2 min)
        self._last_connected: Optional[datetime] = None
        self._last_observation: Optional[datetime] = None
        self._connected_at: Optional[float] = None  # monotonic time
        self._error_message: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._stop_event: asyncio.Event = asyncio.Event()

        self._log = logging.getLogger(
            f"heli.adapters.{config.adapter_type}.{config.adapter_id}"
        )

    # -------------------------------------------------------------------------
    # Config validation — subclasses may override
    # -------------------------------------------------------------------------

    @classmethod
    def required_extra_fields(cls) -> list[str]:
        """
        Return a list of ``extra`` field names that are required for this
        adapter type.  The registry calls this before instantiating adapters
        to surface missing-config errors to the operator.

        Override in subclasses::

            @classmethod
            def required_extra_fields(cls) -> list[str]:
                return ["host", "port"]
        """
        return []

    @classmethod
    def validate_extra(cls, extra: dict) -> list[str]:
        """
        Validate the adapter's ``extra`` config dict.

        Returns a list of human-readable error strings (empty = valid).
        Subclasses may override for richer validation; this base
        implementation checks that all ``required_extra_fields()`` are
        present and non-empty.
        """
        errors: list[str] = []
        for field in cls.required_extra_fields():
            if not extra.get(field):
                errors.append(f"Missing required config field: '{field}'")
        return errors

    # -------------------------------------------------------------------------
    # Abstract interface — subclasses implement these three methods
    # -------------------------------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """
        Open the connection to the signal source.

        Raise any exception on failure; the framework will catch it,
        transition to ERROR status, and retry with backoff.
        """

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close the connection and release all resources held by the adapter.

        Called both on graceful shutdown and before each reconnect attempt.
        Must not raise; swallow exceptions internally.
        """

    @abstractmethod
    async def stream_observations(self) -> AsyncIterator[dict]:
        """
        Yield observation dicts conforming to the Heli.OS observation
        schema (see module docstring).

        The generator runs inside the framework's reconnect loop.  If it
        raises, the framework disconnects and reconnects with backoff.

        Poll-based adapters should sleep between iterations::

            async def stream_observations(self):
                while True:
                    obs = await self._fetch_batch()
                    for o in obs:
                        yield o
                    await asyncio.sleep(self.config.poll_interval_seconds)
        """
        # Satisfy the type checker — subclasses must yield at least once.
        # ``yield`` here makes this an async generator function.
        return
        yield  # pragma: no cover

    # -------------------------------------------------------------------------
    # Framework methods — do not override
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """
        Start the adapter run loop.

        Connects to the source, streams observations, and automatically
        reconnects with exponential backoff on failure.  Returns only after
        ``stop()`` has been called.
        """
        if not self.config.enabled:
            self._status = AdapterStatus.DISABLED
            self._log.info("Adapter disabled by config — not starting.")
            return

        self._log.info(
            "Adapter starting (type=%s, id=%s)",
            self.adapter_type,
            self.config.adapter_id,
        )

        backoff = self._BACKOFF_BASE

        while not self._stop_event.is_set():
            # ── Connect ────────────────────────────────────────────────────
            self._status = AdapterStatus.CONNECTING
            self._error_message = None
            try:
                await self.connect()
                self._status = AdapterStatus.CONNECTED
                self._last_connected = datetime.now(timezone.utc)
                self._connected_at = time.monotonic()
                backoff = self._BACKOFF_BASE  # reset on success
                self._log.info("Adapter connected.")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._status = AdapterStatus.ERROR
                self._error_message = str(exc)
                self._log.error("Connect failed: %s — retrying in %.0fs", exc, backoff)
                await self._interruptible_sleep(backoff)
                backoff = min(backoff * 2, self._BACKOFF_MAX)
                continue

            # ── Stream observations ────────────────────────────────────────
            try:
                async for obs in self.stream_observations():
                    if self._stop_event.is_set():
                        break
                    self._record_observation()
                    await self._publish(obs)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._status = AdapterStatus.ERROR
                self._error_message = str(exc)
                self._log.error(
                    "Stream error: %s — disconnecting and retrying in %.0fs",
                    exc,
                    backoff,
                )

            # ── Disconnect before potential retry ──────────────────────────
            try:
                await self.disconnect()
            except Exception:
                pass

            if self._stop_event.is_set():
                break

            self._status = AdapterStatus.DISCONNECTED
            await self._interruptible_sleep(backoff)
            backoff = min(backoff * 2, self._BACKOFF_MAX)

        # Final disconnect
        try:
            await self.disconnect()
        except Exception:
            pass

        self._status = AdapterStatus.DISCONNECTED
        self._log.info("Adapter stopped.")

    async def stop(self) -> None:
        """
        Signal the adapter to stop and wait for the run loop to exit.

        Safe to call multiple times.
        """
        self._stop_event.set()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    def health(self) -> AdapterHealth:
        """Return a consistent point-in-time health snapshot."""
        now_mono = time.monotonic()
        uptime = (
            now_mono - self._connected_at
            if self._connected_at is not None
            and self._status == AdapterStatus.CONNECTED
            else 0.0
        )
        return AdapterHealth(
            status=self._status,
            last_connected=self._last_connected,
            last_observation=self._last_observation,
            observations_total=self._obs_count,
            observations_per_minute=self._obs_per_minute(),
            error_message=self._error_message,
            uptime_seconds=uptime,
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _record_observation(self) -> None:
        """Increment the observation counter and update the sliding window."""
        now = time.monotonic()
        self._obs_count += 1
        self._obs_window.append(now)
        self._last_observation = datetime.now(timezone.utc)

    async def _publish(self, observation: dict) -> None:
        """
        Publish an observation to MQTT or log it if no client is configured.

        Topic: ``summit/observations/{adapter_type}``
        """
        topic = f"summit/observations/{self.adapter_type}"
        payload = json.dumps(observation, default=str)

        if self.mqtt is not None:
            try:
                self.mqtt.publish(topic, payload, qos=0)
            except Exception as exc:
                self._log.warning("MQTT publish failed: %s", exc)
        else:
            self._log.debug("OBS [%s] %s", topic, payload)

    def _obs_per_minute(self) -> float:
        """
        Calculate observations/minute from the sliding 60-second window.

        Evicts timestamps older than 60 seconds before counting.
        """
        cutoff = time.monotonic() - 60.0
        # Trim stale entries from the left of the deque
        while self._obs_window and self._obs_window[0] < cutoff:
            self._obs_window.popleft()
        count = len(self._obs_window)
        if count == 0:
            return 0.0
        # Extrapolate to per-minute rate based on window span
        if count == 1:
            return count * 1.0  # can't compute rate from a single point
        span = self._obs_window[-1] - self._obs_window[0]
        if span <= 0:
            return 0.0
        return (count / span) * 60.0

    async def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep for ``seconds`` but wake immediately if stop is signalled."""
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass
