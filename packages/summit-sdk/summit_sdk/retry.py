"""
Summit.OS SDK — Retry & Circuit Breaker

Provides:
- RetryPolicy: exponential backoff with jitter
- CircuitBreaker: fail-fast when a service is down
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Set, TypeVar

from .errors import (
    CircuitOpenError,
    RateLimitError,
    ServerError,
    SummitError,
    TimeoutError as SummitTimeout,
)

logger = logging.getLogger("summit.sdk.retry")

T = TypeVar("T")


# ── Retry Policy ────────────────────────────────────────────


@dataclass
class RetryPolicy:
    """Configurable retry with exponential backoff + jitter."""

    max_retries: int = 3
    base_delay: float = 0.5  # seconds
    max_delay: float = 30.0  # seconds
    backoff_factor: float = 2.0
    jitter: float = 0.25  # ± 25%
    retryable_status: Set[int] = field(
        default_factory=lambda: {408, 429, 500, 502, 503, 504}
    )

    def delay_for_attempt(self, attempt: int) -> float:
        """Calculate delay for a given attempt number (0-indexed)."""
        delay = self.base_delay * (self.backoff_factor**attempt)
        delay = min(delay, self.max_delay)
        # Add jitter
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        return max(0, delay)

    def should_retry(
        self, attempt: int, status: int = 0, error: Optional[Exception] = None
    ) -> bool:
        """Decide whether to retry based on attempt count and error type."""
        if attempt >= self.max_retries:
            return False
        if status and status in self.retryable_status:
            return True
        if isinstance(error, (SummitTimeout, ServerError, RateLimitError)):
            return True
        if isinstance(error, (OSError, asyncio.TimeoutError)):
            return True
        return False


# ── Circuit Breaker ─────────────────────────────────────────


class CircuitState(str, Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing — reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """
    Simple circuit breaker.

    - CLOSED: requests flow normally. Failures increment counter.
    - OPEN: after `failure_threshold` failures within `window_s`, reject
      all calls with CircuitOpenError for `recovery_timeout` seconds.
    - HALF_OPEN: after recovery timeout, allow one probe request.
      If it succeeds → CLOSED. If it fails → OPEN again.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # seconds
    window_s: float = 60.0  # sliding window

    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failures: list = field(default_factory=list, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _half_open_in_flight: bool = field(default=False, init=False)

    def _prune_old(self) -> None:
        cutoff = time.monotonic() - self.window_s
        self._failures = [t for t in self._failures if t > cutoff]

    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self._half_open_in_flight = True
                logger.info("Circuit breaker → HALF_OPEN (probing)")
                return True
            return False

        # HALF_OPEN — only one probe at a time
        if self._half_open_in_flight:
            return False
        self._half_open_in_flight = True
        return True

    def record_success(self) -> None:
        """Record a successful request."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker → CLOSED (probe succeeded)")
            self.state = CircuitState.CLOSED
            self._failures.clear()
            self._half_open_in_flight = False
        # In CLOSED, a success just resets naturally via pruning

    def record_failure(self) -> None:
        """Record a failed request."""
        now = time.monotonic()

        if self.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker → OPEN (probe failed)")
            self.state = CircuitState.OPEN
            self._opened_at = now
            self._half_open_in_flight = False
            return

        self._failures.append(now)
        self._prune_old()

        if len(self._failures) >= self.failure_threshold:
            logger.warning(
                f"Circuit breaker → OPEN ({len(self._failures)} failures "
                f"in {self.window_s}s)"
            )
            self.state = CircuitState.OPEN
            self._opened_at = now

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self.state = CircuitState.CLOSED
        self._failures.clear()
        self._half_open_in_flight = False


# ── Retry executor ──────────────────────────────────────────


async def retry_with_circuit(
    fn: Callable[..., Any],
    *args: Any,
    retry: RetryPolicy = RetryPolicy(),
    breaker: Optional[CircuitBreaker] = None,
    **kwargs: Any,
) -> Any:
    """
    Execute `fn` with retry policy and optional circuit breaker.

    Raises the last error if all retries are exhausted.
    """
    last_error: Optional[Exception] = None

    for attempt in range(retry.max_retries + 1):
        # Circuit breaker gate
        if breaker and not breaker.allow_request():
            raise CircuitOpenError()

        try:
            result = await fn(*args, **kwargs)
            if breaker:
                breaker.record_success()
            return result

        except SummitError as e:
            last_error = e
            if breaker:
                breaker.record_failure()
            if not retry.should_retry(attempt, status=e.status, error=e):
                raise
            delay = retry.delay_for_attempt(attempt)
            # For 429, respect server's retry-after if longer
            if isinstance(e, RateLimitError):
                server_delay = e.details.get("retry_after_s", 0)
                delay = max(delay, server_delay)
            logger.debug(
                f"Retry {attempt+1}/{retry.max_retries} "
                f"after {delay:.1f}s — {e.code.value}"
            )
            await asyncio.sleep(delay)

        except (OSError, asyncio.TimeoutError) as e:
            last_error = e
            if breaker:
                breaker.record_failure()
            if attempt >= retry.max_retries:
                raise SummitTimeout(str(e))
            delay = retry.delay_for_attempt(attempt)
            logger.debug(
                f"Retry {attempt+1}/{retry.max_retries} "
                f"after {delay:.1f}s — {type(e).__name__}"
            )
            await asyncio.sleep(delay)

    raise last_error or SummitTimeout("All retries exhausted")
