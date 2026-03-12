"""
Summit.OS MQTT Ingress Rate Limiter

Token bucket rate limiter keyed by source_id (extracted from entity payloads
or MQTT topic). Prevents a misbehaving or compromised adapter from flooding
the world model and taking down the platform.

Configuration (environment variables):
    MQTT_RATE_LIMIT_DEFAULT   - max msgs/sec per source (default: 50)
    MQTT_RATE_LIMIT_BURST     - burst allowance above rate (default: 100)
    MQTT_RATE_LIMIT_ENABLED   - "false" to disable (default: "true")

Source-specific overrides via MQTT_RATE_LIMITS env var (JSON):
    '{"opensky": 200, "modbus-pump-01": 10}'

Behavior:
    - Messages within limit: pass through
    - Messages over limit: DROPPED (not queued) + counter incremented
    - Violations logged at WARNING level with source and drop count
    - Violations exposed as Prometheus counter (if prometheus_client available)
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from typing import Dict, Optional

logger = logging.getLogger("summit.fabric.ratelimit")

_RATE_LIMIT_ENABLED = os.getenv("MQTT_RATE_LIMIT_ENABLED", "true").lower() == "true"
_DEFAULT_RATE = float(os.getenv("MQTT_RATE_LIMIT_DEFAULT", "50"))   # msgs/sec
_DEFAULT_BURST = float(os.getenv("MQTT_RATE_LIMIT_BURST", "100"))    # burst

# Parse per-source overrides
_SOURCE_OVERRIDES: Dict[str, float] = {}
try:
    _overrides_raw = os.getenv("MQTT_RATE_LIMITS", "{}")
    _SOURCE_OVERRIDES = json.loads(_overrides_raw)
except Exception:
    pass

# Prometheus counter (optional)
try:
    from prometheus_client import Counter
    _DROPPED_MSGS = Counter(
        "summit_mqtt_rate_limited_total",
        "Total MQTT messages dropped by rate limiter",
        ["source_id"],
    )
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False


class _TokenBucket:
    """
    Token bucket for a single source.

    Tokens refill at `rate` per second up to `burst`.
    Each message consumes one token.
    """

    __slots__ = ("rate", "burst", "tokens", "_last_refill", "_drop_count")

    def __init__(self, rate: float, burst: float):
        self.rate = rate
        self.burst = burst
        self.tokens = burst          # start full
        self._last_refill = time.monotonic()
        self._drop_count = 0

    def allow(self) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now

        # Refill
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        else:
            self._drop_count += 1
            return False


class MQTTRateLimiter:
    """
    Per-source token bucket rate limiter for MQTT ingress.

    Usage:
        limiter = MQTTRateLimiter()

        def on_message(client, userdata, msg):
            source_id = extract_source_id(msg)
            if not limiter.allow(source_id):
                return   # drop silently
            # ... process message
    """

    def __init__(
        self,
        default_rate: float = _DEFAULT_RATE,
        default_burst: float = _DEFAULT_BURST,
        enabled: bool = _RATE_LIMIT_ENABLED,
    ):
        self.enabled = enabled
        self.default_rate = default_rate
        self.default_burst = default_burst
        self._buckets: Dict[str, _TokenBucket] = defaultdict(self._new_bucket)

        if not enabled:
            logger.info("MQTT rate limiter DISABLED")
        else:
            logger.info(
                f"MQTT rate limiter enabled: default={default_rate} msgs/sec, "
                f"burst={default_burst}, overrides={_SOURCE_OVERRIDES}"
            )

    def _new_bucket(self) -> _TokenBucket:
        return _TokenBucket(self.default_rate, self.default_burst)

    def _get_bucket(self, source_id: str) -> _TokenBucket:
        if source_id not in self._buckets:
            if source_id in _SOURCE_OVERRIDES:
                rate = _SOURCE_OVERRIDES[source_id]
                burst = rate * 2  # burst = 2x rate for per-source overrides
            else:
                rate = self.default_rate
                burst = self.default_burst
            self._buckets[source_id] = _TokenBucket(rate, burst)
        return self._buckets[source_id]

    def allow(self, source_id: str) -> bool:
        """
        Check if a message from source_id is allowed.

        Returns True (allow) or False (drop).
        """
        if not self.enabled:
            return True

        bucket = self._get_bucket(source_id)
        allowed = bucket.allow()

        if not allowed:
            # Log at intervals to avoid log spam
            if bucket._drop_count == 1 or bucket._drop_count % 100 == 0:
                logger.warning(
                    f"Rate limit exceeded: source='{source_id}', "
                    f"drops={bucket._drop_count}, "
                    f"limit={bucket.rate:.0f} msgs/sec"
                )
            if _PROM_AVAILABLE:
                _DROPPED_MSGS.labels(source_id=source_id).inc()

        return allowed

    def get_stats(self) -> Dict[str, dict]:
        """Return per-source stats for the health endpoint."""
        return {
            source_id: {
                "rate_limit": bucket.rate,
                "tokens_remaining": round(bucket.tokens, 2),
                "drop_count": bucket._drop_count,
            }
            for source_id, bucket in self._buckets.items()
        }


def extract_source_id(topic: str, payload: Optional[dict] = None) -> str:
    """
    Extract a source identifier for rate limiting from MQTT topic or payload.

    Priority:
    1. provenance.source_id from entity payload
    2. First segment of MQTT topic path (e.g. "entities" from "entities/drone-01/update")
    3. Fallback: "unknown"
    """
    if payload and isinstance(payload, dict):
        provenance = payload.get("provenance", {})
        source_id = provenance.get("source_id", "")
        if source_id:
            # Use just the adapter prefix (before first hyphen) to group sources
            # e.g., "modbus-pump-01" → "modbus" for rate limiting
            return source_id.split("-")[0] if "-" in source_id else source_id

    if topic:
        parts = topic.split("/")
        # "entities/{entity_id}/update" → use second segment (entity type prefix)
        if len(parts) >= 2:
            entity_id = parts[1]
            return entity_id.split("-")[0] if "-" in entity_id else entity_id
        return parts[0]

    return "unknown"
