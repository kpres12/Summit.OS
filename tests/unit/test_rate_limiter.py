"""Unit tests for the MQTT ingress rate limiter."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "apps", "fabric"))

from rate_limiter import MQTTRateLimiter, extract_source_id


class TestTokenBucket:

    def test_allows_within_limit(self):
        limiter = MQTTRateLimiter(default_rate=100, default_burst=100)
        for _ in range(50):
            assert limiter.allow("source-a") is True

    def test_drops_over_burst(self):
        limiter = MQTTRateLimiter(default_rate=10, default_burst=5)
        allowed = sum(1 for _ in range(20) if limiter.allow("bursty"))
        assert allowed == 5

    def test_tokens_refill_over_time(self):
        limiter = MQTTRateLimiter(default_rate=100, default_burst=5)
        # Exhaust burst
        for _ in range(5):
            limiter.allow("refill-test")
        # Should be empty now
        assert limiter.allow("refill-test") is False
        # Manually advance the bucket's last_refill time
        bucket = limiter._get_bucket("refill-test")
        bucket._last_refill -= 0.5  # simulate 0.5s passing → 50 tokens
        assert limiter.allow("refill-test") is True

    def test_independent_buckets_per_source(self):
        limiter = MQTTRateLimiter(default_rate=10, default_burst=2)
        # Exhaust source-a
        limiter.allow("source-a")
        limiter.allow("source-a")
        assert limiter.allow("source-a") is False
        # source-b should be unaffected
        assert limiter.allow("source-b") is True

    def test_disabled_limiter_always_allows(self):
        limiter = MQTTRateLimiter(default_rate=1, default_burst=1, enabled=False)
        for _ in range(1000):
            assert limiter.allow("flood") is True

    def test_drop_count_increments(self):
        limiter = MQTTRateLimiter(default_rate=1, default_burst=1)
        limiter.allow("drop-test")  # consume burst
        for _ in range(5):
            limiter.allow("drop-test")
        bucket = limiter._get_bucket("drop-test")
        assert bucket._drop_count == 5

    def test_get_stats_returns_per_source(self):
        limiter = MQTTRateLimiter(default_rate=50, default_burst=50)
        limiter.allow("stats-source")
        stats = limiter.get_stats()
        assert "stats-source" in stats
        assert "rate_limit" in stats["stats-source"]
        assert "drop_count" in stats["stats-source"]


class TestExtractSourceId:

    def test_extracts_from_provenance(self):
        payload = {"provenance": {"source_id": "modbus-pump-01"}}
        assert extract_source_id("entities/x/update", payload) == "modbus"

    def test_extracts_from_topic_when_no_provenance(self):
        result = extract_source_id("entities/opensky-abc123/update", {})
        assert result == "opensky"

    def test_fallback_unknown(self):
        result = extract_source_id("", None)
        assert result == "unknown"

    def test_non_prefixed_source_id_returned_as_is(self):
        payload = {"provenance": {"source_id": "celestrak"}}
        assert extract_source_id("", payload) == "celestrak"
