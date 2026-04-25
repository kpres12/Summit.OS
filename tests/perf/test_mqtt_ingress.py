"""MQTT Ingress Throughput Benchmark — Target: 2,000 msgs/sec ingestion."""

import asyncio
import json
import time

import pytest

# ---------------------------------------------------------------------------
# Mock MQTT ingest pipeline
# ---------------------------------------------------------------------------


class _MockMQTTIngestor:
    """
    Simulates the MQTT message callback path:
      - on_message() is called for each arriving MQTT message
      - Messages are enqueued into an asyncio.Queue for downstream processing
    """

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._received = 0

    def on_message(self, topic: str, payload: bytes) -> None:
        """Synchronous MQTT callback — put_nowait to avoid blocking the event loop."""
        self._queue.put_nowait({"topic": topic, "payload": payload})
        self._received += 1

    async def consume_all(self, expected: int, timeout_s: float = 2.0) -> int:
        """Drain the queue until *expected* messages consumed or timeout."""
        consumed = 0
        deadline = time.monotonic() + timeout_s
        while consumed < expected:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(self._queue.get(), timeout=min(remaining, 0.1))
                consumed += 1
            except asyncio.TimeoutError:
                continue
        return consumed

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()


def _make_telemetry(i: int) -> tuple[str, bytes]:
    topic = f"summit/sensor/uav-{i % 50}/telemetry"
    payload = json.dumps(
        {
            "seq": i,
            "ts": time.time(),
            "lat": 37.0 + i * 0.00001,
            "lon": -122.0 - i * 0.00001,
            "alt_m": 100.0,
            "sensor_id": f"uav-{i % 50}",
        }
    ).encode("utf-8")
    return topic, payload


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mqtt_ingress_2k():
    """
    Publish 2,000 fake telemetry messages through the mock MQTT callback
    and assert all are consumed within 2 seconds.
    """
    ingestor = _MockMQTTIngestor()
    num_messages = 2_000

    # Publish phase — synchronous callback simulation
    publish_start = time.perf_counter()
    for i in range(num_messages):
        topic, payload = _make_telemetry(i)
        ingestor.on_message(topic, payload)
    publish_elapsed = time.perf_counter() - publish_start

    assert ingestor._received == num_messages

    # Consume phase — drain the asyncio queue
    consume_start = time.perf_counter()
    consumed = await ingestor.consume_all(num_messages, timeout_s=2.0)
    consume_elapsed = time.perf_counter() - consume_start

    total_elapsed = publish_elapsed + consume_elapsed
    throughput = num_messages / total_elapsed

    print(
        f"\n[mqtt-ingress] {num_messages} msgs: "
        f"publish={publish_elapsed*1000:.1f}ms consume={consume_elapsed*1000:.1f}ms "
        f"total={total_elapsed*1000:.1f}ms — {throughput:,.0f} msgs/sec"
    )

    assert consumed == num_messages, (
        f"Only consumed {consumed}/{num_messages} messages within timeout"
    )
    assert total_elapsed < 2.0, (
        f"MQTT ingress too slow: {total_elapsed:.3f}s (target < 2s, got {throughput:.0f} msgs/sec)"
    )
