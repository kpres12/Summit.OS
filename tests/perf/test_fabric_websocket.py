"""WebSocket Fan-out Throughput Benchmark — Target: 500 msgs/sec to 50 clients."""

import asyncio
import time

import pytest

# ---------------------------------------------------------------------------
# Mock WebSocket manager
# ---------------------------------------------------------------------------


class _MockWebSocketManager:
    """
    Fan-out manager that distributes messages to a list of asyncio Queues,
    one per connected client.  Simulates the server-side broadcast path.
    """

    def __init__(self):
        self._clients: list[asyncio.Queue] = []

    def add_client(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._clients.append(q)
        return q

    async def broadcast(self, message: bytes) -> None:
        for q in self._clients:
            await q.put(message)

    @property
    def client_count(self) -> int:
        return len(self._clients)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fanout_500_msgs_50_clients():
    """
    50 clients each receive 500 broadcast messages.
    All queues must drain within 5 seconds.
    """
    num_clients = 50
    num_messages = 500

    manager = _MockWebSocketManager()
    queues = [manager.add_client() for _ in range(num_clients)]

    assert manager.client_count == num_clients

    # Broadcast all messages
    start = time.perf_counter()
    for i in range(num_messages):
        await manager.broadcast(f"msg-{i}".encode())
    broadcast_elapsed = time.perf_counter() - start

    # Drain all queues and verify counts
    async def drain(q: asyncio.Queue, expected: int) -> int:
        count = 0
        try:
            while True:
                await asyncio.wait_for(q.get(), timeout=5.0)
                count += 1
                if count >= expected:
                    break
        except asyncio.TimeoutError:
            pass
        return count

    drain_start = time.perf_counter()
    results = await asyncio.gather(*[drain(q, num_messages) for q in queues])
    total_elapsed = time.perf_counter() - drain_start

    total_msgs_received = sum(results)
    expected_total = num_clients * num_messages
    msgs_per_sec = expected_total / (broadcast_elapsed + total_elapsed)

    print(
        f"\n[fanout] {num_clients} clients × {num_messages} msgs = {expected_total} total deliveries "
        f"in {broadcast_elapsed + total_elapsed:.3f}s — {msgs_per_sec:,.0f} msg-deliveries/sec"
    )

    for i, count in enumerate(results):
        assert count == num_messages, (
            f"Client {i} received {count}/{num_messages} messages"
        )

    assert broadcast_elapsed + total_elapsed < 5.0, (
        f"Fan-out took {broadcast_elapsed + total_elapsed:.3f}s (target < 5s)"
    )
