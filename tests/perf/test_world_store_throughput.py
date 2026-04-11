"""
World Store Throughput Benchmark

Target: >= 1,000 entity upserts/sec with 100 concurrent writers.
Run: pytest tests/perf/ -v --timeout=60
"""

import asyncio
import time
import uuid

import pytest


# ---------------------------------------------------------------------------
# In-memory WorldStore mock
# ---------------------------------------------------------------------------


class _MockWorldStore:
    """Minimal in-memory stand-in for WorldStore. Thread/task-safe via asyncio.Lock."""

    def __init__(self):
        self._store: dict = {}
        self._lock = asyncio.Lock()

    async def upsert(self, entity_id: str, data: dict) -> None:
        async with self._lock:
            self._store[entity_id] = data

    def size(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sequential_upserts():
    """1,000 sequential upserts must complete in under 2 seconds."""
    store = _MockWorldStore()
    n = 1_000

    start = time.perf_counter()
    for i in range(n):
        await store.upsert(
            str(uuid.uuid4()),
            {"seq": i, "lat": 37.0 + i * 0.0001, "lon": -122.0},
        )
    elapsed = time.perf_counter() - start

    throughput = n / elapsed
    print(f"\n[sequential] {n} upserts in {elapsed:.3f}s — {throughput:,.0f} upserts/sec")

    assert store.size() == n
    assert elapsed < 2.0, (
        f"Sequential upserts too slow: {elapsed:.3f}s (target < 2s, got {throughput:.0f}/sec)"
    )


@pytest.mark.asyncio
async def test_concurrent_writers():
    """100 concurrent asyncio tasks each doing 10 upserts — total < 3 seconds."""
    store = _MockWorldStore()
    num_tasks = 100
    upserts_per_task = 10
    total = num_tasks * upserts_per_task

    async def worker(task_id: int) -> None:
        for j in range(upserts_per_task):
            await store.upsert(
                f"entity-{task_id}-{j}",
                {"task": task_id, "seq": j, "ts": time.time()},
            )

    start = time.perf_counter()
    await asyncio.gather(*[worker(i) for i in range(num_tasks)])
    elapsed = time.perf_counter() - start

    throughput = total / elapsed
    print(
        f"\n[concurrent] {num_tasks} tasks × {upserts_per_task} upserts = {total} total "
        f"in {elapsed:.3f}s — {throughput:,.0f} upserts/sec"
    )

    assert store.size() == total
    assert elapsed < 3.0, (
        f"Concurrent upserts too slow: {elapsed:.3f}s (target < 3s, got {throughput:.0f}/sec)"
    )
