#!/usr/bin/env python3
"""
Heli.OS Entity Stream Load Simulator

Measures throughput ceiling and P95 latency of the backend WebSocket
broadcast by injecting N synthetic entity batches and timing round-trips.

Usage:
    python tools/load_sim.py [--url ws://localhost:8002/ws] [--counts 10,50,100,250,500]

What it tests:
  - How many entities/sec the backend can broadcast before latency degrades
  - P50 / P95 / P99 round-trip latency at each entity count
  - Whether any messages are dropped under load

Architecture:
  - One "sender" task injects entity_batch messages by connecting as a WS
    client and sending directly (simulates N adapters merging into backend)
  - One "receiver" task subscribes and timestamps arrival of each batch
  - Compares send-timestamps embedded in entity payloads with receive time

Note: The backend's entity_broadcast_loop already handles entity updates
internally — this sim patches additional synthetic entities on top to stress
the broadcast path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from typing import Optional

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets")
    raise


# ── Synthetic entity generator ─────────────────────────────────────────────

def _make_entity(i: int, ts_ms: float) -> dict:
    return {
        "entity_id": f"sim-{i:04d}",
        "entity_type": "active",
        "domain": "aerial",
        "classification": "SIM",
        "position": {
            "lat": 37.7749 + (i % 50) * 0.001,
            "lon": -122.4194 + (i // 50) * 0.001,
            "alt": 120.0,
            "heading_deg": random.uniform(0, 360),
        },
        "speed_mps": random.uniform(10, 30),
        "confidence": 0.95,
        "last_seen": ts_ms / 1000.0,
        "source_sensors": ["sim"],
        "_sim_ts_ms": ts_ms,  # embedded timestamp for latency calc
    }


def _make_batch(n: int) -> dict:
    ts = time.time() * 1000
    return {
        "type": "entity_batch",
        "data": [_make_entity(i, ts) for i in range(n)],
    }


# ── Receiver task ──────────────────────────────────────────────────────────

async def receiver(url: str, n_batches: int, latencies: list[float], ready: asyncio.Event):
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "subscribe", "channels": ["entities"]}))
        ready.set()
        received = 0
        while received < n_batches:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                msg = json.loads(raw)
                now_ms = time.time() * 1000
                if msg.get("type") == "entity_batch":
                    # Find embedded sim timestamp to compute latency
                    for entity in msg.get("data", []):
                        sent_ms = entity.get("_sim_ts_ms")
                        if sent_ms:
                            latencies.append(now_ms - sent_ms)
                            break
                    received += 1
            except asyncio.TimeoutError:
                break


# ── Sender task ────────────────────────────────────────────────────────────

async def sender(url: str, n_entities: int, n_batches: int, ready: asyncio.Event):
    await ready.wait()
    await asyncio.sleep(0.1)  # small gap so receiver is fully subscribed
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "subscribe", "channels": []}))
        for _ in range(n_batches):
            batch = _make_batch(n_entities)
            await ws.send(json.dumps(batch))
            await asyncio.sleep(0.05)  # 20 batches/sec


# ── Single run ─────────────────────────────────────────────────────────────

async def run_scenario(url: str, n_entities: int, n_batches: int = 50) -> Optional[dict]:
    latencies: list[float] = []
    ready = asyncio.Event()

    try:
        await asyncio.gather(
            receiver(url, n_batches, latencies, ready),
            sender(url, n_entities, n_batches, ready),
        )
    except Exception as e:
        return {"error": str(e), "n_entities": n_entities}

    if not latencies:
        return {"error": "no latencies recorded", "n_entities": n_entities}

    latencies.sort()
    return {
        "n_entities": n_entities,
        "n_samples": len(latencies),
        "p50_ms":  round(statistics.median(latencies), 1),
        "p95_ms":  round(latencies[int(len(latencies) * 0.95)], 1),
        "p99_ms":  round(latencies[int(len(latencies) * 0.99)], 1) if len(latencies) >= 100 else None,
        "max_ms":  round(max(latencies), 1),
        "mean_ms": round(statistics.mean(latencies), 1),
    }


# ── Main ───────────────────────────────────────────────────────────────────

async def main(url: str, counts: list[int]):
    print(f"\nHeli.OS Load Simulator → {url}")
    print(f"{'Entities':>10}  {'P50 ms':>8}  {'P95 ms':>8}  {'P99 ms':>8}  {'Max ms':>8}  {'Samples':>8}")
    print("-" * 62)

    results = []
    for n in counts:
        result = await run_scenario(url, n_entities=n, n_batches=60)
        results.append(result)
        if "error" in result:
            print(f"{n:>10}  ERROR: {result['error']}")
        else:
            p99 = f"{result['p99_ms']:>8.1f}" if result["p99_ms"] else "       —"
            print(f"{n:>10}  {result['p50_ms']:>8.1f}  {result['p95_ms']:>8.1f}  {p99}  "
                  f"{result['max_ms']:>8.1f}  {result['n_samples']:>8}")
        await asyncio.sleep(1.0)

    # Find ceiling: first n where P95 > 200ms
    print("\nSummary:")
    ceiling = None
    for r in results:
        if "error" not in r and r["p95_ms"] > 200:
            ceiling = r["n_entities"]
            break
    if ceiling:
        print(f"  Throughput ceiling (P95 > 200ms): ~{ceiling} entities/batch")
    else:
        print(f"  All scenarios under 200ms P95 — ceiling > {counts[-1]} entities/batch")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heli.OS load simulator")
    parser.add_argument("--url", default="ws://localhost:8002/ws")
    parser.add_argument("--counts", default="10,50,100,250,500",
                        help="Comma-separated entity counts to test")
    args = parser.parse_args()
    counts = [int(x) for x in args.counts.split(",")]
    asyncio.run(main(args.url, counts))
