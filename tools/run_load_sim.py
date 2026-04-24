#!/usr/bin/env python3
"""
Heli.OS Entity Stream Load Simulation

Tests the production load pattern: server broadcasts one entity_batch
per second (matching the backend tick loop) and measures:
  - Broadcast duration (how long manager.broadcast() takes per tick)
  - Client-side receive latency (server_ts → client_received_ts)
  - Whether any ticks are missed at scale

This matches production: N entities update every 1s and are broadcast
to all connected console clients simultaneously.
"""
import asyncio
import json
import statistics
import time

try:
    import websockets
    from websockets.server import serve
except ImportError:
    print("pip install websockets")
    raise


# ── Broadcast server (mirrors real ConnectionManager) ──────────────────────

class BroadcastServer:
    def __init__(self):
        self.clients: set = set()
        self.broadcast_durations: list[float] = []

    async def handle(self, ws):
        self.clients.add(ws)
        try:
            async for _ in ws:
                pass
        except Exception:
            pass
        finally:
            self.clients.discard(ws)

    async def broadcast(self, message: dict) -> float:
        if not self.clients:
            return 0.0
        payload = json.dumps(message)
        t0 = time.perf_counter()
        dead = set()
        for c in list(self.clients):
            try:
                await c.send(payload)
            except Exception:
                dead.add(c)
        self.clients.difference_update(dead)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.broadcast_durations.append(elapsed_ms)
        return elapsed_ms

    async def tick_loop(self, n_entities: int, n_ticks: int):
        """Simulate the production tick loop: broadcast N entities every 1s."""
        for tick in range(n_ticks):
            ts_ms = time.time() * 1000
            batch = {
                "type": "entity_batch",
                "data": [{
                    "entity_id": f"e-{i:04d}",
                    "entity_type": "active",
                    "domain": "aerial",
                    "position": {"lat": 37.77 + i * 0.0001, "lon": -122.4, "alt": 120, "heading_deg": tick},
                    "speed_mps": 20,
                    "confidence": 0.95,
                    "last_seen": ts_ms / 1000,
                    "_server_ts_ms": ts_ms,
                } for i in range(n_entities)],
            }
            duration = await self.broadcast(batch)
            await asyncio.sleep(max(0, 1.0 - duration / 1000))


# ── Subscriber (console client) ────────────────────────────────────────────

async def subscriber(url: str, n_ticks: int, latencies: list[float], ready: asyncio.Event):
    async with websockets.connect(url) as ws:
        ready.set()
        received = 0
        while received < n_ticks:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                msg = json.loads(raw)
                if msg.get("type") == "entity_batch" and msg["data"]:
                    server_ts = msg["data"][0].get("_server_ts_ms")
                    if server_ts:
                        latencies.append(time.time() * 1000 - server_ts)
                        received += 1
            except asyncio.TimeoutError:
                break


# ── Scenario ───────────────────────────────────────────────────────────────

async def run_scenario(n_entities: int, n_subscribers: int = 3, n_ticks: int = 10) -> dict:
    port = 18766
    server = BroadcastServer()

    async with serve(server.handle, "localhost", port):
        await asyncio.sleep(0.1)
        url = f"ws://localhost:{port}"

        latencies: list[float] = []
        ready_events = [asyncio.Event() for _ in range(n_subscribers)]

        # Connect subscribers first, then start ticking
        sub_tasks = [
            asyncio.create_task(subscriber(url, n_ticks, latencies, ready_events[i]))
            for i in range(n_subscribers)
        ]
        # Wait for all subscribers to connect
        await asyncio.gather(*[e.wait() for e in ready_events])
        await asyncio.sleep(0.05)

        # Run tick loop
        await server.tick_loop(n_entities=n_entities, n_ticks=n_ticks)
        await asyncio.gather(*sub_tasks)

    bd = server.broadcast_durations
    if not bd or not latencies:
        return {"n_entities": n_entities, "error": "no data"}

    latencies.sort()
    n = len(latencies)
    return {
        "n_entities":         n_entities,
        "n_subscribers":      n_subscribers,
        "broadcast_p50_ms":   round(statistics.median(bd), 2),
        "broadcast_p95_ms":   round(sorted(bd)[int(len(bd) * 0.95)], 2),
        "broadcast_max_ms":   round(max(bd), 2),
        "client_p50_ms":      round(statistics.median(latencies), 1),
        "client_p95_ms":      round(latencies[int(n * 0.95)], 1),
        "client_max_ms":      round(max(latencies), 1),
        "ticks_received":     n,
    }


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    entity_counts    = [10, 50, 100, 250, 500]
    n_subscribers    = 3   # simulates 3 open console tabs
    n_ticks          = 10  # 10 seconds of ticks per scenario

    print(f"\nHeli.OS Broadcast Load Simulation")
    print(f"Config: {n_subscribers} subscribers, {n_ticks} ticks @ 1 Hz each")
    print(f"Comparison: Lattice claims 500 sensor nodes (latency unspecified)\n")
    print(f"{'Entities':>10}  {'Bcast P50':>10}  {'Bcast P95':>10}  {'Client P50':>11}  {'Client P95':>11}  {'Recv':>6}")
    print("-" * 72)

    results = []
    for n in entity_counts:
        r = await run_scenario(n_entities=n, n_subscribers=n_subscribers, n_ticks=n_ticks)
        results.append(r)
        if "error" in r:
            print(f"{n:>10}  ERROR: {r['error']}")
        else:
            print(f"{n:>10}  {r['broadcast_p50_ms']:>9.1f}ms  {r['broadcast_p95_ms']:>9.1f}ms  "
                  f"{r['client_p50_ms']:>10.1f}ms  {r['client_p95_ms']:>10.1f}ms  {r['ticks_received']:>6}")
        await asyncio.sleep(0.5)

    print("\nSummary:")
    ok = [r for r in results if "error" not in r and r["client_p95_ms"] < 100]
    ceiling = next((r["n_entities"] for r in results if "error" not in r and r["client_p95_ms"] >= 100), None)
    if ok:
        print(f"  Clean (<100ms P95 client latency) up to: {ok[-1]['n_entities']} entities/tick")
    if ceiling:
        print(f"  Latency crosses 100ms threshold at: {ceiling} entities/tick")
    else:
        print(f"  All scenarios under 100ms P95 — ceiling > {entity_counts[-1]} entities/tick")
    print(f"  (This is 1 Hz broadcast to {n_subscribers} subscribers — production pattern)\n")


if __name__ == "__main__":
    asyncio.run(main())
