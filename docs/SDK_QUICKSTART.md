# Summit.OS SDK — Quickstart

## Install

```bash
# Core SDK (no HTTP dependencies — offline/test mode)
pip install -e packages/summit-sdk

# With HTTP transport (recommended)
pip install -e "packages/summit-sdk[http]"
```

## Connect

```python
import asyncio
from packages.summit_sdk import SummitClient

async def main():
    client = SummitClient(
        api_url="http://localhost:8000",
        api_key="your-api-key",
    )
    await client.connect()

    # Check health
    health = await client.health()
    print(health)  # {"status": "ok", "uptime": 1234}

    await client.disconnect()

asyncio.run(main())
```

## Entities

```python
# Create an entity
entity = await client.entities.create({
    "entity_type": "track",
    "domain": "AIR",
    "lat": 34.0522,
    "lon": -118.2437,
    "alt": 10000,
    "name": "UAL-1234",
})

# List entities by domain
air_tracks = await client.entities.list(domain="AIR", limit=50)

# Update
await client.entities.update(entity["entity_id"], {"speed": 250.0})

# Delete
await client.entities.delete(entity["entity_id"])

# Bulk upsert
await client.entities.bulk_upsert([
    {"entity_type": "track", "domain": "AIR", "lat": 34.0, "lon": -118.0},
    {"entity_type": "track", "domain": "GROUND", "lat": 34.1, "lon": -118.1},
])
```

## Tasks

```python
# Create a task
task = await client.tasks.create({
    "task_type": "navigate",
    "target_lat": 34.0,
    "target_lon": -118.0,
    "mission_id": "mission-alpha",
})

# Assign to an asset
await client.tasks.assign(task["task_id"], assignee_id="drone-01")

# Complete
await client.tasks.complete(task["task_id"], result={"arrived": True})

# Cancel
await client.tasks.cancel(task["task_id"])

# List active tasks
active = await client.tasks.list(state="running")
```

## Sensor Ingestion

```python
await client.sensors.ingest("radar-01", readings=[
    {"lat": 34.05, "lon": -118.24, "range_m": 1200, "bearing_deg": 45},
    {"lat": 34.06, "lon": -118.25, "range_m": 900, "bearing_deg": 90},
])
```

## Integration Client (External Feeds)

For connecting external data sources (ADS-B, radar, AIS):

```python
from packages.summit_sdk import IntegrationClient

feed = IntegrationClient(name="ADS-B West", source_type="adsb")
await feed.start()

# Push tracks from your data source
feed.push_track(
    lat=34.0522, lon=-118.2437, alt=35000,
    callsign="UAL1234", speed=450, heading=90,
    domain="AIR", classification="CIVILIAN_AIRCRAFT",
)

# Flush buffered tracks to Summit.OS
await feed.flush()
await feed.stop()
```

## Error Handling

All errors inherit from `SummitError` with machine-readable codes:

```python
from packages.summit_sdk import SummitClient, NotFoundError, AuthError, RateLimitError

try:
    entity = await client.entities.get("nonexistent-id")
except NotFoundError as e:
    print(e.code)     # SUMMIT-E3001
    print(e.message)  # "Entity 'nonexistent-id' not found"
except AuthError:
    print("Invalid credentials")
except RateLimitError as e:
    print(f"Retry after {e.details['retry_after_s']}s")
```

## Retry & Circuit Breaker

Customize resilience behavior:

```python
from packages.summit_sdk import SummitClient, RetryPolicy, CircuitBreaker

client = SummitClient(
    api_url="http://localhost:8000",
    retry_policy=RetryPolicy(
        max_retries=5,
        base_delay=1.0,
        max_delay=60.0,
        backoff_factor=2.0,
    ),
    circuit_breaker=CircuitBreaker(
        failure_threshold=10,
        recovery_timeout=60.0,
    ),
)
```

## Mesh & Status

```python
# Mesh network status
mesh_status = await client.mesh.status()

# List connected peers
peers = await client.mesh.peers()
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SUMMIT_API_URL` | `http://localhost:8000` | API gateway URL |
| `SUMMIT_API_KEY` | (none) | API key for authentication |
| `SUMMIT_WS_URL` | (derived from API URL) | WebSocket URL |

## Next Steps

- [API Reference](./API.md) — Full REST API contract
- [Architecture](./ARCHITECTURE.md) — System design overview
- [Changelog](../CHANGELOG.md) — Release notes
