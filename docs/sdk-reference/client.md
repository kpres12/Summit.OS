# SDK Reference — `summit-sdk`

Version: `0.1.0-alpha`

## `SummitClient`

Main entry point for the SDK.

### Constructor

```python
SummitClient(
    api_url: str = "http://localhost:8000",
    ws_url: str = "",
    api_key: str = "",
    jwt_token: str = "",
    retry_policy: RetryPolicy | None = None,
    circuit_breaker: CircuitBreaker | None = None,
)
```

**Parameters:**
- `api_url` — Base URL of the Heli.OS API gateway.
- `ws_url` — WebSocket URL. Auto-derived from `api_url` if empty.
- `api_key` — API key sent as `X-API-Key` header.
- `jwt_token` — JWT sent as `Authorization: Bearer` header.
- `retry_policy` — Custom retry configuration. Defaults to 3 retries, 0.5s base delay.
- `circuit_breaker` — Custom circuit breaker. Defaults to 5-failure threshold, 30s recovery.

### Methods

| Method | Returns | Description |
|---|---|---|
| `await connect()` | `bool` | Open HTTP session to API |
| `await disconnect()` | `None` | Close session |
| `await health()` | `dict` | `GET /health` |
| `is_connected` | `bool` | Connection status (property) |

### Sub-clients

- `client.entities` → `EntityClient`
- `client.tasks` → `TaskClient`
- `client.mesh` → `MeshClient`
- `client.sensors` → `SensorClient`

---

## `EntityClient`

Accessed via `client.entities`.

| Method | Returns | Description |
|---|---|---|
| `await get(entity_id)` | `dict` | Fetch single entity |
| `await list(domain?, entity_type?, limit=100)` | `list[dict]` | List entities with optional filters |
| `await create(entity_data)` | `dict` | Create new entity |
| `await update(entity_id, updates)` | `dict` | Partial update |
| `await delete(entity_id)` | `dict` | Delete entity |
| `await bulk_upsert(entities)` | `dict` | Batch create/update |

---

## `TaskClient`

Accessed via `client.tasks`.

| Method | Returns | Description |
|---|---|---|
| `await get(task_id)` | `dict` | Fetch single task |
| `await list(state?, mission_id?)` | `list[dict]` | List tasks with optional filters |
| `await create(task_data)` | `dict` | Create new task |
| `await assign(task_id, assignee_id)` | `dict` | Assign task to asset |
| `await complete(task_id, result?)` | `dict` | Mark task complete |
| `await cancel(task_id)` | `dict` | Cancel task |

---

## `MeshClient`

Accessed via `client.mesh`.

| Method | Returns | Description |
|---|---|---|
| `await status()` | `dict` | Mesh network status |
| `await peers()` | `list[dict]` | Connected peer nodes |

---

## `SensorClient`

Accessed via `client.sensors`.

| Method | Returns | Description |
|---|---|---|
| `await ingest(sensor_id, readings)` | `dict` | Push sensor data |
| `await list()` | `list[dict]` | List registered sensors |

---

## `IntegrationClient`

For connecting external data feeds.

### Constructor

```python
IntegrationClient(
    name: str = "External Integration",
    source_type: str = "custom",    # "adsb", "radar", "ais", "sigint", "custom"
    api_url: str = "http://localhost:8000",
    api_key: str = "",
)
```

### Methods

| Method | Returns | Description |
|---|---|---|
| `await start()` | `None` | Start integration |
| `await stop()` | `None` | Flush + stop |
| `push_track(lat, lon, alt, ...)` | `None` | Buffer a track observation |
| `await flush()` | `int` | Flush buffer to API, returns count sent |
| `buffer_size` | `int` | Current buffer size (property) |
| `stats` | `dict` | Ingestion statistics (property) |

---

## Error Hierarchy

All errors inherit from `SummitError`.

| Exception | Code | HTTP Status | When |
|---|---|---|---|
| `ConnectionError` | `SUMMIT-E1001` | — | Can't reach API |
| `TimeoutError` | `SUMMIT-E1002` | 408 | Request timed out |
| `CircuitOpenError` | `SUMMIT-E1003` | — | Circuit breaker is open |
| `AuthError` | `SUMMIT-E2001/2002` | 401/403 | Invalid credentials |
| `NotFoundError` | `SUMMIT-E3001` | 404 | Resource doesn't exist |
| `ValidationError` | `SUMMIT-E3002` | 422 | Bad request body |
| `RateLimitError` | `SUMMIT-E3004` | 429 | Too many requests |
| `ServerError` | `SUMMIT-E4001/4002` | 500/503 | Server-side failure |
| `NotConnectedError` | `SUMMIT-E5001` | — | `connect()` not called |

### `SummitError` attributes

- `.message: str` — Human-readable message
- `.code: ErrorCode` — Machine-readable enum
- `.status: int` — HTTP status (0 if not HTTP-related)
- `.details: dict` — Extra context
- `.to_dict() → dict` — Serialize for logging

---

## `RetryPolicy`

```python
RetryPolicy(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.25,
    retryable_status: set[int] = {408, 429, 500, 502, 503, 504},
)
```

Retries use exponential backoff: `delay = base_delay × backoff_factor^attempt ± jitter`.
For 429 responses, the SDK respects the server's `retry-after` if longer than the calculated delay.

---

## `CircuitBreaker`

```python
CircuitBreaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    window_s: float = 60.0,
)
```

States: `CLOSED` → `OPEN` (after threshold failures) → `HALF_OPEN` (probe after recovery timeout) → `CLOSED` (on success).

Methods:
- `.state` — Current `CircuitState`
- `.reset()` — Manual reset to `CLOSED`
