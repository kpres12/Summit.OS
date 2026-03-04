# Summit.OS Integration Guide

Connect any device to Summit.OS in under 30 minutes.

## Quick Start

```bash
# 1. Install the SDK
pip install paho-mqtt requests

# 2. Copy the quickstart template
cp examples/quickstart_adapter.py my_adapter.py

# 3. Implement two methods: get_telemetry() and handle_command()
#    See examples/quickstart_adapter.py for the full template

# 4. Run
python my_adapter.py
```

Your device will automatically:
- Register with the Summit.OS gateway
- Appear in the shared world model
- Stream telemetry to the Common Operating Picture
- Accept commands from operators

## Architecture

```
Your Device                    Summit.OS
┌──────────────┐              ┌─────────────────────────────┐
│              │  MQTT        │  MQTT Broker (1883)         │
│  Adapter     │──telemetry──▶│    ↓                        │
│  (your code) │◀──commands───│  Fabric (8001)              │
│              │  heartbeat──▶│    → WorldStore              │
└──────────────┘              │    → Entity CRDT (mesh)     │
       │                      │                             │
       │  HTTP (register)     │  API Gateway (8000)         │
       └─────────────────────▶│    → Node registry          │
                              │                             │
                              │  Console (3000)             │
                              │    ← WebSocket entity stream│
                              └─────────────────────────────┘
```

## The Adapter Pattern

Every device integration subclasses `SummitAdapter`:

```python
from summit_os.adapter import SummitAdapter

class MyDrone(SummitAdapter):
    async def get_telemetry(self) -> dict:
        """Called every 5 seconds. Return device state."""
        return {
            "lat": 37.7749,       # Required
            "lon": -122.4194,     # Required
            "alt": 50.0,          # Required
            "battery": 85.0,      # Recommended
            "status": "ACTIVE",   # Recommended
            "sensors": {},        # Optional
        }

    async def handle_command(self, cmd: str, params: dict) -> bool:
        """Called when the platform sends a command."""
        if cmd == "goto":
            await self.vehicle.goto(params["lat"], params["lon"], params["alt"])
            return True
        return False
```

## Entity Protocol

Every device becomes an **Entity** in the world model:

| Field | Source | Description |
|-------|--------|-------------|
| `entity_id` | `device_id` | Unique device identifier |
| `entity_type` | Always `ASSET` | Type in the world model |
| `domain` | `device_type` | AERIAL, GROUND, MARITIME, etc. |
| `state` | `status` from telemetry | ACTIVE, IDLE, OFFLINE |
| `kinematics` | `lat/lon/alt` from telemetry | Position + velocity |
| `metadata` | Capabilities + extra fields | Device-specific data |

Entities are automatically:
- Stored in the WorldStore (in-memory + Postgres)
- Replicated across mesh nodes via CRDT
- Queryable via REST API and WebSocket
- Visible on the operator console map

## MQTT Topics

Your adapter automatically uses these topics:

**Publishing (device → platform):**
- `health/{device_id}/heartbeat` — Heartbeat (every 30s)
- `devices/{device_id}/telemetry` — Raw telemetry (every 5s)
- `entities/{device_id}/update` — Entity update for WorldStore
- `alerts/{device_id}` — Alerts (via `publish_alert()`)

**Subscribing (platform → device):**
- `tasks/{device_id}/#` — Mission task dispatches
- `control/{device_id}/#` — Direct control commands
- `commands/{device_id}` — Generic commands

## Commands

Commands arrive as JSON on the subscribed topics:

```json
{
  "action": "goto",
  "params": {"lat": 37.78, "lon": -122.42, "alt": 60, "speed": 5.0},
  "task_id": "mission:abc-123",
  "ts_iso": "2025-01-01T00:00:00Z"
}
```

Standard commands your adapter should handle:

| Command | Params | Description |
|---------|--------|-------------|
| `goto` | `{lat, lon, alt, speed}` | Navigate to position |
| `rtl` | `{}` | Return to launch |
| `land` | `{}` | Land immediately |
| `hold` | `{}` | Hold current position |
| `set_mode` | `{mode: str}` | Change operating mode |
| `MISSION_EXECUTE` | `{waypoints: [...]}` | Execute a mission plan |

## Registration

On startup, the adapter calls `POST /api/v1/nodes/register`:

```json
{
  "id": "drone-01",
  "type": "DRONE",
  "capabilities": ["thermal", "rgb_camera"],
  "comm": ["mqtt"]
}
```

Response includes:
- MQTT topic ACLs
- Policy constraints
- JWT token for authenticated operations

## Conformance Testing

Validate your adapter before deployment:

```bash
python -m summit_os.conformance --adapter my_package.MyDrone --device-id test-01
```

This runs 5 tests:
1. **Heartbeat** — Publishes valid heartbeats
2. **Entity Telemetry** — Returns {lat, lon, alt} from get_telemetry()
3. **Registration** — Has device_id, device_type, capabilities
4. **Disconnect/Reconnect** — Has lifecycle hooks
5. **Command Handling** — handle_command() doesn't crash

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MQTT_HOST` | `localhost` | MQTT broker hostname |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `MQTT_USERNAME` | — | MQTT auth username |
| `MQTT_PASSWORD` | — | MQTT auth password |
| `SUMMIT_API_URL` | `http://localhost:8000` | API gateway URL |

## Examples

See `examples/quickstart_adapter.py` for a complete working example.

For real hardware integrations:
- **MAVLink drone**: Use `pymavlink` in `get_telemetry()` and `handle_command()`
- **ROS2 robot**: Use `rclpy` subscriptions for telemetry, action clients for commands
- **IP camera**: Use RTSP/HTTP for frames, publish detections as alerts
- **Serial sensor**: Use `pyserial` to read data, map to telemetry dict
