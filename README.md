# Summit.OS

**Connect any signal. Build any mission.**

Summit.OS is an open-source autonomous systems coordination platform — the integration and coordination layer between your existing sensors, cameras, drones, and assets and the missions you need to run.

You bring your hardware. Summit.OS connects it, fuses the signals into a live world model, and gives your operators one unified interface to understand what's happening and act on it. It works with DJI drones, ONVIF cameras, MAVLink autopilots, ADS-B receivers, weather stations, IoT sensors, and anything else you can write an adapter for.

**Built for:** wildfire response, search & rescue, disaster coordination, commercial UAV operations, maritime safety, critical infrastructure monitoring — any domain where humans need to coordinate autonomous systems in the physical world.

Summit.OS does not build hardware. It makes your hardware work together.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Operator Console (3000)                     │
│              MapLibre map + entity stream + mission feed        │
├─────────────────────────────────────────────────────────────────┤
│                      API Gateway (8000)                         │
│               Routes to backend services, OIDC/RBAC             │
├──────────┬──────────┬──────────────┬────────────┬───────────────┤
│  Fabric  │  Fusion  │ Intelligence │  Tasking   │  Inference    │
│  (8001)  │  (8002)  │   (8003)     │  (8004)    │   (8006)      │
│ WorldStore│ Tracking │  Reasoning   │  Missions  │  Detection    │
│ MQTT/gRPC│ Kalman   │  Advisory    │  State Mach│  YOLOv8/ONNX  │
│ Mesh CRDT│ Correlate│  Anomaly     │  Assignment│  Hot-swap     │
├──────────┴──────────┴──────────────┴────────────┴───────────────┤
│                    Infrastructure                               │
│       Redis · Postgres+PostGIS · MQTT · Prometheus · Grafana    │
└─────────────────────────────────────────────────────────────────┘
```

## Core Concepts

**Everything is an Entity.** Drones, ground robots, cameras, tracks, alerts, missions, geofences — they all share one data model (`packages/entities/core.py`) and live in one WorldStore.

**One World Model.** The WorldStore (`packages/world/store.py`) is the single source of truth. In-memory cache + Postgres persistence + MQTT/WebSocket broadcast. Every service reads from and writes to it.

**Mesh Replication.** CRDT-based entity replication across nodes (`packages/mesh/`). Nodes can disconnect, make conflicting updates, and merge correctly on reconnect.

**Intent-Based Tasking.** Describe what you want done, not how. The assignment engine scores available assets by capability, proximity, battery, and availability, then dispatches through a formal state machine with policy enforcement.

**Plugin AI.** The inference runtime (`apps/inference/`) loads any ONNX model. Ships with YOLOv8n for general object detection. Swap in your own domain-specific model at runtime.

**30-Minute Integration.** Subclass `SummitAdapter`, implement two methods (`get_telemetry`, `handle_command`), and your device appears in the world model. See `INTEGRATION_GUIDE.md`.

## Quick Start

### No Docker (fastest — works on any machine)

```bash
git clone https://github.com/summit-os/summit-os.git
cd summit-os

# Install mock server deps (once)
pip install -r requirements_mock.txt

# Terminal 1: start the mock API + live ADS-B data
make mock

# Terminal 2: start the console
make dev-console

# Open http://localhost:3000
# Live aircraft from OpenSky Network appear on the map immediately.
```

### Full Stack (Docker required)

```bash
# Start everything (infrastructure + all services + console)
make dev

# Verify the stack is healthy
make health

# Run the end-to-end smoke test
make smoke
```

```
# Console:     http://localhost:3000
# API Gateway: http://localhost:8000
# Grafana:     http://localhost:3001
```

## Monorepo Structure

```
apps/
  console/          # Operator console (Next.js + MapLibre)
  fabric/           # Data fabric — WorldStore, MQTT, mesh, WebSocket
  fusion/           # Sensor fusion — tracking, correlation, Kalman filters
  intelligence/     # AI reasoning — advisory, anomaly detection
  tasking/          # Mission planning — state machine, assignment engine, OPA
  inference/        # AI inference runtime — ONNX model serving
  api-gateway/      # Routes requests to backend services

packages/
  entities/         # Core Entity dataclass — the universal data type
  world/            # WorldStore + Entity CRUD API
  mesh/             # CRDT replication, mesh peer, transport
  ai/               # Detection, classification, anomaly, edge inference
  hal/              # Hardware abstraction — MAVLink, ONVIF, SITL drivers
  security/         # mTLS, RBAC, auth, data classification
  autonomy/         # Behavior trees, rules engine
  schemas/          # Shared Pydantic schemas
  observability/    # Metrics, tracing
  summit-os-sdk/    # Integration SDK — adapter base class, conformance tests

models/             # ONNX model files (YOLOv8n reference model)
infra/docker/       # Docker Compose, Mosquitto config, Prometheus config
scripts/            # Demo scripts, smoke tests, mock data generators
tests/              # Integration and E2E tests
examples/           # Quickstart adapter template
```

## Services

| Service | Port | Purpose |
|---------|------|---------|
| Console | 3000 | Operator UI — map, entity list, mission feed, command bar |
| API Gateway | 8000 | Unified REST API, request routing, OIDC enforcement |
| Fabric | 8001 | WorldStore, MQTT bridge, mesh peer, WebSocket streaming |
| Fusion | 8002 | Sensor fusion, multi-target tracking, correlation |
| Intelligence | 8003 | AI reasoning, anomaly detection, advisory generation |
| Tasking | 8004 | Mission lifecycle, assignment engine, OPA policy |
| Inference | 8006 | ONNX model serving, object detection, model hot-swap |

## Development

```bash
# Infrastructure only (Redis, Postgres, MQTT, Prometheus, Grafana)
make dev-services

# Backend Python services only
make dev-backend

# Console only (local Next.js dev server)
make dev-console

# Run all tests
make test

# Lint and format
make lint
make format

# Stream logs
make logs

# Clean up containers and volumes
make clean
```

## Integration

See `INTEGRATION_GUIDE.md` for the full walkthrough. The short version:

```python
from summit_os.adapter import SummitAdapter

class MyDrone(SummitAdapter):
    ENTITY_TYPE = "ASSET"
    DOMAIN = "AERIAL"
    CAPABILITIES = ["camera", "thermal"]

    async def get_telemetry(self):
        return {"lat": 34.05, "lon": -118.24, "alt": 100, "battery_pct": 85}

    async def handle_command(self, command):
        if command["action"] == "goto":
            self.fly_to(command["lat"], command["lon"])

adapter = MyDrone(entity_id="drone-001", gateway_url="http://localhost:8000")
adapter.run()
```

Run conformance tests against your adapter:
```bash
summit-os-conformance --adapter my_adapter:MyDrone --gateway http://localhost:8000
```

## Contributing

See `CONTRIBUTING.md` for setup instructions, branching conventions, and PR process.

## License

Apache 2.0 — see [LICENSE](LICENSE).
