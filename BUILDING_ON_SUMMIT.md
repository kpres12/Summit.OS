# Building on Summit.OS

Summit.OS is an **operational kernel** for autonomous systems — not a finished product. It provides the sensor fusion, entity tracking, mission orchestration, and command-and-control primitives. You build the product on top.

This guide covers:
1. [Creating a custom domain](#1-creating-a-custom-domain)
2. [Building a custom console](#2-building-a-custom-console)
3. [Connecting hardware with the Adapter SDK](#3-connecting-hardware)
4. [Extending the intelligence layer](#4-extending-intelligence)
5. [Deploying your product](#5-deploying-your-product)

---

## 1. Creating a Custom Domain

A **domain** defines how Summit.OS looks and speaks for a specific use case — wildfire, pipeline monitoring, construction safety, maritime, agriculture, or anything you need.

### Zero code required

Drop a JSON file into the reference console's `public/domains/` directory:

```json
{
  "id": "agriculture",
  "name": "Crop Monitoring",
  "description": "Precision agriculture and crop health monitoring",
  "palette": {
    "accent": "#7CB342",
    "accentDim": "#558B2F",
    "accentDark": "#33691E",
    "warning": "#FFB300",
    "critical": "#FF3B3B",
    "nominal": "#4AEDC4",
    "active": "#4FC3F7",
    "backgroundTint": "#0A0C08",
    "panelBg": "#10120D",
    "border": "rgba(124, 179, 66, 0.15)",
    "scanline": "rgba(124, 179, 66, 0.02)"
  },
  "entityLabels": {
    "drone": { "displayName": "Survey Drone", "icon": "●", "color": "#7CB342" },
    "anomaly": { "displayName": "Crop Anomaly", "icon": "▲", "color": "#FFB300" },
    "friendly": { "displayName": "Asset", "icon": "●", "color": "#7CB342" },
    "alert": { "displayName": "Issue", "icon": "▲", "color": "#FF3B3B" }
  },
  "assetTypes": [
    { "type": "DRONE", "label": "SURVEY UAV", "icon": "○" },
    { "type": "SENSOR", "label": "SOIL SENSOR", "icon": "○" }
  ],
  "mapLayers": [
    { "id": "entities", "name": "Assets", "enabled": true, "color": "#7CB342", "icon": "●" },
    { "id": "fields", "name": "Field Boundaries", "enabled": true, "color": "#FFB300", "icon": "⬢" }
  ],
  "commandExamples": [
    "survey field north-40",
    "status all drones",
    "ndvi report section 12"
  ],
  "alertTypes": [
    { "id": "pest", "label": "Pest Detection", "icon": "▲", "color": "#FFB300" },
    { "id": "irrigation", "label": "Irrigation Fault", "icon": "⊘", "color": "#FF3B3B" }
  ],
  "missionTemplates": [
    { "id": "survey_field", "label": "Survey Field", "intent": "survey", "description": "Multispectral survey of crop field" },
    { "id": "inspect_anomaly", "label": "Inspect Anomaly", "intent": "verify", "description": "Close inspection of detected crop anomaly" }
  ],
  "terminology": {
    "mission": "Survey",
    "asset": "Equipment",
    "alert": "Detection",
    "entity": "Asset",
    "operatorView": "Agronomist",
    "supervisorView": "Farm Manager"
  }
}
```

Then register it in `public/domains/index.json`:

```json
[
  { "id": "default", "name": "Summit.OS", "file": "default.json" },
  { "id": "agriculture", "name": "Crop Monitoring", "file": "agriculture.json" }
]
```

The console picks it up on next load. No rebuild, no code changes.

### What the domain config controls

| Field | What it drives |
|---|---|
| `palette` | Every color in the UI — accent, backgrounds, borders, scanline tint |
| `entityLabels` | How entities are named and colored on the map and in lists |
| `assetTypes` | What hardware types appear in the sidebar |
| `mapLayers` | Default map layer toggles |
| `commandExamples` | Rotating placeholder in the command bar |
| `alertTypes` | Alert classification chips |
| `missionTemplates` | Quick-action mission presets |
| `terminology` | UI vocabulary — "Mission" becomes "Survey", "Alert" becomes "Detection" |

### Schema

The full TypeScript type is in `apps/console/lib/domains/types.ts`. JSON files must match this shape exactly. The five bundled domains (`default`, `fire`, `pipeline`, `sar`, `construction`) serve as examples.

---

## 2. Building a Custom Console

The reference console in `apps/console/` is a **starting point, not the product**. It demonstrates the API surface and provides a working operator UI.

### Option A: Fork and customize the reference console

Good for: teams that want a working UI fast and will iterate on it.

```bash
cp -r apps/console my-console/
cd my-console
# Edit components, add your domain, remove what you don't need
npm run dev
```

### Option B: Build from scratch against the API

Good for: teams with specific UX requirements, different frameworks, or mobile targets.

Summit.OS exposes everything through two interfaces:

**REST API** (documented at `/docs` when running):
```
GET  /entities          # All tracked entities
GET  /entities/{id}     # Single entity
GET  /alerts            # Alert queue
POST /alerts/{id}/ack   # Acknowledge alert
GET  /missions          # Mission list
POST /missions          # Create mission
POST /agents            # Dispatch AI agent
GET  /adapters          # Connected hardware
POST /adapters          # Register new adapter
GET  /reasoning/{id}    # AI reasoning for entity
```

**WebSocket** (`ws://localhost:8001/ws`):
```json
{ "type": "entity_update", "data": { "entity_id": "...", "position": {...}, ... } }
{ "type": "alert", "data": { "alert_id": "...", "severity": "CRITICAL", ... } }
{ "type": "mission_update", "data": { "mission_id": "...", "status": "ACTIVE", ... } }
```

Build your console in React, Vue, Svelte, Flutter, Swift — anything that can speak HTTP and WebSocket.

### Option C: Embed in an existing product

If you already have a product and want to add autonomous coordination, use Summit.OS as a headless backend. Your existing UI calls the REST API; the WebSocket pushes real-time updates. No console needed.

---

## 3. Connecting Hardware

The Adapter SDK is the primary integration surface. Any device that can run Python (Raspberry Pi, Jetson, laptop, server) can become a Summit.OS adapter.

### Quick start

```bash
pip install summit-os-sdk
```

```python
from sdk import BaseAdapter, AdapterManifest, EntityBuilder, Protocol, Capability

class MyDrone(BaseAdapter):
    MANIFEST = AdapterManifest(
        name="my-drone",
        version="1.0.0",
        protocol=Protocol.MAVLINK,
        capabilities=[Capability.READ, Capability.COMMAND],
        entity_types=["UAV"],
        description="Custom drone adapter",
    )

    async def run(self):
        while not self.stopped:
            telemetry = await self.read_telemetry()
            entity = (
                EntityBuilder(self.device_id, "MyDrone")
                .asset()
                .aerial()
                .position(telemetry.lat, telemetry.lon, telemetry.alt)
                .speed(telemetry.speed)
                .heading(telemetry.heading)
                .battery(telemetry.battery_pct)
                .build()
            )
            self.publish(entity)
            await self.sleep(1)

    async def handle_command(self, cmd):
        if cmd.action == "RTB":
            await self.return_to_base()
```

### Built-in adapters

Summit.OS ships adapters for common protocols in `adapters/`:

| Adapter | Protocol | Use case |
|---|---|---|
| `mavlink` | MAVLink UDP | Drones (ArduPilot, PX4) |
| `modbus` | Modbus/TCP | PLCs, industrial sensors |
| `opcua` | OPC-UA | SCADA systems |
| `opensky` | ADS-B (OpenSky) | Aircraft tracking |
| `celestrak` | TLE/SGP4 | Satellite tracking |
| `rtsp` | RTSP → HLS | IP cameras, drone gimbals |
| `atak` | CoT UDP | ATAK/TAK interop |
| `ais` | AIS NMEA | Maritime vessel tracking |

### Entity data model

Every adapter publishes entities to the same schema:

```json
{
  "entity_id": "uuid",
  "callsign": "DRONE-01",
  "entity_type": "active",
  "domain": "aerial",
  "classification": "UAV",
  "position": { "lat": 34.12, "lon": -118.34, "alt": 120.0, "heading_deg": 45.0 },
  "speed_mps": 12.5,
  "confidence": 0.95,
  "battery_pct": 78,
  "source_sensors": ["mavlink-01"],
  "last_seen": 1711700000
}
```

Fusion handles correlation, de-duplication, and track management automatically.

---

## 4. Extending Intelligence

### Custom mission types

The mission classifier (`packages/ml/models/mission_classifier.onnx`) maps detections to mission types. To add new types:

1. Add labeled examples to your training data
2. Retrain:
   ```bash
   python packages/ml/train_mission_classifier.py --real-data your_data.csv
   ```
3. The new ONNX model is hot-swapped at runtime

### Custom rules

The intelligence service uses OPA (Open Policy Agent) for policy decisions. Add Rego rules in `infra/opa/` to encode domain-specific logic:

```rego
# Auto-dispatch survey drone for any detection with confidence > 0.8
dispatch_mission {
    input.confidence > 0.8
    input.classification == "smoke"
}
```

### Risk scoring

The risk scorer (`packages/ml/models/risk_scorer.onnx`) rates observations from LOW to CRITICAL. Retrain it on your own escalation history to calibrate for your context.

---

## 5. Deploying Your Product

### Development

```bash
cd infra/docker && docker compose up
```

### Production

See `docs/DEPLOYMENT_GUIDE.md` for:
- Kubernetes deployment (`infra/k8s/`)
- TLS/mTLS configuration
- OIDC provider setup (Keycloak, Auth0, Okta)
- PostgreSQL + PostGIS production config
- MQTT broker clustering
- Monitoring stack (Prometheus + Grafana + Jaeger)
- Secrets management (Infisical or HashiCorp Vault)

### Licensing

Summit.OS is AGPL v3. This means:
- **Free to use, modify, and deploy** — even commercially
- **If you modify the server code and offer it as a service**, you must publish your modifications
- **Adapters you write using the SDK are yours** — the SDK is a separate module, your adapter code is not a derivative work
- **Your domain JSON configs are yours** — data files are not covered by AGPL
- **Your custom console built against the API is yours** — client-side code communicating via HTTP/WebSocket is not a derivative work

For commercial licensing without AGPL obligations, contact the maintainers.

---

## Architecture for Platform Builders

```
Your Product
├── your-console/          ← Your custom UI (any framework)
├── your-adapters/         ← Your hardware integrations (SDK)
├── your-domains/          ← Your domain JSON configs
├── your-models/           ← Your retrained ML models
└── your-policies/         ← Your OPA rules
    │
    ▼
Summit.OS (the kernel)
├── API Gateway            ← Auth, routing, rate limiting
├── Fabric                 ← Entity persistence, MQTT bridge, WebSocket
├── Fusion                 ← Multi-sensor track fusion
├── Intelligence           ← AI dispatch, risk scoring
├── Tasking                ← Mission state machine
└── Inference              ← ONNX model serving
```

You own everything above the line. Summit.OS owns everything below it. Updates to the kernel don't break your product. Your product doesn't require forking the kernel.
