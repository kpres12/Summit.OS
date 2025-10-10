# Summit.OS ↔ WildFire Ops — Single Integration Standard (v1.1)

You are in the **Summit.OS** repo. Do NOT change repo structure.
Enforce ONE interface only:

- **HTTP REST (v1)** at `/api/v1`
- **MQTT over WebSocket** for realtime topics
- Auth headers: `Authorization: Bearer <key>` OR `x-api-key: <key>`
- Timestamps: Unix seconds (float allowed)
- Additive changes only in v1; breaking changes require `/api/v2`

## 0) Non-Goals
❌ No gRPC, no ROS2 bindings, no extra SDKs, no cloud-specific integrations in this contract.  
❌ Do not generate multi-language client libs here. WildFire Ops consumes raw REST + MQTT.

## 1) REST Endpoints (MUST)
Base env for clients: `SUMMIT_API_URL` (e.g., `http://localhost:8000/api`)

- `GET  /api/v1/system/health` → `{ "status":"ok","uptime": <number> }`
- `GET  /api/v1/intelligence/alerts?since=<ts>` → `Alert[]`
- `POST /api/v1/alerts/{id}/ack` → `{ "ok": true, "id": "<id>" }`
- `GET  /api/v1/tasks/active` → `Task[]`
- `POST /api/v1/task/assign` (body: `TaskRequest`) → `Task`
- `POST /api/v1/predict/scenario` (body: `ScenarioRequest`) → **GeoJSON** FeatureCollection

### Canonical Schemas (JSON)
```jsonc
// Alert
{
  "id": "A-001",
  "ts": 1712345678.12,
  "severity": "LOW|MED|HIGH|CRITICAL",
  "title": "Potential ignition near ridge",
  "message": "2.1 km from fuel break, wind 20 mph → SW",
  "location": { "lat": 34.123, "lon": -117.456 },
  "context": { "risk": 0.87, "slope_deg": 15, "fuel_model": "SH5" },
  "acknowledged": false
}

// Task
{
  "id": "T-42",
  "asset": "UGV-Alpha",
  "kind": "PATROL|SURVEY_SMOKE|BUILD_LINE|SUPPRESS|RECON",
  "state": "QUEUED|ENROUTE|ACTIVE|PAUSED|DONE|FAILED",
  "eta_min": 6,
  "params": {}
}

// TaskRequest
{
  "kind": "SURVEY_SMOKE",
  "target": { "lat": 34.123, "lon": -117.456 },
  "params": { "priority": "HIGH" }
}

// Telemetry (MQTT payload too)
{
  "device_id": "UGV-Alpha",
  "ts": 1712345678.12,
  "lat": 34.1229,
  "lon": -117.4551,
  "batt": 91.0,
  "rssi": -60,
  "speed": 2.1,
  "sensors": { "temp_c": 31.2, "wind_mps": 8.9, "smoke_prob": 0.12 }
}

// ScenarioRequest (example)
{
  "aoi": { "type":"Polygon","coordinates":[[[-117.46,34.12],[-117.44,34.12],[-117.44,34.14],[-117.46,34.14],[-117.46,34.12]]] },
  "wind_dir_deg": 235,
  "wind_mps": 10,
  "rh_pct": 22,
  "fuel_model": "SH5",
  "lines": []
}
```

## 2) MQTT Topics (MUST)

Client env: SUMMIT_MQTT_URL (e.g., ws://localhost:1883)
- `alerts/#` → payload: Alert
- `devices/+/telemetry` → payload: Telemetry
- `missions/updates` → payload: Task (state changes)
- `fusion/events` (optional) → payload: free-form event JSON

QoS: 1 for telemetry; 2 for alerts/tasks.
Retain: optional on devices/{id}/telemetry.
LWT: devices/{id}/status → "offline".

## 3) CORS & Headers (dev)
- Allow origin http://localhost:3000
- Accept Authorization and x-api-key headers
- Always return application/json

## 4) OpenAPI
- Serve OpenAPI at /openapi.json and docs at /docs
- Keep models in one place (pydantic) to auto-generate schema

## 5) Acceptance (Contract Tests)

WildFire Ops must, against a vanilla Summit.OS dev instance:
- Health: GET /api/v1/system/health → "ok"
- Alerts list renders; POST /api/v1/alerts/{id}/ack flips acknowledged
- Active tasks render; POST /api/v1/task/assign adds a task and emits on missions/updates
- Subscribes to devices/+/telemetry and updates device views

## 6) Backward-Compat Rules
- v1 is additive only (new fields optional, defaulted).
- Any breaking change requires /api/v2 with parallel support window.

## 7) Dev ENV (reference)

Server:
```
HTTP_PORT=8000
SUMMIT_API_KEY=dev_key_placeholder
MQTT_WS_URL=ws://localhost:1883
```

Client (WildFire Ops):
```
SUMMIT_API_URL=http://localhost:8000/api
SUMMIT_MQTT_URL=ws://localhost:1883
SUMMIT_API_KEY=dev_key_placeholder
```