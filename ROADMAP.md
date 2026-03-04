# Summit.OS Roadmap

## What's Built (v0.1)

**Platform Core**
- WorldStore: unified entity store with CRDT-based mesh replication
- Entity protocol: typed entities (Asset, Track, Alert, Mission, Sensor, etc.) with provenance and relationships
- Data Fabric: MQTT ingestion, Redis streams, WebSocket broadcast to console
- API Gateway: proxied REST for all services, OPA policy checks, OIDC-ready auth
- Operator Console: MapLibre live map, real-time entity stream, layer controls, geofence editor

**Services**
- Fusion: perception pipeline stub, geospatial observation store
- Intelligence: reasoning/advisory service stub
- Tasking: mission planning with state machine, assignment engine
- Inference: ONNX model serving endpoint (YOLOv8n reference model)

**Infrastructure**
- Docker Compose local stack (Redis, Postgres+PostGIS, MQTT, Prometheus, Grafana)
- CI pipeline (lint, test, typecheck, integration)
- SDK with optional extras (mqtt, websocket, ai)

## In Progress

- End-to-end smoke test validation
- Fusion → Inference → WorldStore pipeline (camera frame → detections → map)
- Demo detection script

## Next Up

**Platform Improvements**
- WebSocket authentication (token-based)
- Entity TTL garbage collection background task
- WorldStore persistence benchmarks and tuning
- Geofence enforcement in assignment engine (pre-dispatch containment check)
- Track correlation and deduplication (multi-sensor fusion)

**Console**
- Geofence drawing on MapLibre (click-to-draw polygons)
- Entity history trail (polyline of past positions)
- Alert notification panel with severity-based sorting
- Mission timeline view
- Mobile-responsive layout

**AI / ML**
- Additional ONNX reference models (classification, segmentation)
- Model registry with versioning
- Edge inference adapter (ONNX Runtime on ARM/Jetson)
- Anomaly detection pipeline (time-series sensor data)
- LLM-powered operator assistant (natural language → mission commands)

**Integrations**
- ROS 2 adapter (nav2, sensor_msgs bridge)
- MAVLink adapter (ArduPilot/PX4 via pymavlink)
- ADS-B receiver integration
- AIS maritime data feed
- Weather data overlay (NOAA/OpenWeather)

## Where Contributors Can Help

| Area | Difficulty | Description |
|------|-----------|-------------|
| Console UI | Easy | Add entity history trails, improve mobile layout |
| Documentation | Easy | API docs, architecture diagrams, tutorials |
| Adapters | Medium | ROS 2, MAVLink, ADS-B, AIS integrations |
| Fusion | Medium | Multi-sensor track correlation algorithms |
| AI Models | Medium | Train/ship additional ONNX reference models |
| Edge Runtime | Hard | ONNX inference on ARM/Jetson with model sync |
| CRDT Mesh | Hard | Cross-datacenter entity replication testing |

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and contribution guidelines.
