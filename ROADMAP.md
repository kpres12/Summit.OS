# Summit.OS Roadmap

## What's Built (v0.2)

**Platform Core**
- WorldStore: unified entity store with CRDT-based mesh replication
- Entity protocol: typed entities (Asset, Track, Alert, Mission, Sensor) with provenance and relationships
- Data Fabric: MQTT ingestion, Redis streams, WebSocket broadcast to console
- API Gateway: versioned REST (`X-Summit-API-Version`), OPA policy enforcement, OIDC-ready auth
- Operator Console: MapLibre live map, real-time entity stream, alert queue, entity detail panel, layer controls, geofence editor

**Adapter SDK**
- `BaseAdapter` — abstract base class for all hardware adapters (MQTT self-managed, lifecycle, manifest)
- `AdapterManifest` — capability declaration, approval gating for write-capable adapters
- `EntityBuilder` — fluent builder producing schema-consistent entities with threshold-aware state
- `AdapterPublisher` — retained manifest publishing, QoS-aware entity publishing

**Adapters (all with simulation fallback — no hardware required)**
- OpenSky Network (ADS-B live aircraft positions)
- CelesTrak (satellite tracking via SGP4/TLE)
- Modbus/TCP (PLCs, pumps, pressure sensors, industrial automation)
- OPC-UA (Siemens, GE, Honeywell — modern industrial systems)
- MAVLink (ArduPilot/PX4 drones)

**OPA Safety Policies**
- Geofence enforcement (altitude limits, exclusion zones, inclusion zones)
- Actuator safety pre-flight checks (human approval gates, pressure/temperature interlocks)
- Signed policy files (Ed25519) with startup verification — tamper-evident

**Security & Hardening**
- MQTT rate limiting: token bucket per source, configurable burst, Prometheus metrics
- Per-device X.509 identity: DeviceCA issues certificates, DeviceRegistry maps device → fingerprint
- Vault-backed secrets with env var fallback (zero code change between dev and prod)
- API versioning middleware on all responses

**Testing**
- Unit tests: SDK (AdapterManifest + EntityBuilder), rate limiter, policy signer, secrets client, device identity, all 5 adapters (simulation mode — no hardware)
- E2E tests: API version endpoint, device register/revoke round-trip
- CI pipeline: lint, security scan (bandit), unit tests, adapter simulation tests, integration tests

**Infrastructure**
- Docker Compose local stack (Redis, Postgres+PostGIS, MQTT, Prometheus, Grafana)
- Adapter runner with dynamic loading and per-adapter manifest validation

---

## In Progress

- WebSocket authentication (token-based, ties into device identity)
- Entity TTL garbage collection background task
- Geofence enforcement in assignment engine (pre-dispatch containment check)

---

## Next Up

**Platform**
- Track correlation and deduplication (multi-sensor fusion)
- WorldStore persistence benchmarks and tuning
- Mission timeline view in console
- Entity history trail (polyline of past positions on map)
- Mobile-responsive console layout

**Adapters**
- ROS 2 adapter (nav2, sensor_msgs bridge)
- AIS maritime data feed
- Weather data overlay (NOAA/OpenWeather)
- ONVIF camera adapter (IP cameras, PTZ control)

**AI / ML**
- Additional ONNX reference models (classification, segmentation)
- Model registry with versioning
- Edge inference adapter (ONNX Runtime on ARM/Jetson)
- Anomaly detection pipeline (time-series sensor data)
- LLM-powered operator assistant (natural language → mission commands)

---

## Where Contributors Can Help

| Area | Difficulty | Description |
|------|-----------|-------------|
| Adapters | Easy–Medium | ROS 2, AIS, ONVIF, weather integrations |
| Console UI | Medium | Entity history trails, mission timeline, mobile layout |
| Documentation | Easy | Tutorials, architecture diagrams, adapter how-tos |
| Fusion | Medium | Multi-sensor track correlation algorithms |
| AI Models | Medium | Train/ship additional ONNX reference models |
| Edge Runtime | Hard | ONNX inference on ARM/Jetson with model sync |

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and contribution guidelines.
