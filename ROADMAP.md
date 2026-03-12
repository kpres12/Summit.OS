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
- AIS Maritime (vessel tracking — AISHub API + simulation fallback)

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

**Multi-Latency AI Stack (v0.3)**
- ByteTrack multi-object tracker (`apps/fusion/bytetrack.py`) — replaces SimpleTracker; two-stage Hungarian matching; Kalman prediction; persistent object IDs through occlusion
- Camera adapter (`adapters/camera/`) — RTSP/ONVIF; frame→Inference→ByteTrack pipeline; geo-projected TRACK entities; simulation fallback
- Local LLM brain (`apps/intelligence/brain.py`) — Ollama/Llama3.1; perceive→plan→act loop; tool-calling via physical action definitions; graceful degradation when offline
- Context builder (`apps/intelligence/context_builder.py`) — WorldStore→LLM context; priority-ordered by state; token budget enforcement
- Physical tools (`apps/intelligence/tools.py`) — deploy_asset, create_alert, create_geofence, send_command, open_actuator, query_world, request_human_input
- Mission agent (`apps/intelligence/mission_agent.py`) — autonomous agent lifecycle; multiple concurrent missions; AgentRegistry; auto-completion detection
- Behavior trees (`apps/tasking/behavior_tree.py`) — py-trees integration; Sequence/Selector/Parallel composites; OPA safety gate at every action node; battery and geofence conditions
- Entity history (`packages/world/history.py`) — ring-buffer position trails per entity; haversine dedup; REST API for console map polylines
- Mesh sync (`apps/fabric/mesh_sync.py`) — prioritised delta replication; CRITICAL entities first; bandwidth token bucket; store-and-forward; state-vector anti-entropy
- BATMAN-adv setup (`infra/mesh/batman_setup.sh`) — Layer 2 self-healing mesh; ad-hoc WiFi; DHCP; setup/status/teardown commands
- Brain API endpoints (`GET /brain/status`, `POST /agents`, `GET /agents`, `DELETE /agents/{id}`)

---

## In Progress

- WebSocket authentication (token-based, ties into device identity)
- Entity TTL garbage collection background task
- Console entity history trail visualisation (polyline on MapLibre map)

---

## Next Up

**Platform**
- Track correlation and deduplication (multi-sensor fusion, multi-hypothesis)
- WorldStore persistence benchmarks and tuning
- Mission timeline view in console
- Mobile-responsive console layout

**Adapters**
- ROS 2 adapter (nav2, sensor_msgs bridge)
- AIS maritime data feed
- Weather data overlay (NOAA/OpenWeather)
- ONVIF PTZ control (pan/tilt/zoom commands back to camera)

**AI / ML**
- Additional ONNX reference models (classification, segmentation, fire detection)
- Model registry with versioning
- Edge inference adapter (ONNX Runtime on ARM/Jetson)
- Anomaly detection pipeline (time-series sensor data)
- Fine-tuning pipeline on collected deployment data

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
