# Summit.OS

> **Open source autonomous coordination for the physical world.**
> The operational platform for wildfire response, search & rescue, disaster coordination, and commercial UAV fleets.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org)
[![Docker](https://img.shields.io/badge/docker-compose-blue.svg)](infra/docker/docker-compose.yml)

---

## What it does

At **14:33 on a Tuesday**, a camera on a ridge detects smoke. Here's what Summit.OS does in the next 90 seconds — with no human touching a keyboard:

1. **Fusion** receives the detection (`smoke, confidence 0.91, lat 34.12, lon -118.34`)
2. **Intelligence** scores it CRITICAL, dispatches the nearest available UAV to SURVEY — automatically, using a trained ML model, no LLM required
3. **Tasking** creates a mission, adjusts altitude for terrain, sends waypoints to the drone over MQTT
4. **Fabric** raises an alert, starts escalating if no operator acknowledges within your configured timeout
5. **Console** shows the operator a live map with the smoke location, the drone en route, and a live HLS video feed the moment the drone arrives

The operator's job: watch, verify, decide whether to dispatch ground resources. The software handles everything before that decision.

---

## Why Summit.OS

The coordination software that does this well — Anduril's LatticeOS, Shield AI's platform — is closed-source, defense-export-controlled, and costs millions per year. It is not available to a county fire department, a maritime SAR team, an NGO running conservation drones, or a startup building inspection UAVs.

Summit.OS is the open-source alternative. Same architecture. Built for the civilian world.

|  | Summit.OS | LatticeOS / Defense platforms |
|---|---|---|
| License | AGPL v3 | Proprietary |
| Access | Anyone | Defense contractors |
| Cost | Free to self-host | $M/year |
| Export restrictions | None | ITAR/EAR controlled |
| Use cases | Civilian emergency response, commercial ops | Military |
| AI | Trained on your data, self-improving | Black box |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Operator Console :3002                          │
│  MapLibre live map · Alert queue · Mission feed · Live video · Replay   │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │  REST + WebSocket
┌───────────────────────────▼─────────────────────────────────────────────┐
│                         API Gateway :8000                               │
│              Routing · OIDC/RBAC · Rate limiting · Audit log            │
└──────────┬──────────┬────────────────┬────────────┬─────────────────────┘
           │          │                │            │
    ┌──────▼──┐ ┌─────▼──────┐ ┌──────▼────┐ ┌────▼───────┐ ┌───────────┐
    │  Fabric │ │   Fusion   │ │Intelligence│ │  Tasking   │ │ Inference │
    │  :8001  │ │   :8002    │ │   :8003   │ │   :8004    │ │   :8006   │
    │─────────│ │────────────│ │───────────│ │────────────│ │───────────│
    │WorldSt. │ │Multi-sensor│ │Rule+ML    │ │State mach. │ │ONNX model │
    │MQTT brdg│ │track fusion│ │auto-disp. │ │Asset assign│ │hot-swap   │
    │WS stream│ │Kalman EKF  │ │Risk score │ │OPA policy  │ │YOLOv8n    │
    │Mesh CRDT│ │Re-ID       │ │LLM brain* │ │Replay store│ │           │
    └────┬────┘ └─────┬──────┘ └──────┬────┘ └────┬───────┘ └───────────┘
         │            │               │            │
    ┌────▼────────────▼───────────────▼────────────▼───────────────────────┐
    │               Redis · PostgreSQL+PostGIS · MQTT (Mosquitto)          │
    │               Prometheus · Grafana · Jaeger · OPA                    │
    └───────────────────────────────────────────────────────────────────────┘

    * Optional — system operates fully without it
```

---

## Key Capabilities

**Multi-sensor fusion** — Kalman EKF track fusion across camera, ADS-B, AIS, MAVLink, CoT/ATAK, and any custom sensor. M-of-N track confirmation. Cross-camera re-identification.

**Autonomous mission dispatch** — When a detection arrives, a trained ML model (GradientBoosting, ONNX, <1ms inference) decides the mission type and dispatches immediately. Trained on 87,160 real-world labeled events (NASA FIRMS, NOAA Storm Events, GBIF) plus 20,936 synthetic examples covering edge-case mission types. No LLM required. Rules-based fallback always active.

**Self-improving AI** — Retrain the mission planner on your own operator decisions with one command:
```bash
python packages/ml/train_mission_classifier.py --real-data postgresql://...
```
The more you use it, the better it gets at your specific deployment context.

**Live video** — HLS streaming from any RTSP source (IP cameras, drone gimbal feeds). Sub-5-second latency. Playable in any browser.

**Mission replay** — Every mission is recorded as a time-indexed snapshot stream. Operators can scrub back through any incident for debrief or training.

**Terrain awareness** — SRTM DEM integration. All waypoints automatically adjusted to maintain true AGL altitude over real terrain.

**Alert escalation** — Unacknowledged alerts auto-escalate via webhook + email. Configurable timeout per severity.

**CoT/ATAK compatibility** — Bidirectional UDP bridge. Entities visible to any ATAK device on the network.

**30-minute device integration** — One base class, two methods:
```python
class MyDrone(SummitAdapter):
    async def get_telemetry(self): ...
    async def handle_command(self, cmd): ...
```

---

> [!WARNING]
> **Security defaults are open for local development.** Out of the box, authentication is disabled (`OIDC_ENFORCE=false`, `RBAC_ENFORCE=false`, `API_KEY_ENFORCE=false`) and PII field encryption is off. This is intentional for local dev — running `docker compose up` should just work without configuring an identity provider.
>
> **Before connecting Summit.OS to any real network, real hardware, or real incident data**, read the [Production Hardening](#production-hardening) section below. The API Gateway will print a visible warning banner at startup until you set these values.

---

## Quick Start

**Prerequisites:** Docker Desktop (or Docker + Compose V2), 8 GB RAM, 10 GB disk.

```bash
# 1. Clone
git clone https://github.com/BigMT-Ai/Summit.OS.git
cd Summit.OS

# 2. Configure (defaults work for local dev)
cp .env.example .env

# 3. Start
cd infra/docker
docker compose up
```

That's it. After ~60 seconds:

| URL | What |
|---|---|
| http://localhost:3002 | Operator console |
| http://localhost:8000/docs | API docs (Swagger) |
| http://localhost:3001 | Grafana dashboards (admin / admin) |
| http://localhost:16686 | Jaeger distributed traces |

**Seed demo data (see the pipeline work immediately)**

```bash
pip install httpx redis
python scripts/seed_demo.py
```

This registers five assets on the map, injects a smoke detection, a missing hiker, and a power line alert — the intelligence service scores them, auto-dispatches missions, and your console shows a live active incident within seconds.

Add `--live` to keep assets moving and fire new detections every 30 seconds.

**Optional: enable the LLM reasoning brain**

If you have Ollama installed locally, or want to run it in Docker:
```bash
# With Docker (pulls llama3.1 on first start, ~4 GB)
docker compose --profile llm up

# Then pull the model (one time)
docker compose --profile llm exec ollama ollama pull llama3.1
```
The system works fully without this. The LLM adds natural-language mission reasoning for complex multi-entity scenarios.

---

## How the AI works

Summit.OS ships with two trained ML models in `packages/ml/models/`:

| Model | What it does | Size | Latency |
|---|---|---|---|
| `mission_classifier.onnx` | Maps (class, confidence, location) → mission type | ~200 KB | <1ms |
| `risk_scorer.onnx` | Scores observation severity (LOW → CRITICAL) | ~150 KB | <1ms |

Training data breakdown:
- **35,928** NASA FIRMS active fire detections (global, 7-day) — real-world
- **49,869** NOAA Storm Events (tornadoes, floods, storm surge, 2018–2023) — real-world
- **1,363** GBIF wildlife observations — real-world
- **20,936** synthetic examples covering SEARCH, INSPECT, DELIVER, ORBIT — generated

**Supported mission types:**

| Mission | Triggers | Asset |
|---|---|---|
| SURVEY | Fire, smoke, flood, earthquake damage, crop anomaly | UAV or fixed-wing |
| MONITOR | Person, survivor, vessel, vehicle, wildlife | UAV (loiter) |
| SEARCH | Missing person, distress signal, mayday, overdue vessel | UAV grid pattern |
| PERIMETER | Hazmat spill, tornado, storm surge, intrusion, armed threat | UAV boundary |
| ORBIT | Suspicious drone, vessel tracking, persistent ISR | UAV (continuous orbit) |
| DELIVER | Aid drop, medical supply, cargo, resupply | UAV |
| INSPECT | Pipeline, power line, bridge, solar farm, wind turbine | UAV (close pass) |

**Retrain on your data:**
```bash
# Download fresh public data (NASA, NOAA, GBIF)
python packages/ml/download_real_data.py --years 2020 2021 2022 2023

# Blend with your operator-approved mission history
python packages/ml/train_mission_classifier.py \
  --real-csv packages/ml/data/real_combined.csv \
  --real-data postgresql://summit:password@localhost:5432/summit_os

# New .onnx files drop in place — no redeployment needed
```

---

## Monorepo Structure

```
apps/
  console/          # Next.js 15 operator UI — map, alerts, missions, video
  fabric/           # WorldStore, MQTT bridge, mesh replication, WebSocket
  fusion/           # Sensor fusion, Kalman tracking, re-identification, HLS
  intelligence/     # ML dispatch, risk scoring, advisory, LLM brain (optional)
  tasking/          # Mission lifecycle, assignment, terrain following, replay
  inference/        # ONNX model serving — plug in any detection model
  api-gateway/      # Routing, auth, rate limiting, audit

packages/
  ml/               # Training pipeline, data downloaders, trained models
  entities/         # Core Entity schema — the universal data type
  world/            # WorldStore — in-memory + Postgres + broadcast
  mesh/             # CRDT replication across disconnected nodes
  geo/              # DEM terrain, elevation profiles, line-of-sight
  adapters/         # Adapter registry, built-in adapters (CoT/ATAK, MAVLink)
  observability/    # OpenTelemetry tracing middleware
  security/         # mTLS, RBAC, JWT, data classification

infra/
  docker/           # docker-compose.yml, Mosquitto, Prometheus, Grafana config
  policy/           # OPA authorization policies
  proxy/            # Nginx mTLS proxy configs
```

---

## Services at a glance

| Service | Port | |
|---|---|---|
| Operator Console | 3002 | MapLibre map, alert queue, mission feed, video overlay |
| API Gateway | 8000 | `/docs` for full API reference |
| Fabric | 8001 | WebSocket at `/ws/{org_id}` for live entity stream |
| Fusion | 8002 | `/api/v1/tracks`, `/api/v1/video/hls`, `/api/v1/elevation` |
| Intelligence | 8003 | `/agents`, `/advisories`, `/brain/status` |
| Tasking | 8004 | `/api/v1/missions`, `/api/v1/missions/{id}/replay/timeline` |
| Inference | 8006 | `/detect`, `/models` — swap ONNX models at runtime |
| Grafana | 3001 | Pre-built dashboards for all services |
| Jaeger | 16686 | Distributed traces across the full request path |

---

## Connecting hardware

Summit.OS ships with built-in adapters for:
- **DJI / MAVLink autopilots** — via `pymavlink` (set `TASKING_DIRECT_AUTOPILOT=true`)
- **CoT/ATAK** — bidirectional UDP, multicast at 239.2.3.1:6969
- **ONVIF cameras** — discovery + RTSP stream registration
- **OpenSky Network** — live ADS-B aircraft (no account needed)
- **AIS vessels** — maritime vessel positions (simulation mode; plug in AISHub credentials for live data)
- **CelesTrak** — satellite orbital positions

Custom hardware: subclass `SummitAdapter` in `packages/adapters/`. See `examples/` for a complete template.

---

## Configuration

Copy `.env.example` to `.env`. Defaults work out of the box for local development.

**Minimum for production (change these):**
```bash
POSTGRES_PASSWORD=<strong password>
FABRIC_JWT_SECRET=<64 random hex chars>
FIELD_ENCRYPTION_KEY=<openssl rand -base64 32>
```

**To enable AI brain (optional):**
```bash
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
```

**To enforce auth:**
```bash
OIDC_ENFORCE=true
OIDC_ISSUER=https://your-keycloak/realms/summit
API_KEY_ENFORCE=true
RBAC_ENFORCE=true
```

See `.env.example` for the full reference.

---

## Development

```bash
# Infrastructure only (Postgres, Redis, MQTT, Prometheus)
docker compose -f infra/docker/docker-compose.yml up -d redis postgres mqtt prometheus grafana

# Individual services (from their app directory)
cd apps/fabric && uvicorn main:app --reload --port 8001
cd apps/console && npm run dev

# Run tests (per service — see CONTRIBUTING.md for why)
python -m pytest apps/fabric/tests/
python -m pytest apps/fusion/tests/
python -m pytest apps/tasking/tests/
python -m pytest apps/intelligence/tests/

# Hot reload all Python services
UVICORN_RELOAD=true docker compose -f infra/docker/docker-compose.yml up
```

**Database migrations:**

```bash
# Fresh install — enable PostGIS, then apply all migrations
make db-setup
make db-migrate

# Check migration state
make db-status

# Roll back one revision
make db-rollback
```

Migration files are in `apps/fabric/alembic/versions/`. Every schema change needs an Alembic migration — see [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow.

---

## Deployment

**Self-hosted (recommended start):**
```bash
# Production — builds images, no volume mounts
docker compose -f infra/docker/docker-compose.yml up --build -d
```

**Cloud:** Any Kubernetes cluster. A Helm chart is available at `infra/helm/summit-os/`:
```bash
helm install summit ./infra/helm/summit-os \
  --set secrets.postgresPassword=$(openssl rand -hex 32) \
  --set secrets.fabricJwtSecret=$(openssl rand -hex 32) \
  --set secrets.fieldEncryptionKey=$(openssl rand -base64 32)
```

**Air-gapped / edge:** All ML inference runs locally. No external API calls required. Designed to operate on field hardware with no internet connection.

---

## Production Hardening

> [!IMPORTANT]
> The steps below are **required** before deploying Summit.OS to any environment where it handles real incident data or real hardware.

### 1. Enable authentication

```bash
# In your .env
OIDC_ENFORCE=true
OIDC_ISSUER=https://your-keycloak/realms/summit
OIDC_AUDIENCE=summit-api
OIDC_JWKS_URL=https://your-keycloak/realms/summit/protocol/openid-connect/certs
RBAC_ENFORCE=true
API_KEY_ENFORCE=true
```

Keycloak is the recommended identity provider — `infra/docker/docker-compose.keycloak.yml` stands up a pre-configured instance. Any OIDC-compliant provider (Auth0, Okta, Authentik) works.

### 2. Set strong secrets

```bash
POSTGRES_PASSWORD=$(openssl rand -hex 32)
FABRIC_JWT_SECRET=$(openssl rand -hex 32)
FIELD_ENCRYPTION_KEY=$(openssl rand -base64 32)
GRAFANA_ADMIN_PASSWORD=$(openssl rand -hex 16)
```

### 3. Enable MQTT TLS

The default Mosquitto config uses plaintext port 1883 — fine inside Docker's internal network, not fine if your MQTT broker is reachable externally. To enable TLS:

```bash
# Generate broker certs (writes to infra/docker/certs/)
bash scripts/gen_mqtt_certs.sh

# Then in infra/docker/mosquitto.conf:
# Uncomment the TLS listener block and comment out the dev block
# Update MQTT_PORT=8883 in your .env
```

The TLS configuration block is included (commented out) in `infra/docker/mosquitto.conf`.

### 4. CORS origins

Update `CORS_ORIGINS` to your actual console domain:

```bash
CORS_ORIGINS=https://console.yourdomain.com
```

### 5. Data retention

Configure how long audit logs and mission replays are kept:

```bash
AUDIT_RETENTION_DAYS=90    # audit log entries (default: 90)
```

Mission replay snapshots are stored indefinitely by default. For long-running deployments, implement a periodic cleanup of old `mission_snapshots` rows in Postgres based on your operational and legal requirements. SAR and emergency response operators should consult their jurisdiction's incident record-keeping requirements before reducing retention.

### 6. mTLS (optional, high-security deployments)

Enable the Nginx mTLS proxy layer for service-to-service encryption:

```bash
docker compose --profile mtls up
```

Certificates go in `infra/proxy/certs/`. See `infra/proxy/nginx.conf` for the configuration.

---

## Editions

| | Community | Enterprise |
|---|---|---|
| Core platform | ✓ Open source (AGPL v3) | ✓ On-premise |
| ML models (base) | ✓ Included | ✓ Custom trained |
| Observability stack | ✓ Self-host | ✓ Self-host or managed |
| SLA | — | 99.9% + 4h response |
| Auth (OIDC/RBAC) | ✓ Self-configure | ✓ SSO + MFA enforced |
| Multi-org tenancy | ✓ org_id namespacing | ✓ Row-level isolation + admin UI |
| Custom model training | DIY | ✓ On your operator data |
| Hardware integration support | Community | ✓ Dedicated |
| Compliance (SOC 2 / ISO 27001) | Controls built-in, self-certify | ✓ Audit support |
| **Price** | **Free** | **Contact us** |

---

## Roadmap

- [x] Helm chart for Kubernetes deployment — available at `infra/helm/summit-os/`
- [x] SITL testing environment — `docker compose -f infra/docker/docker-compose.sitl.yml up`
- [x] Pre-built adapters: Skydio, Autel, Parrot — see `adapters/skydio/`, `adapters/autel/`, `adapters/parrot/`
- [ ] USCG SAR data integration (SEARCH model improvement)
- [ ] Behavior tree editor in the console (visual mission programming)
- [ ] Hardware-in-the-loop (HITL) testing with real flight controllers
- [ ] Reach RTK adapter (cm-precision GPS for precision landing)
- [ ] Managed hosting via BigMT.ai (self-hosted is the supported path for v0.1.0)
- [ ] Federated learning — improve shared base models without sharing raw data

---

## Contributing

Issues, PRs, and adapter contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

If you're a fire department, SAR team, or drone operator running Summit.OS in the field — we want to hear from you. Your operational data (anonymized) is what makes the ML models better for everyone.

---

## Who built this

Summit.OS is a [BigMT.ai](https://bigmt.ai) project. Built for civilian operators, not defense contractors.

---

## License

[AGPL v3](LICENSE) — free to self-host, fork, and build on. If you deploy Summit.OS as a network service, the AGPL requires you to make your source available to users. For proprietary deployments where that's not workable, see [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).

The trained ML models in `packages/ml/models/` are custom-built and trained by BigMT.ai and released under the same license.
