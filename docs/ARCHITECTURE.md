# Heli.OS Architecture

## Overview

Heli.OS is an open-source autonomous systems coordination platform. It is the **integration and coordination layer** that sits between your existing signals and the missions you need to run.

The core loop is simple:

```
ANY SIGNAL → world model → operator builds mission → coordinated execution
```

Heli.OS does not build drones, sensors, cameras, or robots. It connects to the ones you already have — or the ones your customers already have — and makes them work together. Every new adapter added to the platform extends what operators can see, understand, and act on.

**Who it's for:** operators, incident commanders, fleet managers, emergency responders, and developers building on top of real-time sensor data. Any domain where humans need to coordinate autonomous or semi-autonomous systems in the physical world.

**What makes it different:** You bring your own hardware. Heli.OS provides the fabric — signal ingestion, sensor fusion, world model, mission framework, and operator interface — as a unified open platform anyone can deploy, extend, and build on top of.

## System Architecture

### Core Principles

1. **Connect Anything**: Standardized adapter framework — if it emits a signal, Heli.OS can ingest it
2. **One World Model**: All signals fused into a single, real-time operational picture regardless of source
3. **Mission-First**: Operators build and execute missions on top of live data — not the other way around
4. **Human-in-the-Loop**: Autonomous coordination with operator authority at every decision point
5. **Open by Default**: Open-source, open APIs, open adapter spec — no vendor lock-in, no black boxes
6. **Edge Resilient**: Offline-capable with local buffering; syncs when connectivity returns

### Architectural Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Operator Interface                       │
│         OPS · COMMAND · DEV  (Console — Next.js)           │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                 Mission Layer                               │
│         Build · Execute · Monitor  (Tasking Service)       │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                Intelligence Layer                           │
│         Alerts · Risk · Recommendations  (Intelligence)    │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                 World Model Layer                           │
│         Entity tracking · Fusion  (Fusion Service)         │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                  Data Fabric Layer                         │
│            (MQTT + Redis Streams + WebSocket)              │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                  Signal Ingestion Layer                     │
│   Adapters: drones · cameras · sensors · APIs · anything   │
└─────────────────────────────────────────────────────────────┘
```

## Microservices Architecture

### Data Fabric Service (`/apps/fabric`)

**Purpose**: Real-time message bus and synchronization layer

**Responsibilities**:
- MQTT message routing and distribution
- Redis Streams for data persistence
- WebSocket connections for real-time updates
- Telemetry aggregation and routing

**Key Components**:
- MQTT Client for message publishing/subscribing
- Redis Client for stream processing
- WebSocket Manager for real-time connections
- Message routing and filtering

**API Endpoints**:
- `POST /telemetry` - Publish telemetry data
- `POST /alerts` - Publish alert data
- `POST /missions` - Publish mission updates
- `GET /metrics` - System metrics
- `WS /ws` - WebSocket for real-time streams

### Sensor Fusion Service (`/apps/fusion`)

**Purpose**: Multi-modal sensor data fusion and world model generation

**Responsibilities**:
- Sensor data normalization and calibration
- Multi-modal data fusion algorithms
- Object detection and tracking
- World model maintenance
- Triangulation and geolocation

**Key Components**:
- Fusion Engine for data processing
- World Model for entity management
- Detection Service for object recognition
- Triangulation Service for position estimation

**API Endpoints**:
- `POST /sensor-data` - Process sensor data
- `POST /detections` - Process detection results
- `GET /world-model` - Get current world state
- `GET /detections` - Get recent detections
- `POST /triangulate` - Triangulate position

### Intelligence Service (`/apps/intelligence`)

**Purpose**: AI reasoning and contextual intelligence generation

**Responsibilities**:
- Risk assessment and prediction
- Situational awareness analysis
- Alert generation and prioritization
- Recommendation engine
- Pattern recognition and anomaly detection

**Key Components**:
- Risk Assessment Engine
- Alert Generation System
- Recommendation Engine
- Pattern Recognition Models
- Contextual Analysis

### Mission Tasking Service (`/apps/tasking`)

**Purpose**: Mission planning and autonomous coordination

**Responsibilities**:
- Mission planning and optimization
- Asset assignment and routing
- Task decomposition and scheduling
- Progress monitoring and adaptation
- Swarm coordination

**Key Components**:
- Mission Planner
- Asset Manager
- Task Scheduler
- Progress Monitor
- Swarm Coordinator

### Edge Agent (`/apps/edge-agent`)

**Purpose**: Edge device integration and local processing

**Responsibilities**:
- ROS 2 integration
- Local inference and processing
- Store-and-forward capabilities
- Device management
- Offline operation support

**Key Components**:
- ROS 2 Bridge
- Local Inference Engine
- Data Buffer
- Device Manager
- Sync Manager

## Data Flow

```
Any Signal Source
  (drone telemetry, camera, ADS-B, weather, IoT sensor, webhook, NMEA GPS, ...)
      │
      ▼
  Adapter Layer  ─── normalizes to Heli.OS observation schema
      │
      ▼
  Data Fabric  ─── MQTT + Redis Streams
      │
      ▼
  Fusion Service  ─── fuses observations into persistent entity records
      │
      ▼
  World Model  ─── live operational picture (all entities, positions, state)
      │
      ├──▶  Intelligence Service  ─── alerts, risk scores, recommendations
      │
      └──▶  Tasking Service  ─── operator builds mission on live world model
                │
                ▼
           Mission Execution  ─── coordinates connected assets
                │
                ▼
           API Gateway → Console  ─── operator sees and controls everything
```

Every signal source, regardless of manufacturer or protocol, ends up as a normalized entity in the world model. Operators never deal with protocol details — they see a unified picture and build missions from it.

## Technology Stack

### Backend Services
- **Language**: Python 3.11
- **Framework**: FastAPI
- **Message Bus**: MQTT + Redis Streams
- **Database**: PostgreSQL + PostGIS + TimescaleDB
- **AI/ML**: PyTorch + Ray
- **Communication**: gRPC + WebSocket

### Frontend
- **Framework**: Next.js 14 + TypeScript
- **Mapping**: MapLibre GL
- **UI**: shadcn/ui + Tailwind CSS
- **State**: Zustand
- **Real-time**: Socket.IO

### Infrastructure
- **Development**: Docker Compose
- **Edge**: k3s
- **Cloud**: EKS
- **IaC**: Terraform
- **Monitoring**: Prometheus + Grafana
- **Security**: Keycloak + mTLS

## Security Architecture

### Authentication & Authorization
- **Identity Provider**: Keycloak (OIDC)
- **Role-Based Access Control**: RBAC with fine-grained permissions
- **API Security**: JWT tokens + API keys
- **Edge Security**: mTLS between agents

### Data Security
- **Encryption**: TLS 1.3 for data in transit
- **Storage**: AES-256 for data at rest
- **Key Management**: HashiCorp Vault
- **Audit Logging**: Comprehensive audit trails

### Network Security
- **Network Segmentation**: Isolated service networks
- **Firewall Rules**: Restrictive ingress/egress
- **VPN Access**: Secure remote access
- **DDoS Protection**: Rate limiting and filtering

## Scalability & Performance

### Horizontal Scaling
- **Microservices**: Independent scaling per service
- **Load Balancing**: Round-robin with health checks
- **Auto-scaling**: Kubernetes HPA based on metrics
- **Database Sharding**: Geographic and temporal partitioning

### Performance Optimization
- **Caching**: Redis for frequently accessed data
- **Stream Processing**: Real-time data processing
- **Edge Computing**: Local processing to reduce latency
- **CDN**: Static asset delivery optimization

### Monitoring & Observability
- **Metrics**: Prometheus + custom metrics
- **Logging**: Structured logging with ELK stack
- **Tracing**: OpenTelemetry distributed tracing
- **Alerting**: Grafana + PagerDuty integration

## Deployment Architecture

### Development Environment
```bash
make dev  # Full local development stack
```

### Production Deployment
- **Edge**: k3s clusters on field devices
- **Cloud**: EKS clusters in AWS/GCP
- **Hybrid**: Edge-cloud synchronization
- **Disaster Recovery**: Multi-region deployment

## Signal Integrations (Adapters)

Adapters are the growth engine of Heli.OS. Each adapter connects a signal source to the platform. Adding a new adapter makes the platform more valuable for every existing deployment.

### Built-in Adapters
| Adapter | Protocol | Use Cases |
|---|---|---|
| OpenSky ADS-B | REST | Aircraft tracking, airspace awareness |
| CelesTrak | REST | Satellite positions, orbital tracking |
| MAVLink | UDP/TCP/Serial | ArduPilot & PX4 drones, UAV telemetry |
| RTSP Camera | RTSP | Any IP camera — surveillance, inspection |
| ONVIF | HTTP | Standards-compliant IP cameras |
| NMEA GPS | Serial/TCP | GPS receivers, vessels, vehicles |
| CAP Alerts | XML/Atom | FEMA, NWS, emergency alert systems |
| Webhook | HTTP | Any system that can POST JSON |
| MQTT Relay | MQTT | Bridge from another MQTT broker |

### Writing a Custom Adapter
Any signal source can be integrated by subclassing `BaseAdapter` from `packages/adapters/base.py`. The framework handles reconnection, health tracking, metrics, and MQTT publishing. An adapter implementation is typically 50–100 lines.

### Use Cases by Domain
- **Wildfire / Disaster Response**: Drone imagery + weather stations + ground sensors + CAP alerts
- **Search & Rescue**: UAV telemetry + NMEA GPS (ground teams) + RTSP cameras + radio relays
- **Commercial UAV Fleets**: MAVLink telemetry from any autopilot + geofence data + ADS-B awareness
- **Critical Infrastructure**: ONVIF cameras + IoT sensors + access control + weather data
- **Maritime**: AIS vessel tracking + NMEA GPS + weather + RTSP dock cameras
- **Smart Cities / Public Safety**: Any combination of the above

## Future Roadmap

- **Mission Templates**: Shareable, parameterized mission blueprints
- **Natural Language Missions**: "Dispatch nearest drone to sector 4 and establish visual contact" → executed
- **Adapter Marketplace**: Community-contributed adapters with one-click install
- **Multi-tenant**: Isolated organizations on shared infrastructure
- **Mobile**: Native operator apps (iOS/Android) with offline support
- **Edge AI**: On-device inference for low-bandwidth deployments
