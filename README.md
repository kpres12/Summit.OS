# Summit.OS - Distributed Intelligence Fabric

Summit.OS is Big Mountain Technologies' flagship command layer — an open, AI-driven "operating system for the physical world," designed to unify sensors, drones, and ground robots into a shared, real-time world model for decision-making and coordinated action.

## 🌍 System Concept

Summit.OS functions like Anduril's Lattice OS but is purpose-built for resilience, wildfire management, and critical-infrastructure protection. It sits above Linux, Mac, or Windows / ROS 2 and provides a unified intelligence fabric for data fusion, situational awareness, and autonomous tasking.

## 🧱 Core Architectural Layers

| Layer | Description | Example Services |
|-------|-------------|------------------|
| Edge Layer | Agents on robots / drones collecting telemetry, video, and sensor data; run local inference | `/edge-agent`, `/ros-bridge`, `/edge-inference` |
| Data Fabric Layer | Real-time message bus & synchronization (MQTT + gRPC + event mesh) | `/fabric`, `/topics`, `/telemetry` |
| Perception & Fusion Layer | Normalizes & fuses multi-modal data (video, weather, IR, lightning, soil) into a world model | `/fusion`, `/classification`, `/tracking` |
| Reasoning & Decision Layer | AI models generate contextual intelligence, predictions, & recommendations | `/intelligence`, `/risk`, `/advisory` |
| Command & Control Layer | Assigns missions & coordinates autonomous behaviors | `/tasking`, `/autonomy`, `/swarm` |
| Operator Interface | Real-time console for map, alerts, & mission management | `/console` (Next.js + MapLibre + 3D terrain) |
| Integration Layer | External systems (ArcGIS, CAD, dispatch, cloud APIs) | `/integrations`, `/reports`, `/api` |

## 🧩 Technical Stack

- **Backend**: FastAPI (Python 3.11) + gRPC microservices
- **Data Fabric**: MQTT + Redis Streams + gRPC streaming
- **Edge**: ROS 2 / micro-ROS, ONNX Runtime, SQLite store-and-forward
- **Frontend**: Next.js 14 + TypeScript + MapLibre GL + shadcn/ui
- **Database**: Postgres + PostGIS + TimescaleDB
- **AI/ML**: PyTorch + Ray for distributed inference
- **Infra**: Docker Compose (dev), k3s (edge), EKS (cloud), Terraform IaC
- **Auth & Security**: Keycloak (OIDC RBAC), mTLS between agents
- **Observability**: OpenTelemetry + Prometheus + Grafana dashboards

## 🧠 AI Intelligence Architecture

Summit.OS is the **distributed intelligence fabric** that serves as the sense-making and autonomy brain for all BigMT robotics systems. AI is not an add-on feature—it's the connective tissue that enables every robot, drone, and sensor to understand, decide, and act in the physical world.

### AI Integration by Layer

**1. Data Fusion & Sensemaking (Perception Layer)**
- **Multimodal Sensor Fusion**: Weather, LiDAR, IR, visual data → spatial context
- **Anomaly Detection**: Smoke, leaks, flooding, temperature spikes
- **Terrain & Object Segmentation**: Brush vs. road vs. water from drone imagery
- **Environmental State Estimation**: Dryness, wind vectors, fire spread rate

**2. Object Detection, Classification & Tracking**
- **Computer Vision Models** (ONNX/TensorRT): Detect smoke, tools, humans, vehicles
- **Multisensor Tracking**: Maintain persistent IDs across video frames and sensor types
- **State Recognition**: "valve open," "fireline constructed," "ditch cleared"
- **Temporal Reasoning**: Predict object motion or environmental changes

**3. Autonomous Operations & Decision Reasoning**
- **Mission Planning**: Graph-based planners + reinforcement learning for optimal task assignment
- **Multi-agent Coordination**: Decide which asset performs which subtask
- **Predictive Modeling**: Physics + ML hybrid models forecasting outcomes
- **Contextual Advisory**: Generate human-readable summaries for ops consoles

**4. Learning & Continuous Improvement**
- **Model Retraining Pipelines**: New sensor data improves detection accuracy
- **Edge-to-Cloud Federated Learning**: Local models train on field data, sync gradients
- **Simulation-to-Real (Sim2Real)**: Reinforcement learning environments for robotics tasks
- **Anomaly Feedback Loops**: Operator confirmations fine-tune detection thresholds

### Cross-Domain AI Applications

| Product | Shared AI Services | Unique Additions |
|---------|-------------------|------------------|
| **FireLine** | Fusion, Intelligence, Tasking, Predict | Fire behavior, smoke detection |
| **DitchBot** | Fusion, Tasking, Predict | Water flow, soil erosion |
| **OilfieldBot** | Fusion, Tasking, Predict | Pressure anomalies, leak detection |
| **GreaseBot** | Fusion, Tasking | Fill-level estimation, route scheduling |
| **TriageBot** | Fusion, Intelligence, Tasking | Object/person recognition, triage prioritization |

## 🎯 Functional Goals

1. **Unified World Model** – fuse all sensor & asset data into a live geospatial graph of fires, terrain, weather, and movement.
2. **Autonomous Coordination** – route multiple UGVs and UAVs dynamically; mission graphs (Patrol → Detect → Suppress → Verify).
3. **Contextual Intelligence** – produce human-readable situational alerts and recommendations.
4. **Edge Resilience** – operate offline with local buffering & sync on reconnect.
5. **Operator Console** – professional dark UI with map layers, device panels, and AI insight feed.
6. **Open Architecture** – allow third-party sensor / platform integration via standardized APIs.

## 🔗 FireLine Console Integration

FireLine Console (Next.js frontend) is the primary user-facing interface for the BigMT ecosystem. It connects to Summit.OS via API and real-time event channels to render live wildfire intelligence, mission status, and AI insights.

### Integration Paths

| Interface | Function | Example |
|-----------|----------|---------|
| REST/gRPC API Gateway | FireLine queries world model data, intelligence feeds, and mission status | `GET /api/v1/intelligence/alerts`, `POST /api/v1/tasks` |
| WebSocket / MQTT Streams | FireLine subscribes to live telemetry, detections, and status updates | `ws://summit-os/fusion/stream` or `topic alerts/#` |
| Event Hooks / Webhooks | Summit.OS pushes event-based updates to FireLine | `POST /fireline/events/alert` |

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd Summit.OS

# Start development environment
make dev

# Access services
# - FireLine Console: http://localhost:3000
# - API Gateway: http://localhost:8000
# - Grafana: http://localhost:3001
```

## 📂 Monorepo Structure

```
/apps
  /console          # FireLine Console (Next.js frontend)
  /fabric           # Data fabric microservice
  /fusion           # Sensor fusion microservice
  /intelligence     # AI reasoning microservice
  /tasking          # Mission planning microservice
  /edge-agent       # Edge agent for robots/drones
  /integrations     # External system integrations
/packages
  /proto            # gRPC protocol definitions
  /schemas          # Shared data schemas
  /ui               # Shared UI components
  /geo              # Geospatial utilities
  /utils            # Common utilities
/infra
  /docker           # Docker configurations
  /k8s              # Kubernetes manifests
  /terraform        # Infrastructure as Code
/docs
  ARCHITECTURE.md
  API.md
  EDGE_PROTOCOL.md
/tests              # Integration and E2E tests
```

## ✅ Development Requirements

- Type-safe, modular, well-tested code
- Example data streams (robot telemetry + camera + weather)
- `make dev` spins up MQTT broker, Postgres, API gateway, and Console
- Unit tests for fusion, triangulation, and intelligence logic
- Mock mission demo: simulated fire detection → alert → automatic dispatch

## 🎯 Acceptance Criteria

1. `make dev` launches the full system locally with mock data
2. Console shows live telemetry and contextual AI alerts
3. Triangulation + fusion produce ignition estimates
4. Mission planner auto-tasks simulated UGVs & UAVs
5. All microservices documented and passing tests
