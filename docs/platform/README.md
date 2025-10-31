# Summit Sentinel Platform — Architecture Overview

Unify Sentinel (sense) and Summit.OS (act) via a shared mission & data fabric. Keep subsystems independently deployable; expose one operator experience and canonical APIs/schemas.

## Diagram
- Mermaid source: docs/platform/diagram.mmd

## Subsystems and Fabric
- Sentinel (Sensors & Detection): multi-sensor ingest at edge → detection → events/track seeds.
- Summit.OS (Mission & Autonomy): tasking, vehicle control, safety, payloads, swarm.
- Shared Fabric (the Platform): canonical object model, message bus, persistence, policy/ROE, APIs, operator UI, digital twin/simulation.

## Text Diagram (logical)
[Edge Sensors] → Sentinel Ingest/Detect → Fusion → Incident/Track Store → Mission Manager/Policy → Tasker → Vehicles (ROS2/DDS)
                                     ↘ Operator UI/API Gateway ↗  Telemetry/Video → Timeseries/Object Store

## Service Components (map to repo + external)
- apps/fabric: real-time fabric (MQTT/Redis/gRPC) + bus bridges (Kafka/NATS↔ROS2/DDS).
- apps/fusion: sensor fusion, geospatial ops (PostGIS), track management.
- apps/intelligence: analytics/ML, advisory, detections enrichment.
- apps/tasking: mission manager, rules/ROE (OPA), tasker, deconfliction.
- apps/api-gateway: public gRPC/REST/WebSocket, authz, schema gateway.
- apps/console: operator UI (Cesium/Mapbox, live video via WebRTC).
- Sentinel (external repo): sensor collectors, edge inference, health.
- Shared persistence: PostGIS + TimescaleDB; object store for imagery/video.

## Platform Interfaces (canonical schemas)
Located in docs/platform/schemas/*.json:
- detection_event.json, track.json, mission_intent.json, task_assignment.json, vehicle_telemetry.json, action_ack.json, with common.json $defs.

## Messaging & Protocols
- Control/teleop low-latency: ROS2/DDS (RTPS) with bridge to fabric.
- Fleet/analytics streams: Kafka or NATS JetStream; replay supported.
- APIs: gRPC (internal), REST/WebSocket (external/UI).

## Policy, Safety, Simulation
- OPA for ROE/policy decisions; human-in-the-loop approvals first.
- Digital twin/sim endpoints mirror real APIs (SIL/HIL testbeds).
- Airspace deconfliction, geofencing, emergency-stop paths.

## Tests & Validation (per service)
- Unit+contract tests for schemas/APIs; load tests on bus and fusion.
- Integration flows: detection→track→mission→task→ack.
- Autonomy: SIL/HIL, latency budgets, failover drills.

## Phasing
1) Sentinel MVP → 2) Summit.OS baseline (human-in-loop) → 3) Integrated ops under strict ROE → 4) Swarm autonomy post-certification.
