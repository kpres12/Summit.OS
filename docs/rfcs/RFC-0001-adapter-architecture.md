# RFC: Device Adapter Architecture for Summit.OS

Status: Draft
Author: Summit.OS Platform

1. Problem
- Vendors expose heterogeneous protocols (MAVLink, ROS 2, ONVIF/RTSP, Modbus/CAN, proprietary).
- We need a consistent way to ingest telemetry, detections, tracks, and media; and optionally send control.

2. Goals
- Adapter-first: any device integrates via a thin adapter translating to Summit contracts.
- Offline-tolerant, secure, observable, versioned, and testable.
- Keep adapters 100–300 LoC where possible; push complexity into shared SDK utilities.

3. Contracts (normalized payloads)
- Telemetry: device_id, ts_iso, location {lat, lon, alt}, status, sensors {}
- Detection/Observation: class, confidence, ts_iso, lat, lon, source, attributes {}
- Track: track_id, ts_iso, lat, lon, vel, heading, source
- Task/Action: task_id, action, params, ts_iso
- Media: uri or chunked payload + metadata

4. Transports
- MQTT topics (preferred):
  - telemetry/<device_id>, detections/<device_id>, tracks/<track_id>, media/<device_id>, health/<device_id>/heartbeat
- HTTP/gRPC fallback via API Gateway if MQTT unavailable.

5. Adapter Runtime Model
- Process: one adapter per device type (can multiplex multiple devices).
- SDK provides BaseAdapter with:
  - lifecycle: start(), stop(), health()
  - publish helpers (MQTT, HTTP fallback)
  - schema validators and backpressure utilities
  - config loading (env, file, remote)

6. Identity & Registry
- Each adapter registers device in Fabric registry (nodes table) with capabilities, comm endpoints, and org_id.
- Identity via:
  - JWT (edge-issued or gateway-issued), and/or
  - mTLS client cert (X-Client-DN → OU=org-id)

7. Configuration
- Env-first with optional YAML file: device_id, org_id, broker url, input endpoints.
- Hot-reload safe where feasible.

8. Observability
- Metrics: published_count, drop_count, reconnects, latency.
- Logs: structured JSON; sensitive data redacted.
- Health endpoints: /health (liveness), /readyz (readiness).

9. Security
- JWT on HTTP, MQTT username/password (or mTLS) where available.
- Rate limiting and payload size limits enforced by SDK.

10. Conformance
- Adapters must pass SDK conformance tests:
  - Contract validation (JSON Schema)
  - Topic naming and QoS
  - Backpressure and retry behavior

11. Versioning
- Contracts use semver; adapters declare supported versions (e.g., contracts>=1.0,<2.0).
- Breaking changes gated by feature flags.

12. Minimal Integration Recipe
- Choose template (MAVLink, ONVIF, Generic-HTTP).
- Implement translate_* functions mapping vendor messages to Summit contracts.
- Configure broker and device_id; run conformance tests.

Appendix A: MQTT Topics
- telemetry/<device_id>
- detections/<device_id>
- tracks/<track_id>
- media/<device_id>
- health/<device_id>/heartbeat
