# Changelog

All notable changes to Summit.OS will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0-alpha] - 2026-02-22
### Added
- **SDK Client** (`packages/summit-sdk`): async Python client with entity, task, mesh, and sensor sub-clients.
- **Integration Client**: batch-based data ingestion for external feeds (ADS-B, radar, AIS).
- **AI/ML Pipeline** (`packages/ai`): detection (YOLO/mock), Bayesian + rule-based classification, anomaly detection (z-score, EMA, isolation forest, ensemble), intent prediction (trajectory, behavior, threat).
- **Security Layer** (`packages/security`): mTLS certificate authority, JWT + API key auth, 5-level RBAC hierarchy, data classification with write-up enforcement.
- **Mesh Transport** (`packages/mesh`): UDP datagram protocol with binary framing, HMAC integrity, AES-256-GCM encryption.
- **gRPC Services** (`packages/grpc_services`): entity store with TTL/versioning/watchers, task store with priority queue and dependency tracking.
- **Sensor Fusion** (`apps/fusion`): Unscented Kalman Filter (8-state, WGS-84), Multi-Hypothesis Tracker with gating and pruning.
- **Hardware Abstraction** (`packages/hal`): MAVLink drone driver, ONVIF camera driver with PTZ and RTSP.
- **Observability** (`packages/observability`): OpenTelemetry-compatible tracing with span context propagation.
- **Helm Chart** (`infra/helm`): Kubernetes deployment with autoscaling, security contexts, and ingress.
- **Operator Console** (`apps/console`): Next.js 14 dashboard with entity map, task panel, and real-time WebSocket feeds.
- **5 Backend Microservices**: fabric, fusion, intelligence, tasking, api-gateway — all with health endpoints and Prometheus metrics.
- **107 Tests**: 73 service-level unit tests + 34 integration tests across all packages.
- **CI Pipeline** (`.github/workflows/ci.yml`): lint, format check, typecheck, unit tests with coverage.
- **Docker Compose** (`infra/docker`): full local dev stack (Redis, Postgres+PostGIS, MQTT, Prometheus, Grafana).

### API
- REST API v1 at `/api/v1` — entities, tasks, alerts, telemetry, drones, sensors.
- MQTT topics: `alerts/#`, `devices/+/telemetry`, `missions/updates`.
- WebSocket: `/ws` for real-time entity and mission updates.

### Known Limitations
- AI models default to mock/fallback implementations when optional dependencies are not installed.
- MAVLink and ONVIF drivers operate in simulation mode without physical hardware.
- SDK version is `0.1.0-alpha` — API surface may change before `1.0.0`.

## Deprecation Policy
- **v0.x**: No backwards-compatibility guarantees. Breaking changes noted in changelog.
- **v1.x** (future): Additive changes only in minor releases. Breaking changes require major version bump with a minimum 90-day deprecation window and `@deprecated` decorator warnings.
