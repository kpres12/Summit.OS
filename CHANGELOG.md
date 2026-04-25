# Changelog

All notable changes to Heli.OS will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Heli.OS uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

_Changes staged for the next release go here._

---

## [0.1.0] — 2026-03-28

### Added
- Core microservices: Fabric, Fusion, Intelligence, Tasking, Inference, API Gateway
- Operator Console (Next.js 15) with MapLibre live map, alert queue, mission feed, HLS video overlay, and mission replay scrubber
- Multi-sensor track fusion: Kalman EKF across camera, ADS-B, AIS, MAVLink, CoT/ATAK
- Autonomous mission dispatch: trained GradientBoosting ONNX model (<1ms inference), rules-based fallback always active
- Risk scoring model: LOW → CRITICAL severity classification
- Training pipeline on 108k real-world labeled events (NASA FIRMS, NOAA Storm Events, GBIF)
- Seven mission types: SURVEY, MONITOR, SEARCH, PERIMETER, ORBIT, DELIVER, INSPECT
- Terrain-following waypoint adjustment via SRTM DEM
- Mission replay: time-indexed snapshot streams for incident debrief
- Alert escalation: unacknowledged alerts auto-escalate via webhook
- CoT/ATAK bidirectional UDP bridge (multicast 239.2.3.1:6969)
- Built-in adapters: OpenSky (ADS-B), CelesTrak (satellites), AIS vessels, ONVIF cameras, MAVLink autopilots
- Mesh CRDT replication for disconnected-network operation
- OPA policy-as-code authorization
- mTLS proxy layer (Nginx, optional profile)
- OpenTelemetry tracing across all services (Jaeger)
- Prometheus metrics + Grafana dashboards
- Multi-Link Communication Manager SDK (RadioMesh, Cellular, Satellite stubs)
- Optional LLM reasoning brain (Ollama, `--profile llm`)
- Docker Compose full-stack orchestration with health checks
- `make` development workflow (dev, test, lint, format, smoke, health)
- Seed demo script: registers assets, injects detections, fires auto-dispatch pipeline
- Initial release

[Unreleased]: https://github.com/Branca-ai/Heli.OS/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Branca-ai/Heli.OS/releases/tag/v0.1.0
