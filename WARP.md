# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project overview
- Monorepo for Summit.OS, a distributed intelligence fabric coordinating robots/drones and fusing multimodal data.
- High-level layout:
  - apps/: microservices and frontend
    - console: Next.js operator console
    - fabric: data fabric (MQTT/Redis/gRPC)
    - fusion: perception/fusion service (geospatial + CV)
    - intelligence: AI reasoning service
    - tasking: mission planning service
    - api-gateway: routes requests to backend services
  - packages/: shared libs
    - summit-os-sdk: Python SDK with optional extras (ros2, mqtt, websocket, ai, dev)
  - infra/docker: local dev stack via docker-compose (Redis, Postgres+PostGIS, MQTT, Prometheus, Grafana + app services)

Common commands
- Quick start (recommended):
  - Start full dev environment (services → apps):
    - make dev
  - Start infrastructure services only:
    - make dev-services
  - Start applications only (expects services running):
    - make dev-apps
  - Frontend only (local Next.js dev server):
    - make dev-console
  - Backend only (all Python microservices + API gateway):
    - make dev-backend
  - Stream logs for all compose services:
    - make logs
  - Health check endpoints (aggregated curl checks):
    - make health
  - Clean up containers/volumes and prune:
    - make clean

- Dependency install
  - Frontend deps (invoked by make dev and make install-deps):
    - (inside apps/console) npm install
  - SDK/dev extras can be installed per your workflow (see packages/summit-os-sdk/setup.py extras_require).

- Build
  - Frontend production build:
    - (inside apps/console) npm run build

- Lint and format
  - Run all lint targets:
    - make lint
  - Run all format targets:
    - make format
  - Direct per-service (Python):
    - (apps/* service) python -m flake8 .
    - (apps/* service) python -m black .
  - Direct frontend (JS/TS):
    - (apps/console) npm run lint
    - Note: apps/console currently does not define a "format" script.

- Tests
  - Run test suite across services:
    - make test
    - Note: apps/console/package.json does not define a "test" script; frontend tests will not run unless you add one.
  - Python (pytest) patterns:
    - Run all tests in a service:
      - (apps/fabric) python -m pytest tests/ -v
    - Run a single file:
      - (apps/fabric) python -m pytest tests/test_example.py -v
    - Run a single test function:
      - (apps/fabric) python -m pytest tests/test_example.py::TestClass::test_case -v
    - Keyword filter:
      - (apps/fabric) python -m pytest -k "fusion and not slow" -v

- Database and demo helpers
  - Enable PostGIS in the dev DB:
    - make db-setup
  - Generate mock data (requires scripts/generate_mock_data.py):
    - make mock-data
  - Start demo mission (requires scripts/demo_mission.py):
    - make demo

High-level architecture and service interactions
- Control plane and data plane (compose):
  - Core infra: Redis (6379), Postgres+PostGIS (5432), MQTT broker (1883/9001), Prometheus (9090), Grafana (3001)
  - App plane (FastAPI + Uvicorn):
    - fabric (8001): message fabric integration (MQTT/Redis/gRPC)
    - fusion (8002): perception + geospatial fusion; writes/reads Postgres/PostGIS
    - intelligence (8003): reasoning/AI services (inference, advisory)
    - tasking (8004): mission planning and coordination
    - api-gateway (8000): routes to backend services via FABRIC_URL/FUSION_URL/INTELLIGENCE_URL/TASKING_URL
  - Frontend: console (3000): Next.js UI using NEXT_PUBLIC_API_URL and NEXT_PUBLIC_WS_URL

- Data flow (typical):
  - Edge/ingest → MQTT topics → fabric → fan-out to fusion/intelligence/tasking via gRPC/HTTP → api-gateway exposes consolidated endpoints → console consumes API/WebSocket and renders map/insights.

- Observability:
  - Prometheus scrapes service metrics; Grafana available at http://localhost:3001.

- Python SDK (packages/summit-os-sdk):
  - Standard setuptools package with optional extras (ros2, mqtt, websocket, ai, dev) and a console entry point: summit-os-cli.

Service ports and environment
- Local ports (compose defaults):
  - console: 3000
  - api-gateway: 8000
  - fabric/fusion/intelligence/tasking: 8001/8002/8003/8004
  - grafana/prometheus: 3001/9090
- Compose config: infra/docker/docker-compose.yml defines service env vars (e.g., REDIS_URL, POSTGRES_URL, MQTT_BROKER), volume mounts for live dev, and uvicorn --reload for Python services.

Notes for Warp
- Prefer using the Makefile targets for common workflows (dev, services, apps, test, lint, format, logs, clean) to ensure consistent orchestration across services.
- When running ad-hoc tests or linters, scope commands to a single service directory (apps/<service>) to avoid cross-env interference.
- Frontend testing/format scripts are not defined in apps/console at the time of writing; use project conventions if/when they’re added.
