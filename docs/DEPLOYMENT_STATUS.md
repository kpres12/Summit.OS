# Deployment Status (Local Dev)

Updated: now

- Infrastructure (Docker Compose)
  - Redis: healthy
  - Postgres+PostGIS: healthy (PostGIS enabled)
  - MQTT (Mosquitto): healthy
  - Prometheus: running
  - Grafana: running
- Applications
  - API Gateway (8000): healthy (health OK)
  - Fabric (8001): starting (migrations run; health endpoint not responding yet)
  - Fusion (8002): healthy (health OK)
  - Intelligence (8003): running
  - Tasking (8004): running
  - Console (3000): running (hydration warning suppressed)

Notes
- Fabric is mid-startup; if it stalls, check logs and DB connectivity. Run `make db-setup` (done).
- Fusion required OpenCV runtime libs; Dockerfile updated and rebuilt.
- Frontend lint/typecheck: fixed; `npm run lint` passes.
- Adapters SDK: MAVLink, ONVIF, Generic HTTP templates added; registry helper implemented.
