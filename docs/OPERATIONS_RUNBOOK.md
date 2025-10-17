# OPERATIONS_RUNBOOK.md

This runbook covers failure modes for upstream integrations and Summit.OS fallbacks.

Common failures and responses
- ROS 2 bridge failure
  - Symptom: no edge telemetry into Fabric.
  - Action: enable direct MAVLink fallback in Tasking.
    - make tasking-direct-on
- Autopilot link loss (PX4/ArduPilot)
  - Symptom: commands not acknowledged, no heartbeats.
  - Action: Tasking keeps mission queue; drones should RTL by failsafe.
- Fusion model unavailable
  - Symptom: 5xx from Fusion, empty detections.
  - Action: set FUSION_ENABLE_VISION_AI=false; use heuristic fusion only.

SITL smoke test (developer)
- PX4 or ArduPilot SITL must be running exposing UDP 14550.
- Register an asset and dispatch a simple loiter:
  ```bash
  make dev-services
  make dev-backend
  python apps/tasking/sim_executor.py --asset drone-001=udp:127.0.0.1:14550 --register-assets --arm --takeoff-alt 20 --loiter-center 37.422,-122.084 --loiter-radius 150 --speed 5 --start
  ```

Security
- Keep GPL components in separate processes and containers.
- Enforce OIDC by setting OIDC_ENFORCE=true on API Gateway/Tasking if available.
