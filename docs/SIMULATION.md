# Summit.OS Simulation Guide (SITL/HITL)

This guide shows how to run a local simulation using PX4 or ArduPilot SITL and drive simulated drones with the Summit.OS Tasking service.

Prerequisites
- MacOS or Linux
- Python 3.10+
- Docker (for Summit services)
- Optional Python packages:
  - pymavlink (MAVLink)
  - httpx (asset registration helper)

Install optional Python deps
- Recommended in a virtualenv or your dev container:
  pip install pymavlink httpx

Start Summit services
- Recommended via Makefile targets:
  make dev-services
  make dev-backend

Start a SITL instance
1) PX4 SITL (UDP 14550)
   - One option (Gazebo/JSBSim vary by setup):
     px4 -d
     (then use your preferred PX4 SITL launcher binding to udp:127.0.0.1:14550)
   - Alternative PX4 launchers exist; ensure a MAVLink endpoint at udp:127.0.0.1:14550

2) ArduPilot SITL (quad)
   - Clone ArduPilot and initialize submodules (outside scope of this doc)
   - Run:
     sim_vehicle.py -v ArduCopter -f quad --console --map
   - By default, MAVLink is typically available on udp:127.0.0.1:14550 (check console output)

Run the Summit.OS sim executor
- One-liner (uses defaults shown):
  make sim

- Or run directly with custom settings:
  python apps/tasking/sim_executor.py \
    --asset drone-001=udp:127.0.0.1:14550 \
    --register-assets --arm --takeoff-alt 20 \
    --loiter-center 37.422,-122.084 --loiter-radius 150 --speed 5 --start

What happens
- sim_executor connects to SITL via MAVLink
- Optionally registers the asset in the Tasking Asset Registry
- Arms and takes off to the target altitude
- Uploads a simple loiter mission (single waypoint) and starts it
- You can watch live mission updates over WebSocket from the Data Fabric

Verify mission updates
- Connect to WebSocket:
  websocat ws://localhost:8001/ws
- Create a mission via API Gateway (optional, separate from sim executor):
  POST http://localhost:8000/v1/missions
  {
    "name": "Area Scan A",
    "objectives": ["loiter/scan"],
    "area": {"center": {"lat": 37.422, "lon": -122.084}, "radius_m": 250},
    "num_drones": 2,
    "planning_params": {"pattern": "grid", "grid_spacing_m": 75, "heading_deg": 0, "altitude": 60, "speed": 5}
  }
- Observe missions/# events flowing to the UI via fabric WebSocket

Tips
- If your broker requires auth, set MQTT_USERNAME/MQTT_PASSWORD before running services.
- To enforce auth on Tasking writes, set:
  OIDC_ENFORCE=true and configure OIDC_ISSUER/OIDC_AUDIENCE
- For multiple SITL vehicles, repeat --asset with distinct UDP ports and unique asset IDs.
- To enable direct autopilot inside the tasking container, set environment and build args before starting services:
  export TASKING_DIRECT_AUTOPILOT=true
  export INCLUDE_PYMAVLINK=true
  make dev-backend  # will rebuild tasking with pymavlink and enable direct autopilot

Planner parameters
- planning_params.pattern: "loiter" or "grid"
- planning_params.grid_spacing_m: lane spacing (m)
- planning_params.heading_deg: grid orientation (degrees)
- planning_params.min_sep_m: minimum separation enforced via spacing
- planning_params.altitude_offset_step_m: per-asset altitude offset (m)
- planning_params.start_delay_step_s: per-asset start delay (s)
- Asset constraints (stored in registry) can clamp altitude and speed:
  constraints: {"min_altitude": 20, "max_altitude": 120, "min_speed": 2, "max_speed": 15}

Troubleshooting
- If pymavlink is missing, install it: pip install pymavlink
- If connection fails, verify your SITL is emitting MAVLink on the expected UDP endpoint.
- If asset registration fails, ensure Tasking is reachable at TASKING_URL (default http://localhost:8004).
