# Summit.OS Mission Planning

This document describes planning parameters and profiles supported by the Tasking service.

Planning request shape
- area
  - center: { lat, lon }
  - radius_m: number
  - polygon: [[lat, lon], ...]  // optional; if provided, lanes are clipped to the polygon
- num_drones: integer (>=1)
- planning_params
  - pattern: "loiter" | "grid" | "spiral" | "expanding_square"
  - altitude: meters AGL
  - speed: m/s
  - grid_spacing_m: lane spacing in meters (grid/expanding_square)
  - heading_deg: grid orientation in degrees (grid)
  - min_sep_m: minimum separation enforced by spacing (grid)
  - altitude_offset_step_m: per-asset altitude offset to layer aircraft (all patterns)
  - start_delay_step_s: per-asset start delay (all patterns)

Asset constraints (asset registry)
- Store per-asset in the "constraints" field:
  {
    "min_altitude": 20,
    "max_altitude": 120,
    "min_speed": 2,
    "max_speed": 15,
    "max_flight_time_s": 900
  }
- Planner clamps altitude/speed to these bounds and trims waypoint lists to avoid exceeding max_flight_time_s (rough approximation based on path length and speed).

Profiles
- loiter
  - Places one waypoint per asset around a circle offset from the center.
- grid
  - Lanes generated in a local meters frame, rotated by heading_deg, converted back to lat/lon.
  - If area.polygon provided, each lane segment is clipped to the polygon (endpoints at boundary intersections).
  - Lanes are assigned round-robin across assets.
- spiral
  - Outward spiral centered at area.center.
- expanding_square
  - Square loops expanding outward from the center.

Examples
- Grid over polygon:
  {
    "area": {
      "center": {"lat": 37.422, "lon": -122.084},
      "polygon": [[37.4225, -122.085], [37.4225, -122.083], [37.4215, -122.083], [37.4215, -122.085]]
    },
    "num_drones": 2,
    "planning_params": {
      "pattern": "grid",
      "grid_spacing_m": 60,
      "heading_deg": 15,
      "min_sep_m": 60,
      "altitude": 70,
      "speed": 5,
      "altitude_offset_step_m": 10,
      "start_delay_step_s": 2
    }
  }

- Spiral search:
  {
    "area": {"center": {"lat": 37.422, "lon": -122.084}, "radius_m": 300},
    "num_drones": 1,
    "planning_params": {"pattern": "spiral", "altitude": 80, "speed": 6}
  }
