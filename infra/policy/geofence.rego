package policy.geofence

import future.keywords.if
import future.keywords.in

# ──────────────────────────────────────────────────────────────────────────────
# Summit.OS Geofence Policy
#
# Evaluated by the OPAClient.evaluate_geofence() method before any mission
# waypoint is dispatched to a vehicle. Ensures assets never leave authorized
# operating boundaries.
#
# Input schema:
#   {
#     "asset_id": "mavlink-drone-01",
#     "waypoints": [
#       { "latitude": 34.05, "longitude": -118.25, "altitude_m": 100.0 }
#     ],
#     "org_id": "acme-corp",
#     "context": {
#       "time": "2026-03-12T14:00:00Z",
#       "check_type": "geofence",
#       "max_altitude_m": 122.0,       // org-configured max (122m = 400ft FAA limit)
#       "geofences": [                  // active geofences from world model
#         {
#           "geofence_id": "gf-001",
#           "type": "exclusion" | "inclusion",
#           "min_lat": 34.0, "max_lat": 34.1,
#           "min_lon": -118.3, "max_lon": -118.2
#         }
#       ]
#     }
#   }
# ──────────────────────────────────────────────────────────────────────────────

default allow := false

allow if {
    count(deny_reasons) == 0
}

deny_reasons[reason] {
    some reason
    _deny(reason)
}

# ── Structural validation ─────────────────────────────────────────────────────

_deny("asset_id is required") if {
    not input.asset_id
}

_deny("waypoints must be a non-empty array") if {
    not input.waypoints
}

_deny("waypoints must be a non-empty array") if {
    count(input.waypoints) == 0
}

# ── Altitude limit ────────────────────────────────────────────────────────────

_default_max_altitude := 122.0  # 400ft FAA default

_max_altitude := input.context.max_altitude_m if {
    input.context.max_altitude_m
} else := _default_max_altitude

_deny(reason) if {
    some wp in input.waypoints
    wp.altitude_m > _max_altitude
    reason := sprintf(
        "Waypoint altitude %.1fm exceeds maximum allowed %.1fm",
        [wp.altitude_m, _max_altitude]
    )
}

_deny("Waypoint altitude below minimum safe altitude (10m)") if {
    some wp in input.waypoints
    wp.altitude_m < 10.0
}

# ── Exclusion zone check ──────────────────────────────────────────────────────
# Deny if any waypoint falls inside an exclusion geofence.

_deny(reason) if {
    some wp in input.waypoints
    some gf in input.context.geofences
    gf.type == "exclusion"
    wp.latitude >= gf.min_lat
    wp.latitude <= gf.max_lat
    wp.longitude >= gf.min_lon
    wp.longitude <= gf.max_lon
    reason := sprintf(
        "Waypoint (%.5f, %.5f) is inside exclusion zone %v",
        [wp.latitude, wp.longitude, gf.geofence_id]
    )
}

# ── Inclusion zone check ──────────────────────────────────────────────────────
# If inclusion geofences are defined, ALL waypoints must be inside at least one.

_inclusion_geofences := [gf | gf := input.context.geofences[_]; gf.type == "inclusion"]

_waypoint_in_any_inclusion(wp) if {
    some gf in _inclusion_geofences
    wp.latitude >= gf.min_lat
    wp.latitude <= gf.max_lat
    wp.longitude >= gf.min_lon
    wp.longitude <= gf.max_lon
}

_deny(reason) if {
    count(_inclusion_geofences) > 0
    some wp in input.waypoints
    not _waypoint_in_any_inclusion(wp)
    reason := sprintf(
        "Waypoint (%.5f, %.5f) is outside all authorized operating zones",
        [wp.latitude, wp.longitude]
    )
}

# ── Coordinate sanity check ───────────────────────────────────────────────────

_deny(reason) if {
    some wp in input.waypoints
    wp.latitude < -90.0
    reason := sprintf("Invalid latitude: %.5f", [wp.latitude])
}

_deny(reason) if {
    some wp in input.waypoints
    wp.latitude > 90.0
    reason := sprintf("Invalid latitude: %.5f", [wp.latitude])
}

_deny(reason) if {
    some wp in input.waypoints
    wp.longitude < -180.0
    reason := sprintf("Invalid longitude: %.5f", [wp.longitude])
}

_deny(reason) if {
    some wp in input.waypoints
    wp.longitude > 180.0
    reason := sprintf("Invalid longitude: %.5f", [wp.longitude])
}
