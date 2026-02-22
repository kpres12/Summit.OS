package policy

import future.keywords.if
import future.keywords.in

default allow := true

# Deny overrides: if any deny rule fires, allow becomes false
allow if {
  not deny
}

# --- Risk / Approval rules ---

# CRITICAL tasks outside geofence are denied
deny if {
  input.risk_level == "CRITICAL"
  not input.in_geofence
}

# HIGH-risk tasks require supervisor approval
deny if {
  input.risk_level == "HIGH"
  not input.approved
}

# --- Geofence rules ---

# Deny any dispatch if explicitly flagged as outside geofence
deny if {
  input.action == "dispatch"
  input.in_geofence == false
}

# --- Altitude rules ---

# Deny if requested altitude exceeds org max (default 400ft / ~122m)
deny if {
  input.mission.planning_params.altitude
  input.mission.planning_params.altitude > input.context.max_altitude_m
}

# Deny if altitude below minimum safe altitude
deny if {
  input.mission.planning_params.altitude
  input.mission.planning_params.altitude < 10
}

# --- Weather rules ---

# Deny if wind speed exceeds safe limit (m/s)
deny if {
  input.context.wind_speed_ms
  input.context.wind_speed_ms > 15
}

# Deny if visibility below minimum (meters)
deny if {
  input.context.visibility_m
  input.context.visibility_m < 500
}

# --- Collect denial reasons for UI feedback ---

deny_reasons["Mission area is outside authorized geofence"] if {
  input.action == "dispatch"
  input.in_geofence == false
}

deny_reasons["HIGH-risk task requires supervisor approval"] if {
  input.risk_level == "HIGH"
  not input.approved
}

deny_reasons["CRITICAL task outside geofence boundary"] if {
  input.risk_level == "CRITICAL"
  not input.in_geofence
}

deny_reasons["Altitude exceeds maximum allowed"] if {
  input.mission.planning_params.altitude
  input.mission.planning_params.altitude > input.context.max_altitude_m
}

deny_reasons["Altitude below minimum safe altitude (10m)"] if {
  input.mission.planning_params.altitude
  input.mission.planning_params.altitude < 10
}

deny_reasons["Wind speed exceeds safe operating limit (15 m/s)"] if {
  input.context.wind_speed_ms
  input.context.wind_speed_ms > 15
}

deny_reasons["Visibility below minimum (500m)"] if {
  input.context.visibility_m
  input.context.visibility_m < 500
}
