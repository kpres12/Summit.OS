package policy

default allow = true

# Example: deny high/critical tasks outside geofence or without supervisor approval
allow {
  input.action == "dispatch"
  not deny
}

deny {
  input.risk_level == "CRITICAL"
  not input.in_geofence
}

deny {
  input.risk_level == "HIGH"
  not input.approved
}
