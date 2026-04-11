package policy.missions

# Default allow
default allow = true

# Collect reasons
deny_reasons[reason] {
  some reason
  deny_reason(reason)
}

# Deny if mission area missing

deny_reason("mission.area missing") {
  not input.mission
} {
  input.mission
  not input.mission.area
}

# Deny if no objectives provided

deny_reason("mission.objectives missing") {
  input.mission
  not input.mission.objectives
}

# Deny unknown objectives

deny_reason(reason) {
  some obj
  obj := input.mission.objectives[_]
  not allowed_objective(obj)
  reason := sprintf("unknown objective: %v", [obj])
}

allowed_objective(obj) {
  obj == "surveillance"
} else {
  obj == "mapping"
} else {
  obj == "containment"
} else {
  obj == "search_and_rescue"
} else {
  obj == "inspection"
} else {
  obj == "delivery"
} else {
  obj == "relay"
} else {
  obj == "monitoring"
} else {
  obj == "survey"
} else {
  obj == "photography"
} else {
  obj == "disaster_response"
} else {
  obj == "wildfire_ops"
}

# Allow if no denies
allow {
  count(deny_reasons) == 0
}
