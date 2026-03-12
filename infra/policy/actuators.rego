package policy.actuators

import future.keywords.if
import future.keywords.in

# ──────────────────────────────────────────────────────────────────────────────
# Summit.OS Actuator Safety Policy
#
# This is the pre-flight check layer for ALL physical actuator commands.
# Before Summit.OS sends any command to physical hardware — closing a valve,
# triggering a relay, moving a motor — this policy is evaluated.
#
# If it returns allow=false, the command is REJECTED before it reaches the wire.
# Every denial is audit-logged by the OPA client.
#
# Input schema (sent by tasking service before command dispatch):
#   {
#     "command": {
#       "type": "set_valve" | "set_relay" | "set_motor_speed" | "emergency_stop" | ...,
#       "target_id": "modbus-pumping-station-01-main-valve",
#       "parameters": { ... command-specific fields ... }
#     },
#     "world_state": {
#       "entities": {
#         "<entity_id>": { "state": "ACTIVE", "metadata": { "value": "...", "unit": "..." } }
#       }
#     },
#     "org_id": "acme-corp",
#     "requested_by": "ai-agent" | "operator" | "mission-planner",
#     "context": {
#       "time": "2026-03-12T14:00:00Z",
#       "mission_id": "...",
#       "approved": true | false
#     }
#   }
# ──────────────────────────────────────────────────────────────────────────────

default allow := false

# Allow only if no deny rules fire
allow if {
    count(deny_reasons) == 0
}

# ── Collect all denial reasons ────────────────────────────────────────────────

deny_reasons[reason] {
    some reason
    _deny(reason)
}

# ── Structural validation ─────────────────────────────────────────────────────

_deny("command.type is required") if {
    not input.command.type
}

_deny("command.target_id is required") if {
    not input.command.target_id
}

_deny("org_id is required") if {
    not input.org_id
}

# ── AI agent commands require approval ───────────────────────────────────────
# An AI agent can SUGGEST an actuator command, but a human operator must
# approve it before it executes on physical hardware.

_deny("AI-initiated actuator commands require human approval") if {
    input.requested_by == "ai-agent"
    not input.context.approved
}

# ── Valve commands: pressure safety check ────────────────────────────────────
# Closing a valve when downstream pressure is already high is dangerous.
# This rule prevents an AI (or human) from creating a pressure surge.

_deny("Valve closure blocked: downstream pressure is CRITICAL") if {
    input.command.type == "set_valve"
    input.command.parameters.state == "closed"
    some entity_id
    entity := input.world_state.entities[entity_id]
    entity.state == "CRITICAL"
    entity.metadata.protocol == "modbus"
    contains(entity.metadata.unit, "PSI")
}

_deny("Valve closure blocked: downstream pressure is WARNING — requires approval") if {
    input.command.type == "set_valve"
    input.command.parameters.state == "closed"
    not input.context.approved
    some entity_id
    entity := input.world_state.entities[entity_id]
    entity.state == "WARNING"
    entity.metadata.protocol == "modbus"
    contains(entity.metadata.unit, "PSI")
}

# ── Motor/pump commands: temperature safety check ────────────────────────────
# Do not start or increase motor speed when motor temperature is critical.

_deny("Motor command blocked: motor temperature is CRITICAL") if {
    input.command.type in {"set_motor_speed", "start_pump"}
    some entity_id
    entity := input.world_state.entities[entity_id]
    entity.state == "CRITICAL"
    entity.metadata.class_label == "temperature_sensor"
}

# ── Emergency stop: always allow, no restrictions ────────────────────────────
# Emergency stop bypasses all other rules. It must never be blocked.

allow if {
    input.command.type == "emergency_stop"
}

# ── Unknown command types: deny by default ────────────────────────────────────

_known_command_types := {
    "set_valve",
    "set_relay",
    "set_motor_speed",
    "start_pump",
    "stop_pump",
    "emergency_stop",
    "set_setpoint",
    "reset_alarm",
}

_deny(reason) if {
    not input.command.type in _known_command_types
    reason := sprintf("Unknown command type: %v", [input.command.type])
}

# ── Rate limiting: same target cannot be commanded more than once per second ──
# (Requires the tasking service to inject last_command_ts into context)

_deny("Command rate limit exceeded for target") if {
    input.context.last_command_ts
    input.context.time
    # If last command was within 1 second, deny
    # (OPA doesn't do time arithmetic natively; tasking service enforces this
    #  by setting a boolean flag in context)
    input.context.rate_limited == true
}
