package policy.canvas.authority

# ──────────────────────────────────────────────────────────────────────────────
# CANVAS TA1 — Conditional-Authority Delegation Policy
# Heli.OS substrate for AFRL/RIK BAA FA8750-24-S-7003 (CANVAS)
#
# Models the Air Force ACE concept: distributed C2 with conditional authority
# delegation to lower-tier nodes when comms degrade. A COCOM-level operator
# edits this policy in the virtual C2 layer (TA1), simulates its effects
# against the operational ontology, signs it with Ed25519, and pushes the
# signed bundle into the decentralized C2 framework (TA2) where individual
# nodes evaluate it locally to determine which actions they're authorized
# to take given their own observed comms state and current commander intent.
#
# Input schema (sent by EngagementAuthorizationGate or a workflow step):
#   {
#     "decision": {
#       "engagement_class":    "counter_uas" | "force_protection" | "ace_strike" | ...,
#       "weapon_class":        "soft_kill" | "hard_kill" | "non_lethal" | ...,
#       "operator_id":         "...",
#       "operator_role":       "operator" | "mission_commander" | "joint_force_commander",
#       "node_tier":           "aoc" | "coc" | "wing" | "fob" | "tactical",
#       "signature_verified":  true | false,
#     },
#     "comms_state": {
#       "uplink_seconds_since": 5,                  // age of last contact w/ next-up tier
#       "pace_active":          "primary" | "alternate" | "contingency" | "emergency",
#       "intent_age_seconds":   120                 // age of last commander-intent push
#     },
#     "intent": {
#       "id":                   "OPLAN-2026-X-INTENT-007",
#       "permits":              ["counter_uas:soft_kill", "force_protection:soft_kill"],
#       "delegated_thresholds": {
#         "uplink_seconds_since": 90,                // delegate if uplink lost > 90s
#         "intent_age_seconds":   600                // intent must be < 10 min old
#       }
#     }
#   }
#
# This file is signed via packages/policy/signer.py before deployment.
# ──────────────────────────────────────────────────────────────────────────────

import future.keywords.if
import future.keywords.in

default allow := false
default reason := "no rule matched"

# Allow only if no deny rule fires AND at least one allow path is satisfied
allow if {
    count(deny_reasons) == 0
    some _ in {true |
        _baseline_authority
    } | _conditional_delegation_active
}

# ── Default deny: missing or invalid signature ─────────────────────────────────

deny_reasons[reason] {
    not input.decision.signature_verified
    reason := "decision signature not verified by gate"
}

# ── Default deny: engagement class outside current intent ──────────────────────

deny_reasons[reason] {
    ec := input.decision.engagement_class
    wc := input.decision.weapon_class
    permitted := input.intent.permits
    not _intent_permits(ec, wc, permitted)
    reason := sprintf("engagement %v:%v not permitted under intent %v",
                      [ec, wc, input.intent.id])
}

_intent_permits(ec, wc, permitted) if {
    fmt := sprintf("%v:%v", [ec, wc])
    fmt in permitted
}

_intent_permits(ec, _, permitted) if {
    fmt := sprintf("%v:any", [ec])
    fmt in permitted
}

# ── Default deny: stale intent ─────────────────────────────────────────────────

deny_reasons[reason] {
    age := input.comms_state.intent_age_seconds
    max_age := input.intent.delegated_thresholds.intent_age_seconds
    age > max_age
    reason := sprintf("intent age %vs exceeds max %vs", [age, max_age])
}

# ── Baseline authority — operator role meets the engagement class minimum ─────

_baseline_authority if {
    role := input.decision.operator_role
    ec := input.decision.engagement_class
    wc := input.decision.weapon_class
    needed := _required_role(ec, wc)
    _role_at_least(role, needed)
    # Top-tier authority always works
    input.decision.node_tier in {"aoc", "coc", "wing"}
}

# ── Conditional delegation — lower tier inherits authority when uplink degraded ──

_conditional_delegation_active if {
    role := input.decision.operator_role
    ec := input.decision.engagement_class
    wc := input.decision.weapon_class
    needed := _required_role(ec, wc)

    # Operator is one role-tier below the normal requirement
    _role_one_below(role, needed)

    # Uplink to next-up tier has been degraded long enough
    age := input.comms_state.uplink_seconds_since
    threshold := input.intent.delegated_thresholds.uplink_seconds_since
    age > threshold
}

# ── Role tiering ───────────────────────────────────────────────────────────────

_role_rank("operator")               := 1
_role_rank("mission_commander")      := 2
_role_rank("joint_force_commander")  := 3
_role_rank("combatant_commander")    := 4

_role_at_least(role, needed) if {
    _role_rank(role) >= _role_rank(needed)
}

_role_one_below(role, needed) if {
    _role_rank(role) == _role_rank(needed) - 1
}

# ── Engagement-class → minimum-role table (matches _ROLE_MATRIX in code) ──────

_required_role("counter_uas",                "soft_kill") := "operator"
_required_role("counter_uas",                "hard_kill") := "mission_commander"
_required_role("force_protection_perimeter", "soft_kill") := "operator"
_required_role("force_protection_perimeter", "hard_kill") := "mission_commander"
_required_role("base_defense",               _)           := "mission_commander"
_required_role("ace_strike",                 _)           := "joint_force_commander"
# default
_required_role(_, _) := "mission_commander"

# ── Reason output (informational) ──────────────────────────────────────────────

reason := sprintf("baseline authority: role=%v tier=%v",
                  [input.decision.operator_role, input.decision.node_tier]) if {
    _baseline_authority
}

reason := sprintf("conditional delegation active: uplink_age=%vs > threshold=%vs",
                  [input.comms_state.uplink_seconds_since,
                   input.intent.delegated_thresholds.uplink_seconds_since]) if {
    _conditional_delegation_active
    not _baseline_authority
}
