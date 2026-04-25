"""
CANVAS Authority Delegation DSL — pure-Python evaluator + bundle signer.

The Rego policy at `infra/policy/canvas/authority_delegation.rego` is the
canonical source of truth. This module provides a faster pure-Python
evaluator (matching Rego semantics exactly) for use inside the workflow
simulator and the engagement-authorization gate's pre-screen.

For production deployment, OPA evaluates the signed Rego file. The DSL
here is for in-process simulation + decision support only — the OPA
result is what counts.

The DSL is also signed with Ed25519 via packages/policy/signer.py before
distribution to TA2 nodes, guaranteeing every node is running the same
intent the COCOM authored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("canvas.authority_dsl")


_ROLE_RANK = {
    "operator":              1,
    "mission_commander":     2,
    "joint_force_commander": 3,
    "combatant_commander":   4,
}


_REQUIRED_ROLE = {
    ("counter_uas",                "soft_kill"):  "operator",
    ("counter_uas",                "hard_kill"):  "mission_commander",
    ("force_protection_perimeter", "soft_kill"):  "operator",
    ("force_protection_perimeter", "hard_kill"):  "mission_commander",
    ("base_defense",               "any"):        "mission_commander",
    ("ace_strike",                 "any"):        "joint_force_commander",
    ("default",                    "any"):        "mission_commander",
}


def required_role(engagement_class: str, weapon_class: str) -> str:
    return (
        _REQUIRED_ROLE.get((engagement_class, weapon_class))
        or _REQUIRED_ROLE.get((engagement_class, "any"))
        or _REQUIRED_ROLE[("default", "any")]
    )


@dataclass
class CommsState:
    uplink_seconds_since: int = 0
    pace_active: str = "primary"        # primary / alternate / contingency / emergency
    intent_age_seconds: int = 0


@dataclass
class CommanderIntent:
    id: str
    permits: list[str] = field(default_factory=list)
    delegated_thresholds_uplink_seconds: int = 60
    delegated_thresholds_intent_age_seconds: int = 600
    signed_by: Optional[str] = None      # COCOM operator id
    signature: Optional[bytes] = None    # Ed25519 sig over (id, permits, thresholds)


@dataclass
class DecisionAuthority:
    """Result of evaluating the DSL on a (decision, comms_state, intent) tuple."""
    allowed: bool
    reason: str
    pathway: str       # "baseline" | "conditional_delegation" | "denied"
    delegated_from: Optional[str] = None
    deny_codes: list[str] = field(default_factory=list)


def evaluate_authority(*, engagement_class: str, weapon_class: str,
                       operator_role: str, node_tier: str,
                       signature_verified: bool,
                       comms: CommsState,
                       intent: CommanderIntent) -> DecisionAuthority:
    """Pure-Python evaluator. Mirrors authority_delegation.rego semantics."""
    # 1. Hard deny: missing signature
    if not signature_verified:
        return DecisionAuthority(
            allowed=False, reason="decision signature not verified by gate",
            pathway="denied", deny_codes=["signature_missing"])

    # 2. Hard deny: engagement outside intent
    permits = set(intent.permits)
    intent_keys = {
        f"{engagement_class}:{weapon_class}",
        f"{engagement_class}:any",
    }
    if not (permits & intent_keys):
        return DecisionAuthority(
            allowed=False,
            reason=f"engagement {engagement_class}:{weapon_class} not permitted "
                   f"under intent {intent.id}",
            pathway="denied", deny_codes=["intent_does_not_permit"])

    # 3. Hard deny: stale intent
    if comms.intent_age_seconds > intent.delegated_thresholds_intent_age_seconds:
        return DecisionAuthority(
            allowed=False,
            reason=f"intent age {comms.intent_age_seconds}s exceeds max "
                   f"{intent.delegated_thresholds_intent_age_seconds}s",
            pathway="denied", deny_codes=["intent_stale"])

    needed = required_role(engagement_class, weapon_class)
    rank_have = _ROLE_RANK.get(operator_role, 0)
    rank_need = _ROLE_RANK.get(needed, 0)

    # 4. Baseline authority — operator role meets the engagement requirement
    #    (regardless of tier; doctrine grants role-based authority at any node)
    if rank_have >= rank_need:
        return DecisionAuthority(
            allowed=True,
            reason=f"baseline authority: role={operator_role} meets required "
                   f"{needed} at {node_tier}",
            pathway="baseline")

    # 5. Conditional delegation — operator one rank below required, but the
    #    uplink to next-up tier has been degraded longer than the
    #    delegation threshold the COCOM signed into the intent. The
    #    lower-tier operator temporarily inherits authority so distributed
    #    operations continue under contested comms.
    if rank_have == rank_need - 1 and node_tier in {"fob", "tactical"}:
        if comms.uplink_seconds_since > intent.delegated_thresholds_uplink_seconds:
            return DecisionAuthority(
                allowed=True,
                reason=f"conditional delegation active: uplink_age="
                       f"{comms.uplink_seconds_since}s > threshold="
                       f"{intent.delegated_thresholds_uplink_seconds}s; "
                       f"role {operator_role} acquired {needed} authority",
                pathway="conditional_delegation",
                delegated_from=needed)
        return DecisionAuthority(
            allowed=False,
            reason=f"role {operator_role} below required {needed}; uplink "
                   f"{comms.uplink_seconds_since}s under delegation threshold "
                   f"{intent.delegated_thresholds_uplink_seconds}s",
            pathway="denied", deny_codes=["role_insufficient", "uplink_healthy"])

    return DecisionAuthority(
        allowed=False,
        reason=f"role {operator_role} insufficient for {needed} (no delegation "
               f"path available at tier {node_tier})",
        pathway="denied", deny_codes=["role_insufficient"])
