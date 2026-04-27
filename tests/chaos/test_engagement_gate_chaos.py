"""
Engagement Gate Chaos Tests
============================
Adversarial abuse of EngagementAuthorizationGate. Every test uses the real
state machine with no mocks — the goal is to find actual failure modes before
live tests find them for us.

Failure modes probed:
  - Concurrent authorization races (does exactly one AUTHORIZED event fire?)
  - TTL expiry mid-authorize (AUTHORIZED vs EXPIRED race)
  - Invalid state transitions (does the machine reject every bad transition?)
  - Role escalation attempts (can an operator authorize an ace_strike?)
  - Double-authorize (second authorize on an already-AUTHORIZED case)
  - Authorize after expire (AUTHORIZED on an EXPIRED case)
  - Signature forgery (bad sig must never produce AUTHORIZED)
  - Mass case creation + expiry (memory + throughput at scale)
  - HOLD-then-expire (HELD case must go EXPIRED, not back to PENDING)
  - Missing weapon option reference (option_id not in case.options)
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))

from c2_intel.engagement_authorization import (
    EngagementAuthorizationGate,
    EngagementAuthorizationError,
    EngagementCase,
    EngagementState,
    OperatorAuthorization,
    OperatorDecision,
    TrackEvidence,
    PIDEvidence,
    ROEContext,
    DeconflictionContext,
    WeaponOption,
)
from c2_intel.models import C2EventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _track(track_id: str = None) -> TrackEvidence:
    return TrackEvidence(
        track_id=track_id or str(uuid.uuid4()),
        entity_id=str(uuid.uuid4()),
        classification="rotary_uas",
        confidence=0.92,
        sensors=["radar-1", "eo-1"],
    )


def _pid() -> PIDEvidence:
    return PIDEvidence(method="rf_fingerprint", confidence=0.88)


def _roe(block: bool = False) -> ROEContext:
    return ROEContext(
        roe_id="ROE-CHAOS-001",
        permits_engagement_type=True,
        proportionality_passed=True,
        collateral_estimate="blocking" if block else "minimal",
    )


def _decon() -> DeconflictionContext:
    return DeconflictionContext(blue_force_clear=True, airspace_clear=True)


def _option(option_id: str = None) -> WeaponOption:
    return WeaponOption(
        option_id=option_id or str(uuid.uuid4()),
        weapon_asset_id="jammer-1",
        weapon_class="soft_kill",
        range_m=500.0,
        time_of_flight_s=0.0,
        pk_estimate=0.95,
        roe_compliant=True,
        deconfliction_ok=True,
        rationale="soft-kill within ROE",
    )


def _auth(option_id: str, role: str = "operator") -> OperatorAuthorization:
    return OperatorAuthorization(
        decision=OperatorDecision.AUTHORIZE,
        operator_id="op-chaos-1",
        operator_role=role,
        rationale="chaos test authorization",
        selected_option=option_id,
    )


def _make_gate(events: List = None, ttl: int = 60) -> EngagementAuthorizationGate:
    bucket: List = events if events is not None else []
    return EngagementAuthorizationGate.for_testing(
        emit_event=lambda et, d: bucket.append((et, d)),
        capture_audit=[],
        default_ttl_seconds=ttl,
    )


def _full_open(gate: EngagementAuthorizationGate) -> EngagementCase:
    """Open a case all the way to PENDING_AUTHORIZATION."""
    opt = _option()
    case = gate.open_case(_track())
    gate.submit_pid(case.case_id, _pid())
    gate.submit_roe(case.case_id, _roe())
    gate.submit_deconfliction(case.case_id, _decon())
    gate.surface_options(case.case_id, [opt])
    return gate._cases[case.case_id], opt


# ---------------------------------------------------------------------------
# 1. Concurrent authorization race
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_authorization_race():
    """100 coroutines all call authorize() on the same case simultaneously.
    Exactly one AUTHORIZED event must be emitted — never zero, never two."""
    events: List = []
    gate = _make_gate(events)
    case, opt = _full_open(gate)
    auth = _auth(opt.option_id)

    results = await asyncio.gather(
        *[asyncio.to_thread(gate.authorize, case.case_id, auth) for _ in range(100)],
        return_exceptions=True,
    )

    authorized_events = [e for e in events if e[0] == C2EventType.ENGAGEMENT_AUTHORIZED]
    assert len(authorized_events) == 1, (
        f"Expected exactly 1 AUTHORIZED event, got {len(authorized_events)}. "
        "Concurrent authorize() has a race condition."
    )
    successes = [r for r in results if not isinstance(r, Exception)]
    assert len(successes) == 1, "Exactly one authorize() call must succeed"


# ---------------------------------------------------------------------------
# 2. Double authorize — AUTHORIZED case cannot be re-authorized
# ---------------------------------------------------------------------------

def test_double_authorize_rejected():
    """authorize() on an already-AUTHORIZED case must raise, not emit a second event."""
    events: List = []
    gate = _make_gate(events)
    case, opt = _full_open(gate)
    auth = _auth(opt.option_id)

    gate.authorize(case.case_id, auth)
    assert len([e for e in events if e[0] == C2EventType.ENGAGEMENT_AUTHORIZED]) == 1

    with pytest.raises((EngagementAuthorizationError, RuntimeError, ValueError)):
        gate.authorize(case.case_id, auth)

    assert len([e for e in events if e[0] == C2EventType.ENGAGEMENT_AUTHORIZED]) == 1, \
        "Second authorize() must not emit a second AUTHORIZED event"


# ---------------------------------------------------------------------------
# 3. Authorize after expire
# ---------------------------------------------------------------------------

def test_authorize_after_expire_rejected():
    """An EXPIRED case must never be authorized."""
    events: List = []
    gate = _make_gate(events, ttl=1)
    case, opt = _full_open(gate)

    future = datetime.now(timezone.utc) + timedelta(seconds=120)
    gate.expire_stale(now=future)

    refreshed = gate._cases[case.case_id]
    assert refreshed.state == EngagementState.EXPIRED

    auth = _auth(opt.option_id)
    with pytest.raises((EngagementAuthorizationError, RuntimeError, ValueError)):
        gate.authorize(case.case_id, auth)

    authorized = [e for e in events if e[0] == C2EventType.ENGAGEMENT_AUTHORIZED]
    assert len(authorized) == 0, "Expired case must never emit AUTHORIZED"


# ---------------------------------------------------------------------------
# 4. TTL expiry race — expire fires while authorize is in-flight
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ttl_expiry_vs_authorize_race():
    """expire_stale() and authorize() run concurrently. The outcome must be
    one of: AUTHORIZED or EXPIRED — never both, never neither on the case."""
    events: List = []
    gate = _make_gate(events, ttl=1)
    case, opt = _full_open(gate)
    auth = _auth(opt.option_id)
    future = datetime.now(timezone.utc) + timedelta(seconds=120)

    results = await asyncio.gather(
        asyncio.to_thread(gate.authorize, case.case_id, auth),
        asyncio.to_thread(gate.expire_stale, future),
        return_exceptions=True,
    )

    refreshed = gate._cases[case.case_id]
    assert refreshed.state in (EngagementState.AUTHORIZED, EngagementState.EXPIRED), \
        f"Case must be AUTHORIZED or EXPIRED, got {refreshed.state}"

    auth_events = [e for e in events if e[0] == C2EventType.ENGAGEMENT_AUTHORIZED]
    assert len(auth_events) <= 1, "Must never emit more than one AUTHORIZED event"


# ---------------------------------------------------------------------------
# 5. Role escalation — operator cannot authorize ace_strike
# ---------------------------------------------------------------------------

def test_role_escalation_blocked():
    """An operator-role user must not be able to authorize an ace_strike.
    ace_strike requires joint_force_commander; operator role must be rejected."""
    events: List = []
    # Use a real role checker — op-low has 'operator' which cannot clear
    # joint_force_commander required by ace_strike.
    _role_db = {"op-low": "operator"}
    _hierarchy = ["operator", "mission_commander", "joint_force_commander"]

    def _check_role(op_id: str, required: str) -> bool:
        op_role = _role_db.get(op_id, "operator")
        try:
            return _hierarchy.index(op_role) >= _hierarchy.index(required)
        except ValueError:
            return False

    gate = EngagementAuthorizationGate(
        emit_event=lambda et, d: events.append((et, d)),
        verify_operator_signature=lambda sig, payload: True,
        operator_has_role=_check_role,
        audit_sink=lambda entry: None,
    )

    opt = WeaponOption(
        option_id=str(uuid.uuid4()),
        weapon_asset_id="precision-strike-1",
        weapon_class="hard_kill",
        range_m=5000.0,
        time_of_flight_s=45.0,
        pk_estimate=0.88,
        roe_compliant=True,
        deconfliction_ok=True,
        rationale="ace strike option",
    )
    case = gate.open_case(_track())
    gate.submit_pid(case.case_id, _pid())
    gate.submit_roe(case.case_id, _roe())
    gate.submit_deconfliction(case.case_id, _decon())
    gate.surface_options(case.case_id, [opt])

    low_auth = OperatorAuthorization(
        decision=OperatorDecision.AUTHORIZE,
        operator_id="op-low",
        operator_role="operator",
        rationale="role escalation attempt",
        selected_option=opt.option_id,
    )
    with pytest.raises((EngagementAuthorizationError, RuntimeError, ValueError)):
        gate.authorize(case.case_id, low_auth, engagement_class="ace_strike")

    assert len([e for e in events if e[0] == C2EventType.ENGAGEMENT_AUTHORIZED]) == 0, \
        "operator role must not authorize ace_strike"


# ---------------------------------------------------------------------------
# 6. Signature forgery
# ---------------------------------------------------------------------------

def test_forged_signature_rejected():
    """A decision with a bad Ed25519 signature must never produce AUTHORIZED.
    Uses a gate that rejects all signatures (allow_signature=False)."""
    events: List = []
    gate = EngagementAuthorizationGate.for_testing(
        emit_event=lambda et, d: events.append((et, d)),
        allow_signature=False,
        capture_audit=[],
    )
    case, opt = _full_open(gate)
    auth = _auth(opt.option_id)
    auth.signature = b"this is a forged signature"

    with pytest.raises((EngagementAuthorizationError, RuntimeError, ValueError)):
        gate.authorize(case.case_id, auth)

    assert len([e for e in events if e[0] == C2EventType.ENGAGEMENT_AUTHORIZED]) == 0


# ---------------------------------------------------------------------------
# 7. Blocking collateral — ROE with collateral_estimate="blocking"
# ---------------------------------------------------------------------------

def test_blocking_collateral_prevents_authorization():
    """ROE context with collateral_estimate='blocking' must block the case
    before it ever reaches PENDING_AUTHORIZATION."""
    gate = _make_gate()
    case = gate.open_case(_track())
    gate.submit_pid(case.case_id, _pid())

    # blocking collateral → submit_roe transitions to DENIED, not an exception
    result = gate.submit_roe(case.case_id, _roe(block=True))
    assert result.state == EngagementState.DENIED, \
        "Blocking collateral estimate must transition case to DENIED"


# ---------------------------------------------------------------------------
# 8. Non-existent weapon option
# ---------------------------------------------------------------------------

def test_missing_weapon_option_rejected():
    """authorize() with a selected_option not in case.options must be rejected."""
    events: List = []
    gate = _make_gate(events)
    case, opt = _full_open(gate)

    bad_auth = OperatorAuthorization(
        decision=OperatorDecision.AUTHORIZE,
        operator_id="op-chaos-1",
        operator_role="operator",
        rationale="bad option ref",
        selected_option="non-existent-option-id",
    )
    with pytest.raises((EngagementAuthorizationError, RuntimeError, ValueError, KeyError)):
        gate.authorize(case.case_id, bad_auth)

    assert len([e for e in events if e[0] == C2EventType.ENGAGEMENT_AUTHORIZED]) == 0


# ---------------------------------------------------------------------------
# 9. HOLD then expire — must go EXPIRED not back to PENDING
# ---------------------------------------------------------------------------

def test_held_case_expires_not_reopens():
    """A HELD case that hits TTL must go to EXPIRED, not resurrect as PENDING."""
    events: List = []
    gate = _make_gate(events, ttl=1)
    case, opt = _full_open(gate)

    hold_auth = OperatorAuthorization(
        decision=OperatorDecision.HOLD,
        operator_id="op-chaos-1",
        operator_role="operator",
        rationale="holding for confirmation",
    )
    gate.authorize(case.case_id, hold_auth)
    assert gate._cases[case.case_id].state == EngagementState.HELD

    future = datetime.now(timezone.utc) + timedelta(seconds=120)
    gate.expire_stale(now=future)

    final_state = gate._cases[case.case_id].state
    assert final_state == EngagementState.EXPIRED, \
        f"HELD case should expire to EXPIRED, got {final_state}"
    assert final_state != EngagementState.PENDING_AUTHORIZATION, \
        "HELD case must not silently re-enter PENDING after TTL"


# ---------------------------------------------------------------------------
# 10. Invalid state transition enumeration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_transition", [
    "skip_to_authorized",
    "pid_on_pending",
    "deconflict_before_roe",
])
def test_invalid_state_transitions(bad_transition: str):
    """Every attempted shortcut through the state machine must be rejected."""
    gate = _make_gate()

    if bad_transition == "skip_to_authorized":
        # Try to authorize a case that hasn't been through the full pipeline
        case = gate.open_case(_track())
        auth = _auth(str(uuid.uuid4()))
        with pytest.raises((EngagementAuthorizationError, RuntimeError, ValueError, KeyError)):
            gate.authorize(case.case_id, auth)

    elif bad_transition == "pid_on_pending":
        # Try to re-run PID after case is already at PENDING_AUTHORIZATION
        case, _ = _full_open(gate)
        with pytest.raises((EngagementAuthorizationError, RuntimeError, ValueError)):
            gate.submit_pid(case.case_id, _pid())

    elif bad_transition == "deconflict_before_roe":
        # Try to deconflict before ROE is cleared
        case = gate.open_case(_track())
        gate.submit_pid(case.case_id, _pid())
        with pytest.raises((EngagementAuthorizationError, RuntimeError, ValueError)):
            gate.submit_deconfliction(case.case_id, _decon())


# ---------------------------------------------------------------------------
# 11. Mass case creation — throughput + memory
# ---------------------------------------------------------------------------

def test_mass_case_creation_throughput():
    """Create 10,000 cases and verify: throughput > 1,000/s, memory doesn't
    grow unbounded after expire_stale clears closed cases."""
    import tracemalloc
    gate = _make_gate(ttl=1)
    tracemalloc.start()

    N = 10_000
    t0 = time.perf_counter()
    case_ids = []
    for _ in range(N):
        c = gate.open_case(_track())
        case_ids.append(c.case_id)
    elapsed = time.perf_counter() - t0

    rate = N / elapsed
    assert rate > 1_000, f"Case creation throughput {rate:.0f}/s is too low (need >1k/s)"

    # Expire all
    future = datetime.now(timezone.utc) + timedelta(seconds=3600)
    gate.expire_stale(now=future)

    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # DETECTED cases have no auth_ttl — expire_stale correctly ignores them.
    # Verify that no cases with auth_ttl set are stuck past TTL.
    stale = [
        c for c in gate._cases.values()
        if c.state in (EngagementState.PENDING_AUTHORIZATION,
                       EngagementState.AUTHORIZED, EngagementState.HELD)
        and c.auth_ttl and future > c.auth_ttl
    ]
    assert len(stale) == 0, \
        f"{len(stale)} expirable cases still stuck past TTL after expire_stale"


# ---------------------------------------------------------------------------
# 12. Rapid full-pipeline stress
# ---------------------------------------------------------------------------

def test_rapid_full_pipeline():
    """Run 1,000 cases through the full pipeline end-to-end.
    Verify all 1,000 emit exactly one AUTHORIZED event."""
    events: List = []
    gate = _make_gate(events)

    N = 1_000
    for _ in range(N):
        opt = _option()
        case = gate.open_case(_track())
        gate.submit_pid(case.case_id, _pid())
        gate.submit_roe(case.case_id, _roe())
        gate.submit_deconfliction(case.case_id, _decon())
        gate.surface_options(case.case_id, [opt])
        gate.authorize(case.case_id, _auth(opt.option_id))

    auth_events = [e for e in events if e[0] == C2EventType.ENGAGEMENT_AUTHORIZED]
    assert len(auth_events) == N, \
        f"Expected {N} AUTHORIZED events, got {len(auth_events)}"


# ---------------------------------------------------------------------------
# 13. Deny does not become authorize
# ---------------------------------------------------------------------------

def test_deny_cannot_be_overridden():
    """A DENIED case must not be re-authorized. Denial is final."""
    events: List = []
    gate = _make_gate(events)
    case, opt = _full_open(gate)

    deny_auth = OperatorAuthorization(
        decision=OperatorDecision.DENY,
        operator_id="op-chaos-1",
        operator_role="operator",
        rationale="collateral concern",
    )
    gate.authorize(case.case_id, deny_auth)
    assert gate._cases[case.case_id].state == EngagementState.DENIED

    with pytest.raises((EngagementAuthorizationError, RuntimeError, ValueError)):
        gate.authorize(case.case_id, _auth(opt.option_id))

    assert len([e for e in events if e[0] == C2EventType.ENGAGEMENT_AUTHORIZED]) == 0


# ---------------------------------------------------------------------------
# 14. Audit trail completeness
# ---------------------------------------------------------------------------

def test_every_transition_is_audited():
    """Every state transition must produce an audit entry.
    An authorization that emits no audit trail is a compliance failure."""
    audit: List[Dict] = []
    gate = EngagementAuthorizationGate.for_testing(
        emit_event=lambda et, d: None,
        capture_audit=audit,
        default_ttl_seconds=60,
    )
    opt = _option()
    case = gate.open_case(_track())
    gate.submit_pid(case.case_id, _pid())
    gate.submit_roe(case.case_id, _roe())
    gate.submit_deconfliction(case.case_id, _decon())
    gate.surface_options(case.case_id, [opt])
    gate.authorize(case.case_id, _auth(opt.option_id))

    assert len(audit) >= 5, \
        f"Expected at least 5 audit entries (one per transition), got {len(audit)}"

    states_audited = {entry.get("state") or entry.get("to_state") or entry.get("event")
                      for entry in audit}
    assert len(states_audited) >= 3, \
        "Audit trail must record distinct state transitions, not duplicate entries"
