"""
Engagement Authorization Gate — full test suite.

Covers the load-bearing invariants for DoDD 3000.09 / LoAC compliance:

  - Construction refuses pass-through defaults (no fail-open by accident)
  - State machine cannot skip transitions (no DETECTED → AUTHORIZED)
  - PID confidence threshold enforced
  - ROE / proportionality / collateral / deconfliction deny paths fire
  - Operator role check refuses insufficient roles
  - Signature verification refuses tampered signatures
  - TTL on AUTHORIZED auto-emits ENGAGEMENT_DENIED on expiry
  - ENGAGEMENT_AUTHORIZED is only emitted from authorize()
  - Production wiring (RBAC + Ed25519 + chained-HMAC audit) end-to-end
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from packages.c2_intel.engagement_authorization import (   # noqa: E402
    DeconflictionContext,
    EngagementAuthorizationError,
    EngagementAuthorizationGate,
    EngagementState,
    OperatorAuthorization,
    OperatorDecision,
    PIDEvidence,
    ROEContext,
    TrackEvidence,
    WeaponOption,
    required_role,
)
from packages.c2_intel.engagement_wiring import (   # noqa: E402
    ChainedHMACAuditSink,
    build_production_gate,
    make_rbac_role_check,
)
from packages.c2_intel.models import C2EventType   # noqa: E402
from packages.security.rbac import RBACEngine   # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _track() -> TrackEvidence:
    return TrackEvidence(
        track_id="t-1", entity_id="e-1", classification="rotary_uas",
        confidence=0.92, sensors=["radar-1", "rf-1"],
        last_position={"lat": 34.5, "lon": -118.0, "alt_m": 200.0},
        last_seen=datetime.now(timezone.utc),
    )


def _good_pid() -> PIDEvidence:
    return PIDEvidence(method="visual_isr", confidence=0.85,
                       source_asset_id="isr-1")


def _good_roe() -> ROEContext:
    return ROEContext(roe_id="ROE-1", permits_engagement_type=True,
                      proportionality_passed=True, collateral_estimate="low")


def _good_deconf() -> DeconflictionContext:
    return DeconflictionContext(blue_force_clear=True, airspace_clear=True,
                                nearby_civilians_count=0)


def _good_options() -> list[WeaponOption]:
    return [WeaponOption(option_id="opt-1", weapon_asset_id="cuas-1",
                         weapon_class="soft_kill", range_m=1400.0,
                         time_of_flight_s=4.0, pk_estimate=0.85,
                         roe_compliant=True, deconfliction_ok=True,
                         rationale="ground-fixed cuas, in range")]


def _walk_to_pending(gate: EngagementAuthorizationGate):
    track = _track()
    case = gate.open_case(track)
    gate.submit_pid(case.case_id, _good_pid())
    gate.submit_roe(case.case_id, _good_roe())
    gate.submit_deconfliction(case.case_id, _good_deconf())
    gate.surface_options(case.case_id, _good_options())
    return case


# ---------------------------------------------------------------------------
# Construction safety
# ---------------------------------------------------------------------------


class TestConstructionSafety:
    def test_refuses_none_emit_event(self):
        with pytest.raises((ValueError, TypeError)):
            EngagementAuthorizationGate(
                emit_event=None,
                verify_operator_signature=lambda s, p: True,
                operator_has_role=lambda o, r: True,
                audit_sink=lambda e: None,
            )

    def test_refuses_none_verifier(self):
        with pytest.raises((ValueError, TypeError)):
            EngagementAuthorizationGate(
                emit_event=lambda et, p: None,
                verify_operator_signature=None,
                operator_has_role=lambda o, r: True,
                audit_sink=lambda e: None,
            )

    def test_refuses_none_role_check(self):
        with pytest.raises((ValueError, TypeError)):
            EngagementAuthorizationGate(
                emit_event=lambda et, p: None,
                verify_operator_signature=lambda s, p: True,
                operator_has_role=None,
                audit_sink=lambda e: None,
            )

    def test_refuses_none_audit_sink(self):
        with pytest.raises((ValueError, TypeError)):
            EngagementAuthorizationGate(
                emit_event=lambda et, p: None,
                verify_operator_signature=lambda s, p: True,
                operator_has_role=lambda o, r: True,
                audit_sink=None,
            )

    def test_refuses_non_callable(self):
        with pytest.raises((ValueError, TypeError)):
            EngagementAuthorizationGate(
                emit_event=lambda et, p: None,
                verify_operator_signature="not a function",  # type: ignore
                operator_has_role=lambda o, r: True,
                audit_sink=lambda e: None,
            )

    def test_for_testing_helper_works(self):
        gate = EngagementAuthorizationGate.for_testing()
        assert gate is not None
        case = gate.open_case(_track())
        assert case.state == EngagementState.DETECTED


# ---------------------------------------------------------------------------
# State machine integrity
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_cannot_skip_pid(self):
        gate = EngagementAuthorizationGate.for_testing()
        case = gate.open_case(_track())
        with pytest.raises(EngagementAuthorizationError):
            gate.submit_roe(case.case_id, _good_roe())  # skip PID

    def test_cannot_skip_roe(self):
        gate = EngagementAuthorizationGate.for_testing()
        case = gate.open_case(_track())
        gate.submit_pid(case.case_id, _good_pid())
        with pytest.raises(EngagementAuthorizationError):
            gate.submit_deconfliction(case.case_id, _good_deconf())  # skip ROE

    def test_cannot_skip_deconfliction(self):
        gate = EngagementAuthorizationGate.for_testing()
        case = gate.open_case(_track())
        gate.submit_pid(case.case_id, _good_pid())
        gate.submit_roe(case.case_id, _good_roe())
        with pytest.raises(EngagementAuthorizationError):
            gate.surface_options(case.case_id, _good_options())  # skip decon

    def test_cannot_authorize_without_options(self):
        gate = EngagementAuthorizationGate.for_testing()
        case = gate.open_case(_track())
        gate.submit_pid(case.case_id, _good_pid())
        gate.submit_roe(case.case_id, _good_roe())
        gate.submit_deconfliction(case.case_id, _good_deconf())
        with pytest.raises(EngagementAuthorizationError):
            gate.authorize(case.case_id, OperatorAuthorization(
                decision=OperatorDecision.AUTHORIZE,
                operator_id="op", operator_role="mission_commander",
                rationale="x", selected_option="opt-1", signature=b"sig"),
                engagement_class="counter_uas")

    def test_unknown_case_id(self):
        gate = EngagementAuthorizationGate.for_testing()
        with pytest.raises(EngagementAuthorizationError):
            gate.submit_pid("nonexistent", _good_pid())


# ---------------------------------------------------------------------------
# PID confidence threshold
# ---------------------------------------------------------------------------


class TestPIDThreshold:
    def test_low_confidence_pid_rejected(self):
        gate = EngagementAuthorizationGate.for_testing()
        case = gate.open_case(_track())
        with pytest.raises(EngagementAuthorizationError):
            gate.submit_pid(case.case_id,
                            PIDEvidence(method="visual_isr", confidence=0.5))

    def test_pid_at_threshold_accepted(self):
        gate = EngagementAuthorizationGate.for_testing()
        case = gate.open_case(_track())
        gate.submit_pid(case.case_id,
                        PIDEvidence(method="iff_squawk", confidence=0.71))
        assert gate.get_case(case.case_id).state == EngagementState.PID_CONFIRMED


# ---------------------------------------------------------------------------
# ROE / proportionality / collateral
# ---------------------------------------------------------------------------


class TestROEDenials:
    def test_roe_does_not_permit_engagement(self):
        gate = EngagementAuthorizationGate.for_testing()
        case = gate.open_case(_track())
        gate.submit_pid(case.case_id, _good_pid())
        gate.submit_roe(case.case_id, ROEContext(
            roe_id="ROE-1", permits_engagement_type=False,
            proportionality_passed=True, collateral_estimate="low"))
        assert gate.get_case(case.case_id).state == EngagementState.DENIED

    def test_proportionality_failed(self):
        gate = EngagementAuthorizationGate.for_testing()
        case = gate.open_case(_track())
        gate.submit_pid(case.case_id, _good_pid())
        gate.submit_roe(case.case_id, ROEContext(
            roe_id="ROE-1", permits_engagement_type=True,
            proportionality_passed=False, collateral_estimate="moderate"))
        assert gate.get_case(case.case_id).state == EngagementState.DENIED

    def test_collateral_blocking(self):
        gate = EngagementAuthorizationGate.for_testing()
        case = gate.open_case(_track())
        gate.submit_pid(case.case_id, _good_pid())
        gate.submit_roe(case.case_id, ROEContext(
            roe_id="ROE-1", permits_engagement_type=True,
            proportionality_passed=True, collateral_estimate="blocking"))
        assert gate.get_case(case.case_id).state == EngagementState.DENIED


# ---------------------------------------------------------------------------
# Deconfliction
# ---------------------------------------------------------------------------


class TestDeconflictionDenials:
    def test_blue_force_in_volume(self):
        gate = EngagementAuthorizationGate.for_testing()
        case = gate.open_case(_track())
        gate.submit_pid(case.case_id, _good_pid())
        gate.submit_roe(case.case_id, _good_roe())
        gate.submit_deconfliction(case.case_id, DeconflictionContext(
            blue_force_clear=False, airspace_clear=True,
            conflicts=["bf-asset-7"]))
        assert gate.get_case(case.case_id).state == EngagementState.DENIED

    def test_airspace_conflict(self):
        gate = EngagementAuthorizationGate.for_testing()
        case = gate.open_case(_track())
        gate.submit_pid(case.case_id, _good_pid())
        gate.submit_roe(case.case_id, _good_roe())
        gate.submit_deconfliction(case.case_id, DeconflictionContext(
            blue_force_clear=True, airspace_clear=False,
            conflicts=["civ-airliner-1"]))
        assert gate.get_case(case.case_id).state == EngagementState.DENIED

    def test_no_viable_options(self):
        gate = EngagementAuthorizationGate.for_testing()
        case = gate.open_case(_track())
        gate.submit_pid(case.case_id, _good_pid())
        gate.submit_roe(case.case_id, _good_roe())
        gate.submit_deconfliction(case.case_id, _good_deconf())
        # All options non-compliant
        gate.surface_options(case.case_id, [WeaponOption(
            option_id="bad-1", weapon_asset_id="x", weapon_class="hard_kill",
            range_m=100, time_of_flight_s=1, pk_estimate=0.9,
            roe_compliant=False, deconfliction_ok=True, rationale="")])
        assert gate.get_case(case.case_id).state == EngagementState.DENIED


# ---------------------------------------------------------------------------
# Operator role + signature verification
# ---------------------------------------------------------------------------


class TestRoleAndSignatureChecks:
    def test_insufficient_role_refused(self):
        gate = EngagementAuthorizationGate(
            emit_event=lambda et, p: None,
            verify_operator_signature=lambda s, p: True,
            operator_has_role=lambda op, role: role == "operator",  # only operator
            audit_sink=lambda e: None,
        )
        case = _walk_to_pending(gate)
        with pytest.raises(EngagementAuthorizationError) as exc:
            gate.authorize(case.case_id, OperatorAuthorization(
                decision=OperatorDecision.AUTHORIZE,
                operator_id="junior", operator_role="operator",
                rationale="x", selected_option="opt-1", signature=b"sig"),
                engagement_class="base_defense")  # needs mission_commander
        assert "required role" in str(exc.value)

    def test_tampered_signature_refused(self):
        gate = EngagementAuthorizationGate(
            emit_event=lambda et, p: None,
            verify_operator_signature=lambda s, p: False,  # always reject
            operator_has_role=lambda op, role: True,
            audit_sink=lambda e: None,
        )
        case = _walk_to_pending(gate)
        with pytest.raises(EngagementAuthorizationError) as exc:
            gate.authorize(case.case_id, OperatorAuthorization(
                decision=OperatorDecision.AUTHORIZE,
                operator_id="op", operator_role="mission_commander",
                rationale="x", selected_option="opt-1", signature=b"bad-sig"),
                engagement_class="counter_uas")
        assert "signature" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Happy path + event emission
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_full_authorize_flow(self):
        events = []
        gate = EngagementAuthorizationGate(
            emit_event=lambda et, p: events.append((et.value, p)),
            verify_operator_signature=lambda s, p: True,
            operator_has_role=lambda op, role: True,
            audit_sink=lambda e: None,
        )
        case = _walk_to_pending(gate)
        gate.authorize(case.case_id, OperatorAuthorization(
            decision=OperatorDecision.AUTHORIZE,
            operator_id="op", operator_role="mission_commander",
            rationale="confirmed rotary UAS over restricted area",
            selected_option="opt-1", signature=b"sig"),
            engagement_class="counter_uas")
        assert gate.get_case(case.case_id).state == EngagementState.AUTHORIZED
        gate.mark_complete(case.case_id, bda={"effect": "lost_link"})
        assert gate.get_case(case.case_id).state == EngagementState.COMPLETE

        types = [t for t, _ in events]
        assert C2EventType.THREAT_IDENTIFIED.value in types
        assert C2EventType.ENGAGEMENT_AUTHORIZED.value in types
        assert C2EventType.ENGAGEMENT_COMPLETE.value in types

    def test_authorized_emitted_only_via_authorize(self):
        """Critical invariant: ENGAGEMENT_AUTHORIZED must not be emitted by
        any other code path. We verify by walking through deny / hold paths
        and confirming only THREAT_IDENTIFIED and ENGAGEMENT_DENIED fire."""
        events = []
        gate = EngagementAuthorizationGate(
            emit_event=lambda et, p: events.append((et.value, p)),
            verify_operator_signature=lambda s, p: True,
            operator_has_role=lambda op, role: True,
            audit_sink=lambda e: None,
        )
        case = _walk_to_pending(gate)
        gate.authorize(case.case_id, OperatorAuthorization(
            decision=OperatorDecision.DENY, operator_id="op",
            operator_role="mission_commander", rationale="abort"),
            engagement_class="counter_uas")
        types = [t for t, _ in events]
        assert C2EventType.ENGAGEMENT_AUTHORIZED.value not in types
        assert C2EventType.ENGAGEMENT_DENIED.value in types


# ---------------------------------------------------------------------------
# TTL / expiry
# ---------------------------------------------------------------------------


class TestTTLExpiry:
    def test_expire_stale_emits_denied(self):
        events = []
        gate = EngagementAuthorizationGate(
            emit_event=lambda et, p: events.append((et.value, p)),
            verify_operator_signature=lambda s, p: True,
            operator_has_role=lambda op, role: True,
            audit_sink=lambda e: None,
            default_ttl_seconds=1,
        )
        case = _walk_to_pending(gate)
        gate.authorize(case.case_id, OperatorAuthorization(
            decision=OperatorDecision.AUTHORIZE,
            operator_id="op", operator_role="mission_commander",
            rationale="x", selected_option="opt-1", signature=b"sig"),
            engagement_class="counter_uas")

        # Force-expire by passing future timestamp
        future = datetime.now(timezone.utc) + timedelta(seconds=10)
        expired = gate.expire_stale(now=future)
        assert case.case_id in expired
        assert gate.get_case(case.case_id).state == EngagementState.EXPIRED
        types = [t for t, _ in events]
        assert C2EventType.ENGAGEMENT_DENIED.value in types


# ---------------------------------------------------------------------------
# Required-role doctrine table
# ---------------------------------------------------------------------------


class TestRequiredRoleMatrix:
    def test_counter_uas_soft_kill(self):
        assert required_role("counter_uas", "soft_kill") == "operator"

    def test_counter_uas_hard_kill(self):
        assert required_role("counter_uas", "hard_kill") == "mission_commander"

    def test_ace_strike_any(self):
        assert required_role("ace_strike", "any") == "joint_force_commander"

    def test_unknown_class_falls_back_to_default(self):
        assert required_role("unknown_class", "any") == "mission_commander"


# ---------------------------------------------------------------------------
# Production wiring: RBAC role check + audit chain
# ---------------------------------------------------------------------------


class TestRBACRoleCheck:
    def test_operator_passes_operator_check(self):
        rbac = RBACEngine()
        rbac.assign_role("op-1", "OPERATOR")
        check = make_rbac_role_check(rbac)
        assert check("op-1", "operator") is True

    def test_mission_commander_passes_operator_check_via_inheritance(self):
        rbac = RBACEngine()
        rbac.assign_role("cmdr-1", "MISSION_COMMANDER")
        check = make_rbac_role_check(rbac)
        assert check("cmdr-1", "operator") is True
        assert check("cmdr-1", "mission_commander") is True

    def test_operator_fails_mission_commander_check(self):
        rbac = RBACEngine()
        rbac.assign_role("op-1", "OPERATOR")
        check = make_rbac_role_check(rbac)
        assert check("op-1", "mission_commander") is False

    def test_unknown_user_fails(self):
        rbac = RBACEngine()
        check = make_rbac_role_check(rbac)
        assert check("unknown", "operator") is False


class TestChainedHMACAuditSink:
    def test_chain_grows_and_verifies(self, tmp_path):
        sink = ChainedHMACAuditSink(tmp_path / "audit.jsonl",
                                    hmac_key=b"\x42" * 32)
        sink({"transition": "OPEN", "case_id": "c1"})
        sink({"transition": "PID_CONFIRMED", "case_id": "c1"})
        sink({"transition": "ROE_CLEARED", "case_id": "c1"})

        ok, n, bad = sink.verify_chain()
        assert ok is True
        assert n == 3
        assert bad is None

    def test_tampered_row_breaks_chain(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        sink = ChainedHMACAuditSink(path, hmac_key=b"\x42" * 32)
        sink({"transition": "OPEN", "case_id": "c1"})
        sink({"transition": "AUTHORIZED", "case_id": "c1"})

        # Tamper with the file: change the case_id in row 0
        rows = path.read_text().splitlines()
        rec = json.loads(rows[0])
        rec["case_id"] = "TAMPERED"
        rows[0] = json.dumps(rec)
        path.write_text("\n".join(rows) + "\n")

        # Re-create sink with same key to verify
        sink2 = ChainedHMACAuditSink(path, hmac_key=b"\x42" * 32)
        ok, _, bad_idx = sink2.verify_chain()
        assert ok is False
        assert bad_idx == 0


class TestProductionGate:
    def test_build_production_gate_e2e(self, tmp_path):
        events = []
        rbac = RBACEngine()
        rbac.assign_role("cmdr-1", "MISSION_COMMANDER")

        gate = build_production_gate(
            rbac_engine=rbac,
            emit_event=lambda et, p: events.append((et.value, p)),
            keys_dir=str(tmp_path / "keys"),
            audit_log_path=tmp_path / "audit.jsonl",
        )

        # Without keys, signature verify will fail closed → authorize refused
        case = _walk_to_pending(gate)
        with pytest.raises(EngagementAuthorizationError):
            gate.authorize(case.case_id, OperatorAuthorization(
                decision=OperatorDecision.AUTHORIZE,
                operator_id="cmdr-1", operator_role="mission_commander",
                rationale="x", selected_option="opt-1", signature=b"sig"),
                engagement_class="counter_uas")
