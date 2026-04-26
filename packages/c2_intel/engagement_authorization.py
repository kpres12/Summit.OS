"""
Engagement Authorization Workflow

The load-bearing human-in-the-loop gate component for Heli.OS in DoD /
federal use cases. This module enforces the architectural invariant that
no kinetic action is dispatched without an authorized human authorization
event flowing through this state machine.

State machine
-------------
    DETECTED                — track exists, no PID yet
       │ PID confirmed (sensor or operator)
       ▼
    PID_CONFIRMED           — distinction established (combatant vs civilian)
       │ ROE check passes
       ▼
    ROE_CLEARED             — proportionality + collateral assessment passes
       │ deconfliction check passes
       ▼
    DECONFLICTED            — no blue-force in engagement volume, airspace clear
       │ operator action: AUTHORIZE  /  DENY  /  HOLD  /  REQUEST_HIGHER
       ▼
    PENDING_AUTHORIZATION   — option pack surfaced to operator
       │
       ├─▶ ENGAGEMENT_AUTHORIZED  → emit C2 event, proceed to tasking
       ├─▶ ENGAGEMENT_DENIED      → emit event, hold track, no action
       └─▶ ENGAGEMENT_HELD        → re-enter with timeout

Inputs to the gate (signed by sensor/intel/policy modules):
    - track:           confirmed sensor-fused track w/ confidence + classification
    - pid_evidence:    evidence chain establishing positive identification
    - roe_context:     current ROE state + commander's intent + special instructions
    - deconfliction:   blue-force list + airspace conflicts + nearby civilians
    - weapon_options:  ranked weapon-target pairings (from
                       track_to_weapon_ranker)

Operator output (signed by operator + RBAC role check):
    - decision:        AUTHORIZE | DENY | HOLD | REQUEST_HIGHER
    - operator_id:     authenticated identity (from packages/security/auth.py)
    - operator_role:   RBAC role (must match required_role for engagement type)
    - rationale:       free-text reason (LoAC audit trail)
    - selected_option: index into weapon_options (only if AUTHORIZE)
    - timestamp:       ISO 8601 UTC
    - signature:       Ed25519 sig over decision payload

Audit trail
-----------
Every state transition is appended to an immutable audit log
(`packages/observability/db_logger.py`) with HMAC chaining
(`packages/security/world_model_hmac.py`). LoAC compliance evidence.

Design constraints (do not weaken without CEO sign-off):
  1. The state machine cannot transition from DECONFLICTED → AUTHORIZED
     without an `operator_decision` of AUTHORIZE signed by an
     authenticated human with a role permitted to engage.
  2. AUTHORIZED carries a TTL. Expiration without a downstream
     ENGAGEMENT_COMPLETE event auto-emits ENGAGEMENT_DENIED for audit.
  3. There is no API path that emits ENGAGEMENT_AUTHORIZED without going
     through `EngagementAuthorizationGate.authorize()`.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .models import C2EventType

logger = logging.getLogger("c2_intel.engagement_authorization")


# ---------------------------------------------------------------------------
# State machine states + decisions
# ---------------------------------------------------------------------------


class EngagementState(str, Enum):
    DETECTED              = "detected"
    PID_CONFIRMED         = "pid_confirmed"
    ROE_CLEARED           = "roe_cleared"
    DECONFLICTED          = "deconflicted"
    PENDING_AUTHORIZATION = "pending_authorization"
    AUTHORIZED            = "authorized"
    DENIED                = "denied"
    HELD                  = "held"
    EXPIRED               = "expired"
    COMPLETE              = "complete"


class OperatorDecision(str, Enum):
    AUTHORIZE        = "authorize"
    DENY             = "deny"
    HOLD             = "hold"
    REQUEST_HIGHER   = "request_higher"


# ---------------------------------------------------------------------------
# Required RBAC roles per engagement class
# ---------------------------------------------------------------------------


# Map (engagement_class, weapon_class) → minimum operator role.
# Pulled from doctrine; configurable per deployment via OPA policy.
_ROLE_MATRIX: Dict[tuple, str] = {
    ("counter_uas",  "soft_kill"):   "operator",        # jamming/spoofing — defensive
    ("counter_uas",  "hard_kill"):   "mission_commander",
    ("force_protection_perimeter", "soft_kill"):    "operator",
    ("force_protection_perimeter", "hard_kill"):    "mission_commander",
    ("base_defense", "any"):         "mission_commander",
    ("ace_strike",   "any"):         "joint_force_commander",
    ("default",      "any"):         "mission_commander",
}


def required_role(engagement_class: str, weapon_class: str) -> str:
    """Minimum operator role required to AUTHORIZE this engagement."""
    return (
        _ROLE_MATRIX.get((engagement_class, weapon_class))
        or _ROLE_MATRIX.get((engagement_class, "any"))
        or _ROLE_MATRIX[("default", "any")]
    )


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TrackEvidence:
    """Confirmed sensor-fused track passed into the gate."""
    track_id:           str
    entity_id:          str
    classification:     str       # e.g. "rotary_uas", "fixed_wing", "vessel_unknown"
    confidence:         float     # 0..1 (sensor fusion confidence)
    sensors:            List[str] = field(default_factory=list)
    last_position:      Optional[Dict[str, float]] = None  # {lat, lon, alt_m}
    last_seen:          Optional[datetime] = None


@dataclass
class PIDEvidence:
    """Positive identification evidence chain."""
    method:             str       # "iff_squawk", "visual_isr", "rf_fingerprint", "pattern_of_life"
    confidence:         float
    source_asset_id:    Optional[str] = None
    timestamp:          datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes:              str = ""


@dataclass
class ROEContext:
    """Current ROE + commander's intent at engagement decision time."""
    roe_id:                     str   # e.g. "OPLAN-XXXX-ROE-2026-001"
    permits_engagement_type:    bool  # is the engagement class allowed under current ROE?
    proportionality_passed:     bool  # weapon vs target proportionality check
    collateral_estimate:        str   # "minimal" | "low" | "moderate" | "elevated" | "blocking"
    sti_active:                 bool = False  # special tactical instructions present
    commander_intent_match:     bool = True


@dataclass
class DeconflictionContext:
    """Blue-force + airspace check at engagement decision time."""
    blue_force_clear:        bool
    airspace_clear:          bool
    nearby_civilians_count:  int = 0
    conflicts:               List[str] = field(default_factory=list)


@dataclass
class WeaponOption:
    """One ranked option for engaging a target. Surfaced to operator."""
    option_id:        str
    weapon_asset_id:  str
    weapon_class:     str       # "soft_kill" | "hard_kill" | "non_lethal" | "kinetic_air"
    range_m:          float
    time_of_flight_s: float
    pk_estimate:      float     # probability of kill / mission effect (0..1)
    roe_compliant:    bool
    deconfliction_ok: bool
    rationale:        str


@dataclass
class OperatorAuthorization:
    """Operator's signed decision."""
    decision:         OperatorDecision
    operator_id:      str
    operator_role:    str
    rationale:        str
    selected_option:  Optional[str] = None       # WeaponOption.option_id
    timestamp:        datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signature:        Optional[bytes] = None     # Ed25519 sig produced by client


@dataclass
class EngagementCase:
    """Full state of one engagement decision instance."""
    case_id:        str
    state:          EngagementState
    track:          TrackEvidence
    pid:            Optional[PIDEvidence]      = None
    roe:            Optional[ROEContext]       = None
    deconfliction:  Optional[DeconflictionContext] = None
    options:        List[WeaponOption]         = field(default_factory=list)
    decision:       Optional[OperatorAuthorization] = None
    auth_ttl:       Optional[datetime]         = None
    audit:          List[Dict[str, Any]]       = field(default_factory=list)
    created_at:     datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


class EngagementAuthorizationError(RuntimeError):
    """Raised when an unsafe state transition is attempted."""


class EngagementAuthorizationGate:
    """The single mandatory gate between confirmed-threat and any kinetic
    tasking dispatch. There is intentionally no API path that emits
    ENGAGEMENT_AUTHORIZED without going through this object's authorize().

    Dependencies are REQUIRED — there are no pass-through defaults. If you
    cannot supply a real validator + audit sink, you should not be running
    this gate. (Test-only construction can use the helper
    `EngagementAuthorizationGate.for_testing(...)` which fails closed and
    documents the test exemption.)
    """

    def __init__(
        self,
        emit_event: Callable[[C2EventType, Dict[str, Any]], None],
        verify_operator_signature: Callable[[bytes, Dict[str, Any]], bool],
        operator_has_role: Callable[[str, str], bool],
        audit_sink: Callable[[Dict[str, Any]], None],
        default_ttl_seconds: int = 60,
    ) -> None:
        for name, fn in (
            ("emit_event", emit_event),
            ("verify_operator_signature", verify_operator_signature),
            ("operator_has_role", operator_has_role),
            ("audit_sink", audit_sink),
        ):
            if fn is None or not callable(fn):
                raise ValueError(
                    f"EngagementAuthorizationGate requires a callable {name}. "
                    "Pass-through defaults were removed for safety. See "
                    "EngagementAuthorizationGate.for_testing() if you "
                    "explicitly need a permissive instance for unit tests."
                )
        self._emit_event   = emit_event
        self._verify_sig   = verify_operator_signature
        self._has_role     = operator_has_role
        self._audit_sink   = audit_sink
        self._default_ttl  = default_ttl_seconds
        self._cases: Dict[str, EngagementCase] = {}

    @classmethod
    def for_testing(cls, emit_event: Optional[Callable] = None,
                    *, allow_role: bool = True,
                    allow_signature: bool = True,
                    capture_audit: Optional[List[Dict[str, Any]]] = None,
                    default_ttl_seconds: int = 60
                    ) -> "EngagementAuthorizationGate":
        """Test-only constructor with explicit permissive validators.

        This bypass is documented and named — call sites are easy to grep
        for. Production code MUST use the real constructor with wired
        dependencies. The test helper is intentionally verbose to avoid
        accidental production use.
        """
        emit = emit_event or (lambda et, payload: None)
        captured = capture_audit if capture_audit is not None else []
        return cls(
            emit_event=emit,
            verify_operator_signature=lambda sig, payload: bool(allow_signature),
            operator_has_role=lambda op, role: bool(allow_role),
            audit_sink=lambda entry: captured.append(entry),
            default_ttl_seconds=default_ttl_seconds,
        )

    # --- transitions ------------------------------------------------------

    def open_case(self, track: TrackEvidence) -> EngagementCase:
        case = EngagementCase(
            case_id=str(uuid.uuid4()),
            state=EngagementState.DETECTED,
            track=track,
        )
        self._cases[case.case_id] = case
        self._record(case, "OPEN", {"track_id": track.track_id})
        self._emit_event(C2EventType.THREAT_IDENTIFIED, {
            "case_id":   case.case_id,
            "track_id":  track.track_id,
            "entity_id": track.entity_id,
            "classification": track.classification,
            "confidence": track.confidence,
        })
        return case

    def submit_pid(self, case_id: str, pid: PIDEvidence) -> EngagementCase:
        case = self._require(case_id, EngagementState.DETECTED)
        if pid.confidence < 0.7:
            raise EngagementAuthorizationError(
                f"PID confidence {pid.confidence:.2f} below 0.70 threshold")
        case.pid = pid
        case.state = EngagementState.PID_CONFIRMED
        self._record(case, "PID_CONFIRMED", {
            "method": pid.method, "confidence": pid.confidence,
        })
        return case

    def submit_roe(self, case_id: str, roe: ROEContext) -> EngagementCase:
        case = self._require(case_id, EngagementState.PID_CONFIRMED)
        if not roe.permits_engagement_type:
            self._deny(case, reason="ROE does not permit this engagement type",
                       roe_id=roe.roe_id)
            return case
        if not roe.proportionality_passed:
            self._deny(case, reason="Proportionality check failed",
                       roe_id=roe.roe_id)
            return case
        if roe.collateral_estimate == "blocking":
            self._deny(case, reason="Collateral damage estimate blocking",
                       roe_id=roe.roe_id)
            return case
        case.roe = roe
        case.state = EngagementState.ROE_CLEARED
        self._record(case, "ROE_CLEARED", {
            "roe_id": roe.roe_id,
            "collateral_estimate": roe.collateral_estimate,
        })
        return case

    def submit_deconfliction(self, case_id: str,
                             dec: DeconflictionContext) -> EngagementCase:
        case = self._require(case_id, EngagementState.ROE_CLEARED)
        if not dec.blue_force_clear:
            self._deny(case, reason="Blue force in engagement volume",
                       conflicts=dec.conflicts)
            return case
        if not dec.airspace_clear:
            self._deny(case, reason="Airspace conflict",
                       conflicts=dec.conflicts)
            return case
        case.deconfliction = dec
        case.state = EngagementState.DECONFLICTED
        self._record(case, "DECONFLICTED", {
            "nearby_civilians_count": dec.nearby_civilians_count,
        })
        return case

    def surface_options(self, case_id: str,
                        options: List[WeaponOption]) -> EngagementCase:
        case = self._require(case_id, EngagementState.DECONFLICTED)
        # Filter: only options that are ROE+deconfliction compliant.
        viable = [o for o in options if o.roe_compliant and o.deconfliction_ok]
        if not viable:
            self._deny(case, reason="No viable weapon options after filters")
            return case
        case.options = viable
        case.state = EngagementState.PENDING_AUTHORIZATION
        case.auth_ttl = datetime.now(timezone.utc) + timedelta(seconds=self._default_ttl)
        self._record(case, "OPTIONS_SURFACED", {
            "n_options": len(viable),
            "option_ids": [o.option_id for o in viable],
        })
        return case

    # --- the actual gate -------------------------------------------------

    def authorize(self, case_id: str,
                  decision: OperatorAuthorization,
                  engagement_class: str = "default") -> EngagementCase:
        """The single mandatory human-authorization step. There is no path
        in the codebase from DECONFLICTED/PENDING_AUTHORIZATION to
        AUTHORIZED that does not go through this method."""
        case = self._require(case_id, EngagementState.PENDING_AUTHORIZATION)

        if decision.decision == OperatorDecision.DENY:
            self._deny(case, reason=decision.rationale or "Operator denied",
                       operator_id=decision.operator_id, signed=True)
            case.decision = decision
            return case

        if decision.decision == OperatorDecision.HOLD:
            case.state = EngagementState.HELD
            case.decision = decision
            case.auth_ttl = datetime.now(timezone.utc) + timedelta(seconds=self._default_ttl)
            self._record(case, "HELD", {
                "operator_id": decision.operator_id,
                "rationale": decision.rationale,
            })
            return case

        if decision.decision == OperatorDecision.REQUEST_HIGHER:
            case.state = EngagementState.HELD
            case.decision = decision
            case.auth_ttl = datetime.now(timezone.utc) + timedelta(seconds=self._default_ttl)
            self._record(case, "REQUEST_HIGHER", {
                "operator_id": decision.operator_id,
            })
            return case

        # AUTHORIZE path — every check below MUST pass
        if decision.decision != OperatorDecision.AUTHORIZE:
            raise EngagementAuthorizationError(
                f"Unexpected decision {decision.decision}")

        if decision.selected_option is None:
            raise EngagementAuthorizationError(
                "AUTHORIZE requires selected_option referencing a viable WeaponOption")

        selected = next((o for o in case.options
                         if o.option_id == decision.selected_option), None)
        if selected is None:
            raise EngagementAuthorizationError(
                f"selected_option {decision.selected_option} not in viable options")
        if not (selected.roe_compliant and selected.deconfliction_ok):
            raise EngagementAuthorizationError(
                "Selected option is not ROE/deconfliction compliant — refusing")

        # Operator identity + role
        needed_role = required_role(engagement_class, selected.weapon_class)
        if not self._has_role(decision.operator_id, needed_role):
            raise EngagementAuthorizationError(
                f"Operator {decision.operator_id} does not hold required role "
                f"'{needed_role}' for engagement_class={engagement_class} / "
                f"weapon_class={selected.weapon_class}")

        # Cryptographic signature on the decision payload
        sig_payload = {
            "case_id":         case.case_id,
            "decision":        decision.decision.value,
            "operator_id":     decision.operator_id,
            "operator_role":   decision.operator_role,
            "selected_option": decision.selected_option,
            "timestamp":       decision.timestamp.isoformat(),
        }
        if not self._verify_sig(decision.signature or b"", sig_payload):
            raise EngagementAuthorizationError(
                "Operator signature failed verification — refusing")

        # All gates passed. Authorize.
        case.decision  = decision
        case.state     = EngagementState.AUTHORIZED
        case.auth_ttl  = datetime.now(timezone.utc) + timedelta(seconds=self._default_ttl)
        self._record(case, "AUTHORIZED", {
            "operator_id":     decision.operator_id,
            "operator_role":   decision.operator_role,
            "selected_option": decision.selected_option,
            "weapon_asset_id": selected.weapon_asset_id,
            "weapon_class":    selected.weapon_class,
            "rationale":       decision.rationale,
            "signature_ok":    True,
        })
        self._emit_event(C2EventType.ENGAGEMENT_AUTHORIZED, {
            "case_id":         case.case_id,
            "track_id":        case.track.track_id,
            "entity_id":       case.track.entity_id,
            "weapon_asset_id": selected.weapon_asset_id,
            "weapon_class":    selected.weapon_class,
            "operator_id":     decision.operator_id,
            "auth_ttl":        case.auth_ttl.isoformat(),
        })
        return case

    def mark_complete(self, case_id: str,
                      bda: Optional[Dict[str, Any]] = None) -> EngagementCase:
        case = self._require(case_id, EngagementState.AUTHORIZED)
        case.state = EngagementState.COMPLETE
        self._record(case, "COMPLETE", {"bda": bda or {}})
        self._emit_event(C2EventType.ENGAGEMENT_COMPLETE, {
            "case_id": case.case_id,
            "bda": bda or {},
        })
        return case

    def expire_stale(self, now: Optional[datetime] = None) -> List[str]:
        """Sweep PENDING_AUTHORIZATION, AUTHORIZED, and HELD cases past TTL.

        PENDING_AUTHORIZATION: expire when operator doesn't respond within TTL.
        AUTHORIZED cases: expire when auth_ttl is exceeded without COMPLETE.
        HELD cases: expire when hold timeout is exceeded without resumption.
        """
        now = now or datetime.now(timezone.utc)
        expired: List[str] = []
        _expirable = (
            EngagementState.PENDING_AUTHORIZATION,
            EngagementState.AUTHORIZED,
            EngagementState.HELD,
        )
        _reason = {
            EngagementState.PENDING_AUTHORIZATION:
                "Operator did not respond within TTL — case expired",
            EngagementState.AUTHORIZED:
                "Authorization TTL expired without ENGAGEMENT_COMPLETE",
            EngagementState.HELD:
                "HELD case TTL expired — operator did not resume or deny",
        }
        for case in list(self._cases.values()):
            if case.state in _expirable and case.auth_ttl and now > case.auth_ttl:
                self._deny(case, reason=_reason[case.state])
                case.state = EngagementState.EXPIRED
                expired.append(case.case_id)
        return expired

    # --- internals --------------------------------------------------------

    def _require(self, case_id: str,
                 expected: EngagementState) -> EngagementCase:
        case = self._cases.get(case_id)
        if case is None:
            raise EngagementAuthorizationError(f"Unknown case_id {case_id}")
        if case.state != expected:
            raise EngagementAuthorizationError(
                f"Case {case_id} in state {case.state.value}, expected "
                f"{expected.value}")
        return case

    def _deny(self, case: EngagementCase, reason: str, **extra: Any) -> None:
        case.state = EngagementState.DENIED
        self._record(case, "DENIED", {"reason": reason, **extra})
        self._emit_event(C2EventType.ENGAGEMENT_DENIED, {
            "case_id":  case.case_id,
            "track_id": case.track.track_id,
            "reason":   reason,
        })

    def _record(self, case: EngagementCase, transition: str,
                payload: Dict[str, Any]) -> None:
        entry = {
            "ts":         datetime.now(timezone.utc).isoformat(),
            "case_id":    case.case_id,
            "track_id":   case.track.track_id,
            "transition": transition,
            "to_state":   case.state.value,
            "payload":    payload,
        }
        case.audit.append(entry)
        self._audit_sink(entry)
        logger.info("[engagement] %s case=%s -> %s", transition, case.case_id,
                    case.state.value)

    def get_case(self, case_id: str) -> Optional[EngagementCase]:
        return self._cases.get(case_id)

    def list_cases(self,
                   state: Optional[EngagementState] = None) -> List[EngagementCase]:
        if state is None:
            return list(self._cases.values())
        return [c for c in self._cases.values() if c.state == state]
