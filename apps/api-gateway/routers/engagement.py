"""
Heli.OS Engagement Authorization Router
==========================================
The HTTP surface for the human-in-the-loop kinetic-action gate.

This is the SINGLE production API path that drives EngagementAuthorizationGate.
Every endpoint mutates state through gate methods only — there is no
backdoor to the AUTHORIZED state.

Endpoints
---------
POST   /engagement/cases                    Open a new case from a confirmed track
POST   /engagement/cases/{case_id}/pid      Submit positive identification evidence
POST   /engagement/cases/{case_id}/roe      Submit ROE / proportionality / collateral
POST   /engagement/cases/{case_id}/decon    Submit deconfliction context
POST   /engagement/cases/{case_id}/options  Surface ranked weapon options
POST   /engagement/cases/{case_id}/decide   Operator decision (AUTHORIZE / DENY / HOLD)
POST   /engagement/cases/{case_id}/complete Mark engagement complete (post-action BDA)
GET    /engagement/cases                    List cases (filterable by state)
GET    /engagement/cases/{case_id}          Inspect a single case (full audit)
POST   /engagement/expire-stale             Sweep + auto-deny TTL-expired cases

Auth model
----------
Every mutating endpoint requires:
  Authorization: Bearer <JWT>           (operator OIDC identity, set elsewhere)
  X-Operator-Signature: <base64 Ed25519> (required ONLY on /decide AUTHORIZE)

The /decide endpoint is the only one that emits ENGAGEMENT_AUTHORIZED, and
only when:
  1. The decision payload's `selected_option` is in the case's viable options
  2. The operator's RBAC role meets _ROLE_MATRIX[engagement_class][weapon_class]
  3. The X-Operator-Signature verifies via Ed25519 against the operator's
     public key in SUMMIT_SENSOR_KEYS_DIR

If any of those fail, the gate raises EngagementAuthorizationError → HTTP 403.

The full state-transition audit chain is written via the gate's audit_sink
(chained-HMAC by default, db_logger if available).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger("api-gateway.engagement")

router = APIRouter(prefix="/engagement", tags=["engagement"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class TrackInput(BaseModel):
    track_id: str
    entity_id: str
    classification: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sensors: List[str] = Field(default_factory=list)
    last_position: Optional[Dict[str, float]] = None


class PIDInput(BaseModel):
    method: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_asset_id: Optional[str] = None
    notes: str = ""


class ROEInput(BaseModel):
    roe_id: str
    permits_engagement_type: bool
    proportionality_passed: bool
    collateral_estimate: str
    sti_active: bool = False
    commander_intent_match: bool = True


class DeconflictionInput(BaseModel):
    blue_force_clear: bool
    airspace_clear: bool
    nearby_civilians_count: int = 0
    conflicts: List[str] = Field(default_factory=list)


class WeaponOptionInput(BaseModel):
    option_id: str
    weapon_asset_id: str
    weapon_class: str
    range_m: float
    time_of_flight_s: float
    pk_estimate: float = Field(..., ge=0.0, le=1.0)
    roe_compliant: bool
    deconfliction_ok: bool
    rationale: str


class DecisionInput(BaseModel):
    decision: str   # AUTHORIZE | DENY | HOLD | REQUEST_HIGHER
    operator_id: str
    operator_role: str
    rationale: str
    selected_option: Optional[str] = None
    engagement_class: str = "default"


class CaseSummary(BaseModel):
    case_id: str
    state: str
    track_id: str
    entity_id: str
    classification: str
    created_at: str
    n_audit_entries: int
    auth_ttl: Optional[str] = None


class CaseDetail(CaseSummary):
    pid: Optional[Dict[str, Any]] = None
    roe: Optional[Dict[str, Any]] = None
    deconfliction: Optional[Dict[str, Any]] = None
    options: List[Dict[str, Any]] = Field(default_factory=list)
    decision: Optional[Dict[str, Any]] = None
    audit: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Gate accessor (set during app startup)
# ---------------------------------------------------------------------------


_GATE = None


def set_gate(gate) -> None:
    """Called from app startup to inject the production-wired gate."""
    global _GATE
    _GATE = gate


def _gate():
    if _GATE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="EngagementAuthorizationGate not initialized. "
                   "Start the API gateway with engagement_wiring configured.")
    return _GATE


def _summarize(case) -> CaseSummary:
    return CaseSummary(
        case_id=case.case_id,
        state=case.state.value,
        track_id=case.track.track_id,
        entity_id=case.track.entity_id,
        classification=case.track.classification,
        created_at=case.created_at.isoformat(),
        n_audit_entries=len(case.audit),
        auth_ttl=case.auth_ttl.isoformat() if case.auth_ttl else None,
    )


def _detail(case) -> CaseDetail:
    return CaseDetail(
        **_summarize(case).model_dump(),
        pid=case.pid.__dict__ if case.pid else None,
        roe=case.roe.__dict__ if case.roe else None,
        deconfliction=case.deconfliction.__dict__ if case.deconfliction else None,
        options=[o.__dict__ for o in case.options],
        decision=case.decision.__dict__ if case.decision else None,
        audit=case.audit,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/cases", status_code=status.HTTP_201_CREATED)
async def open_case(track: TrackInput) -> CaseSummary:
    from packages.c2_intel.engagement_authorization import TrackEvidence
    g = _gate()
    case = g.open_case(TrackEvidence(
        track_id=track.track_id, entity_id=track.entity_id,
        classification=track.classification, confidence=track.confidence,
        sensors=track.sensors, last_position=track.last_position,
        last_seen=datetime.now(timezone.utc),
    ))
    return _summarize(case)


@router.post("/cases/{case_id}/pid")
async def submit_pid(case_id: str, pid: PIDInput) -> CaseSummary:
    from packages.c2_intel.engagement_authorization import (
        EngagementAuthorizationError, PIDEvidence,
    )
    try:
        case = _gate().submit_pid(case_id, PIDEvidence(
            method=pid.method, confidence=pid.confidence,
            source_asset_id=pid.source_asset_id, notes=pid.notes,
        ))
    except EngagementAuthorizationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _summarize(case)


@router.post("/cases/{case_id}/roe")
async def submit_roe(case_id: str, roe: ROEInput) -> CaseSummary:
    from packages.c2_intel.engagement_authorization import (
        EngagementAuthorizationError, ROEContext,
    )
    try:
        case = _gate().submit_roe(case_id, ROEContext(**roe.model_dump()))
    except EngagementAuthorizationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _summarize(case)


@router.post("/cases/{case_id}/decon")
async def submit_decon(case_id: str, dec: DeconflictionInput) -> CaseSummary:
    from packages.c2_intel.engagement_authorization import (
        DeconflictionContext, EngagementAuthorizationError,
    )
    try:
        case = _gate().submit_deconfliction(
            case_id, DeconflictionContext(**dec.model_dump()))
    except EngagementAuthorizationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _summarize(case)


@router.post("/cases/{case_id}/options")
async def surface_options(case_id: str,
                          options: List[WeaponOptionInput]) -> CaseSummary:
    from packages.c2_intel.engagement_authorization import (
        EngagementAuthorizationError, WeaponOption,
    )
    try:
        case = _gate().surface_options(
            case_id, [WeaponOption(**o.model_dump()) for o in options])
    except EngagementAuthorizationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _summarize(case)


@router.post("/cases/{case_id}/decide")
async def operator_decision(case_id: str, decision: DecisionInput,
                            request: Request) -> CaseDetail:
    """The single mandatory operator-decision surface for kinetic action.

    For AUTHORIZE: requires X-Operator-Signature header (base64 Ed25519
    signature over the canonical decision payload). The gate verifies
    against the operator's public key. Insufficient role, missing
    signature, or signature-verify failure → HTTPException 403.
    """
    from packages.c2_intel.engagement_authorization import (
        EngagementAuthorizationError, OperatorAuthorization, OperatorDecision,
    )

    try:
        op_decision = OperatorDecision(decision.decision.lower())
    except ValueError:
        raise HTTPException(status_code=400,
                            detail=f"unknown decision: {decision.decision}")

    sig_header = request.headers.get("X-Operator-Signature", "")
    sig_bytes = b""
    if sig_header:
        try:
            # Header is the base64-encoded raw signature; we keep it ascii-encoded
            # and let the verifier decode it itself.
            sig_bytes = sig_header.encode("ascii")
        except Exception:
            raise HTTPException(status_code=400,
                                detail="malformed X-Operator-Signature header")

    if op_decision == OperatorDecision.AUTHORIZE and not sig_bytes:
        raise HTTPException(
            status_code=403,
            detail="X-Operator-Signature header is required for AUTHORIZE")

    op_auth = OperatorAuthorization(
        decision=op_decision,
        operator_id=decision.operator_id,
        operator_role=decision.operator_role,
        rationale=decision.rationale,
        selected_option=decision.selected_option,
        signature=sig_bytes,
    )
    try:
        case = _gate().authorize(case_id, op_auth,
                                  engagement_class=decision.engagement_class)
    except EngagementAuthorizationError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return _detail(case)


@router.post("/cases/{case_id}/complete")
async def mark_complete(case_id: str,
                        bda: Optional[Dict[str, Any]] = None) -> CaseDetail:
    from packages.c2_intel.engagement_authorization import EngagementAuthorizationError
    try:
        case = _gate().mark_complete(case_id, bda=bda)
    except EngagementAuthorizationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _detail(case)


@router.get("/cases")
async def list_cases(state: Optional[str] = None) -> List[CaseSummary]:
    from packages.c2_intel.engagement_authorization import EngagementState
    s = None
    if state:
        try:
            s = EngagementState(state.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"unknown state: {state}")
    return [_summarize(c) for c in _gate().list_cases(state=s)]


@router.get("/cases/{case_id}")
async def get_case(case_id: str) -> CaseDetail:
    case = _gate().get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="case not found")
    return _detail(case)


@router.post("/expire-stale")
async def expire_stale() -> Dict[str, Any]:
    """Operator/scheduler triggers the TTL sweep. Cases past TTL transition
    AUTHORIZED → EXPIRED with an auto ENGAGEMENT_DENIED for audit."""
    expired = _gate().expire_stale()
    return {"expired_case_ids": expired, "count": len(expired)}
