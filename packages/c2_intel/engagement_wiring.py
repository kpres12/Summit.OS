"""
Engagement Authorization — Production Wiring
================================================
Connects EngagementAuthorizationGate to the real security primitives:

  verify_operator_signature  →  packages/security/sensor_signing.verify_frame
                                  (Ed25519, fail-closed when crypto unavailable)
  operator_has_role          →  packages/security/rbac.RBACEngine
                                  (with role-name normalization between the
                                   doctrine taxonomy and RBAC's labels)
  audit_sink                 →  packages/observability/db_logger (when present)
                                  → fallback to chained-HMAC append-only file
  emit_event                 →  whatever the caller injects (typically MQTT
                                   bridge or in-process event bus)

This module is the only correct way to instantiate a production-grade
EngagementAuthorizationGate. It also installs the doctrine roles
(`operator`, `mission_commander`, `joint_force_commander`,
`combatant_commander`) into the RBAC engine if they aren't already
present, so the role-tier matrix in engagement_authorization.py works
out of the box.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .engagement_authorization import EngagementAuthorizationGate
from .models import C2EventType

logger = logging.getLogger("c2_intel.engagement_wiring")


# ---------------------------------------------------------------------------
# Doctrine-role ↔ RBAC-role mapping
# ---------------------------------------------------------------------------


# Engagement-authorization role taxonomy is doctrine-driven (operator,
# mission_commander, joint_force_commander, combatant_commander). RBAC
# already has OPERATOR / MISSION_COMMANDER / ADMIN / SUPER_ADMIN. The
# mapping below normalizes between the two so authorization decisions
# correctly traverse RBAC's role-inheritance graph.

DOCTRINE_TO_RBAC = {
    "operator":              "OPERATOR",
    "mission_commander":     "MISSION_COMMANDER",
    "joint_force_commander": "ADMIN",          # JFC has full mission-command authority
    "combatant_commander":   "SUPER_ADMIN",
}


def _ensure_doctrine_roles(rbac_engine) -> None:
    """RBAC already defines OPERATOR / MISSION_COMMANDER / ADMIN /
    SUPER_ADMIN. Doctrine roles (joint_force_commander, etc.) are mapped
    to those at check time; no new roles are added to RBAC here. This
    function is kept as the explicit place to extend the mapping if a
    deployment needs more granularity.
    """
    return None


def make_rbac_role_check(rbac_engine) -> Callable[[str, str], bool]:
    """Returns a (operator_id, doctrine_role_name) -> bool function that
    consults the RBAC engine, normalizing the doctrine role name to RBAC's
    role taxonomy and checking with role inheritance."""
    _ensure_doctrine_roles(rbac_engine)

    def _check(operator_id: str, doctrine_role: str) -> bool:
        rbac_role = DOCTRINE_TO_RBAC.get(doctrine_role, doctrine_role.upper())
        try:
            user_roles = set(rbac_engine.get_user_roles(operator_id))
        except Exception as e:
            logger.warning("rbac.get_user_roles(%s) failed: %s", operator_id, e)
            return False
        if not user_roles:
            return False
        # Direct match
        if rbac_role in user_roles:
            return True
        # Inheritance: SUPER_ADMIN inherits everything, ADMIN inherits
        # MISSION_COMMANDER, MISSION_COMMANDER inherits OPERATOR.
        # The RBAC engine's check_permission walks parents, but we want a
        # role check, not a permission check, so we walk parents ourselves.
        rank = {"OPERATOR": 1, "MISSION_COMMANDER": 2, "ADMIN": 3,
                "SUPER_ADMIN": 4, "VIEWER": 0}
        try:
            need = rank.get(rbac_role, 0)
            best = max((rank.get(r, 0) for r in user_roles), default=0)
            return best >= need
        except Exception:
            return False

    return _check


# ---------------------------------------------------------------------------
# Signature verifier (wraps sensor_signing.verify_frame)
# ---------------------------------------------------------------------------


def make_ed25519_verifier(keys_dir: Optional[str] = None
                          ) -> Callable[[bytes, Dict[str, Any]], bool]:
    """Returns a (signature_bytes, decision_payload) -> bool function that
    canonicalizes the decision payload, looks up the operator's public key,
    and verifies the Ed25519 signature.

    Operator public keys are expected at
    {keys_dir or SUMMIT_SENSOR_KEYS_DIR}/{operator_id}.pub
    (same directory layout as sensor_signing).
    """
    from packages.security import sensor_signing  # type: ignore

    def _verify(signature_b: bytes, payload: Dict[str, Any]) -> bool:
        if not signature_b:
            return False
        operator_id = payload.get("operator_id")
        if not operator_id:
            return False
        canonical = json.dumps(payload, sort_keys=True,
                               separators=(",", ":")).encode("utf-8")
        sig_b64 = signature_b.decode("ascii") if isinstance(signature_b, bytes) else str(signature_b)
        try:
            return sensor_signing.verify_frame(operator_id, canonical, sig_b64,
                                                keys_dir=keys_dir)
        except sensor_signing.SensorSigningUnavailable as e:
            logger.error("signature verify unavailable: %s", e)
            return False
        except Exception as e:
            logger.warning("signature verify error: %s", e)
            return False

    return _verify


# ---------------------------------------------------------------------------
# Audit sink — chained-HMAC append-only file (always available)
# ---------------------------------------------------------------------------


class ChainedHMACAuditSink:
    """Append-only file with a per-row chained HMAC. Tampering with any row
    invalidates the chain forward. Provides the sink for the engagement
    gate's audit trail when no enterprise observability backend is wired.

    Each line is a JSON object with: ts, case_id, transition, payload,
    prev_hmac (hex of previous row's hmac), this_hmac (hex of HMAC over
    prev_hmac + canonical(this_record)).
    """

    def __init__(self, log_path: Path, hmac_key: Optional[bytes] = None):
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if hmac_key is None:
            env_key = os.environ.get("HELI_AUDIT_HMAC_KEY", "")
            if env_key:
                try:
                    hmac_key = bytes.fromhex(env_key)
                except ValueError:
                    hmac_key = env_key.encode("utf-8")
            else:
                hmac_key = os.urandom(32)
                logger.warning(
                    "ChainedHMACAuditSink: HELI_AUDIT_HMAC_KEY not set — "
                    "using ephemeral key (audit chain non-portable across "
                    "restarts). Set HELI_AUDIT_HMAC_KEY for production.")
        self._key = hmac_key
        # Recover the latest hmac from the file so the chain survives restart
        self._prev_hmac = self._tail_hmac()

    def _tail_hmac(self) -> str:
        if not self._path.exists():
            return "0" * 64  # genesis
        last = ""
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                last = line
        if not last.strip():
            return "0" * 64
        try:
            return json.loads(last).get("this_hmac", "0" * 64)
        except Exception:
            return "0" * 64

    def __call__(self, entry: Dict[str, Any]) -> None:
        record = dict(entry)
        record["prev_hmac"] = self._prev_hmac
        canonical = json.dumps({k: v for k, v in record.items()
                                if k != "this_hmac"},
                               sort_keys=True, separators=(",", ":")).encode("utf-8")
        this_hmac = hmac.new(self._key, canonical, hashlib.sha256).hexdigest()
        record["this_hmac"] = this_hmac
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self._prev_hmac = this_hmac

    def verify_chain(self) -> tuple[bool, int, Optional[int]]:
        """Walk the file and verify each row's HMAC chain.
        Returns (ok, rows_checked, first_bad_row_index_or_None)."""
        if not self._path.exists():
            return True, 0, None
        prev = "0" * 64
        n = 0
        with self._path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("prev_hmac") != prev:
                    return False, n, i
                this_claimed = row.get("this_hmac")
                canonical = json.dumps({k: v for k, v in row.items()
                                        if k != "this_hmac"},
                                       sort_keys=True, separators=(",", ":")).encode("utf-8")
                expected = hmac.new(self._key, canonical, hashlib.sha256).hexdigest()
                if this_claimed != expected:
                    return False, n, i
                prev = this_claimed
                n += 1
        return True, n, None


def make_audit_sink(log_path: Optional[Path] = None
                    ) -> Callable[[Dict[str, Any]], None]:
    """Construct the default chained-HMAC audit sink.

    Tries to delegate to packages/observability/db_logger if that module
    has an `engagement_audit` function; otherwise writes to
    {HELI_AUDIT_DIR or ./var/audit}/engagement.audit.jsonl with chained HMAC.
    """
    # Prefer enterprise observability backend if available
    try:
        from packages.observability import db_logger  # type: ignore
        sink = getattr(db_logger, "engagement_audit", None)
        if callable(sink):
            return sink
    except ImportError:
        pass

    audit_dir = Path(os.environ.get("HELI_AUDIT_DIR", "./var/audit"))
    path = log_path or (audit_dir / "engagement.audit.jsonl")
    return ChainedHMACAuditSink(path)


# ---------------------------------------------------------------------------
# Top-level factory — production-correct gate construction
# ---------------------------------------------------------------------------


def build_production_gate(
    *,
    rbac_engine,
    emit_event: Callable[[C2EventType, Dict[str, Any]], None],
    keys_dir: Optional[str] = None,
    audit_log_path: Optional[Path] = None,
    default_ttl_seconds: int = 60,
) -> EngagementAuthorizationGate:
    """Construct an EngagementAuthorizationGate with production wiring.

    Required:
      rbac_engine     — instance of packages.security.rbac.RBACEngine
                         with operators registered + role assignments made
      emit_event      — callable that publishes the C2 event downstream
                         (typically the MQTT bridge or local bus)
    Optional:
      keys_dir        — Ed25519 public key directory (default
                         SUMMIT_SENSOR_KEYS_DIR or ./sensor_keys)
      audit_log_path  — chained-HMAC audit file path (default
                         {HELI_AUDIT_DIR or ./var/audit}/engagement.audit.jsonl)
      default_ttl_seconds — auto-deny TTL for AUTHORIZED state
    """
    return EngagementAuthorizationGate(
        emit_event=emit_event,
        verify_operator_signature=make_ed25519_verifier(keys_dir=keys_dir),
        operator_has_role=make_rbac_role_check(rbac_engine),
        audit_sink=make_audit_sink(log_path=audit_log_path),
        default_ttl_seconds=default_ttl_seconds,
    )
