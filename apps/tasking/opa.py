"""
OPA Policy Engine Client for Summit.OS Tasking

Fail-closed by default: if OPA is unreachable, deny the action.
All policy evaluations are audit-logged to a local SQLite database.

Toggle behavior via env:
  OPA_FAIL_MODE=closed (default) | open
  OPA_URL=http://opa:8181
  OPA_TIMEOUT=3.0
  OPA_AUDIT_DB=policy_audit.db
"""

from __future__ import annotations

import httpx
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tasking.opa")

DEFAULT_OPA_URL = os.getenv("OPA_URL", "http://opa:8181")

# Policy signature verification (non-fatal if PyNaCl not installed)
_policy_signer = None
_POLICY_DIR = os.getenv("POLICY_DIR", "/infra/policy")


def _get_policy_signer():
    global _policy_signer
    if _policy_signer is None:
        try:
            import sys as _sys
            from pathlib import Path as _Path

            _pkgs = str(_Path(__file__).resolve().parents[2] / "packages")
            if _pkgs not in _sys.path:
                _sys.path.insert(0, _pkgs)
            from policy.signer import PolicySigner

            _policy_signer = PolicySigner()
        except Exception:
            pass
    return _policy_signer


def verify_policies(policy_dir: str = _POLICY_DIR) -> bool:
    """
    Verify all policy file signatures before Summit.OS loads them into OPA.
    Call this at service startup.

    Returns True if all policies are valid (or verification is unavailable).
    Raises PolicyVerificationError if any policy is tampered (enforce mode).
    """
    import os

    if not os.path.isdir(policy_dir):
        logger.debug(f"Policy dir not found at {policy_dir} — skipping verification")
        return True
    signer = _get_policy_signer()
    if signer is None:
        logger.debug("Policy signer unavailable — skipping verification")
        return True
    try:
        signer.verify_directory(policy_dir)
        logger.info(f"All policies in {policy_dir} verified successfully")
        return True
    except Exception as e:
        logger.critical(f"Policy verification FAILED: {e}")
        raise


OPA_FAIL_MODE = os.getenv("OPA_FAIL_MODE", "closed").lower()  # "closed" or "open"
OPA_TIMEOUT = float(os.getenv("OPA_TIMEOUT", "3.0"))
OPA_AUDIT_DB = os.getenv("OPA_AUDIT_DB", "policy_audit.db")


def _init_audit_db(db_path: str):
    """Create audit table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS policy_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            rule TEXT NOT NULL,
            input_summary TEXT,
            result TEXT NOT NULL,
            allowed INTEGER NOT NULL,
            reasons TEXT,
            fail_mode TEXT,
            latency_ms REAL,
            error TEXT
        )
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_policy_audit_ts ON policy_audit (ts)
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_policy_audit_allowed ON policy_audit (allowed)
    """
    )
    conn.commit()
    conn.close()


# Initialize audit DB at import time
try:
    _init_audit_db(OPA_AUDIT_DB)
except Exception as e:
    logger.warning(f"Could not initialize audit DB: {e}")


def _audit_log(
    rule: str,
    input_summary: str,
    result: str,
    allowed: bool,
    reasons: List[str],
    fail_mode: str,
    latency_ms: float,
    error: Optional[str] = None,
):
    """Write an audit record to the local SQLite DB."""
    try:
        conn = sqlite3.connect(OPA_AUDIT_DB)
        conn.execute(
            """
            INSERT INTO policy_audit (ts, rule, input_summary, result, allowed, reasons, fail_mode, latency_ms, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                rule,
                input_summary[:2048] if input_summary else "",
                result[:4096] if result else "",
                1 if allowed else 0,
                json.dumps(reasons),
                fail_mode,
                latency_ms,
                error,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Audit log write failed: {e}")


class OPAClient:
    """
    OPA policy evaluation client.

    Default behavior is FAIL-CLOSED: if OPA cannot be reached,
    the action is denied. Set OPA_FAIL_MODE=open to restore
    the previous fail-open behavior (development only).
    """

    def __init__(
        self,
        base_url: str | None = None,
        policy_path: str = "/v1/data/policy",
        fail_mode: str | None = None,
    ) -> None:
        self.base_url = base_url or DEFAULT_OPA_URL
        self.policy_path = policy_path
        self.fail_mode = fail_mode or OPA_FAIL_MODE

    async def evaluate(self, rule: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate an OPA policy rule.

        Args:
            rule: e.g., "missions/allow"
            input_data: input JSON for policy
        Returns: result dict with keys: allow(bool), deny_reasons(list[str])
        """
        url = f"{self.base_url}{self.policy_path}/{rule.lstrip('/')}"
        start = time.monotonic()
        error_msg: Optional[str] = None

        try:
            async with httpx.AsyncClient(timeout=OPA_TIMEOUT) as client:
                r = await client.post(url, json={"input": input_data})
                r.raise_for_status()
                data = r.json() or {}
                latency = (time.monotonic() - start) * 1000

                # OPA returns {"result": {...}} or a raw value
                if isinstance(data, dict) and "result" in data:
                    result = data["result"] or {}
                else:
                    result = data

                allowed = bool(result.get("allow", False))
                reasons = result.get("deny_reasons") or result.get("reasons") or []
                if isinstance(reasons, str):
                    reasons = [reasons]

                # Build summary for audit
                input_summary = json.dumps(
                    {k: str(v)[:256] for k, v in input_data.items()},
                    default=str,
                )

                _audit_log(
                    rule=rule,
                    input_summary=input_summary,
                    result=json.dumps(result, default=str),
                    allowed=allowed,
                    reasons=reasons,
                    fail_mode="normal",
                    latency_ms=latency,
                )

                return result

        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            error_msg = str(exc)

            if self.fail_mode == "open":
                logger.warning(f"OPA unreachable, FAIL-OPEN: {error_msg}")
                _audit_log(
                    rule=rule,
                    input_summary=json.dumps(input_data, default=str)[:2048],
                    result="{}",
                    allowed=True,
                    reasons=[],
                    fail_mode="open",
                    latency_ms=latency,
                    error=error_msg,
                )
                return {"allow": True, "deny_reasons": []}
            else:
                logger.error(f"OPA unreachable, FAIL-CLOSED: {error_msg}")
                _audit_log(
                    rule=rule,
                    input_summary=json.dumps(input_data, default=str)[:2048],
                    result="{}",
                    allowed=False,
                    reasons=["OPA unreachable — fail-closed"],
                    fail_mode="closed",
                    latency_ms=latency,
                    error=error_msg,
                )
                return {
                    "allow": False,
                    "deny_reasons": ["OPA unreachable — fail-closed"],
                }

    async def evaluate_pre_dispatch(
        self,
        mission_id: str,
        asset_id: str,
        plan: Dict[str, Any],
        org_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Pre-dispatch policy check: evaluate whether a specific
        asset dispatch is allowed BEFORE sending MQTT commands.

        This is the last gate before a real-world effect.
        """
        input_data = {
            "mission_id": mission_id,
            "asset_id": asset_id,
            "plan": plan,
            "org_id": org_id or "dev",
            "context": {
                "time": datetime.now(timezone.utc).isoformat(),
                "check_type": "pre_dispatch",
            },
        }
        return await self.evaluate("missions/dispatch", input_data)

    async def evaluate_geofence(
        self,
        asset_id: str,
        waypoints: List[Dict[str, float]],
        org_id: Optional[str] = None,
        geofences: Optional[List[Dict[str, Any]]] = None,
        max_altitude_m: float = 122.0,
    ) -> Dict[str, Any]:
        """Check waypoints against geofence policy."""
        input_data = {
            "asset_id": asset_id,
            "waypoints": waypoints,
            "org_id": org_id or "dev",
            "context": {
                "time": datetime.now(timezone.utc).isoformat(),
                "check_type": "geofence",
                "geofences": geofences or [],
                "max_altitude_m": max_altitude_m,
            },
        }
        return await self.evaluate("geofence/check", input_data)

    async def evaluate_actuator_command(
        self,
        command_type: str,
        target_id: str,
        parameters: Dict[str, Any],
        world_state_entities: Dict[str, Any],
        org_id: Optional[str] = None,
        requested_by: str = "operator",
        approved: bool = False,
        mission_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Pre-flight safety check for physical actuator commands.

        This is the last gate before Summit.OS sends a command to real
        hardware. If this returns allow=False, the command is rejected
        and the denial reason is surfaced to the operator.

        Args:
            command_type: e.g. "set_valve", "set_motor_speed", "emergency_stop"
            target_id: entity_id of the target actuator
            parameters: command-specific parameters (e.g. {"state": "closed"})
            world_state_entities: current world model snapshot (entity_id -> entity)
            requested_by: "operator", "ai-agent", or "mission-planner"
            approved: whether a human operator has approved this command
            mission_id: optional mission context
        """
        input_data = {
            "command": {
                "type": command_type,
                "target_id": target_id,
                "parameters": parameters,
            },
            "world_state": {
                "entities": world_state_entities,
            },
            "org_id": org_id or "dev",
            "requested_by": requested_by,
            "context": {
                "time": datetime.now(timezone.utc).isoformat(),
                "check_type": "actuator_command",
                "approved": approved,
                "mission_id": mission_id or "",
            },
        }
        return await self.evaluate("actuators/allow", input_data)


async def get_audit_log(
    limit: int = 100,
    allowed_only: Optional[bool] = None,
    since: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query the local policy audit log."""
    conn = sqlite3.connect(OPA_AUDIT_DB)
    conn.row_factory = sqlite3.Row
    query = "SELECT * FROM policy_audit"
    conditions = []
    params: List[Any] = []

    if allowed_only is not None:
        conditions.append("allowed = ?")
        params.append(1 if allowed_only else 0)
    if since:
        conditions.append("ts >= ?")
        params.append(since)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    cursor = conn.execute(query, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
