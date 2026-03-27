"""
middleware/rbac.py — Role-based access control FastAPI dependency for Summit.OS.

Provides require_role(*roles) — a factory returning a FastAPI Depends() that
checks JWT role claims. When RBAC_ENFORCE=false (default) all checks are no-ops,
mirroring the API_KEY_ENFORCE pattern.

JWT role claims checked (in order):
  claims["roles"]  → list[str] or space-separated str
  claims["role"]   → str
  claims["groups"] → list[str]

Recognised role names (case-insensitive):
  VIEWER, OPERATOR, MISSION_COMMANDER, ADMIN, SUPER_ADMIN
  + UI aliases: viewer, ops → OPERATOR, command → MISSION_COMMANDER, dev → ADMIN

Role hierarchy (each role inherits everything below):
  SUPER_ADMIN > ADMIN > MISSION_COMMANDER > OPERATOR > VIEWER
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Optional

from fastapi import HTTPException, Request

logger = logging.getLogger("api-gateway.rbac")

RBAC_ENFORCE = os.getenv("RBAC_ENFORCE", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Role hierarchy — maps role → set of roles it inherits
# ---------------------------------------------------------------------------
_HIERARCHY: dict[str, frozenset[str]] = {
    "SUPER_ADMIN": frozenset({"ADMIN", "MISSION_COMMANDER", "OPERATOR", "VIEWER"}),
    "ADMIN": frozenset({"MISSION_COMMANDER", "OPERATOR", "VIEWER"}),
    "MISSION_COMMANDER": frozenset({"OPERATOR", "VIEWER"}),
    "OPERATOR": frozenset({"VIEWER"}),
    "VIEWER": frozenset(),
}

# UI / OIDC group name aliases → canonical role
_ALIAS_MAP: dict[str, str] = {
    "ops": "OPERATOR",
    "command": "MISSION_COMMANDER",
    "dev": "ADMIN",
    "admin": "ADMIN",
    "superadmin": "SUPER_ADMIN",
    "super_admin": "SUPER_ADMIN",
    "viewer": "VIEWER",
    "operator": "OPERATOR",
    "mission_commander": "MISSION_COMMANDER",
}


def _extract_roles(claims: dict) -> frozenset[str]:
    """Return the set of canonical role names present in a JWT claims dict."""
    raw: list[str] = []
    for field in ("roles", "role", "groups"):
        val = claims.get(field)
        if isinstance(val, list):
            raw.extend(str(v) for v in val)
        elif isinstance(val, str):
            raw.extend(val.split())

    result: set[str] = set()
    for token in raw:
        upper = token.upper()
        if upper in _HIERARCHY:
            result.add(upper)
        alias = _ALIAS_MAP.get(token.lower())
        if alias:
            result.add(alias)
    return frozenset(result)


def _effective_roles(user_roles: frozenset[str]) -> frozenset[str]:
    """Expand user's roles with all inherited roles."""
    effective: set[str] = set(user_roles)
    for role in user_roles:
        effective.update(_HIERARCHY.get(role, frozenset()))
    return frozenset(effective)


def _decode_jwt_payload(authorization: str) -> dict:
    """Base64url-decode the JWT payload section without signature verification."""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise ValueError("not a bearer token")
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("not a JWT")
    payload_b64 = parts[1]
    padding = 4 - len(payload_b64) % 4
    if padding != 4:
        payload_b64 += "=" * padding
    return json.loads(base64.urlsafe_b64decode(payload_b64))


def require_role(*roles: str):
    """
    Return a FastAPI Depends() callable that enforces at least one of *roles*.

    When RBAC_ENFORCE=false (default) the check is a no-op — safe for development.

    Example::

        @app.post("/v1/missions")
        async def create_mission(
            _role = Depends(require_role("MISSION_COMMANDER")),
        ): ...
    """
    required = frozenset(r.upper() for r in roles)

    async def _check(request: Request) -> Optional[dict[str, Any]]:
        if not RBAC_ENFORCE:
            return None

        authorization = request.headers.get("authorization", "")
        if not authorization:
            raise HTTPException(status_code=403, detail="Authorization header required")

        try:
            claims = _decode_jwt_payload(authorization)
        except Exception as exc:
            raise HTTPException(
                status_code=403, detail=f"Cannot decode token: {exc}"
            ) from exc

        user_roles = _extract_roles(claims)
        if not (required & _effective_roles(user_roles)):
            logger.warning(
                "RBAC denied — user_roles=%s required=%s path=%s",
                user_roles,
                required,
                request.url.path,
            )
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient role. Required: {sorted(required)}",
            )

        return claims

    return _check
