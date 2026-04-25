"""Pure utilities, auth helpers, org helpers, and MQTT helpers."""
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import HTTPException, Request

import state

logger = logging.getLogger("tasking")


def _safe_json(val):
    """Parse JSON string to dict/list if needed (SQLite returns JSON columns as strings)."""
    if val is None:
        return None
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return None
    return val


def _safe_isoformat(val):
    """Return ISO string from datetime or passthrough string."""
    if val is None:
        return None
    if isinstance(val, str):
        return val
    return val.isoformat()


def _to_asyncpg_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


async def _require_auth(request: Request):
    if not state.OIDC_ENFORCE:
        return
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth.split(" ", 1)[1]
    if not state.OIDC_AVAILABLE or not state.OIDC_ISSUER:
        # If enforcement is enabled but libs/config missing, deny
        raise HTTPException(status_code=401, detail="OIDC unavailable")
    try:
        # NOTE: In production, fetch JWKS and verify signature and claims properly
        # This is a placeholder decode without verification context
        from jose import jwt
        jwt.get_unverified_claims(token)
        return
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


def _get_org_id(request: Request) -> str:
    """Extract org_id from X-Org-ID header or JWT claims. Returns 'default' in Community mode."""
    if not state._ENTERPRISE_MULTI_TENANT:
        return "default"
    org_id = request.headers.get("X-Org-ID") or request.headers.get("x-org-id")
    if org_id:
        return org_id.strip()
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and state.OIDC_AVAILABLE:
        try:
            from jose import jwt
            claims = jwt.get_unverified_claims(auth[7:])
            org_id = claims.get("org_id") or claims.get("org") or claims.get("tenant")
            if org_id:
                return str(org_id).strip()
        except Exception:
            pass
    return "default"


async def _publish_mission_update(mission_id: str, event: Dict[str, Any]):
    if not state.mqtt_client:
        return
    payload = json.dumps(
        {
            "mission_id": mission_id,
            **event,
            "ts_iso": datetime.now(timezone.utc).isoformat(),
        }
    )
    state.mqtt_client.publish("missions/updates", payload, qos=1)
    state.mqtt_client.publish(f"missions/{mission_id}", payload, qos=1)
