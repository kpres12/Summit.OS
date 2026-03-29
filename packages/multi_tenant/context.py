"""
packages/multi_tenant/context.py — org_id extraction from incoming requests.

Extraction order:
  1. X-Org-ID HTTP header  (preferred — explicit, cacheable at proxy)
  2. JWT claim: "org_id", "org", or "tenant"  (token-based clients)

In Community mode (ENTERPRISE_MULTI_TENANT=false) always returns "default"
so existing code paths work without any caller changes.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import Request

logger = logging.getLogger("multi_tenant.context")

ENTERPRISE_MULTI_TENANT: bool = (
    os.getenv("ENTERPRISE_MULTI_TENANT", "false").lower() == "true"
)

if ENTERPRISE_MULTI_TENANT:
    logger.info("Enterprise multi-tenancy ENABLED")
else:
    logger.info("Enterprise multi-tenancy DISABLED — single-org Community mode")


def extract_org_id(request: Request) -> Optional[str]:
    """
    Extract org_id from the request without raising.

    Returns None when Enterprise mode is off (callers should treat as "default").
    Returns None when no org_id can be found in Enterprise mode (caller raises 400).
    """
    if not ENTERPRISE_MULTI_TENANT:
        return "default"

    # 1. Explicit header (fastest, works for service-to-service and browser clients)
    org_id = request.headers.get("X-Org-ID") or request.headers.get("x-org-id")
    if org_id:
        return org_id.strip()

    # 2. JWT claim fallback
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        try:
            # Decode without signature verification — auth middleware has already
            # validated the signature by the time we reach this point.
            from jose import jwt as _jwt

            claims = _jwt.get_unverified_claims(token)
            org_id = (
                claims.get("org_id")
                or claims.get("org")
                or claims.get("tenant")
            )
            if org_id:
                return str(org_id).strip()
        except Exception as exc:
            logger.debug("JWT claim extraction failed: %s", exc)

    return None
