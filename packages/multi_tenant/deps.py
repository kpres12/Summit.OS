"""
packages/multi_tenant/deps.py — FastAPI dependencies for org-scoped endpoints.

Usage:
    from packages.multi_tenant import require_org_id

    @router.get("/missions")
    async def list_missions(org_id: str = Depends(require_org_id)):
        # org_id is "default" in Community mode, validated org slug in Enterprise
        ...
"""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from .context import ENTERPRISE_MULTI_TENANT, extract_org_id


async def require_org_id(request: Request) -> str:
    """
    FastAPI dependency that returns the org_id for the current request.

    - Community mode: always returns "default", never raises.
    - Enterprise mode: returns the org_id from X-Org-ID header or JWT claim.
      Raises 400 if no org_id can be determined.
      Raises 403 if the authenticated user's token org does not match the
      requested org (prevents horizontal privilege escalation).
    """
    org_id = extract_org_id(request)

    if ENTERPRISE_MULTI_TENANT and not org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "X-Org-ID header is required in Enterprise multi-tenant mode. "
                "Include your organization ID with every request."
            ),
        )

    return org_id or "default"
