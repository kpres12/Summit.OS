"""
packages/multi_tenant — Enterprise multi-tenancy for Heli.OS.

Active only when ENTERPRISE_MULTI_TENANT=true. When false (Community mode),
all data implicitly belongs to the single "default" organization and no
tenant isolation is enforced.

Public API:
    extract_org_id(request)  — pull org_id from X-Org-ID header or JWT claim
    require_org_id           — FastAPI dependency that returns the validated org_id
    ENTERPRISE_MULTI_TENANT  — bool flag, read once at startup
"""

from .context import ENTERPRISE_MULTI_TENANT, extract_org_id
from .deps import require_org_id

__all__ = ["ENTERPRISE_MULTI_TENANT", "extract_org_id", "require_org_id"]
