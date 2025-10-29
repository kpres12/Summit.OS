import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
import httpx
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from typing import Any

# Optional OIDC/JWT enforcement
try:
    from jose import jwt
    OIDC_JOSE_AVAILABLE = True
except Exception:
    OIDC_JOSE_AVAILABLE = False

# Globals
engine: Optional[AsyncEngine] = None
SessionLocal: Optional[sessionmaker] = None

# DB Table for task approvals
metadata = MetaData()
approvals = Table(
    "approvals",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("task_id", String(128), nullable=False, unique=True),
    Column("asset_id", String(128)),
    Column("action", String(256)),
    Column("risk_level", String(32)),
    Column("status", String(32)),  # PENDING_APPROVAL, APPROVED, REJECTED
    Column("approved_by", String(128)),
    Column("created_at", DateTime(timezone=True)),
    Column("approved_at", DateTime(timezone=True))
)

def _to_asyncpg_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, SessionLocal
    
    # Setup DB for approvals
    pg_url = _to_asyncpg_url(os.getenv("POSTGRES_URL", "postgresql://summit:summit_password@localhost:5432/summit_os"))
    engine = create_async_engine(pg_url, echo=False, future=True)
    SessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    
    yield
    
    if engine:
        await engine.dispose()

app = FastAPI(title="Summit API Gateway", version="0.4.1", lifespan=lifespan)

# Unified error response shape
from fastapi.responses import JSONResponse
from fastapi.requests import Request as _Request

@app.exception_handler(HTTPException)
async def http_exc_handler(_req: _Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content=ErrorResponse(error="HTTPException", detail=exc.detail).model_dump())
FABRIC_URL = os.getenv("FABRIC_URL", "http://fabric:8001")
FUSION_URL = os.getenv("FUSION_URL", "http://fusion:8002")
TASKING_URL = os.getenv("TASKING_URL", "http://tasking:8004")
INTELLIGENCE_URL = os.getenv("INTELLIGENCE_URL", "http://intelligence:8003")

# OIDC config (optional)
OIDC_ENFORCE = os.getenv("OIDC_ENFORCE", "false").lower() == "true"
OIDC_ISSUER = os.getenv("OIDC_ISSUER")
OIDC_AUDIENCE = os.getenv("OIDC_AUDIENCE")
OIDC_JWKS_URL = os.getenv("OIDC_JWKS_URL")
_JWKS_CACHE: dict[str, Any] | None = None

# Optional mission plugins (e.g., sentinel)
MISSIONS = os.getenv("MISSIONS", "").split(",") if os.getenv("MISSIONS") else []
try:
    if any(m.strip().lower() == "sentinel" for m in MISSIONS):
        from .plugins.sentinel import register as register_sentinel
        register_sentinel(app, FUSION_URL)
    elif any(m.strip().lower() == "wildfire" for m in MISSIONS):  # backward-compatible
        from .plugins.wildfire import register as register_wildfire
        register_wildfire(app, FUSION_URL)
except Exception:
    # Plugins are optional; don't fail API startup if missing
    pass


@app.get("/health")
async def health():
    return {"status": "ok", "service": "api-gateway"}

@app.get("/readyz")
async def readyz():
    try:
        assert SessionLocal is not None
        async with SessionLocal() as session:
            await session.execute(text("SELECT 1"))
        # Probe downstream health endpoints quickly
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.get(f"{FABRIC_URL}/health")
            await client.get(f"{FUSION_URL}/health")
            await client.get(f"{INTELLIGENCE_URL}/health")
            await client.get(f"{TASKING_URL}/health")
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {e}")

@app.get("/livez")
async def livez():
    return {"status": "alive"}

# -------------------------
# Org scoping and security helpers
# -------------------------
from pydantic import BaseModel as _BaseModel

class ErrorResponse(_BaseModel):
    error: str
    detail: dict | list | str | None = None

async def get_org_id(request: Request) -> str | None:
    # Prefer explicit header set by mTLS proxy; fallback to None
    org = request.headers.get("X-Org-ID") or request.headers.get("x-org-id")
    return org

# -------------------------
# Security helpers (optional OIDC)
# -------------------------
async def _load_jwks() -> dict | None:
    global _JWKS_CACHE
    if not (OIDC_ENFORCE and OIDC_JOSE_AVAILABLE and OIDC_JWKS_URL):
        return None
    if _JWKS_CACHE is None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(OIDC_JWKS_URL)
                r.raise_for_status()
                _JWKS_CACHE = r.json()
        except Exception:
            _JWKS_CACHE = None
    return _JWKS_CACHE

def _select_jwk(jwks: dict, kid: str | None) -> dict | None:
    try:
        keys = jwks.get("keys", []) if isinstance(jwks, dict) else []
        if kid:
            for k in keys:
                if k.get("kid") == kid:
                    return k
        return keys[0] if keys else None
    except Exception:
        return None

async def verify_bearer(authorization: str | None = None) -> dict | None:
    """FastAPI dependency to optionally enforce OIDC JWTs on protected routes."""
    if not OIDC_ENFORCE:
        return None
    if not OIDC_JOSE_AVAILABLE:
        raise HTTPException(status_code=500, detail="OIDC enforcement enabled but jose not installed")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    jwks = await _load_jwks()
    try:
        headers = jwt.get_unverified_header(token)
        claims = jwt.get_unverified_claims(token)
        if OIDC_AUDIENCE and claims.get("aud") and OIDC_AUDIENCE not in (claims.get("aud") if isinstance(claims.get("aud"), list) else [claims.get("aud")]):
            raise HTTPException(status_code=401, detail="Invalid audience")
        if OIDC_ISSUER and claims.get("iss") != OIDC_ISSUER:
            raise HTTPException(status_code=401, detail="Invalid issuer")
        # Verify signature if JWKS available
        if jwks:
            jwk = _select_jwk(jwks, headers.get("kid"))
            if jwk is None:
                raise HTTPException(status_code=401, detail="No matching JWK")
            from jose.utils import base64url_decode
            try:
                jwt.decode(token, jwk, algorithms=[headers.get("alg") or "RS256"], audience=OIDC_AUDIENCE, issuer=OIDC_ISSUER)
            except Exception as e:
                raise HTTPException(status_code=401, detail=f"Signature verification failed: {e}")
        return claims
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


class TaskSubmitRequest(BaseModel):
    asset_id: str
    action: str
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    waypoints: list = []

class TaskApproveRequest(BaseModel):
    approved_by: str


async def _opa_check(payload: dict) -> bool:
    opa_url = os.getenv("OPA_URL", "http://opa:8181/v1/data/policy/allow")
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.post(opa_url, json={"input": payload})
            r.raise_for_status()
            data = r.json()
            return bool(data.get("result", False))
    except Exception:
        # Fail open if OPA not reachable
        return True

@app.post("/v1/tasks")
async def submit_task(req: TaskSubmitRequest, _claims: dict | None = Depends(verify_bearer)):
    """Submit a task. High-risk tasks require approval before dispatch."""
    assert SessionLocal is not None
    
    task_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)
    
    # High-risk tasks require human approval
    if req.risk_level in ["HIGH", "CRITICAL"]:
        status = "PENDING_APPROVAL"
        
        # Store approval record
        async with SessionLocal() as session:
            await session.execute(
                approvals.insert().values(
                    task_id=task_id,
                    asset_id=req.asset_id,
                    action=req.action,
                    risk_level=req.risk_level,
                    status=status,
                    created_at=created_at
                )
            )
            await session.commit()
        
        return {
            "task_id": task_id,
            "status": status,
            "message": "Task requires approval due to high risk level"
        }
    else:
        # Policy check (OPA) before dispatch
        approved = True
        try:
            approved = await _opa_check({
                "action": "dispatch",
                "risk_level": req.risk_level,
                "approved": True,  # low/med implicit
                "in_geofence": True  # assume OK; future: query fabric
            })
        except Exception:
            approved = True
        if not approved:
            raise HTTPException(status_code=403, detail="Policy denied dispatch")

        # Low/Medium risk: dispatch immediately
        status = "DISPATCHED"
        
        # Dispatch to Tasking service
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(
                    f"{TASKING_URL}/dispatch",
                    json={
                        "task_id": task_id,
                        "asset_id": req.asset_id,
                        "action": req.action,
                        "waypoints": req.waypoints
                    }
                )
                r.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Tasking service error: {e}")
        
        return {"task_id": task_id, "status": status}


@app.post("/v1/tasks/{task_id}/approve")
async def approve_task(task_id: str, req: TaskApproveRequest, _claims: dict | None = Depends(verify_bearer)):
    """Approve a pending task and dispatch it."""
    assert SessionLocal is not None

    # Basic RBAC: require supervisor/admin if OIDC enforced
    if OIDC_ENFORCE:
        roles = set((_claims or {}).get("roles") or (_claims or {}).get("realm_access", {}).get("roles", []) or [])
        if not ("supervisor" in roles or "admin" in roles):
            raise HTTPException(status_code=403, detail="Insufficient role")
    
    # Fetch approval record
    async with SessionLocal() as session:
        result = await session.execute(
            text("SELECT task_id, asset_id, action, risk_level, status FROM approvals WHERE task_id = :tid"),
            {"tid": task_id}
        )
        row = result.first()
        
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if row.status != "PENDING_APPROVAL":
            raise HTTPException(status_code=400, detail=f"Task already {row.status}")
        
        # Update approval
        await session.execute(
            approvals.update().where(approvals.c.task_id == task_id).values(
                status="APPROVED",
                approved_by=req.approved_by,
                approved_at=datetime.now(timezone.utc)
            )
        )
        await session.commit()
    # Policy check (OPA): require approved + geofence
    try:
        geo_ok = True
        try:
            # optional: check geofence containment if waypoints present
            if req and req.model_dump().get("waypoints"):
                # sample first waypoint
                wp = req.model_dump()["waypoints"][0] if req.model_dump()["waypoints"] else None
                if wp and isinstance(wp, dict):
                    lat = wp.get("lat")
                    lon = wp.get("lon")
                    if lat is not None and lon is not None:
                        async with httpx.AsyncClient(timeout=3.0) as client:
                            rr = await client.get(f"{FABRIC_URL}/api/v1/geofences/contains", params={"lat": lat, "lon": lon})
                            rr.raise_for_status()
                            geo_ok = bool(rr.json().get("contains", True))
        except Exception:
            geo_ok = True
        pol_ok = await _opa_check({
            "action": "dispatch",
            "risk_level": "APPROVED",
            "approved": True,
            "in_geofence": geo_ok,
        })
        if not pol_ok:
            raise HTTPException(status_code=403, detail="Policy denied dispatch")
    except HTTPException:
        raise
    except Exception:
        pass
    
    # Dispatch to Tasking
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(
                f"{TASKING_URL}/dispatch",
                json={
                    "task_id": task_id,
                    "asset_id": row.asset_id,
                    "action": row.action
                }
            )
            r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Tasking service error: {e}")
    
    return {
        "task_id": task_id,
        "status": "APPROVED",
        "approved_by": req.approved_by,
        "approved_at": datetime.now(timezone.utc).isoformat()
    }


@app.get("/v1/tasks/pending")
async def list_pending_tasks(_claims: dict | None = Depends(verify_bearer)):
    """List all tasks pending approval."""
    assert SessionLocal is not None
    
    async with SessionLocal() as session:
        result = await session.execute(
            text("SELECT task_id, asset_id, action, risk_level, created_at FROM approvals WHERE status = 'PENDING_APPROVAL' ORDER BY created_at DESC")
        )
        rows = result.all()
        
        return {
            "pending_tasks": [
                {
                    "task_id": r.task_id,
                    "asset_id": r.asset_id,
                    "action": r.action,
                    "risk_level": r.risk_level,
                    "created_at": r.created_at.isoformat()
                }
                for r in rows
            ]
        }


@app.get("/v1/alerts")
async def list_alerts(limit: int = 100):
    """Return recent alerts from Fabric's worldstate."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{FABRIC_URL}/api/v1/worldstate")
            r.raise_for_status()
            data = r.json()
            alerts = data.get("alerts", [])
            return {"alerts": alerts[:limit]}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fabric upstream error: {e}")


@app.get("/v1/mission/{asset_id}/status")
async def get_mission_status(asset_id: str, _claims: dict | None = Depends(verify_bearer)):
    return {
        "asset_id": asset_id,
        "status": "IDLE",
        "ts_iso": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/v1/worldstate")
async def get_world_state(org_id: str | None = Depends(get_org_id)):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            headers = {"X-Org-ID": org_id} if org_id else None
            r = await client.get(f"{FABRIC_URL}/api/v1/worldstate", headers=headers)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fabric upstream error: {e}")

# -------------------------
# Node registry proxy to Fabric
# -------------------------
class NodeRegisterRequest(BaseModel):
    id: str
    type: str
    pubkey: str | None = None
    fw_version: str | None = None
    location: dict | None = None
    capabilities: list[str] = []
    comm: list[str] = []

@app.post("/api/v1/nodes/register")
async def proxy_register_node(req: NodeRegisterRequest, _claims: dict | None = Depends(verify_bearer), org_id: str | None = Depends(get_org_id)):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            headers = {"X-Org-ID": org_id} if org_id else None
            r = await client.post(f"{FABRIC_URL}/api/v1/nodes/register", json=req.model_dump(), headers=headers)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fabric upstream error: {e}")

@app.delete("/api/v1/nodes/{node_id}")
async def proxy_retire_node(node_id: str, _claims: dict | None = Depends(verify_bearer), org_id: str | None = Depends(get_org_id)):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            headers = {"X-Org-ID": org_id} if org_id else None
            r = await client.delete(f"{FABRIC_URL}/api/v1/nodes/{node_id}", headers=headers)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fabric upstream error: {e}")


@app.get("/v1/observations")
async def get_observations(cls: str | None = None, limit: int = 50, org_id: str | None = Depends(get_org_id)):
    params = {"limit": str(limit)}
    if cls:
        params["cls"] = cls
    url = f"{FUSION_URL}/observations"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            headers = {"X-Org-ID": org_id} if org_id else None
            r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            return {"observations": r.json()}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fusion upstream error: {e}")


@app.get("/v1/advisories")
async def get_advisories(risk_level: str | None = None, limit: int = 50, org_id: str | None = Depends(get_org_id)):
    params = {"limit": str(limit)}
    if risk_level:
        params["risk_level"] = risk_level
    url = f"{INTELLIGENCE_URL}/advisories"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            headers = {"X-Org-ID": org_id} if org_id else None
            r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            return {"advisories": r.json()}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Intelligence upstream error: {e}")


# Proxy missions to tasking (authoritative) with schema validation
@app.post("/v1/missions")
async def create_mission_proxy(payload: dict, _claims: dict | None = Depends(verify_bearer), org_id: str | None = Depends(get_org_id)):
    # Validate against mission schema if present
    try:
        import json, os
        from jsonschema import validate
        schema_path = os.path.join(os.path.dirname(__file__), "..", "..", "packages", "contracts", "jsonschemas", "mission.schema.json")
        schema_path = os.path.abspath(schema_path)
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        validate(instance=payload, schema=schema)
    except Exception:
        pass
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {"X-Org-ID": org_id} if org_id else None
            r = await client.post(f"{TASKING_URL}/api/v1/missions", json=payload, headers=headers)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Tasking upstream error: {e}")


@app.get("/v1/missions/{mission_id}")
async def get_mission_proxy(mission_id: str, _claims: dict | None = Depends(verify_bearer)):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{TASKING_URL}/api/v1/missions/{mission_id}")
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Tasking upstream error: {e}")
