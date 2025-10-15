import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

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

app = FastAPI(title="Summit API Gateway", version="0.3.0", lifespan=lifespan)
FABRIC_URL = os.getenv("FABRIC_URL", "http://fabric:8001")
FUSION_URL = os.getenv("FUSION_URL", "http://fusion:8002")
TASKING_URL = os.getenv("TASKING_URL", "http://tasking:8004")

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


class TaskSubmitRequest(BaseModel):
    asset_id: str
    action: str
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    waypoints: list = []

class TaskApproveRequest(BaseModel):
    approved_by: str


@app.post("/v1/tasks")
async def submit_task(req: TaskSubmitRequest):
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
async def approve_task(task_id: str, req: TaskApproveRequest):
    """Approve a pending task and dispatch it."""
    assert SessionLocal is not None
    
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
async def list_pending_tasks():
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
async def list_alerts():
    return {"alerts": []}


@app.get("/v1/mission/{asset_id}/status")
async def get_mission_status(asset_id: str):
    return {
        "asset_id": asset_id,
        "status": "IDLE",
        "ts_iso": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/v1/worldstate")
async def get_world_state():
    return {"entities": []}

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
async def proxy_register_node(req: NodeRegisterRequest):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(f"{FABRIC_URL}/api/v1/nodes/register", json=req.model_dump())
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fabric upstream error: {e}")

@app.delete("/api/v1/nodes/{node_id}")
async def proxy_retire_node(node_id: str):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.delete(f"{FABRIC_URL}/api/v1/nodes/{node_id}")
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fabric upstream error: {e}")


@app.get("/v1/observations")
async def get_observations(cls: str | None = None, limit: int = 50):
    params = {"limit": str(limit)}
    if cls:
        params["cls"] = cls
    url = f"{FUSION_URL}/observations"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return {"observations": r.json()}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Fusion upstream error: {e}")


# Proxy missions to tasking (authoritative)
@app.post("/v1/missions")
async def create_mission_proxy(payload: dict):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{TASKING_URL}/api/v1/missions", json=payload)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Tasking upstream error: {e}")


@app.get("/v1/missions/{mission_id}")
async def get_mission_proxy(mission_id: str):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{TASKING_URL}/api/v1/missions/{mission_id}")
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Tasking upstream error: {e}")
