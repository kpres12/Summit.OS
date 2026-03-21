import logging
import os
import sys
import json
import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, List
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import redis.asyncio as aioredis

# Make intelligence module importable by sub-modules
sys.path.insert(0, os.path.dirname(__file__))

logger = logging.getLogger("intelligence")
logging.basicConfig(level=logging.INFO)
from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, and_, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Optional XGBoost
try:
    import xgboost as xgb  # type: ignore
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# Test mode
INTELLIGENCE_TEST_MODE = os.getenv("INTELLIGENCE_TEST_MODE", "false").lower() == "true"

# Brain / agent registry (optional — only active when Ollama is available)
try:
    from mission_agent import AgentRegistry
    from brain import Brain
    _BRAIN_AVAILABLE = True
except Exception as _brain_err:
    _BRAIN_AVAILABLE = False
    logger.debug(f"Brain not loaded: {_brain_err}")

# Globals
engine: Optional[AsyncEngine] = None
SessionLocal: Optional[sessionmaker] = None
redis_client: Optional[aioredis.Redis] = None
agent_registry: Optional["AgentRegistry"] = None  # type: ignore

# Optional XGBoost model
risk_model: Optional["xgb.Booster"] = None  # type: ignore
INTELLIGENCE_ENABLE_XGB = (os.getenv("INTELLIGENCE_ENABLE_XGB", "false").lower() == "true")
MODEL_REGISTRY = os.getenv("MODEL_REGISTRY", "/models")
RISK_MODEL_PATH = os.getenv("INTELLIGENCE_MODEL_PATH") or ""

# DB Table for advisories
metadata = MetaData()
advisories = Table(
    "advisories",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("advisory_id", String(128), nullable=False, unique=True),
    Column("observation_id", Integer),
    Column("risk_level", String(32)),
    Column("message", String(512)),
    Column("confidence", Float),
    Column("ts", DateTime(timezone=True)),
    Column("org_id", String(128))
)

def _to_asyncpg_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, SessionLocal, redis_client

    # DB setup
    if INTELLIGENCE_TEST_MODE:
        pg_url = "sqlite+aiosqlite://"
    else:
        pg_url = _to_asyncpg_url(os.getenv("POSTGRES_URL", "postgresql://summit:summit_password@localhost:5432/summit_os"))
    engine = create_async_engine(pg_url, echo=False, future=True)
    SessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        if not INTELLIGENCE_TEST_MODE:
            # Optional Timescale hypertable setup
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
                await conn.execute(text("SELECT create_hypertable('advisories','ts', if_not_exists => TRUE)"))
            except Exception as e:
                logger.debug("Suppressed error", exc_info=True)  # was: pass

    if not INTELLIGENCE_TEST_MODE:
        # Redis setup (consume observations from Fusion)
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = await aioredis.from_url(redis_url, decode_responses=True)
        # Ensure consumer group
        try:
            await redis_client.xgroup_create("observations_stream", "intelligence", id="$", mkstream=True)
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                pass

    # Optional: load XGBoost model
    if INTELLIGENCE_ENABLE_XGB and XGB_AVAILABLE:
        model_path = RISK_MODEL_PATH or os.path.join(MODEL_REGISTRY, "risk_model.json")
        try:
            globals()['risk_model'] = xgb.Booster()  # type: ignore
            risk_model.load_model(model_path)  # type: ignore
        except Exception:
            globals()['risk_model'] = None

    # Start background risk scoring processor (skip in test mode)
    task = None
    prune_task = None
    if not INTELLIGENCE_TEST_MODE:
        task = asyncio.create_task(_risk_scoring_processor())

    # Initialise agent registry
    global agent_registry
    if _BRAIN_AVAILABLE:
        agent_registry = AgentRegistry()
        prune_task = asyncio.create_task(_agent_prune_loop())
        logger.info("AI brain initialised (AgentRegistry ready)")
    else:
        logger.info("AI brain not loaded — intelligence running in rule-based mode")

    try:
        yield
    finally:
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if prune_task:
            prune_task.cancel()
            try:
                await prune_task
            except asyncio.CancelledError:
                pass
        if redis_client:
            await redis_client.close()
        if engine:
            await engine.dispose()


async def _agent_prune_loop():
    """Periodically prune finished agents to prevent unbounded memory growth."""
    while True:
        await asyncio.sleep(300)
        if agent_registry:
            n = agent_registry.prune_finished(keep=50)
            if n > 0:
                logger.info(f"Pruned {n} finished agents")

app = FastAPI(title="Summit Intelligence", version="0.2.0", lifespan=lifespan)

# ── OpenTelemetry tracing middleware ──────────────────────────────────────────
try:
    _otel_root = os.path.join(os.path.dirname(__file__), "../..")
    if _otel_root not in sys.path:
        sys.path.insert(0, _otel_root)
    from packages.observability.tracing import get_tracer, create_tracing_middleware
    _tracer = get_tracer("summit-intelligence")
    app.middleware("http")(create_tracing_middleware(_tracer))
except Exception as _e:
    logging.warning("OTel middleware not wired: %s", _e)

class Advisory(BaseModel):
    advisory_id: str
    observation_id: int | None
    risk_level: str
    message: str
    confidence: float
    ts: datetime

@app.get("/health")
async def health():
    return {"status": "ok", "service": "intelligence"}

@app.get("/readyz")
async def readyz():
    try:
        assert SessionLocal is not None
        async with SessionLocal() as session:
            await session.execute(text("SELECT 1"))
        if redis_client:
            await redis_client.ping()
        return {"status": "ready"}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"Not ready: {e}")

@app.get("/livez")
async def livez():
    return {"status": "alive"}

from fastapi import Request as _Req, HTTPException

# ── AI Brain endpoints ────────────────────────────────────────────────────────

@app.get("/brain/status")
async def brain_status():
    """Check if the local LLM (Ollama) brain is available."""
    if not _BRAIN_AVAILABLE:
        return {"available": False, "reason": "brain module not loaded"}
    import httpx
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{ollama_url}/api/tags")
            if r.status_code == 200:
                tags = r.json().get("models", [])
                names = [t.get("name", "").split(":")[0] for t in tags]
                model_base = model.split(":")[0]
                return {
                    "available": model_base in names,
                    "ollama_url": ollama_url,
                    "model": model,
                    "available_models": names,
                }
    except Exception as e:
        return {"available": False, "ollama_url": ollama_url, "reason": str(e)}
    return {"available": False}


class AgentCreateRequest(BaseModel):
    mission_objective: str
    mission_id: Optional[str] = None
    tick_interval: float = float(os.getenv("AGENT_TICK_INTERVAL", "30"))
    max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "20"))


@app.post("/agents", status_code=201)
async def create_agent(req: AgentCreateRequest):
    """Launch an autonomous mission agent with a natural-language objective."""
    if not _BRAIN_AVAILABLE or agent_registry is None:
        raise HTTPException(status_code=503, detail="AI brain not available — check Ollama")
    agent = await agent_registry.create(
        mission_objective=req.mission_objective,
        mission_id=req.mission_id,
        tick_interval=req.tick_interval,
        max_steps=req.max_steps,
        world_url=os.getenv("FABRIC_URL", "http://localhost:8001"),
        tasking_url=os.getenv("TASKING_URL", "http://localhost:8004"),
    )
    return agent.to_dict()


@app.get("/agents")
async def list_agents(status: Optional[str] = None):
    """List all mission agents."""
    if agent_registry is None:
        return {"agents": [], "brain_available": False}
    return {
        "agents": agent_registry.list_agents(status=status),
        "brain_available": _BRAIN_AVAILABLE,
    }


@app.get("/agents/{mission_id}")
async def get_agent(mission_id: str):
    """Get details for a specific mission agent."""
    if agent_registry is None:
        raise HTTPException(status_code=503, detail="Agent registry not available")
    agent = agent_registry.get(mission_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent {mission_id} not found")
    return agent.to_dict()


@app.get("/reasoning/{entity_id}")
async def get_entity_reasoning(entity_id: str):
    """
    Return AI reasoning context for an entity.
    Queries recent advisories and generates a thought log the console can display.
    Falls back to rule-based reasoning when no advisories exist.
    """
    thoughts = []
    # Query recent advisories mentioning this entity
    if SessionLocal is not None:
        try:
            async with SessionLocal() as session:
                rows = (await session.execute(
                    text("SELECT message, confidence, ts FROM advisories ORDER BY id DESC LIMIT 5")
                )).all()
                for row in rows:
                    ts_val = row.ts
                    ts_str = ts_val.strftime("%H:%M:%SZ") if hasattr(ts_val, "strftime") else str(ts_val)[:19] + "Z"
                    thoughts.append({
                        "ts": ts_str,
                        "msg": row.message,
                        "confidence": float(row.confidence or 0.8),
                    })
        except Exception:
            pass

    if not thoughts:
        # Rule-based fallback — fetch entity from fabric and reason over it
        entity = None
        fabric_url = os.getenv("FABRIC_URL", "http://localhost:8001")
        try:
            import httpx as _httpx
            async with _httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{fabric_url}/api/v1/entities/{entity_id}")
                if r.status_code == 200:
                    entity = r.json()
        except Exception:
            pass

        from datetime import datetime as _dt
        now = _dt.utcnow()
        if entity:
            speed = entity.get("speed_mps", 0) or entity.get("kinematics", {}).get("speed_mps", 0)
            etype = entity.get("entity_type", entity.get("state", "unknown"))
            batt = entity.get("battery_pct")
            pos = entity.get("position", {})
            lat = pos.get("lat", 0) if pos else 0
            lon = pos.get("lon", 0) if pos else 0
            if etype in ("alert", "CRITICAL", "HIGH"):
                thoughts = [
                    {"ts": now.strftime("%H:%M:%SZ"), "msg": f"Anomalous state detected on {entity_id} — velocity {speed:.1f} m/s", "confidence": 0.89},
                    {"ts": now.strftime("%H:%M:%SZ"), "msg": "No matching authorized profile. Operator review recommended.", "confidence": 0.84},
                ]
            elif batt is not None and float(batt) < 25:
                thoughts = [
                    {"ts": now.strftime("%H:%M:%SZ"), "msg": f"Battery at {float(batt):.0f}% — RTB evaluation initiated", "confidence": 0.95},
                ]
            else:
                thoughts = [
                    {"ts": now.strftime("%H:%M:%SZ"), "msg": f"Entity nominal at ({lat:.4f}, {lon:.4f}), speed {speed:.1f} m/s", "confidence": 0.97},
                ]
        else:
            thoughts = [{"ts": now.strftime("%H:%M:%SZ"), "msg": f"Entity {entity_id} — awaiting data", "confidence": 0.5}]

    return {"entity_id": entity_id, "thoughts": thoughts}


@app.delete("/agents/{mission_id}")
async def cancel_agent(mission_id: str):
    """Cancel a running mission agent."""
    if agent_registry is None:
        raise HTTPException(status_code=503, detail="Agent registry not available")
    ok = await agent_registry.cancel(mission_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Agent {mission_id} not found")
    return {"mission_id": mission_id, "status": "CANCELLED"}


async def _get_org_id(req: _Req) -> str | None:
    return req.headers.get("X-Org-ID") or req.headers.get("x-org-id")

@app.get("/advisories", response_model=List[Advisory])
async def list_advisories(risk_level: Optional[str] = None, limit: int = 50, org_id: str | None = Depends(_get_org_id)):
    """List recent advisories, optionally filtered by risk level."""
    assert SessionLocal is not None

    # Clamp limit to prevent abuse (e.g. limit=999999)
    safe_limit = max(1, min(int(limit), 500))

    async with SessionLocal() as session:
        # Build query using SQLAlchemy ORM — fully parameterized, no string concat
        conditions = []
        if risk_level:
            conditions.append(advisories.c.risk_level == risk_level)
        if org_id:
            conditions.append(advisories.c.org_id == org_id)

        stmt = (
            select(
                advisories.c.advisory_id,
                advisories.c.observation_id,
                advisories.c.risk_level,
                advisories.c.message,
                advisories.c.confidence,
                advisories.c.ts,
            )
            .where(and_(*conditions) if conditions else text("1=1"))
            .order_by(advisories.c.id.desc())
            .limit(safe_limit)
        )
        rows = (await session.execute(stmt)).all()

        return [Advisory(**dict(r._mapping)) for r in rows]


async def _risk_scoring_processor():
    """Background processor that consumes observations and generates risk-scored advisories."""
    assert SessionLocal is not None
    assert redis_client is not None
    
    import socket
    consumer_name = f"intel-{socket.gethostname()}"

    while True:
        try:
            # Read from observations_stream via consumer group
            messages = await redis_client.xreadgroup("intelligence", consumer_name, {"observations_stream": ">"}, block=1000, count=10)
            
            if not messages:
                continue
            
            for stream_name, stream_messages in messages:
                for msg_id, fields in stream_messages:
                    
                    try:
                        payload = fields.get("payload")
                        if not payload:
                            continue
                        
                        data = json.loads(payload)
                        
                        # Calculate risk score (XGB optional)
                        risk_level = _calculate_risk_level_ml(data) if (INTELLIGENCE_ENABLE_XGB and risk_model is not None) else _calculate_risk_level(data)
                        
                        # Generate advisory message
                        message = _generate_advisory_message(data, risk_level)
                        
                        advisory_id = str(uuid.uuid4())
                        
                        # Persist advisory
                        async with SessionLocal() as session:
                            await session.execute(
                                advisories.insert().values(
                                    advisory_id=advisory_id,
                                    observation_id=None,  # Could link to observation.id if needed
                                    risk_level=risk_level,
                                    message=message,
                                    confidence=float(data.get("confidence", 0.0)),
                                    ts=datetime.now(timezone.utc),
                                    org_id=None,
                                )
                            )
                            await session.commit()
                        
                        # Optionally publish to MQTT for real-time alerts
                        # await mqtt_client.publish(f"advisories/{data.get('class')}", ...)
                        # Ack after success
                        try:
                            await redis_client.xack("observations_stream", "intelligence", msg_id)
                        except Exception as e:
                            logger.debug("Suppressed error", exc_info=True)  # was: pass
                    
                    except Exception as e:
                        logger.warning("", exc_info=True)
                        # DLQ: record failing payload
                        try:
                            if redis_client is not None:
                                bad_payload = fields.get("payload") if isinstance(fields, dict) else None
                                await redis_client.xadd(
                                    "intelligence_dlq",
                                    {
                                        "error": str(e),
                                        "payload": bad_payload or "",
                                        "ts": datetime.now(timezone.utc).isoformat(),
                                    },
                                )
                                try:
                                    await redis_client.xack("observations_stream", "intelligence", msg_id)
                                except Exception as e:
                                    logger.debug("Suppressed error", exc_info=True)  # was: pass
                        except Exception as e:
                            logger.debug("Suppressed error", exc_info=True)  # was: pass
                        continue
        
        except Exception as e:
            logger.error("", exc_info=True)
            try:
                if redis_client is not None:
                    await redis_client.xadd(
                        "intelligence_dlq",
                        {
                            "error": str(e),
                            "payload": "<stream_error>",
                            "ts": datetime.now(timezone.utc).isoformat(),
                        },
                    )
            except Exception as e:
                logger.debug("Suppressed error", exc_info=True)  # was: pass
            await asyncio.sleep(1)

def _calculate_risk_level(data: dict) -> str:
    """Calculate risk level based on observation data (rules fallback)."""
    confidence = float(data.get("confidence", 0.0))
    # Class-agnostic default: map confidence to LOW/MEDIUM/HIGH
    if confidence >= 0.85:
        return "CRITICAL"
    if confidence >= 0.7:
        return "HIGH"
    if confidence >= 0.5:
        return "MEDIUM"
    return "LOW"


def _calculate_risk_level_ml(data: dict) -> str:
    """Score risk using XGBoost model -> map to levels."""
    try:
        import numpy as _np  # local import to avoid global hard dep
        features = _extract_features(data)
        dmat = xgb.DMatrix(_np.array([features], dtype=float))  # type: ignore
        preds = risk_model.predict(dmat)  # type: ignore
        score = float(preds[0]) if isinstance(preds, (list, tuple, _np.ndarray)) else float(preds)
        # Map score [0,1] to levels
        if score >= 0.9:
            return "CRITICAL"
        if score >= 0.75:
            return "HIGH"
        if score >= 0.5:
            return "MEDIUM"
        return "LOW"
    except Exception:
        return _calculate_risk_level(data)


def _extract_features(data: dict) -> list[float]:
    """Simple, generic feature vector from observation payload."""
    lat = float(data.get("lat") or 0.0)
    lon = float(data.get("lon") or 0.0)
    conf = float(data.get("confidence") or 0.0)
    # Encode class length and simple hash to avoid one-hot explosion
    cls = str(data.get("class") or "")
    cls_len = float(len(cls))
    cls_hash = float((hash(cls) % 1000) / 1000.0)
    # If bbox present
    bbox = data.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        bw = float(bbox[2])
        bh = float(bbox[3])
    else:
        bw = 0.0
        bh = 0.0
    return [lat, lon, conf, cls_len, cls_hash, bw, bh]

def _generate_advisory_message(data: dict, risk_level: str) -> str:
    """Generate human-readable advisory message."""
    obs_class = data.get("class", "unknown")
    confidence = float(data.get("confidence", 0.0))
    lat = data.get("lat")
    lon = data.get("lon")
    
    location_str = f" at ({lat:.4f}, {lon:.4f})" if lat and lon else ""
    
    return f"{risk_level} risk: {obs_class} detected with {confidence:.0%} confidence{location_str}"
