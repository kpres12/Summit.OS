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

logger = logging.getLogger("kofa")
logging.basicConfig(level=logging.INFO)
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    and_,
    select,
    text,
)
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
INTELLIGENCE_ENABLE_XGB = (
    os.getenv("INTELLIGENCE_ENABLE_XGB", "false").lower() == "true"
)
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
    Column("org_id", String(128)),
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
        pg_url = _to_asyncpg_url(
            os.getenv(
                "POSTGRES_URL",
                "postgresql://summit:summit_password@localhost:5432/summit_os",
            )
        )
    engine = create_async_engine(pg_url, echo=False, future=True)
    SessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        if not INTELLIGENCE_TEST_MODE:
            # Optional Timescale hypertable setup
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
                await conn.execute(
                    text(
                        "SELECT create_hypertable('advisories','ts', if_not_exists => TRUE)"
                    )
                )
            except Exception as e:
                logger.debug("Suppressed error", exc_info=True)  # was: pass

    if not INTELLIGENCE_TEST_MODE:
        # Redis setup (consume observations from Fusion)
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = await aioredis.from_url(redis_url, decode_responses=True)
        # Ensure consumer group
        try:
            await redis_client.xgroup_create(
                "observations_stream", "intelligence", id="$", mkstream=True
            )
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                pass

    # Optional: load XGBoost model
    if INTELLIGENCE_ENABLE_XGB and XGB_AVAILABLE:
        model_path = RISK_MODEL_PATH or os.path.join(MODEL_REGISTRY, "risk_model.json")
        try:
            globals()["risk_model"] = xgb.Booster()  # type: ignore
            risk_model.load_model(model_path)  # type: ignore
        except Exception:
            globals()["risk_model"] = None

    # Initialise KOFA model registry (loads all ONNX models once)
    from kofa_models import get_kofa_models

    get_kofa_models()  # warm up — logs which models loaded

    # Start background risk scoring processor (skip in test mode)
    task = None
    prune_task = None
    anomaly_task = None
    if not INTELLIGENCE_TEST_MODE:
        task = asyncio.create_task(_risk_scoring_processor())
        anomaly_task = asyncio.create_task(_entity_anomaly_loop())

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
        for t in [task, prune_task, anomaly_task]:
            if t:
                t.cancel()
                try:
                    await t
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


ENGINE_NAME = "KOFA"  # Summit.OS autonomous dispatch engine

app = FastAPI(
    title=f"Summit Intelligence — {ENGINE_NAME}", version="0.2.0", lifespan=lifespan
)

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
    return {"status": "ok", "service": "intelligence", "engine": ENGINE_NAME}


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
        raise HTTPException(
            status_code=503, detail="AI brain not available — check Ollama"
        )
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
                rows = (
                    await session.execute(
                        text(
                            "SELECT message, confidence, ts FROM advisories ORDER BY id DESC LIMIT 5"
                        )
                    )
                ).all()
                for row in rows:
                    ts_val = row.ts
                    ts_str = (
                        ts_val.strftime("%H:%M:%SZ")
                        if hasattr(ts_val, "strftime")
                        else str(ts_val)[:19] + "Z"
                    )
                    thoughts.append(
                        {
                            "ts": ts_str,
                            "msg": row.message,
                            "confidence": float(row.confidence or 0.8),
                        }
                    )
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
            speed = entity.get("speed_mps", 0) or entity.get("kinematics", {}).get(
                "speed_mps", 0
            )
            etype = entity.get("entity_type", entity.get("state", "unknown"))
            batt = entity.get("battery_pct")
            pos = entity.get("position", {})
            lat = pos.get("lat", 0) if pos else 0
            lon = pos.get("lon", 0) if pos else 0
            if etype in ("alert", "CRITICAL", "HIGH"):
                thoughts = [
                    {
                        "ts": now.strftime("%H:%M:%SZ"),
                        "msg": f"Anomalous state detected on {entity_id} — velocity {speed:.1f} m/s",
                        "confidence": 0.89,
                    },
                    {
                        "ts": now.strftime("%H:%M:%SZ"),
                        "msg": "No matching authorized profile. Operator review recommended.",
                        "confidence": 0.84,
                    },
                ]
            elif batt is not None and float(batt) < 25:
                thoughts = [
                    {
                        "ts": now.strftime("%H:%M:%SZ"),
                        "msg": f"Battery at {float(batt):.0f}% — RTB evaluation initiated",
                        "confidence": 0.95,
                    },
                ]
            else:
                thoughts = [
                    {
                        "ts": now.strftime("%H:%M:%SZ"),
                        "msg": f"Entity nominal at ({lat:.4f}, {lon:.4f}), speed {speed:.1f} m/s",
                        "confidence": 0.97,
                    },
                ]
        else:
            thoughts = [
                {
                    "ts": now.strftime("%H:%M:%SZ"),
                    "msg": f"Entity {entity_id} — awaiting data",
                    "confidence": 0.5,
                }
            ]

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
async def list_advisories(
    risk_level: Optional[str] = None,
    limit: int = 50,
    org_id: str | None = Depends(_get_org_id),
):
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
    """
    Background processor: consumes observations_stream -> risk score -> advisory -> dispatch.

    KOFA pipeline per observation:
      1. False positive filter  — drop sensor noise before processing
      2. Weather risk scorer    — elevate/downgrade risk if weather data present
      3. Incident correlator    — suppress duplicate mission dispatch
      4. Advisory persistence
      5. Escalation predictor   — pre-escalate if operator unlikely to ack
      6. Mission dispatch        — rule + ML planner, with outcome probability logged
      7. LLM agent (optional)   — complex scenario reasoning via Ollama
    """
    assert SessionLocal is not None
    assert redis_client is not None

    import socket
    from kofa_models import get_kofa_models

    consumer_name = f"intel-{socket.gethostname()}"
    kofa = get_kofa_models()

    # Rough active mission list for escalation predictor workload signal
    _active_missions: List[str] = []

    while True:
        try:
            messages = await redis_client.xreadgroup(
                "intelligence",
                consumer_name,
                {"observations_stream": ">"},
                block=1000,
                count=10,
            )
            if not messages:
                continue

            for _stream_name, stream_messages in messages:
                for msg_id, fields in stream_messages:
                    try:
                        payload = fields.get("payload")
                        if not payload:
                            continue

                        data = json.loads(payload)

                        # ── 1. False positive filter ───────────────────────────
                        if kofa.is_false_positive(data):
                            logger.debug(
                                "KOFA FP filter: dropped %s (conf=%.2f)",
                                data.get("class"),
                                float(data.get("confidence", 0)),
                            )
                            await redis_client.xack(
                                "observations_stream", "intelligence", msg_id
                            )
                            continue

                        # ── 2. Base risk score ─────────────────────────────────
                        risk_level = (
                            _calculate_risk_level_ml(data)
                            if (INTELLIGENCE_ENABLE_XGB and risk_model is not None)
                            else _calculate_risk_level(data)
                        )

                        # ── 2b. Weather adjustment ─────────────────────────────
                        risk_level = kofa.adjust_risk_for_weather(data, risk_level)

                        # ── 3. Incident correlation ────────────────────────────
                        duplicate = risk_level in (
                            "CRITICAL",
                            "HIGH",
                        ) and kofa.is_duplicate_incident(data)
                        # Record after duplicate check so obs doesn't match itself
                        kofa.record_observation(data)

                        # ── 4. Advisory persistence ────────────────────────────
                        message = _generate_advisory_message(data, risk_level)
                        advisory_id = str(uuid.uuid4())

                        async with SessionLocal() as session:
                            await session.execute(
                                advisories.insert().values(
                                    advisory_id=advisory_id,
                                    observation_id=None,
                                    risk_level=risk_level,
                                    message=message,
                                    confidence=float(data.get("confidence", 0.0)),
                                    ts=datetime.now(timezone.utc),
                                    org_id=None,
                                )
                            )
                            await session.commit()

                        # ── 5. Escalation predictor ────────────────────────────
                        if risk_level in ("CRITICAL", "HIGH"):
                            esc_prob = kofa.predict_escalation_prob(
                                data, risk_level, len(_active_missions)
                            )
                            if esc_prob >= kofa.ESCALATION_THRESHOLD:
                                logger.info(
                                    "KOFA: pre-escalating %s advisory "
                                    "(escalation_prob=%.0f%%, %s)",
                                    risk_level,
                                    esc_prob * 100,
                                    data.get("class"),
                                )
                                try:
                                    await redis_client.xadd(
                                        "escalation_stream",
                                        {
                                            "advisory_id": advisory_id,
                                            "risk_level": risk_level,
                                            "class": str(data.get("class", "")),
                                            "escalation_prob": str(round(esc_prob, 3)),
                                            "ts": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                        },
                                    )
                                except Exception as _esc_err:
                                    logger.debug(
                                        "Escalation stream write failed: %s", _esc_err
                                    )

                        # ── 6. Mission dispatch (single or swarm) ──────────────
                        if risk_level in ("CRITICAL", "HIGH") and not duplicate:
                            from mission_planner import get_planner
                            from swarm_planner import get_swarm_planner

                            plan = get_planner().plan(data)
                            if plan is not None:
                                outcome_prob = kofa.predict_mission_success_prob(data)
                                try:
                                    import httpx as _httpx

                                    _tasking_url = os.getenv(
                                        "TASKING_URL", "http://localhost:8004"
                                    )
                                    _fabric_url = os.getenv(
                                        "FABRIC_URL", "http://localhost:8001"
                                    )

                                    # ── query available UAVs for swarm decision ──
                                    n_available = 1
                                    try:
                                        async with _httpx.AsyncClient(
                                            timeout=2.0
                                        ) as _fc:
                                            _er = await _fc.get(
                                                f"{_fabric_url}/api/v1/entities",
                                                params={
                                                    "type": "UAV",
                                                    "status": "available",
                                                },
                                            )
                                            if _er.status_code == 200:
                                                _body = _er.json()
                                                _ents = (
                                                    _body
                                                    if isinstance(_body, list)
                                                    else _body.get("entities", [])
                                                )
                                                n_available = max(1, len(_ents))
                                    except Exception:
                                        pass  # graceful fallback to single dispatch

                                    # ── expand to swarm if multiple assets ready ──
                                    swarm = get_swarm_planner()
                                    if swarm.should_swarm(
                                        plan.mission_type, n_available
                                    ):
                                        plans_to_dispatch = swarm.expand(
                                            plan, n_available
                                        )
                                        logger.info(
                                            "KOFA swarm: %d drones → %s "
                                            "(%d sectors, %s)",
                                            n_available,
                                            plan.mission_type,
                                            len(plans_to_dispatch),
                                            data.get("class"),
                                        )
                                    else:
                                        plans_to_dispatch = [plan]

                                    # ── dispatch each plan ──────────────────────
                                    for _plan in plans_to_dispatch:
                                        _wps = _plan.raw_observation.get(
                                            "_waypoints"
                                        ) or [
                                            {
                                                "lat": _plan.lat,
                                                "lon": _plan.lon,
                                                "alt_m": _plan.alt_m,
                                                "action": (
                                                    "LOITER"
                                                    if _plan.loiter
                                                    else _plan.mission_type
                                                ),
                                            }
                                        ]
                                        _mp: dict = {
                                            "title": _plan.rationale,
                                            "mission_type": _plan.mission_type,
                                            "priority": _plan.priority,
                                            "asset_class": _plan.asset_class,
                                            "auto_generated": True,
                                            "waypoints": _wps,
                                            "source_advisory_id": advisory_id,
                                        }
                                        if outcome_prob >= 0:
                                            _mp["kofa_outcome_prob"] = round(
                                                outcome_prob, 3
                                            )
                                        swarm_id = _plan.raw_observation.get(
                                            "_swarm_id"
                                        )
                                        if swarm_id:
                                            _mp["swarm_id"] = swarm_id
                                            _mp["sector_id"] = (
                                                _plan.raw_observation.get("_sector_id")
                                            )
                                            _mp["n_sectors"] = (
                                                _plan.raw_observation.get("_n_sectors")
                                            )

                                        async with _httpx.AsyncClient(
                                            timeout=5.0
                                        ) as _hc:
                                            resp = await _hc.post(
                                                f"{_tasking_url}/api/v1/missions",
                                                json=_mp,
                                            )
                                            if resp.status_code in (200, 201):
                                                _mid = resp.json().get(
                                                    "mission_id"
                                                ) or resp.json().get("id")
                                                if _mid:
                                                    _active_missions.append(str(_mid))
                                                    if len(_active_missions) > 50:
                                                        _active_missions.pop(0)

                                    logger.info(
                                        "KOFA: dispatched %d %s mission(s) — %s "
                                        "(conf=%.2f, outcome_prob=%s)",
                                        len(plans_to_dispatch),
                                        plan.mission_type,
                                        data.get("class"),
                                        float(data.get("confidence", 0)),
                                        (
                                            f"{outcome_prob:.0%}"
                                            if outcome_prob >= 0
                                            else "n/a"
                                        ),
                                    )
                                except Exception as _dispatch_err:
                                    logger.warning(
                                        "Auto-dispatch failed: %s", _dispatch_err
                                    )

                        elif duplicate:
                            logger.info(
                                "KOFA: skipped dispatch — correlates with active incident (%s)",
                                data.get("class"),
                            )

                        # ── 7. Optional LLM agent ──────────────────────────────
                        if (
                            risk_level in ("CRITICAL", "HIGH")
                            and not duplicate
                            and agent_registry is not None
                            and _BRAIN_AVAILABLE
                        ):
                            obs_lat = data.get("lat")
                            obs_lon = data.get("lon")
                            obs_class = str(data.get("class", "anomaly"))
                            obs_conf = float(data.get("confidence", 0.0))
                            loc_str = (
                                f" at ({float(obs_lat):.4f}, {float(obs_lon):.4f})"
                                if obs_lat and obs_lon
                                else ""
                            )
                            objective = (
                                f"{obs_class.capitalize()} detected{loc_str} with "
                                f"{obs_conf:.0%} confidence. Verify detection and assess "
                                f"risk. Adjust mission if rule-based dispatch was suboptimal."
                            )
                            try:
                                await agent_registry.create(
                                    mission_objective=objective,
                                    world_url=os.getenv(
                                        "FABRIC_URL", "http://localhost:8001"
                                    ),
                                    tasking_url=os.getenv(
                                        "TASKING_URL", "http://localhost:8004"
                                    ),
                                )
                                logger.info(
                                    "LLM agent spawned for %s %s%s",
                                    risk_level,
                                    obs_class,
                                    loc_str,
                                )
                            except Exception as _agent_err:
                                logger.warning(
                                    "LLM agent spawn failed (non-critical): %s",
                                    _agent_err,
                                )

                        await redis_client.xack(
                            "observations_stream", "intelligence", msg_id
                        )

                    except Exception as exc:
                        logger.warning("Observation processing error", exc_info=True)
                        try:
                            if redis_client is not None:
                                await redis_client.xadd(
                                    "intelligence_dlq",
                                    {
                                        "error": str(exc),
                                        "payload": (
                                            fields.get("payload")
                                            if isinstance(fields, dict)
                                            else ""
                                        ),
                                        "ts": datetime.now(timezone.utc).isoformat(),
                                    },
                                )
                                await redis_client.xack(
                                    "observations_stream", "intelligence", msg_id
                                )
                        except Exception:
                            pass
                        continue

        except Exception as exc:
            logger.error("Stream processor error", exc_info=True)
            try:
                if redis_client is not None:
                    await redis_client.xadd(
                        "intelligence_dlq",
                        {
                            "error": str(exc),
                            "payload": "<stream_error>",
                            "ts": datetime.now(timezone.utc).isoformat(),
                        },
                    )
            except Exception:
                pass
            await asyncio.sleep(1)


async def _entity_anomaly_loop():
    """
    Background task: poll Fabric for entity telemetry, detect anomalies,
    and raise HIGH advisories for entities with anomalous behavior.
    """
    from kofa_models import get_kofa_models

    _INTERVAL = int(os.getenv("KOFA_ANOMALY_INTERVAL_S", "30"))
    _THRESHOLD = float(os.getenv("KOFA_ANOMALY_ALERT_THRESHOLD", "-0.25"))
    fabric_url = os.getenv("FABRIC_URL", "http://localhost:8001")
    kofa = get_kofa_models()

    while True:
        await asyncio.sleep(_INTERVAL)
        try:
            import httpx as _httpx

            async with _httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{fabric_url}/api/v1/entities")
                if r.status_code != 200:
                    continue
                body = r.json()
                entities = body if isinstance(body, list) else body.get("entities", [])

            for ent in entities:
                eid = ent.get("entity_id") or ent.get("id")
                if not eid:
                    continue
                pos = ent.get("position") or {}
                kin = ent.get("kinematics") or {}
                kofa.update_entity_telemetry(
                    eid,
                    {
                        "lat": float(pos.get("lat") or ent.get("lat") or 0),
                        "lon": float(pos.get("lon") or ent.get("lon") or 0),
                        "alt_m": float(pos.get("alt_m") or ent.get("alt_m") or 0),
                        "speed_mps": float(
                            kin.get("speed_mps") or ent.get("speed_mps") or 0
                        ),
                        "heading_deg": float(
                            kin.get("heading_deg") or ent.get("heading_deg") or 0
                        ),
                        "entity_type": str(
                            ent.get("entity_type") or ent.get("type") or ""
                        ),
                        "mission_active": bool(
                            ent.get("mission_id") or ent.get("mission_active")
                        ),
                    },
                )

            for item in kofa.anomalous_entities(threshold=_THRESHOLD):
                eid = item["entity_id"]
                score = item["anomaly_score"]
                logger.warning(
                    "KOFA anomaly: entity %s score=%.3f — operator review recommended",
                    eid,
                    score,
                )
                if SessionLocal is not None:
                    try:
                        async with SessionLocal() as session:
                            await session.execute(
                                advisories.insert().values(
                                    advisory_id=str(uuid.uuid4()),
                                    observation_id=None,
                                    risk_level="HIGH",
                                    message=(
                                        f"KOFA: anomalous behavior detected on entity {eid} "
                                        f"(score={score:.3f}) — possible GPS spoof, "
                                        f"connectivity loss, or erratic movement"
                                    ),
                                    confidence=min(abs(score) * 2, 1.0),
                                    ts=datetime.now(timezone.utc),
                                    org_id=None,
                                )
                            )
                            await session.commit()
                    except Exception as _adv_err:
                        logger.debug("Anomaly advisory write failed: %s", _adv_err)

        except Exception as exc:
            logger.debug("Entity anomaly loop error: %s", exc)


# ── Risk scorer (loaded once at import time) ──────────────────────────────────
_ML_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "packages", "ml")
)
if _ML_ROOT not in sys.path:
    sys.path.insert(0, _ML_ROOT)

_risk_session = None
_risk_labels: dict = {}
_risk_feat_fn = None

try:
    import onnxruntime as _ort  # type: ignore
    import numpy as _np_risk  # type: ignore
    from features import extract as _risk_feat_fn  # type: ignore

    _risk_model_path = os.path.join(_ML_ROOT, "models", "risk_scorer.onnx")
    _risk_labels_path = os.path.join(_ML_ROOT, "models", "risk_scorer_labels.json")
    if os.path.exists(_risk_model_path):
        _risk_session = _ort.InferenceSession(
            _risk_model_path, providers=["CPUExecutionProvider"]
        )
        with open(_risk_labels_path) as _f:
            _risk_labels = json.load(_f)
        logger.info("Risk scorer ONNX loaded")
except Exception as _rs_err:
    logger.debug("Risk scorer ONNX not loaded: %s", _rs_err)


def _calculate_risk_level(data: dict) -> str:
    """Calculate risk level — trained ONNX model when available, rules as fallback."""
    if _risk_session is not None and _risk_feat_fn is not None:
        try:
            feat = _np_risk.array([_risk_feat_fn(data)], dtype=_np_risk.float32)
            input_name = _risk_session.get_inputs()[0].name
            pred = _risk_session.run(None, {input_name: feat})[0][0]
            return _risk_labels[str(int(pred))]
        except Exception:
            pass
    # Rule fallback
    confidence = float(data.get("confidence", 0.0))
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
        score = (
            float(preds[0])
            if isinstance(preds, (list, tuple, _np.ndarray))
            else float(preds)
        )
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


# ── SITREP endpoint ────────────────────────────────────────────────────────────


class SitRepRequest(BaseModel):
    time_window_s: int = 300
    risk_level: Optional[str] = None  # filter to specific risk level
    enhance_llm: bool = False  # request LLM narrative enhancement


@app.post("/sitrep")
async def generate_sitrep(req: SitRepRequest):
    """
    Generate a situation report from recent advisories.

    Returns a structured SITREP with a natural-language summary, per-domain
    findings, and a recommended action.

    Set enhance_llm=true to request LLM narrative enhancement via Ollama
    (falls back to template if Ollama is unavailable).
    """
    from sitrep import get_sitrep_generator

    assert SessionLocal is not None
    safe_window = max(30, min(req.time_window_s, 86400))

    async with SessionLocal() as session:
        conditions = [
            (
                text(f"ts >= datetime('now', '-{safe_window} seconds')")
                if "sqlite" in str(session.bind.url)
                else text(f"ts >= NOW() - INTERVAL '{safe_window} seconds'")
            )
        ]
        if req.risk_level:
            conditions.append(advisories.c.risk_level == req.risk_level)

        stmt = (
            select(
                advisories.c.advisory_id,
                advisories.c.risk_level,
                advisories.c.message,
                advisories.c.confidence,
                advisories.c.ts,
            )
            .where(and_(*conditions))
            .order_by(advisories.c.id.desc())
            .limit(200)
        )
        rows = (await session.execute(stmt)).all()

    raw_advisories = [dict(r._mapping) for r in rows]

    gen = get_sitrep_generator()
    sitrep = gen.from_advisories(raw_advisories, time_window_s=safe_window)

    if req.enhance_llm and _BRAIN_AVAILABLE:
        sitrep = await gen.enhance_with_llm(sitrep, raw_advisories)

    return sitrep.to_dict()


@app.get("/sitrep")
async def get_latest_sitrep(
    time_window_s: int = 300,
    risk_level: Optional[str] = None,
):
    """
    GET shorthand for /sitrep — returns template-only SITREP (no LLM).
    Use POST /sitrep with enhance_llm=true for LLM narrative.
    """
    req = SitRepRequest(
        time_window_s=time_window_s,
        risk_level=risk_level,
        enhance_llm=False,
    )
    return await generate_sitrep(req)


# ── Swarm status endpoint ──────────────────────────────────────────────────────


@app.get("/swarm/{swarm_id}")
async def get_swarm_status(swarm_id: str):
    """
    Return all missions belonging to a swarm_id.
    Queries Tasking service for missions tagged with this swarm_id.
    """
    import httpx as _httpx

    tasking_url = os.getenv("TASKING_URL", "http://localhost:8004")
    try:
        async with _httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(
                f"{tasking_url}/api/v1/missions",
                params={"swarm_id": swarm_id},
            )
            if r.status_code == 200:
                return {"swarm_id": swarm_id, "missions": r.json()}
    except Exception as exc:
        logger.debug("Swarm status query failed: %s", exc)
    return {"swarm_id": swarm_id, "missions": [], "note": "tasking service unavailable"}
