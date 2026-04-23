"""
Heli.OS Data Fabric Service

Real-time message bus and synchronization layer for Heli.OS.
Handles MQTT, Redis Streams, and gRPC streaming for distributed intelligence.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

import structlog
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json
from datetime import datetime, timezone, timedelta

from config import Settings
from mqtt_client import MQTTClient
from redis_client import RedisClient
from websocket_manager import WebSocketManager
from models import (
    TelemetryMessage,
    AlertMessage,
    MissionUpdate,
    Location,
    SeverityLevel,
    MissionStatus,
)

# WorldStore (unified entity store)
try:
    import sys as _sys

    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from packages.world.store import WorldStore
    from packages.world.api import create_world_router
    from packages.world.history import HistoryStore, register_routes as register_history_routes

    WORLD_STORE_AVAILABLE = True
except Exception:
    WORLD_STORE_AVAILABLE = False

# c2_intel — live observation processing pipeline
try:
    from packages.c2_intel.models import C2Observation, C2EventType, SensorSource, ObservationPriority
    from packages.c2_intel.priority import C2PriorityMatrix
    from packages.c2_intel.chains import get_chain_detector
    from packages.c2_intel.dedup import ObservationDeduplicator
    from packages.c2_intel.evidence import C2EvidenceAggregator

    _c2_priority   = C2PriorityMatrix()
    _c2_chains     = get_chain_detector()
    _c2_dedup      = ObservationDeduplicator(window_seconds=30.0)
    _c2_evidence   = C2EvidenceAggregator()

    # Rolling window of recent observations per entity for evidence clustering
    # entity_id → deque of C2Observation (last 20)
    from collections import defaultdict, deque as _deque
    _c2_obs_window: dict = defaultdict(lambda: _deque(maxlen=20))

    C2_INTEL_AVAILABLE = True
except Exception as _c2_err:
    C2_INTEL_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logging.getLogger(__name__).warning("c2_intel unavailable: %s", _c2_err)

# Mesh peer
try:
    from packages.mesh.peer import MeshPeer
    from packages.mesh.entity_crdt import EntityCRDTMap

    MESH_AVAILABLE = True
except Exception:
    MESH_AVAILABLE = False

# SQLAlchemy (async) for registry persistence
from sqlalchemy import (
    MetaData,
    Table,
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    JSON,
    text,
)
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Optional geometry
try:
    from geoalchemy2 import Geometry

    GEO_AVAILABLE = True
except Exception:
    Geometry = None  # type: ignore
    GEO_AVAILABLE = False
from jose import jwt
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig

# Test mode: disable geospatial columns BEFORE table definitions to avoid SpatiaLite
FABRIC_TEST_MODE = os.getenv("FABRIC_TEST_MODE", "false").lower() == "true"
if FABRIC_TEST_MODE:
    GEO_AVAILABLE = False

# ── Capability status at startup ──────────────────────────────────────────────
def _log_capabilities() -> None:
    _OK  = "✓"
    _OFF = "✗"
    _log = logging.getLogger("fabric")
    _log.info("─── Fabric service capabilities ────────────────────────────────")
    _log.info("  %s  World store   (entity stream + world model API)", _OK if WORLD_STORE_AVAILABLE else _OFF)
    _log.info("  %s  Mesh peer     (CRDT gossip sync for disconnected ops)", _OK if MESH_AVAILABLE else _OFF)
    _log.info("  %s  GeoSpatial    (PostGIS geometry columns)", _OK if GEO_AVAILABLE else _OFF)
    if not WORLD_STORE_AVAILABLE:
        _log.warning("  World store unavailable — /v1/entities and real-time entity stream will be empty")
    if not MESH_AVAILABLE:
        _log.warning("  Mesh peer unavailable — multi-node CRDT sync disabled")
    _log.info("────────────────────────────────────────────────────────────────")

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Global services
mqtt_client: Optional[MQTTClient] = None
redis_client: Optional[RedisClient] = None
websocket_manager: Optional[WebSocketManager] = None
engine: Optional[AsyncEngine] = None
SessionLocal: Optional[sessionmaker] = None
world_store: Optional["WorldStore"] = None
history_store: Optional["HistoryStore"] = None

# WebSocket token auth — set WS_AUTH_REQUIRED=true in production
WS_AUTH_REQUIRED = os.getenv("WS_AUTH_REQUIRED", "false").lower() == "true"
mesh_peer = None
entity_crdt = None

# In-memory world state cache (thin-slice)
world_state: Dict[str, Any] = {
    "devices": {},  # device_id -> {lat, lon, alt, ts_iso, status, sensors}
    "alerts": [],  # recent alerts (most recent first)
}
MAX_ALERTS = int(os.getenv("FABRIC_MAX_ALERTS", "200"))

# Shared alert dict for AlertEscalationService (alert_id → alert dict, kept in sync)
_escalation_alerts: Dict[str, Any] = {}

# In-memory REST stores (survive process lifetime, reset on restart)
import uuid as _uuid
from datetime import datetime as _dt, timezone as _tz

_alerts_store: Dict[str, Any] = {}
_missions_store: Dict[str, Any] = {}
_tasks_store: Dict[str, Any] = {}
_assets_store: Dict[str, Any] = {}
_agents_store: Dict[str, Any] = {}


def _now_iso() -> str:
    return _dt.now(_tz.utc).isoformat()


# Database metadata and tables (registry)
metadata = MetaData()

nodes = Table(
    "nodes",
    metadata,
    Column("id", String(128), primary_key=True),
    Column("type", String(32), nullable=False),
    Column("pubkey", String(4096)),
    Column("fw_version", String(64)),
    Column("location", JSON),
    Column("capabilities", JSON),
    Column("comm", JSON),
    Column("policy", JSON),
    Column("status", String(32), default="OFFLINE"),
    Column("last_seen", DateTime(timezone=True)),
    Column("retired", Boolean, default=False),
    Column("created_at", DateTime(timezone=True)),
    Column("updated_at", DateTime(timezone=True)),
)

coverages = Table(
    "coverages",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("node_id", String(128), nullable=False),
    Column("viewshed_geojson", JSON),
    Column("version", String(128)),
    Column("updated_at", DateTime(timezone=True)),
)

# World model tables
# Optional multi-tenant org_id
try:
    ORG_ID_FIELD = os.getenv("ORG_ID_FIELD", "org")
except Exception:
    ORG_ID_FIELD = "org"

world_entities = Table(
    "world_entities",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("entity_id", String(128), nullable=False),  # device_id
    Column("type", String(64), nullable=False, default="DEVICE"),
    Column("properties", JSON),
    Column("updated_at", DateTime(timezone=True)),
    Column("org_id", String(128)),
    # geometry point WGS84 if available
    *(
        [Column("geom", Geometry(geometry_type="POINT", srid=4326))]
        if GEO_AVAILABLE
        else []
    ),
)

world_alerts = Table(
    "world_alerts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("alert_id", String(128), nullable=False),
    Column("severity", String(32), nullable=False),
    Column("description", String(512)),
    Column("source", String(128)),
    Column("ts", DateTime(timezone=True)),
    Column("properties", JSON),
    Column("org_id", String(128)),
    *(
        [Column("geom", Geometry(geometry_type="POINT", srid=4326))]
        if GEO_AVAILABLE
        else []
    ),
)

# Settings
HEARTBEAT_STALE_SECS = 120  # 2 minutes
HEARTBEAT_OFFLINE_SECS = 600  # 10 minutes
FABRIC_JWT_SECRET = os.getenv("FABRIC_JWT_SECRET", "dev_secret")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    global mqtt_client, redis_client, websocket_manager, engine, SessionLocal, mesh_peer, entity_crdt, world_store, history_store, FABRIC_JWT_SECRET

    settings = Settings()

    _log_capabilities()
    logger.info("Starting Heli.OS Data Fabric Service")

    # Resolve sensitive secrets via secrets client (Vault → env var fallback)
    try:
        _root = str(Path(__file__).resolve().parents[2])
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from packages.secret_store.client import get_secret as _get_secret_fabric

        _resolved_jwt = await _get_secret_fabric(
            "FABRIC_JWT_SECRET", default=FABRIC_JWT_SECRET
        )
        if _resolved_jwt:
            FABRIC_JWT_SECRET = _resolved_jwt
        logger.info("Fabric secrets resolved")
    except Exception as _e:
        logger.warning("Secrets client unavailable, using env var: %s", _e)

    # DB connection
    if FABRIC_TEST_MODE:
        pg_url = "sqlite+aiosqlite://"
    else:
        pg_url = os.getenv(
            "POSTGRES_URL",
            "postgresql+asyncpg://summit:summit_password@localhost:5432/heli_os",
        )
        if pg_url.startswith("postgresql://"):
            pg_url = pg_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(pg_url, echo=False, future=True)
    SessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    if FABRIC_TEST_MODE:
        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
    else:
        # Ensure extensions (PostGIS optional — Render free tier doesn't have it)
        try:
            async with engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        except Exception as _pg_ex:
            logger.warning("PostGIS extension unavailable, geospatial features disabled: %s", _pg_ex)
            global GEO_AVAILABLE
            GEO_AVAILABLE = False
        if os.getenv("FABRIC_SKIP_MIGRATIONS", "false").lower() != "true":
            try:
                _run_migrations()
            except Exception as _mig_err:
                import traceback
                logger.error("Migration failed — continuing with existing schema: %s\n%s", _mig_err, traceback.format_exc())
        # Create geospatial and org_id indexes if available (skip if migrations disabled)
        if os.getenv("FABRIC_SKIP_MIGRATIONS", "false").lower() != "true":
            try:
                async with engine.begin() as conn:
                    await conn.execute(
                        text(
                            "ALTER TABLE world_entities ADD COLUMN IF NOT EXISTS org_id varchar(128)"
                        )
                    )
                    await conn.execute(
                        text(
                            "ALTER TABLE world_alerts ADD COLUMN IF NOT EXISTS org_id varchar(128)"
                        )
                    )
                    await conn.execute(
                        text(
                            "CREATE INDEX IF NOT EXISTS idx_world_entities_org ON world_entities (org_id)"
                        )
                    )
                    await conn.execute(
                        text(
                            "CREATE INDEX IF NOT EXISTS idx_world_alerts_org ON world_alerts (org_id)"
                        )
                    )
                    if GEO_AVAILABLE:
                        await conn.execute(
                            text(
                                "CREATE INDEX IF NOT EXISTS idx_world_entities_geom ON world_entities USING GIST (geom)"
                            )
                        )
                        await conn.execute(
                            text(
                                "CREATE INDEX IF NOT EXISTS idx_world_alerts_geom ON world_alerts USING GIST (geom)"
                            )
                        )
            except Exception as _idx_ex:
                logger.warning("Schema post-migration steps skipped (tables not yet created): %s", _idx_ex)

    # WebSocket manager
    websocket_manager = WebSocketManager()

    # Initialize WorldStore (unified entity store)
    if WORLD_STORE_AVAILABLE:
        try:
            world_store = WorldStore(org_id=os.getenv("FABRIC_ORG_ID", "default"))
            await world_store.initialize(
                engine=engine if not FABRIC_TEST_MODE else None,
                session_factory=SessionLocal if not FABRIC_TEST_MODE else None,
                ws_manager=websocket_manager,
            )
            logger.info("WorldStore initialized")
            history_store = HistoryStore()
            logger.info("HistoryStore initialized")
        except Exception as _ws_err:
            import traceback
            logger.error("WorldStore initialization failed — running without persistence: %s\n%s", _ws_err, traceback.format_exc())
            world_store = None
            history_store = None

    if not FABRIC_TEST_MODE:
        # Redis connection — optional, graceful degradation without it
        try:
            redis_client = RedisClient(settings.redis_url)
            await redis_client.connect()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Redis unavailable — running without telemetry streams: {e}")
            redis_client = None

        # MQTT client — optional
        try:
            mqtt_client = MQTTClient(
                broker=settings.mqtt_broker,
                port=settings.mqtt_port,
                username=settings.mqtt_username,
                password=settings.mqtt_password,
            )
            await mqtt_client.connect()
            logger.info("Connected to MQTT broker")
            await mqtt_client.subscribe("observations/#", _handle_observation)
            await mqtt_client.subscribe("detections/#", _handle_observation)
            await mqtt_client.subscribe("missions/#", _handle_mission)
            await mqtt_client.subscribe("health/+/heartbeat", _handle_heartbeat)
            await mqtt_client.subscribe("plainview/leaks", _handle_plainview_leak)
            await mqtt_client.subscribe("valves/+/status", _handle_valve_status)
            await mqtt_client.subscribe("pipeline/pressure/+", _handle_pipeline_pressure)
        except Exception as e:
            logger.warning(f"MQTT unavailable — running without hardware messaging: {e}")
            mqtt_client = None

        # Wire MQTT into WorldStore
        if WORLD_STORE_AVAILABLE and world_store:
            world_store._mqtt_client = mqtt_client

        # Start mesh peer for entity replication
        if MESH_AVAILABLE and not FABRIC_TEST_MODE:
            try:
                mesh_bind = os.getenv("MESH_BIND", "0.0.0.0")
                mesh_port = int(os.getenv("MESH_PORT", "7946"))
                mesh_seeds = [
                    s.strip()
                    for s in os.getenv("MESH_SEEDS", "").split(",")
                    if s.strip()
                ]
                mesh_peer = MeshPeer(
                    node_id=os.getenv("FABRIC_NODE_ID", "fabric-primary"),
                    bind_addr=mesh_bind,
                    bind_port=mesh_port,
                    seed_nodes=[
                        (s.split(":")[0], int(s.split(":")[1]))
                        for s in mesh_seeds
                        if ":" in s
                    ],
                )
                entity_crdt = EntityCRDTMap(node_id=mesh_peer.node_id)
                # When mesh receives remote entity state, merge into WorldStore
                if world_store:

                    def _on_mesh_merge(entity_id, entity_dict):
                        asyncio.create_task(
                            world_store.merge_remote(entity_id, entity_dict)
                        )

                    entity_crdt.on_merge(_on_mesh_merge)
                await mesh_peer.start()
                logger.info(f"Mesh peer started on {mesh_bind}:{mesh_port}")
            except Exception as e:
                logger.warning(f"Mesh peer failed to start: {e}")

        # Start background tasks
        asyncio.create_task(telemetry_processor())
        asyncio.create_task(alert_processor())
        asyncio.create_task(heartbeat_watcher())
        asyncio.create_task(_ttl_gc_task())
        asyncio.create_task(_opensky_poll_loop())
        asyncio.create_task(_mavlink_sim_loop())

        # Alert escalation service (uses live shared dict, updated as alerts arrive)
        try:
            from alert_escalation import AlertEscalationService

            _esc_svc = AlertEscalationService(_escalation_alerts)
            asyncio.create_task(_esc_svc.run())
            logger.info("AlertEscalationService started")
        except Exception as _esc_err:
            logger.warning(f"AlertEscalationService not started: {_esc_err}")

        # Subscribe to entity update topics for WorldStore ingestion
        if mqtt_client and WORLD_STORE_AVAILABLE and world_store:
            await mqtt_client.subscribe("entities/+/update", _handle_entity_update)

    # Mount entity REST routes now that world_store is ready
    try:
        _mount_world_router()
    except Exception as _mwr_err:
        logger.warning("World router mount failed: %s", _mwr_err)

    yield

    # Cleanup
    if mesh_peer:
        try:
            await mesh_peer.stop()
        except Exception:
            pass
    if mqtt_client:
        await mqtt_client.disconnect()
    if redis_client:
        await redis_client.disconnect()
    if engine:
        await engine.dispose()
    logger.info("Shutting down Heli.OS Data Fabric Service")


app = FastAPI(
    title="Heli.OS Data Fabric",
    description="Real-time message bus and synchronization layer",
    version="1.1.0",
    lifespan=lifespan,
)

# Mount WorldStore entity API (deferred — router created once world_store exists)
# The router is added inside lifespan; see _mount_world_router below.
_world_router_mounted = False


def _mount_world_router():
    global _world_router_mounted
    if WORLD_STORE_AVAILABLE and world_store and not _world_router_mounted:
        router = create_world_router(world_store)
        app.include_router(router, prefix="/api/v1", tags=["world"])
        if history_store:
            register_history_routes(app, history_store)
        _world_router_mounted = True


# CORS middleware
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class TelemetryData(BaseModel):
    device_id: str
    timestamp: datetime
    location: Dict[str, float]  # lat, lon, alt
    sensors: Dict[str, Any]
    status: str


class AlertData(BaseModel):
    alert_id: str
    timestamp: datetime
    severity: str
    location: Dict[str, float]
    description: str
    source: str


class MissionData(BaseModel):
    mission_id: str
    timestamp: datetime
    status: str
    assets: List[str]
    objectives: List[str]


# Registry models
class NodeRegisterRequest(BaseModel):
    id: str
    type: str  # TOWER, DRONE, UGV, GATEWAY
    pubkey: str | None = None
    fw_version: str | None = None
    location: Dict[str, Any] | None = None
    capabilities: List[str] = []
    comm: List[str] = []


class NodeRegisterResponse(BaseModel):
    status: str
    mqtt_topics: Dict[str, List[str]]
    policy: Dict[str, Any]
    token: str


# Org helper
from fastapi import Request as _Req


async def _get_org_id(req: _Req) -> str | None:
    return req.headers.get("X-Org-ID") or req.headers.get("x-org-id")


# API Endpoints
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "fabric"}


@app.get("/readyz")
async def readyz():
    """Readiness probe: checks DB and Redis connectivity."""
    try:
        assert SessionLocal is not None
        async with SessionLocal() as session:
            await session.execute(text("SELECT 1"))
        if redis_client and redis_client.redis:
            await redis_client.redis.ping()
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {e}")


@app.get("/livez")
async def livez():
    return {"status": "alive"}


# WebSocket endpoint — Community mode (no org isolation)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global websocket_manager
    if websocket_manager is None:
        websocket_manager = WebSocketManager()
    if not await _verify_ws_token(websocket):
        await websocket.close(code=4001, reason="Unauthorized")
        return
    await websocket_manager.connect(websocket, org_id="default")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


# WebSocket endpoint — Enterprise mode (org-scoped, requires JWT with matching org claim)
@app.websocket("/ws/{org_id}")
async def websocket_endpoint_org(websocket: WebSocket, org_id: str):
    """
    Org-scoped WebSocket for Enterprise multi-tenant deployments.

    The client must present a Bearer token whose org_id / org / tenant claim
    matches the org_id path parameter. Mismatches are rejected before the
    handshake completes.
    """
    global websocket_manager
    if websocket_manager is None:
        websocket_manager = WebSocketManager()

    # Validate org claim when Enterprise mode is on
    _enterprise = os.getenv("ENTERPRISE_MULTI_TENANT", "false").lower() == "true"
    if _enterprise:
        _token_org: str | None = None
        _auth = websocket.headers.get("Authorization") or websocket.query_params.get("token", "")
        if _auth.startswith("Bearer "):
            _auth = _auth[7:]
        if _auth:
            try:
                from jose import jwt as _jwt
                _claims = _jwt.get_unverified_claims(_auth)
                _token_org = (
                    _claims.get("org_id")
                    or _claims.get("org")
                    or _claims.get("tenant")
                )
            except Exception:
                pass
        if _token_org and _token_org != org_id:
            await websocket.close(code=4003, reason="org_id mismatch")
            return

    await websocket_manager.connect(websocket, org_id=org_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


# Simple COP/world-state endpoint (thin-slice)
@app.get("/api/v1/worldstate")
async def get_world_state(
    limit_devices: int = 200,
    limit_alerts: int = 200,
    org_id: str | None = Depends(_get_org_id),
):
    """Aggregate world state from DB (latest position per entity) and recent alerts. Optional org filtering."""
    assert SessionLocal is not None
    devices: list[dict] = []
    alerts: list[dict] = []
    try:
        async with SessionLocal() as session:
            # Latest per entity using DISTINCT ON
            q_devices = text(
                """
                SELECT DISTINCT ON (entity_id)
                    entity_id, type, properties, updated_at
                    {geom}
                FROM world_entities
                {where}
                ORDER BY entity_id, updated_at DESC
                LIMIT :lim
                """.format(
                    geom=(
                        ", ST_X(geom) AS lon, ST_Y(geom) AS lat"
                        if GEO_AVAILABLE
                        else ""
                    ),
                    where="WHERE org_id = :org_id" if org_id else "",
                )
            )
            params = {"lim": limit_devices}
            if org_id:
                params["org_id"] = org_id
            drows = (await session.execute(q_devices, params)).mappings().all()
            for r in drows:
                d = {
                    "device_id": r["entity_id"],
                    "type": r.get("type"),
                    "properties": r.get("properties"),
                    "ts_iso": (
                        r.get("updated_at").isoformat() if r.get("updated_at") else None
                    ),
                }
                if (
                    GEO_AVAILABLE
                    and r.get("lon") is not None
                    and r.get("lat") is not None
                ):
                    d.update({"lon": float(r["lon"]), "lat": float(r["lat"])})
                devices.append(d)
            # Recent alerts
            q_alerts = text(
                """
                SELECT alert_id, severity, description, source, ts
                    {geom}
                FROM world_alerts
                {where}
                ORDER BY id DESC
                LIMIT :lim
                """.format(
                    geom=(
                        ", ST_X(geom) AS lon, ST_Y(geom) AS lat"
                        if GEO_AVAILABLE
                        else ""
                    ),
                    where="WHERE org_id = :org_id" if org_id else "",
                )
            )
            aparams = {"lim": limit_alerts}
            if org_id:
                aparams["org_id"] = org_id
            arows = (await session.execute(q_alerts, aparams)).mappings().all()
            for r in arows:
                a = {
                    "alert_id": r["alert_id"],
                    "severity": r.get("severity"),
                    "description": r.get("description"),
                    "source": r.get("source"),
                    "ts_iso": r.get("ts").isoformat() if r.get("ts") else None,
                }
                if (
                    GEO_AVAILABLE
                    and r.get("lon") is not None
                    and r.get("lat") is not None
                ):
                    a.update({"lon": float(r["lon"]), "lat": float(r["lat"])})
                alerts.append(a)
    except Exception:
        # Fallback to in-memory cache
        try:
            devices = list(world_state.get("devices", {}).values())
            alerts = world_state.get("alerts", [])
        except Exception:
            devices, alerts = [], []
    return {
        "devices": devices,
        "alerts": alerts,
        "counts": {"devices": len(devices), "alerts": len(alerts)},
        "ts_iso": datetime.now(timezone.utc).isoformat(),
    }


# Replay endpoint for observations stream
@app.get("/api/v1/replay/observations")
async def replay_observations(from_id: str = "$", count: int = 100):
    if not redis_client or not redis_client.redis:
        raise HTTPException(status_code=503, detail="Redis not connected")
    try:
        records = await redis_client.redis.xread(
            {"observations_stream": from_id}, count=count, block=100
        )
        result: list[dict] = []
        for stream, msgs in records:
            for msg_id, fields in msgs:
                result.append({"id": msg_id, "fields": fields})
        return {"records": result, "next": result[-1]["id"] if result else from_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Optional OIDC enforcement
OIDC_ENFORCE = os.getenv("OIDC_ENFORCE", "false").lower() == "true"
try:
    from jose import jwt as _jwt

    OIDC_JOSE_AVAILABLE = True
except Exception:
    OIDC_JOSE_AVAILABLE = False

from fastapi import Header


async def _verify_bearer_fabric(
    authorization: str | None = Header(default=None),
) -> dict | None:
    if not OIDC_ENFORCE:
        return None
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    # For now, accept unsigned token (issuer verification could be added)
    token = authorization.split(" ", 1)[1]
    try:
        claims = _jwt.get_unverified_claims(token) if OIDC_JOSE_AVAILABLE else {}
        return claims
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/telemetry")
async def publish_telemetry(
    telemetry: "TelemetryData",
    request: _Req,
    _claims: dict | None = Depends(_verify_bearer_fabric),
):
    if not mqtt_client or not redis_client:
        raise HTTPException(status_code=503, detail="Services not connected")
    try:
        topic = f"telemetry/{telemetry.device_id}"
        await mqtt_client.publish(topic, telemetry.model_dump_json())
        # Convert to internal model for Redis
        loc = Location(
            latitude=float(telemetry.location.get("lat")),
            longitude=float(telemetry.location.get("lon")),
            altitude=telemetry.location.get("alt"),
        )
        tm = TelemetryMessage(
            device_id=telemetry.device_id,
            timestamp=telemetry.timestamp,
            location=loc,
            sensors=telemetry.sensors,
        )
        await redis_client.add_telemetry(tm)
        # Update in-memory world state
        try:
            world_state["devices"][telemetry.device_id] = {
                "device_id": telemetry.device_id,
                "lat": float(loc.latitude),
                "lon": float(loc.longitude),
                "alt": loc.altitude,
                "ts_iso": telemetry.timestamp.isoformat(),
                "status": telemetry.status,
                "sensors": telemetry.sensors,
            }
        except Exception as e:
            logger.debug("Suppressed error", exc_info=True)  # was: pass
        # Persist to world_entities
        try:
            assert SessionLocal is not None
            async with SessionLocal() as session:
                org_id = None
                try:
                    org_id = (
                        (_claims or {}).get("org")
                        or (_claims or {}).get("org_id")
                        or (_claims or {}).get("tenant")
                    )
                except Exception:
                    org_id = None
                # Fallback to mTLS client DN parsing (e.g., 'OU=org-123, CN=device')
                if not org_id:
                    try:
                        x_client_dn = request.headers.get("X-Client-DN")
                        if x_client_dn:
                            parts = [p.strip() for p in str(x_client_dn).split(",")]
                            for p in parts:
                                if p.startswith("OU="):
                                    org_id = p.split("=", 1)[1]
                                    break
                    except Exception as e:
                        logger.debug("Suppressed error", exc_info=True)  # was: pass
                if GEO_AVAILABLE:
                    await session.execute(
                        world_entities.insert().values(
                            entity_id=telemetry.device_id,
                            type="DEVICE",
                            properties={
                                "status": telemetry.status,
                                "sensors": telemetry.sensors,
                            },
                            updated_at=telemetry.timestamp,
                            org_id=org_id,
                            geom=text(
                                f"ST_SetSRID(ST_MakePoint({float(loc.longitude)}, {float(loc.latitude)}), 4326)"
                            ),
                        )
                    )
                else:
                    await session.execute(
                        world_entities.insert().values(
                            entity_id=telemetry.device_id,
                            type="DEVICE",
                            properties={
                                "status": telemetry.status,
                                "sensors": telemetry.sensors,
                                "lon": float(loc.longitude),
                                "lat": float(loc.latitude),
                            },
                            updated_at=telemetry.timestamp,
                            org_id=org_id,
                        )
                    )
                await session.commit()
        except Exception as e:
            logger.debug("Suppressed error", exc_info=True)  # was: pass
        if websocket_manager:
            await websocket_manager.broadcast_to_org(
                json.dumps({"type": "telemetry", "data": tm.model_dump()}),
                org_id=org_id or "default",
            )
        return {"status": "published", "device_id": telemetry.device_id}
    except Exception as e:
        logger.error("Failed to publish telemetry", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to publish telemetry")


@app.post("/alerts")
async def publish_alert(
    alert: "AlertData",
    request: _Req,
    _claims: dict | None = Depends(_verify_bearer_fabric),
):
    if not mqtt_client or not redis_client:
        raise HTTPException(status_code=503, detail="Services not connected")
    try:
        topic = f"alerts/{alert.alert_id}"
        await mqtt_client.publish(topic, alert.model_dump_json())
        # Convert to internal model for Redis
        aloc = Location(
            latitude=float(alert.location.get("lat")),
            longitude=float(alert.location.get("lon")),
        )
        sev = (
            SeverityLevel(alert.severity.lower())
            if isinstance(alert.severity, str)
            else alert.severity
        )
        am = AlertMessage(
            alert_id=alert.alert_id,
            timestamp=alert.timestamp,
            severity=sev,
            location=aloc,
            description=alert.description,
            source=alert.source,
        )
        await redis_client.add_alert(am)
        # Update in-memory world state (alerts ring buffer)
        try:
            alert_dict = {
                "alert_id": alert.alert_id,
                "severity": str(sev.value if hasattr(sev, "value") else sev),
                "lat": float(aloc.latitude),
                "lon": float(aloc.longitude),
                "description": alert.description,
                "source": alert.source,
                "ts_iso": alert.timestamp.isoformat(),
            }
            world_state["alerts"].insert(0, alert_dict)
            del world_state["alerts"][MAX_ALERTS:]
            # Keep escalation dict in sync (shared reference — AlertEscalationService watches this)
            _escalation_alerts[alert.alert_id] = alert_dict
        except Exception as e:
            logger.debug("Suppressed error", exc_info=True)  # was: pass
        # Persist to world_alerts
        try:
            assert SessionLocal is not None
            async with SessionLocal() as session:
                org_id = None
                try:
                    org_id = (
                        (_claims or {}).get("org")
                        or (_claims or {}).get("org_id")
                        or (_claims or {}).get("tenant")
                    )
                except Exception:
                    org_id = None
                if not org_id:
                    try:
                        x_client_dn = request.headers.get("X-Client-DN")
                        if x_client_dn:
                            parts = [p.strip() for p in str(x_client_dn).split(",")]
                            for p in parts:
                                if p.startswith("OU="):
                                    org_id = p.split("=", 1)[1]
                                    break
                    except Exception as e:
                        logger.debug("Suppressed error", exc_info=True)  # was: pass
                if GEO_AVAILABLE:
                    await session.execute(
                        world_alerts.insert().values(
                            alert_id=alert.alert_id,
                            severity=str(sev.value if hasattr(sev, "value") else sev),
                            description=alert.description,
                            source=alert.source,
                            ts=alert.timestamp,
                            properties={"source": alert.source},
                            org_id=org_id,
                            geom=text(
                                f"ST_SetSRID(ST_MakePoint({float(aloc.longitude)}, {float(aloc.latitude)}), 4326)"
                            ),
                        )
                    )
                else:
                    await session.execute(
                        world_alerts.insert().values(
                            alert_id=alert.alert_id,
                            severity=str(sev.value if hasattr(sev, "value") else sev),
                            description=alert.description,
                            source=alert.source,
                            ts=alert.timestamp,
                            properties={
                                "source": alert.source,
                                "lon": float(aloc.longitude),
                                "lat": float(aloc.latitude),
                            },
                            org_id=org_id,
                        )
                    )
                await session.commit()
        except Exception as e:
            logger.debug("Suppressed error", exc_info=True)  # was: pass
        if websocket_manager:
            await websocket_manager.broadcast_to_org(
                json.dumps({"type": "alert", "data": am.model_dump()}),
                org_id=org_id or "default",
            )
        # Store in memory for GET /v1/alerts
        _alerts_store[alert.alert_id] = {
            "alert_id": alert.alert_id,
            "severity": alert.severity,
            "description": alert.description,
            "source": alert.source,
            "ts_iso": alert.ts_iso or _now_iso(),
            "acknowledged": False,
        }
        return {"status": "published", "alert_id": alert.alert_id}
    except Exception as e:
        logger.error("Failed to publish alert", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to publish alert")


@app.post("/api/v1/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Mark an alert as acknowledged, stopping escalation."""
    found = False
    for a in world_state["alerts"]:
        if a.get("alert_id") == alert_id:
            a["acknowledged"] = True
            a["acknowledged_at"] = datetime.now(timezone.utc).isoformat()
            found = True
            break
    # Also update the escalation-watched dict
    if alert_id in _escalation_alerts:
        _escalation_alerts[alert_id]["acknowledged"] = True
        found = True
    if not found:
        raise HTTPException(
            status_code=404, detail=f"Alert {alert_id} not found in active state"
        )
    return {"status": "acknowledged", "alert_id": alert_id}


@app.get("/api/v1/elevation")
async def get_elevation(lat: float, lon: float):
    """Return terrain elevation (metres MSL) at lat/lon from SRTM DEM tiles."""
    try:
        import sys as _sys

        _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
        from packages.geo.dem import get_provider

        elev = get_provider().get_elevation(lat, lon)
        return {"lat": lat, "lon": lon, "elevation_m": elev}
    except Exception as e:
        return {"lat": lat, "lon": lon, "elevation_m": 0.0, "note": str(e)}


@app.post("/missions")
async def publish_mission_update(mission: "MissionData"):
    if not mqtt_client or not redis_client:
        raise HTTPException(status_code=503, detail="Services not connected")
    try:
        topic = f"missions/{mission.mission_id}"
        await mqtt_client.publish(topic, mission.model_dump_json())
        # Convert to internal model for Redis
        mstatus = (
            mission.status.lower()
            if isinstance(mission.status, str)
            else str(mission.status)
        )
        try:
            ms = MissionStatus(mstatus)
        except Exception:
            ms = MissionStatus.ACTIVE
        mu = MissionUpdate(
            mission_id=mission.mission_id,
            timestamp=mission.timestamp,
            status=ms,
            assets=mission.assets,
            objectives=mission.objectives,
        )
        await redis_client.add_mission_update(mu)
        if websocket_manager:
            _mission_org = getattr(mission, "org_id", None) or "default"
            await websocket_manager.broadcast_to_org(
                json.dumps({"type": "mission", "data": mu.model_dump()}),
                org_id=_mission_org,
            )
        return {"status": "published", "mission_id": mission.mission_id}
    except Exception as e:
        logger.error("Failed to publish mission update", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to publish mission update")


@app.post("/api/v1/nodes/register", response_model=NodeRegisterResponse)
async def register_node(req: NodeRegisterRequest):
    assert SessionLocal is not None
    now = datetime.now(timezone.utc)
    topics = {
        "pub": [f"devices/{req.id}/telemetry", f"alerts/{req.id}"],
        "sub": [f"tasks/{req.id}/#", f"control/{req.id}/#"],
    }
    policy = {"max_bitrate_kbps": 600, "retention_hours": 48}

    async with SessionLocal() as session:
        # Upsert node
        await _upsert_node(session, req, now)
        await session.commit()

    # Trigger coverage calc for towers (stub)
    if req.type.upper() == "TOWER":
        asyncio.create_task(_compute_viewshed_stub(req.id))

    token = _issue_token(req.id, topics, policy, ttl_seconds=600)
    return NodeRegisterResponse(
        status="accepted",
        mqtt_topics=topics,
        policy=policy,
        token=token,
    )


@app.delete("/api/v1/nodes/{node_id}")
async def retire_node(node_id: str):
    assert SessionLocal is not None
    async with SessionLocal() as session:
        await session.execute(
            nodes.update()
            .where(nodes.c.id == node_id)
            .values(
                retired=True,
                status="RETIRED",
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()
    return {"status": "retired", "id": node_id}


# Helpers
async def _upsert_node(
    session: AsyncSession, req: "NodeRegisterRequest", now: datetime
):
    existing = await session.execute(
        text("SELECT id FROM nodes WHERE id = :id"), {"id": req.id}
    )
    if existing.first():
        await session.execute(
            nodes.update()
            .where(nodes.c.id == req.id)
            .values(
                type=req.type,
                pubkey=req.pubkey,
                fw_version=req.fw_version,
                location=req.location,
                capabilities=req.capabilities,
                comm=req.comm,
                policy={"geofence_id": None, "risk_gate": None},
                retired=False,
                updated_at=now,
            )
        )
    else:
        await session.execute(
            nodes.insert().values(
                id=req.id,
                type=req.type,
                pubkey=req.pubkey,
                fw_version=req.fw_version,
                location=req.location,
                capabilities=req.capabilities,
                comm=req.comm,
                policy={"geofence_id": None, "risk_gate": None},
                status="ONLINE",
                last_seen=now,
                retired=False,
                created_at=now,
                updated_at=now,
            )
        )


@app.get("/api/v1/nodes/{node_id}")
async def get_node(node_id: str):
    assert SessionLocal is not None
    async with SessionLocal() as session:
        res = await session.execute(
            text("SELECT * FROM nodes WHERE id = :id"), {"id": node_id}
        )
        row = res.mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="Node not found")
        return {**row}


@app.get("/api/v1/coverage")
async def list_coverages():
    """List tower coverages; UI can union polygons client-side until server union is available."""
    assert SessionLocal is not None
    async with SessionLocal() as session:
        res = await session.execute(
            text(
                "SELECT node_id, viewshed_geojson, version, updated_at FROM coverages ORDER BY updated_at DESC"
            )
        )
        rows = [dict(r) for r in res.mappings().all()]
        return {"coverages": rows}


@app.get("/api/v1/coverage/union")
async def coverage_union():
    """Return union of all viewshed polygons as GeoJSON (or null if none)."""
    assert SessionLocal is not None
    # Shapely is optional; return null union if not available
    try:
        import shapely.geometry as sgeom
        from shapely.ops import unary_union
        from shapely.geometry import mapping
    except Exception:
        return {"union": None, "count": 0}
    async with SessionLocal() as session:
        res = await session.execute(
            text(
                "SELECT viewshed_geojson FROM coverages WHERE viewshed_geojson IS NOT NULL"
            )
        )
        geoms = []
        for r in res.mappings().all():
            try:
                geoms.append(sgeom.shape(r["viewshed_geojson"]))
            except Exception:
                continue
        if not geoms:
            return {"union": None, "count": 0}
        u = unary_union(geoms)
        return {"union": mapping(u), "count": len(geoms)}


# Handlers
def _raw_obs_to_c2(data: Dict[str, Any]) -> Optional["C2Observation"]:
    """
    Convert a raw adapter observation dict to a C2Observation.
    Returns None if the event_type is unrecognised or missing.
    """
    if not C2_INTEL_AVAILABLE:
        return None

    raw_evt = data.get("event_type", "")
    if not raw_evt:
        # Infer from adapter telemetry fields
        battery = data.get("metadata", {}).get("battery_remaining",
                  data.get("metadata", {}).get("battery_pct"))
        if battery is not None:
            battery = float(battery)
            if battery <= 15:
                raw_evt = "battery_critical"
            elif battery <= 25:
                raw_evt = "battery_low"
        if not raw_evt:
            return None

    try:
        evt = C2EventType(raw_evt.lower())
    except ValueError:
        return None  # not a recognised C2 event type

    raw_src = data.get("adapter_type", data.get("source", "unknown"))
    try:
        src = SensorSource(raw_src.lower())
    except ValueError:
        src = SensorSource.UNKNOWN

    entity_id = data.get("entity_id", "unknown")

    return C2Observation(
        event_type=evt,
        node_id=entity_id,
        title=data.get("callsign", entity_id),
        description=data.get("classification"),
        source=src,
        confidence=float(data.get("confidence", 0.5)),
        score=int(data.get("score", 50)),
    )


async def _handle_observation(topic: str, data: Dict[str, Any]):
    try:
        entity_id = data.get("entity_id", "")

        # ── 1. Forward raw observation to Redis Stream ─────────────────────
        if redis_client and redis_client.redis:
            await redis_client.redis.xadd("observations_stream", {
                "topic": topic,
                "payload": json.dumps(data),
                "ts": datetime.now(timezone.utc).isoformat(),
            })

        # ── 2. Run through c2_intel pipeline ──────────────────────────────
        if C2_INTEL_AVAILABLE and entity_id:
            c2_obs = _raw_obs_to_c2(data)
            if c2_obs:
                # Deduplication — drop redundant observations from multi-source sensors
                deduped = _c2_dedup.deduplicate([c2_obs])
                if not deduped:
                    return  # duplicate — stop processing

                c2_obs = deduped[0]

                # Accumulate per-entity observation window
                _c2_obs_window[entity_id].append(c2_obs)
                recent = list(_c2_obs_window[entity_id])

                # Priority scoring (single + composite)
                single_priority, actions = _c2_priority.score_observation(c2_obs)
                node_result = _c2_priority.score_node_observations(entity_id, recent)
                composite_priority = node_result.get("composite_priority", single_priority)
                promotion_reason   = node_result.get("promotion_reason")
                composite_score    = node_result.get("composite_score", c2_obs.score)

                # Chain detection — predict cascading events
                predictions = _c2_chains.predict(recent)

                # Evidence clustering — roll up observations into compound insights
                clusters = _c2_evidence.aggregate(recent, entity_id=entity_id)

                # Build enriched observation record for WorldStore / WebSocket
                enriched = {
                    "event_type":          c2_obs.event_type.value,
                    "confidence":          c2_obs.confidence,
                    "lat":  data.get("lat") or (data.get("position") or {}).get("lat"),
                    "lon":  data.get("lon") or (data.get("position") or {}).get("lon"),
                    "ts":   data.get("ts_iso", datetime.now(timezone.utc).isoformat()),
                    "source":              c2_obs.source.value,
                    "priority":            composite_priority.value,
                    "score":               composite_score,
                    "promotion_reason":    promotion_reason,
                    "actions":             [a.value for a in actions],
                    "predicted_events":    [
                        {"event": p.event, "minutes": p.minutes_from_now,
                         "confidence": p.confidence}
                        for p in predictions[:3]
                    ],
                    "evidence_clusters":   [c.to_dict() for c in clusters[:2]],
                }

                # ── 3. Write back into WorldStore entity ──────────────────
                if WORLD_STORE_AVAILABLE and world_store:
                    try:
                        entity = world_store.get(entity_id) or {}
                        existing = list(entity.get("_observations", []))
                        existing.append(enriched)
                        entity["_observations"] = existing[-20:]
                        entity["_composite_priority"] = composite_priority.value
                        entity["_composite_score"]    = composite_score
                        if predictions:
                            entity["_predicted_events"] = [
                                {"event": p.event, "minutes": p.minutes_from_now}
                                for p in predictions[:3]
                            ]
                        world_store.upsert(entity, source="c2_intel")
                    except Exception as ws_err:
                        logger.debug("WorldStore c2_intel update skipped: %s", ws_err)

                # ── 4. Broadcast high-priority events to WebSocket ─────────
                if websocket_manager:
                    from packages.c2_intel.models import ObservationPriority
                    is_high = composite_priority in (
                        ObservationPriority.CRITICAL, ObservationPriority.HIGH
                    )
                    if is_high or c2_obs.event_type.value == "entity_detected":
                        await websocket_manager.broadcast(json.dumps({
                            "type": "c2_event",
                            "data": {
                                "entity_id":        entity_id,
                                "event_type":       c2_obs.event_type.value,
                                "priority":         composite_priority.value,
                                "score":            composite_score,
                                "promotion_reason": promotion_reason,
                                "predicted_events": enriched["predicted_events"],
                                "evidence_clusters": enriched["evidence_clusters"],
                                "actions":          enriched["actions"],
                            },
                        }))

                # ── 5. Log CRITICAL compound promotions ───────────────────
                if promotion_reason:
                    logger.warning(
                        "c2_intel COMPOUND PROMOTION [%s] → %s: %s",
                        entity_id, composite_priority.value, promotion_reason,
                    )

        # Fallback: propagate any raw event_type even without full c2_intel
        elif entity_id and WORLD_STORE_AVAILABLE and world_store:
            raw_evt = data.get("event_type", "")
            if raw_evt:
                try:
                    entity = world_store.get(entity_id) or {}
                    existing = list(entity.get("_observations", []))
                    existing.append({
                        "event_type": raw_evt,
                        "confidence": data.get("confidence", 0.5),
                        "lat": data.get("lat") or (data.get("position") or {}).get("lat"),
                        "lon": data.get("lon") or (data.get("position") or {}).get("lon"),
                        "ts":  data.get("ts_iso", datetime.now(timezone.utc).isoformat()),
                        "source": data.get("adapter_type", "unknown"),
                    })
                    entity["_observations"] = existing[-20:]
                    world_store.upsert(entity, source="raw")
                except Exception:
                    pass

            if raw_evt == "entity_detected" and websocket_manager:
                await websocket_manager.broadcast(json.dumps({
                    "type": "entity_detected", "data": data,
                }))

    except Exception as e:
        logger.error("Failed to handle observation: %s", e)


async def _handle_mission(topic: str, data: Dict[str, Any]):
    try:
        # Broadcast to UI subscribers (org-scoped in Enterprise mode)
        if websocket_manager:
            await websocket_manager.broadcast_to_org(
                json.dumps({"type": "mission_event", "topic": topic, "data": data}),
                org_id=data.get("org_id", "default"),
            )
        logger.info("Forwarded mission to ws", topic=topic)
    except Exception as e:
        logger.error(f"Failed to handle mission: {e}")


async def _handle_heartbeat(topic: str, data: Dict[str, Any]):
    """health/{node_id}/heartbeat payload: {"ts":..., "status":"OK", ...}"""
    try:
        node_id = topic.split("/")[1]
        now = datetime.now(timezone.utc)
        assert SessionLocal is not None
        async with SessionLocal() as session:
            await session.execute(
                nodes.update()
                .where(nodes.c.id == node_id)
                .values(
                    status="ONLINE",
                    last_seen=now,
                    updated_at=now,
                )
            )
            await session.commit()
        if websocket_manager:
            await websocket_manager.broadcast_to_org(
                json.dumps({"type": "heartbeat", "node_id": node_id, "ts": data.get("ts")}),
                org_id=data.get("org_id", "default"),
            )
    except Exception as e:
        logger.error(f"Failed to handle heartbeat: {e}")


async def _handle_plainview_leak(topic: str, data: Dict[str, Any]):
    """Handle Plainview leak/spill detection events and map to alert_stream + world_alerts."""
    try:
        if not redis_client:
            return
        # Parse fields
        aid = str(data.get("id") or data.get("alert_id") or "")
        sev_raw = str(data.get("severity") or "low").lower()
        try:
            sev = SeverityLevel(sev_raw)
        except Exception:
            sev = SeverityLevel.LOW
        loc = data.get("location") or {}
        latitude = float(loc.get("lat")) if loc.get("lat") is not None else None
        longitude = float(loc.get("lon")) if loc.get("lon") is not None else None
        if latitude is None or longitude is None:
            return
        aloc = Location(latitude=latitude, longitude=longitude)
        am = AlertMessage(
            alert_id=aid or f"LEAK-{int(datetime.now(timezone.utc).timestamp())}",
            timestamp=datetime.now(timezone.utc),
            severity=sev,
            location=aloc,
            description=str(data.get("class") or "Leak/Spill detected"),
            source=str(data.get("source") or "plainview"),
            category="plainview.leak",
            tags=["plainview", "leak"],
            metadata={
                k: v for k, v in data.items() if k not in {"id", "severity", "location"}
            },
        )
        await redis_client.add_alert(am)
        # Persist minimal world_alerts row (reuse existing path via /alerts would be heavier)
        try:
            assert SessionLocal is not None
            async with SessionLocal() as session:
                if GEO_AVAILABLE:
                    await session.execute(
                        world_alerts.insert().values(
                            alert_id=am.alert_id,
                            severity=str(sev.value if hasattr(sev, "value") else sev),
                            description=am.description,
                            source=am.source,
                            ts=am.timestamp,
                            properties=am.metadata,
                            org_id=None,
                            geom=text(
                                f"ST_SetSRID(ST_MakePoint({float(longitude)}, {float(latitude)}), 4326)"
                            ),
                        )
                    )
                else:
                    await session.execute(
                        world_alerts.insert().values(
                            alert_id=am.alert_id,
                            severity=str(sev.value if hasattr(sev, "value") else sev),
                            description=am.description,
                            source=am.source,
                            ts=am.timestamp,
                            properties={
                                **am.metadata,
                                "lon": float(longitude),
                                "lat": float(latitude),
                            },
                            org_id=None,
                        )
                    )
                await session.commit()
        except Exception as e:
            logger.debug("Suppressed error", exc_info=True)  # was: pass
        if websocket_manager:
            await websocket_manager.broadcast_to_org(
                json.dumps({"type": "alert", "data": am.model_dump()}),
                org_id=data.get("org_id", "default"),
            )
    except Exception as e:
        logger.error(f"Failed to handle plainview leak: {e}")


async def _handle_valve_status(topic: str, data: Dict[str, Any]):
    """Handle valve status telemetry and forward to operations_stream and WS."""
    try:
        asset_id = topic.split("/")[1] if "/" in topic else None
        record = {
            "topic": topic,
            "asset_id": asset_id or data.get("asset_id") or "unknown",
            "payload": json.dumps(data),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if redis_client:
            await redis_client.add_operation_event(record)
        if websocket_manager:
            await websocket_manager.broadcast_to_org(
                json.dumps({"type": "valve_status", "topic": topic, "data": data}),
                org_id=data.get("org_id", "default"),
            )
    except Exception as e:
        logger.error(f"Failed to handle valve status: {e}")


async def _handle_pipeline_pressure(topic: str, data: Dict[str, Any]):
    """Handle pipeline pressure taps; forward as operations events."""
    try:
        segment_id = topic.split("/")[-1]
        record = {
            "topic": topic,
            "segment_id": segment_id,
            "payload": json.dumps(data),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if redis_client:
            await redis_client.add_operation_event(record)
    except Exception as e:
        logger.error(f"Failed to handle pipeline pressure: {e}")


async def _handle_entity_update(topic: str, data: Dict[str, Any]):
    """Handle entities/{entity_id}/update from SDK adapters — ingest into WorldStore."""
    try:
        if not WORLD_STORE_AVAILABLE or not world_store:
            return
        from packages.entities.core import Entity

        entity_id = data.get("entity_id") or topic.split("/")[1]
        # Build Entity from MQTT dict and upsert into WorldStore
        data["id"] = entity_id
        entity = Entity.from_dict(data)
        world_store.upsert(entity, source="mqtt")
        # Record position in history trail
        if history_store:
            history_store.record_from_entity(data)
        # Also feed into mesh CRDT for replication
        if entity_crdt:
            entity_crdt.update(entity_id, data)
    except Exception as e:
        logger.error(f"Failed to handle entity update: {e}")


async def _ttl_gc_task():
    """Periodically prune expired entities from the WorldStore (every 30 s)."""
    while True:
        await asyncio.sleep(30)
        if world_store:
            pruned = world_store.prune_expired()
            if pruned:
                logger.info(f"TTL GC pruned {pruned} expired entities")
                if history_store:
                    # Evict history for entities that no longer exist
                    active_ids = set(world_store._entities.keys())
                    for eid in list(history_store.entity_ids()):
                        if eid not in active_ids:
                            history_store.evict(eid)


async def _verify_ws_token(websocket: WebSocket) -> bool:
    """Return True if the WebSocket connection carries a valid Fabric JWT.

    Checks Authorization header first, then ?token= query param.
    When WS_AUTH_REQUIRED is False (default dev mode) always returns True.
    """
    if not WS_AUTH_REQUIRED:
        return True
    auth = websocket.headers.get("Authorization", "") or websocket.query_params.get("token", "")
    if auth.startswith("Bearer "):
        auth = auth[7:]
    if not auth:
        return False
    try:
        jwt.decode(auth, FABRIC_JWT_SECRET, algorithms=["HS256"])
        return True
    except Exception:
        return False


# Background processors
async def telemetry_processor():
    while True:
        try:
            if redis_client:
                await redis_client.process_telemetry_stream()
            await asyncio.sleep(1)
        except Exception as e:
            logger.error("Telemetry processor error", error=str(e))
            await asyncio.sleep(5)


async def alert_processor():
    while True:
        try:
            if redis_client:
                await redis_client.process_alert_stream()
            await asyncio.sleep(1)
        except Exception as e:
            logger.error("Alert processor error", error=str(e))
            await asyncio.sleep(5)


async def heartbeat_watcher():
    """Flip nodes to STALE/OFFLINE based on last_seen."""
    assert SessionLocal is not None
    while True:
        try:
            now = datetime.now(timezone.utc)
            async with SessionLocal() as session:
                # STALE
                await session.execute(
                    text(
                        """
                        UPDATE nodes
                        SET status = 'STALE', updated_at = :now
                        WHERE status = 'ONLINE' AND last_seen IS NOT NULL AND last_seen < :stale_cutoff AND retired = FALSE;
                        """
                    ),
                    {
                        "now": now,
                        "stale_cutoff": now - timedelta(seconds=HEARTBEAT_STALE_SECS),
                    },
                )
                # OFFLINE
                await session.execute(
                    text(
                        """
                        UPDATE nodes
                        SET status = 'OFFLINE', updated_at = :now
                        WHERE (status = 'STALE' OR status = 'ONLINE') AND last_seen IS NOT NULL AND last_seen < :offline_cutoff AND retired = FALSE;
                        """
                    ),
                    {
                        "now": now,
                        "offline_cutoff": now
                        - timedelta(seconds=HEARTBEAT_OFFLINE_SECS),
                    },
                )
                await session.commit()
            await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"Heartbeat watcher error: {e}")
            await asyncio.sleep(10)


# Geofences table and endpoints
geofences = Table(
    "geofences",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("org_id", String(128)),
    Column("name", String(128)),
    Column("props", JSON),
    *(
        [Column("geom", Geometry(geometry_type="POLYGON", srid=4326))]
        if GEO_AVAILABLE
        else []
    ),
)


@app.post("/api/v1/geofences")
async def create_geofence(
    payload: dict, _claims: dict | None = Depends(_verify_bearer_fabric)
):
    assert SessionLocal is not None
    org_id = None
    try:
        org_id = (
            (_claims or {}).get("org")
            or (_claims or {}).get("org_id")
            or (_claims or {}).get("tenant")
        )
    except Exception:
        org_id = None
    name = str(payload.get("name") or "geofence")
    props = payload.get("props") or {}
    coords = payload.get("coordinates")  # GeoJSON-like: [[lon,lat], ...]
    if not coords or not GEO_AVAILABLE:
        # store props only
        async with SessionLocal() as session:
            await session.execute(
                geofences.insert().values(org_id=org_id, name=name, props=props)
            )
            await session.commit()
        return {"status": "created", "name": name}
    # Build polygon WKT
    try:
        ring = ",".join([f"{float(x)} {float(y)}" for x, y in coords])
        wkt = f"POLYGON(({ring}))"
        async with SessionLocal() as session:
            await session.execute(
                geofences.insert().values(
                    org_id=org_id,
                    name=name,
                    props=props,
                    geom=text(f"ST_GeomFromText('{wkt}', 4326)"),
                )
            )
            await session.commit()
        return {"status": "created", "name": name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/geofences")
async def list_geofences(org_id: str | None = Depends(_get_org_id)):
    assert SessionLocal is not None
    async with SessionLocal() as session:
        where = "WHERE org_id = :org_id" if org_id else ""
        res = await session.execute(
            text(f"SELECT id, name, props FROM geofences {where} ORDER BY id DESC"),
            {"org_id": org_id} if org_id else {},
        )
        rows = [dict(r) for r in res.mappings().all()]
        return {"geofences": rows}


@app.get("/api/v1/geofences/contains")
async def geofence_contains(
    lat: float, lon: float, org_id: str | None = Depends(_get_org_id)
):
    if not GEO_AVAILABLE:
        return {"contains": True}
    assert SessionLocal is not None
    async with SessionLocal() as session:
        where = "WHERE org_id = :org_id" if org_id else ""
        q = text(
            f"SELECT EXISTS (SELECT 1 FROM geofences {where} WHERE ST_Contains(geom, ST_SetSRID(ST_MakePoint(:lon,:lat),4326))) AS inside"
        )
        params = {"lat": lat, "lon": lon}
        if org_id:
            params["org_id"] = org_id
        r = (await session.execute(q, params)).mappings().first()
        return {"contains": bool(r.get("inside")) if r else False}


# Coverage stub
async def _compute_viewshed_stub(node_id: str):
    """Placeholder viewshed compute; stores a null geometry and notifies UI."""
    try:
        assert SessionLocal is not None
        async with SessionLocal() as session:
            await session.execute(
                coverages.insert().values(
                    node_id=node_id,
                    viewshed_geojson=None,
                    version="demSRTM_v4_1",
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
        if websocket_manager:
            await websocket_manager.broadcast_to_org(
                json.dumps({"type": "coverage_updated", "node_id": node_id}),
                org_id="default",  # coverage events are infrastructure-scoped, not org-scoped
            )
    except Exception as e:
        logger.error(f"Viewshed stub failed: {e}")


# JWT helper and token refresh
def _issue_token(
    node_id: str,
    topics: Dict[str, List[str]],
    policy: Dict[str, Any],
    ttl_seconds: int = 600,
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": node_id,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl_seconds)).timestamp()),
        "topics": topics,
        "policy": policy,
    }
    return jwt.encode(payload, FABRIC_JWT_SECRET, algorithm="HS256")


class TokenResponse(BaseModel):
    token: str
    expires_in: int


@app.post("/api/v1/nodes/{node_id}/token", response_model=TokenResponse)
async def refresh_token(node_id: str):
    topics = {
        "pub": [f"devices/{node_id}/telemetry", f"alerts/{node_id}"],
        "sub": [f"tasks/{node_id}/#", f"control/{node_id}/#"],
    }
    policy = {"max_bitrate_kbps": 600, "retention_hours": 48}
    token = _issue_token(node_id, topics, policy, ttl_seconds=600)
    return TokenResponse(token=token, expires_in=600)


def _run_migrations():
    try:
        ini_path = os.path.join(os.path.dirname(__file__), "alembic.ini")
        cfg = AlembicConfig(ini_path)
        # Ensure env picks up the DB URL
        if not os.getenv("POSTGRES_URL"):
            os.environ["POSTGRES_URL"] = (
                "postgresql://heli:summit_password@localhost:5432/heli_os"
            )
        alembic_command.upgrade(cfg, "head")
        logger.info("Alembic migrations applied")
    except Exception as e:
        logger.error(f"Failed to run migrations: {e}")
        raise


# ── REST API endpoints (/v1/ prefix — matches frontend lib/api.ts) ───────────

# ── Alerts ────────────────────────────────────────────────────────────────────

@app.get("/v1/alerts")
async def list_alerts(limit: int = 100):
    alerts = sorted(_alerts_store.values(), key=lambda a: a.get("ts_iso", ""), reverse=True)
    return {"alerts": alerts[:limit]}


@app.post("/v1/alerts/{alert_id}/acknowledge")
async def ack_alert_v1(alert_id: str):
    if alert_id in _alerts_store:
        _alerts_store[alert_id]["acknowledged"] = True
    return {"ok": True, "alert_id": alert_id}


# ── Missions ──────────────────────────────────────────────────────────────────

@app.get("/v1/missions")
async def list_missions(limit: int = 50):
    missions = sorted(_missions_store.values(), key=lambda m: m.get("created_at", ""), reverse=True)
    return missions[:limit]


@app.post("/v1/missions")
async def create_mission_v1(payload: Dict[str, Any]):
    mission_id = str(_uuid.uuid4())
    mission = {
        "mission_id": mission_id,
        "name": payload.get("name"),
        "objectives": payload.get("objectives", []),
        "status": "PENDING",
        "created_at": _now_iso(),
        "started_at": None,
        "completed_at": None,
    }
    _missions_store[mission_id] = mission
    if websocket_manager:
        await websocket_manager.broadcast(json.dumps({"type": "mission_created", "data": mission}))
    return mission


@app.post("/v1/missions/parse")
async def parse_mission_nlp(payload: Dict[str, Any]):
    text = payload.get("text", "")
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return {"mission_type": "SURVEY", "pattern": "grid", "altitude_m": 120,
                "asset_hint": None, "objectives": [text], "confidence": 0.5,
                "interpretation": text}
    import httpx as _httpx
    prompt = (
        "Parse this mission command into JSON with fields: "
        "mission_type (SURVEY/RECON/ESCORT/PATROL/INTERCEPT), pattern (grid/lawnmower/spiral/orbit), "
        "altitude_m (number), asset_hint (string or null), objectives (array of strings), "
        "confidence (0-1), interpretation (plain English summary). "
        f"Command: {text}\nRespond with only valid JSON."
    )
    try:
        async with _httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={"model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.1, "max_tokens": 300},
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
    except Exception as e:
        logger.warning("Mission NLP parse failed: %s", e)
        return {"mission_type": "SURVEY", "pattern": "grid", "altitude_m": 120,
                "asset_hint": None, "objectives": [text], "confidence": 0.4,
                "interpretation": text}


@app.post("/v1/missions/preview")
async def preview_mission_waypoints(payload: Dict[str, Any]):
    pattern = payload.get("pattern", "grid")
    alt = float(payload.get("altitude_m", 120))
    area = payload.get("area", [])
    if not area:
        return {"waypoints": [], "pattern": pattern, "count": 0}
    lats = [p["lat"] for p in area]
    lons = [p["lon"] for p in area]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    waypoints = []
    steps = 5
    for i in range(steps + 1):
        t = i / steps
        lat = min_lat + t * (max_lat - min_lat)
        lon = (min_lon if i % 2 == 0 else max_lon)
        waypoints.append({"lat": lat, "lon": lon, "alt": alt})
    return {"waypoints": waypoints, "pattern": pattern, "count": len(waypoints)}


@app.get("/v1/missions/{mission_id}/replay/timeline")
async def replay_timeline(mission_id: str):
    return {"mission_id": mission_id, "count": 0, "start": None, "end": None, "timestamps": []}


@app.get("/v1/missions/{mission_id}/replay/snapshot")
async def replay_snapshot(mission_id: str, ts: str = "", index: int = 0):
    return {"mission_id": mission_id, "ts": ts, "entities": []}


# ── Mission Orchestrator (NL → devices execute) ───────────────────────────────

from .mission_orchestrator import launch_mission, stop_mission, get_mission_status, list_active_missions


async def _orchestrator_world_store_fn():
    """Pull current entities from WorldStore for the orchestrator."""
    if WORLD_STORE_AVAILABLE and world_store:
        try:
            entities = await world_store.get_all()
            return [e if isinstance(e, dict) else e.__dict__ for e in entities]
        except Exception:
            pass
    return []


async def _orchestrator_dispatch_fn(entity_id: str, command: dict):
    """Route orchestrator commands to the right adapter."""
    cmd_type = command.get("type", "")
    if websocket_manager:
        await websocket_manager.broadcast(json.dumps({
            "type": "command_sent",
            "data": {"entity_id": entity_id, "command": cmd_type, "payload": command},
        }))
    # MAVLink assets
    if entity_id.startswith("mavlink-"):
        try:
            from adapters.mavlink_adapter import MAVLinkAdapter
            adapter = MAVLinkAdapter()
            await adapter.send_command(entity_id, command)
            return
        except Exception as e:
            logger.warning("MAVLink dispatch failed for %s: %s", entity_id, e)

    # DJI assets — route via DJI Cloud API MQTT broker
    if entity_id.startswith("dji-"):
        try:
            if mqtt_client:
                import uuid as _uuid
                sn = entity_id.replace("dji-", "")
                topic = f"thing/product/{sn}/services"
                await mqtt_client.publish(topic, json.dumps({
                    **command,
                    "bid": str(_uuid.uuid4()),
                    "tid": str(_uuid.uuid4()),
                }))
                return
        except Exception as e:
            logger.warning("DJI dispatch failed for %s: %s", entity_id, e)

    # ONVIF cameras — route via MQTT (adapter instances own the connection)
    if entity_id.startswith("onvif-") or cmd_type.startswith("PTZ"):
        try:
            if mqtt_client:
                await mqtt_client.publish(
                    f"heli/commands/{entity_id}",
                    json.dumps({"entity_id": entity_id, "command": command}),
                )
        except Exception as e:
            logger.warning("ONVIF dispatch failed for %s: %s", entity_id, e)


async def _orchestrator_broadcast_fn(event: dict):
    if websocket_manager:
        await websocket_manager.broadcast(json.dumps(event))


@app.post("/v1/missions/execute")
async def execute_mission(payload: Dict[str, Any]):
    """
    Parse a natural-language mission and immediately begin autonomous execution.

    Body:
      {
        "text": "Find the missing hiker in grid sector 7",
        "area": [{"lat": 37.1, "lon": -122.0}, ...],   // search polygon
        "altitude_m": 50                                 // optional, default 50
      }

    Returns mission_id. Poll GET /v1/missions/active/{mission_id} for status.
    """
    text = payload.get("text", "")
    area = payload.get("area", [])
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if not area or len(area) < 2:
        raise HTTPException(status_code=400, detail="area must have at least 2 points")

    # Parse natural language → structured mission
    nlp_result = await parse_mission_nlp({"text": text})
    if payload.get("altitude_m"):
        nlp_result["altitude_m"] = payload["altitude_m"]

    mission_id = await launch_mission(
        nlp_result=nlp_result,
        area=area,
        world_store_fn=_orchestrator_world_store_fn,
        dispatch_fn=_orchestrator_dispatch_fn,
        broadcast_fn=_orchestrator_broadcast_fn,
    )
    return {
        "mission_id": mission_id,
        "status": "executing",
        "interpretation": nlp_result.get("interpretation", text),
        "pattern": nlp_result.get("pattern", "grid"),
        "mission_type": nlp_result.get("mission_type", "SURVEY"),
    }


@app.delete("/v1/missions/active/{mission_id}")
async def abort_mission(mission_id: str):
    """Abort an active autonomous mission."""
    stopped = await stop_mission(mission_id)
    if not stopped:
        raise HTTPException(status_code=404, detail=f"No active mission: {mission_id}")
    return {"ok": True, "mission_id": mission_id, "status": "aborted"}


@app.get("/v1/missions/active")
async def list_active():
    """List all currently executing missions."""
    return list_active_missions()


@app.get("/v1/missions/active/{mission_id}")
async def mission_status(mission_id: str):
    """Get real-time status of an executing mission."""
    status = get_mission_status(mission_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"No active mission: {mission_id}")
    return status


# ── Tasks ─────────────────────────────────────────────────────────────────────

@app.get("/v1/tasks")
async def list_tasks(limit: int = 100):
    tasks = sorted(_tasks_store.values(), key=lambda t: t.get("created_at", ""), reverse=True)
    return tasks[:limit]


@app.get("/v1/tasks/pending")
async def list_pending_tasks():
    return [t for t in _tasks_store.values() if t.get("status") == "pending"]


@app.post("/v1/tasks")
async def create_task(payload: Dict[str, Any]):
    task_id = str(_uuid.uuid4())
    task = {
        "task_id": task_id,
        "asset_id": payload.get("asset_id", ""),
        "action": payload.get("action", ""),
        "status": "pending",
        "risk_level": payload.get("risk_level", "LOW"),
        "waypoints": payload.get("waypoints", []),
        "mission_id": payload.get("mission_id"),
        "objective": payload.get("objective"),
        "created_at": _now_iso(),
        "started_at": None,
        "completed_at": None,
    }
    _tasks_store[task_id] = task
    if websocket_manager:
        await websocket_manager.broadcast(json.dumps({"type": "task_created", "data": task}))
    return {"task_id": task_id, "status": "pending"}


@app.post("/v1/tasks/{task_id}/approve")
async def approve_task(task_id: str, payload: Dict[str, Any]):
    if task_id in _tasks_store:
        _tasks_store[task_id]["status"] = "active"
        _tasks_store[task_id]["started_at"] = _now_iso()
        _tasks_store[task_id]["approved_by"] = payload.get("approved_by")
    return {"ok": True, "task_id": task_id}


# ── Assets ────────────────────────────────────────────────────────────────────

@app.get("/v1/assets")
async def list_assets():
    return {"assets": list(_assets_store.values())}


# ── Worldstate + geofence aliases at /v1/ (frontend calls /v1/ not /api/v1/) ─

@app.get("/v1/worldstate")
async def get_world_state_alias():
    return {
        "entity_count": len(world_state.get("devices", {})),
        "alert_count": len(_alerts_store),
        "mission_count": len(_missions_store),
        "entities": list(world_state.get("devices", {}).values()),
    }


@app.get("/v1/geofences")
async def list_geofences_alias():
    if engine and SessionLocal:
        try:
            async with SessionLocal() as session:
                rows = await session.execute(text("SELECT * FROM geofences ORDER BY id DESC"))
                return {"geofences": [dict(r._mapping) for r in rows.all()]}
        except Exception:
            pass
    return {"geofences": []}


@app.post("/v1/geofences")
async def create_geofence_alias(payload: Dict[str, Any]):
    if engine and SessionLocal:
        try:
            async with SessionLocal() as session:
                await session.execute(
                    text("INSERT INTO geofences (name, type, props) VALUES (:name, :type, :props)"),
                    {"name": payload.get("name", ""), "type": payload.get("type", "EXCLUSION"),
                     "props": json.dumps(payload.get("props", {}))})
                await session.commit()
        except Exception as e:
            return {"ok": False, "error": str(e)}
    return {"ok": True}


@app.delete("/v1/geofences/{gf_id}")
async def delete_geofence_alias(gf_id: int):
    if engine and SessionLocal:
        try:
            async with SessionLocal() as session:
                await session.execute(text("DELETE FROM geofences WHERE id = :id"), {"id": gf_id})
                await session.commit()
        except Exception:
            pass
    return {"ok": True}


# ── AI Agent (Groq-powered) ───────────────────────────────────────────────────

_GROQ_SYSTEM_PROMPT = """You are HELIOS, an AI mission coordinator for Heli.OS — an autonomous systems platform for disaster response, search & rescue, and wildfire operations. You help operators plan and coordinate UAV and ground asset missions.

When given a mission objective:
1. Analyze the situation briefly
2. Recommend specific assets and actions
3. Identify risks
4. Provide a concise execution plan

Keep responses tactical, concise, and actionable. Use operator-style language."""


@app.post("/agents")
async def create_agent(payload: Dict[str, Any]):
    agent_id = str(_uuid.uuid4())[:8].upper()
    objective = payload.get("mission_objective") or payload.get("command") or ""
    agent = {
        "agent_id": agent_id,
        "objective": objective,
        "status": "RUNNING",
        "created_at": _now_iso(),
        "result": None,
    }
    _agents_store[agent_id] = agent

    async def _run_agent():
        import httpx as _httpx

        async def _broadcast_result(result: str):
            _agents_store[agent_id]["status"] = "COMPLETED"
            _agents_store[agent_id]["result"] = result
            if websocket_manager:
                await websocket_manager.broadcast(json.dumps({
                    "type": "agent_result",
                    "data": {"agent_id": agent_id, "result": result, "objective": objective}
                }))

        # ── Try Groq first ────────────────────────────────────────────────────
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                async with _httpx.AsyncClient(timeout=30.0) as c:
                    r = await c.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                        json={"model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                              "messages": [
                                  {"role": "system", "content": _GROQ_SYSTEM_PROMPT},
                                  {"role": "user", "content": objective},
                              ],
                              "temperature": 0.7, "max_tokens": 500},
                    )
                    r.raise_for_status()
                    result = r.json()["choices"][0]["message"]["content"].strip()
                    await _broadcast_result(result)
                    return
            except Exception as e:
                logger.warning("Groq agent failed, falling back to Ollama: %s", e)

        # ── Ollama fallback (local LLM) ───────────────────────────────────────
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
        try:
            async with _httpx.AsyncClient(timeout=60.0) as c:
                r = await c.post(
                    f"{ollama_url}/api/chat",
                    json={"model": ollama_model, "stream": False,
                          "messages": [
                              {"role": "system", "content": _GROQ_SYSTEM_PROMPT},
                              {"role": "user", "content": objective},
                          ]},
                )
                r.raise_for_status()
                result = r.json()["message"]["content"].strip()
                await _broadcast_result(f"[Ollama/{ollama_model}] {result}")
                return
        except Exception as e:
            logger.warning("Ollama agent failed: %s", e)

        _agents_store[agent_id]["status"] = "FAILED"
        _agents_store[agent_id]["result"] = "AI brain offline — no Groq API key and Ollama unreachable."

    asyncio.create_task(_run_agent())
    return {"ok": True, "agent_id": agent_id}


@app.get("/agents")
async def list_agents():
    return {"agents": list(_agents_store.values())}


# Also wire incoming alerts from the existing /alerts POST into _alerts_store
# (patch the publish_alert handler to also store in memory)


# ── OpenSky ADS-B poller (direct, no MQTT) ───────────────────────────────────

_OPENSKY_TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network"
    "/protocol/openid-connect/token"
)

async def _opensky_poll_loop():
    """Poll OpenSky every OPENSKY_POLL_INTERVAL seconds and push entities to WebSocket."""
    import time as _time

    if os.getenv("OPENSKY_ENABLED", "true").lower() != "true":
        logger.info("OpenSky poller disabled (OPENSKY_ENABLED != true)")
        return

    poll_interval = max(float(os.getenv("OPENSKY_POLL_INTERVAL", "10")), 5.0)
    client_id = os.getenv("OPENSKY_CLIENT_ID")
    client_secret = os.getenv("OPENSKY_CLIENT_SECRET")
    bbox = os.getenv("OPENSKY_BBOX", "")

    _token: Optional[str] = None
    _token_expires: float = 0.0

    async def _get_token(client: "httpx.AsyncClient") -> Optional[str]:
        nonlocal _token, _token_expires
        if _token and _time.monotonic() < _token_expires:
            return _token
        try:
            resp = await client.post(
                _OPENSKY_TOKEN_URL,
                data={"grant_type": "client_credentials",
                      "client_id": client_id, "client_secret": client_secret},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15.0,
            )
            resp.raise_for_status()
            d = resp.json()
            _token = d["access_token"]
            _token_expires = _time.monotonic() + int(d.get("expires_in", 1800)) - 30
            return _token
        except Exception as e:
            logger.warning("OpenSky token refresh failed: %s", e)
            return None

    logger.info("OpenSky poller starting (interval=%ss)", poll_interval)
    import httpx as _httpx
    async with _httpx.AsyncClient(timeout=30.0) as client:
        while True:
            try:
                params: Dict[str, Any] = {}
                if bbox and bbox.strip():
                    parts = [float(x) for x in bbox.split(",")]
                    if len(parts) == 4:
                        params = {"lamin": parts[0], "lomin": parts[1],
                                  "lamax": parts[2], "lomax": parts[3]}

                headers: Dict[str, str] = {}
                if client_id and client_secret:
                    tok = await _get_token(client)
                    if tok:
                        headers["Authorization"] = f"Bearer {tok}"

                resp = await client.get(
                    "https://opensky-network.org/api/states/all",
                    params=params, headers=headers,
                )
                resp.raise_for_status()
                states = resp.json().get("states") or []

                now = _time.time()
                batch = []
                for sv in states:
                    try:
                        icao24 = sv[0]
                        lat, lon = sv[6], sv[5]
                        if not icao24 or lat is None or lon is None:
                            continue
                        callsign = (sv[1] or "").strip() or icao24.upper()
                        alt = float(sv[7] or sv[13] or 0)
                        speed = float(sv[9] or 0)
                        heading = float(sv[10] or 0)
                        on_ground = bool(sv[8])
                        batch.append({
                            "entity_id": f"opensky-{icao24}",
                            "entity_type": "neutral",
                            "domain": "aerial",
                            "classification": "aircraft",
                            "position": {
                                "lat": float(lat), "lon": float(lon),
                                "alt": alt, "heading_deg": heading,
                            },
                            "speed_mps": speed,
                            "confidence": 0.95,
                            "last_seen": now,
                            "source_sensors": ["opensky-adsb"],
                            "callsign": callsign,
                            "properties": {"on_ground": on_ground, "icao24": icao24},
                        })
                    except Exception:
                        continue

                if batch and websocket_manager:
                    await websocket_manager.broadcast(
                        json.dumps({"type": "entity_batch", "data": batch})
                    )
                    logger.info("OpenSky: broadcast %d aircraft", len(batch))

            except Exception as e:
                logger.error("OpenSky poll error: %s", e)

            await asyncio.sleep(poll_interval)


# ── MAVLink simulated vehicle loop ───────────────────────────────────────────

async def _mavlink_sim_loop():
    """Broadcast simulated drone entities for demo/dev. Disable with MAVLINK_SIM_DISABLED=true."""
    import math as _math
    import time as _time

    if os.getenv("MAVLINK_SIM_DISABLED", "false").lower() == "true":
        logger.info("MAVLink simulation disabled")
        return

    logger.info("MAVLink simulation started — 2 simulated drones")

    _vehicles = [
        {"id": "HELI-SIM-01", "clat": 34.052235, "clon": -118.243683,
         "r": 0.020, "spd": 0.15, "alt": 150.0, "vel": 12.0, "batt": 85.0, "drain": 0.002},
        {"id": "HELI-SIM-02", "clat": 34.068920, "clon": -118.445100,
         "r": 0.015, "spd": 0.10, "alt": 120.0, "vel": 9.0, "batt": 62.0, "drain": 0.0015},
    ]
    angles = [0.0, 1.5]
    tick = 5.0

    while True:
        try:
            now = _time.time()
            batch = []
            for i, v in enumerate(_vehicles):
                angles[i] = (angles[i] + v["spd"] * tick) % (2 * _math.pi)
                lat = v["clat"] + v["r"] * _math.sin(angles[i])
                lon = v["clon"] + v["r"] * _math.cos(angles[i])
                hdg = (_math.degrees(angles[i] + _math.pi / 2)) % 360
                v["batt"] = max(0.0, v["batt"] - v["drain"] * tick)
                state = "critical" if v["batt"] < 10 else ("warning" if v["batt"] < 20 else "active")
                batch.append({
                    "entity_id": f"mavlink-{v['id'].lower()}",
                    "entity_type": state,
                    "domain": "aerial",
                    "classification": "drone",
                    "callsign": v["id"],
                    "position": {"lat": lat, "lon": lon, "alt": v["alt"], "heading_deg": hdg},
                    "speed_mps": v["vel"],
                    "confidence": 1.0,
                    "last_seen": now,
                    "source_sensors": ["mavlink-sim"],
                    "battery_pct": v["batt"],
                    "properties": {"controllable": True, "vehicle_type": "quadcopter", "sim": True},
                })
            if batch and websocket_manager:
                await websocket_manager.broadcast(
                    json.dumps({"type": "entity_batch", "data": batch})
                )
        except Exception as e:
            logger.error("MAVLink sim error: %s", e)
        await asyncio.sleep(tick)


# ── Dispatch command endpoint ─────────────────────────────────────────────────

@app.post("/v1/dispatch/{entity_id}")
async def dispatch_entity_command(entity_id: str, payload: Dict[str, Any]):
    """Send a command to an entity (MAVLink adapter or broadcast)."""
    command = str(payload.get("command", "dispatch")).upper()
    waypoints = payload.get("waypoints")
    result_detail = "command queued"

    # Try to route via MAVLink adapter if entity is a MAVLink vehicle
    if entity_id.startswith("mavlink-"):
        try:
            import sys as _sys
            _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
            from adapters.mavlink.adapter import MAVLinkAdapter
            adapter = MAVLinkAdapter()
            vehicle_id = entity_id.replace("mavlink-", "").upper()
            if command == "HALT":
                await adapter.send_command(vehicle_id, {"action": "HALT"})
            elif command in ("RTB", "RTL"):
                await adapter.send_command(vehicle_id, {"action": "RTL"})
            elif command == "GOTO" and waypoints:
                await adapter.send_command(vehicle_id, {"action": "GOTO", "waypoints": waypoints})
            result_detail = "command sent to MAVLink adapter"
        except Exception as _mav_err:
            logger.warning("MAVLink dispatch unavailable: %s", _mav_err)

    if websocket_manager:
        await websocket_manager.broadcast(json.dumps({
            "type": "command_sent",
            "data": {"entity_id": entity_id, "command": command, "ts": _now_iso()}
        }))
    return {"ok": True, "entity_id": entity_id, "command": command, "detail": result_detail}


# ── Next.js reverse proxy (single-service Docker deployment) ─────────────────
# When NEXTJS_INTERNAL_URL is set, catch all unmatched routes and proxy them
# to the Next.js standalone server running on an internal port.
_NEXTJS_URL = os.getenv("NEXTJS_INTERNAL_URL")
if _NEXTJS_URL:
    import httpx
    from fastapi import Request
    from fastapi.responses import Response as _Response

    @app.api_route("/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"])
    async def _proxy_nextjs(path: str, request: Request):
        url = f"{_NEXTJS_URL}/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        skip = {"host", "content-length", "transfer-encoding"}
        headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                resp = await client.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    content=await request.body(),
                )
            drop = {"content-encoding", "content-length", "transfer-encoding"}
            out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in drop}
            return _Response(content=resp.content, status_code=resp.status_code,
                             media_type=resp.headers.get("content-type"), headers=out_headers)
        except Exception as e:
            logger.error(f"Proxy error: {e}")
            return _Response(content=b"Frontend unavailable", status_code=502)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info",
    )
