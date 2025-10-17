"""
Summit.OS Data Fabric Service

Real-time message bus and synchronization layer for Summit.OS.
Handles MQTT, Redis Streams, and gRPC streaming for distributed intelligence.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

import structlog
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json
from datetime import datetime, timezone, timedelta

from .config import Settings
from .mqtt_client import MQTTClient
from .redis_client import RedisClient
from .websocket_manager import WebSocketManager
from .models import TelemetryMessage, AlertMessage, MissionUpdate, Location, SeverityLevel, MissionStatus

# SQLAlchemy (async) for registry persistence
from sqlalchemy import (
    MetaData, Table, Column, Integer, String, DateTime, Boolean, JSON, text
)
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from jose import jwt
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig

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
        structlog.processors.JSONRenderer()
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

# In-memory world state cache (thin-slice)
world_state: Dict[str, Any] = {
    "devices": {},  # device_id -> {lat, lon, alt, ts_iso, status, sensors}
    "alerts": [],   # recent alerts (most recent first)
}
MAX_ALERTS = int(os.getenv("FABRIC_MAX_ALERTS", "200"))

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

# Settings
HEARTBEAT_STALE_SECS = 120  # 2 minutes
HEARTBEAT_OFFLINE_SECS = 600  # 10 minutes
FABRIC_JWT_SECRET = os.getenv("FABRIC_JWT_SECRET", "dev_secret")
FABRIC_TEST_MODE = os.getenv("FABRIC_TEST_MODE", "false").lower() == "true"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    global mqtt_client, redis_client, websocket_manager, engine, SessionLocal

    settings = Settings()

    logger.info("Starting Summit.OS Data Fabric Service")

    # DB connection
    if FABRIC_TEST_MODE:
        pg_url = "sqlite+aiosqlite://"
    else:
        pg_url = os.getenv("POSTGRES_URL", "postgresql+asyncpg://summit:summit_password@localhost:5432/summit_os")
        if pg_url.startswith("postgresql://"):
            pg_url = pg_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(pg_url, echo=False, future=True)
    SessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    if FABRIC_TEST_MODE:
        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
    else:
        _run_migrations()

    # WebSocket manager
    websocket_manager = WebSocketManager()

    if not FABRIC_TEST_MODE:
        # Redis connection
        redis_client = RedisClient(settings.redis_url)
        await redis_client.connect()
        logger.info("Connected to Redis")

        # MQTT client
        mqtt_client = MQTTClient(
            broker=settings.mqtt_broker,
            port=settings.mqtt_port,
            username=settings.mqtt_username,
            password=settings.mqtt_password,
        )
        await mqtt_client.connect()
        logger.info("Connected to MQTT broker")

        # Subscribe to topics
        await mqtt_client.subscribe("observations/#", _handle_observation)
        await mqtt_client.subscribe("detections/#", _handle_observation)  # legacy
        await mqtt_client.subscribe("missions/#", _handle_mission)
        await mqtt_client.subscribe("health/+/heartbeat", _handle_heartbeat)

        # Start background tasks
        asyncio.create_task(telemetry_processor())
        asyncio.create_task(alert_processor())
        asyncio.create_task(heartbeat_watcher())

    yield

    # Cleanup
    if mqtt_client:
        await mqtt_client.disconnect()
    if redis_client:
        await redis_client.disconnect()
    if engine:
        await engine.dispose()
    logger.info("Shutting down Summit.OS Data Fabric Service")

app = FastAPI(
    title="Summit.OS Data Fabric",
    description="Real-time message bus and synchronization layer",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
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

# API Endpoints
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "fabric"}

# WebSocket endpoint for real-time multiplexed events
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global websocket_manager
    if websocket_manager is None:
        websocket_manager = WebSocketManager()
    await websocket_manager.connect(websocket)
    try:
        # Keep the connection alive; ignore incoming messages for now
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

# Simple COP/world-state endpoint (thin-slice)
@app.get("/api/v1/worldstate")
async def get_world_state():
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

@app.post("/telemetry")
async def publish_telemetry(telemetry: "TelemetryData"):
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
        except Exception:
            pass
        if websocket_manager:
            await websocket_manager.broadcast(json.dumps({"type": "telemetry", "data": tm.model_dump()}))
        return {"status": "published", "device_id": telemetry.device_id}
    except Exception as e:
        logger.error("Failed to publish telemetry", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to publish telemetry")

@app.post("/alerts")
async def publish_alert(alert: "AlertData"):
    if not mqtt_client or not redis_client:
        raise HTTPException(status_code=503, detail="Services not connected")
    try:
        topic = f"alerts/{alert.alert_id}"
        await mqtt_client.publish(topic, alert.model_dump_json())
        # Convert to internal model for Redis
        aloc = Location(latitude=float(alert.location.get("lat")), longitude=float(alert.location.get("lon")))
        sev = SeverityLevel(alert.severity.lower()) if isinstance(alert.severity, str) else alert.severity
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
            world_state["alerts"].insert(0, {
                "alert_id": alert.alert_id,
                "severity": str(sev.value if hasattr(sev, "value") else sev),
                "lat": float(aloc.latitude),
                "lon": float(aloc.longitude),
                "description": alert.description,
                "source": alert.source,
                "ts_iso": alert.timestamp.isoformat(),
            })
            del world_state["alerts"][MAX_ALERTS:]
        except Exception:
            pass
        if websocket_manager:
            await websocket_manager.broadcast(json.dumps({"type": "alert", "data": am.model_dump()}))
        return {"status": "published", "alert_id": alert.alert_id}
    except Exception as e:
        logger.error("Failed to publish alert", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to publish alert")

@app.post("/missions")
async def publish_mission_update(mission: "MissionData"):
    if not mqtt_client or not redis_client:
        raise HTTPException(status_code=503, detail="Services not connected")
    try:
        topic = f"missions/{mission.mission_id}"
        await mqtt_client.publish(topic, mission.model_dump_json())
        # Convert to internal model for Redis
        mstatus = mission.status.lower() if isinstance(mission.status, str) else str(mission.status)
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
            await websocket_manager.broadcast(json.dumps({"type": "mission", "data": mu.model_dump()}))
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
            nodes.update().where(nodes.c.id == node_id).values(
                retired=True,
                status="RETIRED",
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()
    return {"status": "retired", "id": node_id}

# Helpers
async def _upsert_node(session: AsyncSession, req: "NodeRegisterRequest", now: datetime):
    existing = await session.execute(text("SELECT id FROM nodes WHERE id = :id"), {"id": req.id})
    if existing.first():
        await session.execute(
            nodes.update().where(nodes.c.id == req.id).values(
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
        res = await session.execute(text("SELECT * FROM nodes WHERE id = :id"), {"id": node_id})
        row = res.mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="Node not found")
        return {**row}

@app.get("/api/v1/coverage")
async def list_coverages():
    """List tower coverages; UI can union polygons client-side until server union is available."""
    assert SessionLocal is not None
    async with SessionLocal() as session:
        res = await session.execute(text("SELECT node_id, viewshed_geojson, version, updated_at FROM coverages ORDER BY updated_at DESC"))
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
        res = await session.execute(text("SELECT viewshed_geojson FROM coverages WHERE viewshed_geojson IS NOT NULL"))
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
async def _handle_observation(topic: str, data: Dict[str, Any]):
    try:
        # Forward to Redis Stream for consumers
        if redis_client and redis_client.redis:
            stream_data = {
                "topic": topic,
                "payload": json.dumps(data),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            await redis_client.redis.xadd("observations_stream", stream_data)
            logger.info("Forwarded observation to stream", topic=topic)
    except Exception as e:
        logger.error(f"Failed to handle observation: {e}")

async def _handle_mission(topic: str, data: Dict[str, Any]):
    try:
        # Broadcast to UI subscribers
        if websocket_manager:
            await websocket_manager.broadcast(json.dumps({"type": "mission_event", "topic": topic, "data": data}))
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
                nodes.update().where(nodes.c.id == node_id).values(
                    status="ONLINE",
                    last_seen=now,
                    updated_at=now,
                )
            )
            await session.commit()
        if websocket_manager:
            await websocket_manager.broadcast(json.dumps({"type": "heartbeat", "node_id": node_id, "ts": data.get("ts")}))
    except Exception as e:
        logger.error(f"Failed to handle heartbeat: {e}")

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
                    {"now": now, "stale_cutoff": now - timedelta(seconds=HEARTBEAT_STALE_SECS)},
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
                    {"now": now, "offline_cutoff": now - timedelta(seconds=HEARTBEAT_OFFLINE_SECS)},
                )
                await session.commit()
            await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"Heartbeat watcher error: {e}")
            await asyncio.sleep(10)

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
            await websocket_manager.broadcast(json.dumps({"type": "coverage_updated", "node_id": node_id}))
    except Exception as e:
        logger.error(f"Viewshed stub failed: {e}")

# JWT helper and token refresh
def _issue_token(node_id: str, topics: Dict[str, List[str]], policy: Dict[str, Any], ttl_seconds: int = 600) -> str:
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
        ini_path = os.path.join(os.path.dirname(__file__), 'alembic.ini')
        cfg = AlembicConfig(ini_path)
        # Ensure env picks up the DB URL
        if not os.getenv('POSTGRES_URL'):
            os.environ['POSTGRES_URL'] = 'postgresql://summit:summit_password@localhost:5432/summit_os'
        alembic_command.upgrade(cfg, 'head')
        logger.info("Alembic migrations applied")
    except Exception as e:
        logger.error(f"Failed to run migrations: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info",
    )
