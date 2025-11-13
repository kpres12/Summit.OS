"""
Summit.OS Sensor Fusion Service (thin-slice)

Consumes smoke detections from MQTT, validates against JSON Schema,
triangulates (stub), persists ignition estimates to Postgres, and exposes APIs.
"""

import os
import json
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from queue import Queue
from typing import Any, Dict, List, Optional

import base64
import io

import redis.asyncio as aioredis
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from jsonschema import Draft202012Validator
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Float, Integer, MetaData, Table, text, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

import paho.mqtt.client as mqtt
import numpy as np
import cv2
from vision_inference import VisionInference
from model_registry import list_models, select_model

# Globals initialized in lifespan
engine: Optional[AsyncEngine] = None
SessionLocal: Optional[sessionmaker] = None
redis_client: Optional[aioredis.Redis] = None
detection_queue: Queue = Queue(maxsize=1000)
smoke_schema: Optional[Dict[str, Any]] = None
validator: Optional[Draft202012Validator] = None

# Optional vision AI
mqtt_client: Optional[mqtt.Client] = None
vision: Optional[VisionInference] = None

# Optional tracking
try:
    from tracking import SimpleTracker  # lightweight local tracker
except Exception:
    SimpleTracker = None  # type: ignore
tracker: Optional["SimpleTracker"] = None  # type: ignore

# Simple triangulation
try:
    from triangulation import bearing_intersection
except Exception:
    bearing_intersection = None  # type: ignore

# Cache last bearings per source
_last_bearings: dict[str, dict] = {}


def _to_asyncpg_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


# Define table using SQLAlchemy Core
metadata = MetaData()
observations = Table(
    "observations",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("class", String(128), nullable=False),
    Column("lat", Float),
    Column("lon", Float),
    Column("confidence", Float, nullable=False),
    Column("ts", DateTime(timezone=True), nullable=False),
    Column("source", String(128)),
    Column("attributes", JSONB),
    Column("org_id", String(128))
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, SessionLocal, redis_client, smoke_schema, validator

    # Allow disabling startup for unit tests
    if os.getenv("FUSION_DISABLE_STARTUP") == "1":
        yield
        return

    # Database setup
    pg_url = _to_asyncpg_url(os.getenv("POSTGRES_URL", "postgresql://summit:summit_password@localhost:5432/summit_os"))
    engine = create_async_engine(pg_url, echo=False, future=True)
    SessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        # Optional Timescale hypertable setup
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
            await conn.execute(text("SELECT create_hypertable('observations','ts', if_not_exists => TRUE)"))
        except Exception:
            pass

    # Load JSON schema for generic Observation
    schema_path = os.getenv("OBSERVATION_SCHEMA_PATH", "/contracts/jsonschemas/observation.schema.json")
    try:
        with open(schema_path, "r") as f:
            smoke_schema = json.load(f)
        validator = Draft202012Validator(smoke_schema)
    except Exception:
        # Fallback minimal schema if contracts not mounted
        smoke_schema = {
            "type": "object",
            "properties": {
                "class": {"type": "string"},
                "ts_iso": {"type": "string"},
                "confidence": {"type": "number"},
                "lat": {"type": "number"},
                "lon": {"type": "number"}
            },
            "required": ["class", "ts_iso", "confidence"],
        }
        validator = Draft202012Validator(smoke_schema)

    # Redis setup (consume from Fabric's streams)
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = await aioredis.from_url(redis_url, decode_responses=True)

    # Ensure consumer group for observations
    try:
        await redis_client.xgroup_create("observations_stream", "fusion", id="$", mkstream=True)
    except Exception as e:
        # BUSYGROUP means it exists; ignore others for now
        if "BUSYGROUP" not in str(e):
            pass

    # Optional: start vision AI if enabled
    enable_vision = os.getenv("FUSION_ENABLE_VISION_AI", "false").lower() == "true"
    tasks: list[asyncio.Task] = []
    if enable_vision:
        model_path = os.getenv("FUSION_MODEL_PATH") or None
        conf = float(os.getenv("FUSION_CONF_THRESHOLD", "0.6"))
        # Lazy create inference
        globals()['vision'] = VisionInference(model_path=model_path, conf_threshold=conf)
        # Optional tracking
        enable_tracking = os.getenv("FUSION_ENABLE_TRACKING", "false").lower() == "true"
        if enable_tracking and SimpleTracker is not None:
            globals()['tracker'] = SimpleTracker(iou_threshold=float(os.getenv("FUSION_TRACK_IOU", "0.3")),
                                                max_age_s=float(os.getenv("FUSION_TRACK_MAX_AGE_S", "1.0")))
        # MQTT client to receive images
        broker = os.getenv("MQTT_BROKER", "localhost")
        port = int(os.getenv("MQTT_PORT", "1883"))
        globals()['mqtt_client'] = mqtt.Client()
        mqtt_client.connect(broker, port, 60)
        mqtt_client.on_message = _on_mqtt_image
        mqtt_client.subscribe("images/#", qos=0)
        mqtt_client.loop_start()

    # Start async consumer from Redis Stream
    task = asyncio.create_task(_redis_stream_consumer())
    tasks.append(task)

    try:
        yield
    finally:
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        if redis_client:
            await redis_client.close()
        if engine:
            await engine.dispose()


app = FastAPI(
    title="Summit.OS Fusion (Thin Slice)",
    version="0.3.0",
    lifespan=lifespan,
)

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Observation(BaseModel):
    id: int
    cls: str
    lat: float | None = None
    lon: float | None = None
    confidence: float
    ts: datetime
    source: str | None = None
    attributes: dict | None = None


@app.get("/health")
async def health():
    return {"status": "ok", "service": "fusion"}

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


from fastapi import Request as _Req

async def _get_org_id(req: _Req) -> str | None:
    return req.headers.get("X-Org-ID") or req.headers.get("x-org-id")

@app.get("/observations", response_model=List[Observation])
async def list_observations(cls: Optional[str] = None, limit: int = 50, org_id: str | None = Depends(_get_org_id)):
    assert SessionLocal is not None
    async with SessionLocal() as session:
        base = "SELECT id, class as cls, lat, lon, confidence, ts, source, attributes FROM observations"
        where = []
        params: dict = {"lim": limit}
        if cls:
            where.append("class = :cls")
            params["cls"] = cls
        if org_id:
            where.append("org_id = :org_id")
            params["org_id"] = org_id
        sql = base + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id DESC LIMIT :lim"
        rows = (await session.execute(text(sql), params)).all()
        return [Observation(**dict(r._mapping)) for r in rows]


# Model registry endpoints
@app.get("/models")
async def get_models():
    root = os.getenv("MODEL_REGISTRY", "/models")
    return {"models": list_models(root)}

@app.post("/models/select")
async def set_model(payload: dict):
    path = payload.get("path")
    if not path:
        return {"status": "error", "message": "path required"}
    try:
        select_model(path)
        return {"status": "ok", "path": path}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))

# MQTT image handler and vision inference path
def _on_mqtt_image(client, userdata, msg):
    try:
        payload = msg.payload
        data = None
        try:
            # JSON with base64 image: {"image_b64": "...", "device_id":"...", "ts_iso":"..."}
            data = json.loads(payload.decode("utf-8"))
        except Exception:
            data = None
        if data and "image_b64" in data:
            img_bytes = base64.b64decode(data["image_b64"])  # type: ignore
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            _process_vision_image(img, data)
    except Exception:
        pass


def _process_vision_image(img: np.ndarray, meta: Dict[str, Any]):
    try:
        if not vision:
            return
        detections = vision.detect(img)
        if not detections:
            return
        # Optional tracking
        if tracker is not None:
            ts = None
            try:
                ts_iso = meta.get("ts_iso")
                if ts_iso:
                    from datetime import datetime
                    ts = datetime.fromisoformat(str(ts_iso).replace("Z", "+00:00")).timestamp()
            except Exception:
                ts = None
            detections = tracker.update(detections, timestamp=ts)
        # Persist detections as observations
        loop = asyncio.get_event_loop()
        loop.create_task(_persist_observations_from_detections(detections, meta))
    except Exception:
        pass


async def _persist_observations_from_detections(detections: List[Dict[str, Any]], meta: Dict[str, Any]):
    assert SessionLocal is not None
    assert redis_client is not None
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc)
    async with SessionLocal() as session:
        for det in detections:
            cls = det.get("class", "object")
            conf = float(det.get("confidence", 0.0))
            lat = float(meta.get("lat")) if meta.get("lat") is not None else None
            lon = float(meta.get("lon")) if meta.get("lon") is not None else None
            attrs = det
            await session.execute(
                observations.insert().values(
                    **{"class": cls},
                    lat=lat,
                    lon=lon,
                    confidence=conf,
                    ts=ts,
                    source=str(meta.get("device_id") or "vision"),
                    attributes=attrs,
                    org_id=None,
                )
            )
        await session.commit()
    # Also push to Redis Stream for downstream consumers (Intelligence)
    for det in detections:
        try:
            record = {
                "topic": "vision/detection",
                "payload": json.dumps({**det, "ts_iso": ts.isoformat()}),
                "ts": ts.isoformat(),
            }
            await redis_client.xadd("observations_stream", record)
        except Exception:
            pass


# Simple debounce registry to avoid spamming Sentinel with repeated nearby detections
_last_simulation_ts: float | None = None
_last_simulation_point: tuple[float, float] | None = None

async def _redis_stream_consumer():
    """Consume observations from Fabric's Redis Stream using consumer groups."""
    assert SessionLocal is not None
    assert redis_client is not None

    import socket
    consumer_name = f"fusion-{socket.gethostname()}"

    while True:
        try:
            # Read from consumer group; '>' means new messages for this consumer
            messages = await redis_client.xreadgroup("fusion", consumer_name, {"observations_stream": ">"}, count=10, block=1000)

            if not messages:
                continue

            for stream_name, stream_messages in messages:
                for msg_id, fields in stream_messages:
                    
                    try:
                        # Parse payload from Fabric
                        payload = fields.get("payload")
                        if not payload:
                            continue
                        
                        data = json.loads(payload)
                        
                        # Validate
                        assert validator is not None
                        validator.validate(data)
                        
                        # Extract fields
                        lat = float(data.get("lat")) if data.get("lat") is not None else None
                        lon = float(data.get("lon")) if data.get("lon") is not None else None
                        conf = float(data.get("confidence", 0.0))
                        ts_str = data.get("ts_iso")
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else datetime.now(timezone.utc)
                        except Exception:
                            ts = datetime.now(timezone.utc)
                        
                        source = data.get("source")
                        attributes = {k: v for k, v in data.items() if k not in {"class", "lat", "lon", "confidence", "ts_iso", "source"}}

                        # Optional triangulation: if bearing present, combine with previous from different source
                        try:
                            if bearing_intersection is not None and attributes and lat is not None and lon is not None:
                                bdeg = attributes.get("bearing") or attributes.get("bearing_deg")
                                if bdeg is not None:
                                    src = str(source or "")
                                    now_ts = datetime.now(timezone.utc).timestamp()
                                    prev = None
                                    # find previous from different source within 120s
                                    for s, rec in list(_last_bearings.items()):
                                        if s != src and now_ts - rec.get("ts", 0) < 120:
                                            prev = rec
                                            break
                                    _last_bearings[src] = {"lat": lat, "lon": lon, "bearing": float(bdeg), "ts": now_ts}
                                    if prev:
                                        inter = bearing_intersection(lat, lon, float(bdeg), prev["lat"], prev["lon"], float(prev["bearing"]))
                                        if inter:
                                            ilat, ilon = inter
                                            # push as observation ignition estimate
                                            async with SessionLocal() as session2:
                                                await session2.execute(
                                                    observations.insert().values(
                                                        **{"class": "ignition_estimate", "lat": ilat, "lon": ilon, "confidence": max(conf, 0.8), "ts": ts, "source": "triangulation", "attributes": {"from": [src, "prev" ]}}
                                                    )
                                                )
                                                await session2.commit()
                                            # Also notify downstream via stream
                                            try:
                                                await redis_client.xadd("observations_stream", {"topic": "fusion/triangulation", "payload": json.dumps({"class":"ignition_estimate","lat": ilat, "lon": ilon, "confidence": max(conf,0.8), "ts_iso": ts.isoformat(), "source":"triangulation"}), "ts": ts.isoformat()})
                                            except Exception:
                                                pass
                        except Exception:
                            pass
                        
                        # Check for duplicates
                        try:
                            import hashlib
                            sig_src = {
                                "class": data.get("class"),
                                "lat": lat,
                                "lon": lon,
                                "confidence": conf,
                                "ts": ts.isoformat(),
                                "source": source,
                            }
                            sig = hashlib.sha1(json.dumps(sig_src, sort_keys=True).encode("utf-8")).hexdigest()
                            key = f"obs_seen:{sig}"
                            # setex 600s; if already exists, skip
                            if await redis_client.set(key, "1", ex=600, nx=True) is None:
                                # duplicate
                                try:
                                    await redis_client.xack("observations_stream", "fusion", msg_id)
                                except Exception:
                                    pass
                                continue
                        except Exception:
                            pass

                        # Persist
                        async with SessionLocal() as session:
                            await session.execute(
                                observations.insert().values(
                                    **{"class": data["class"], "lat": lat, "lon": lon, "confidence": conf, "ts": ts, "source": source, "attributes": attributes, "org_id": None}
                                )
                            )
                            await session.commit()
                        # Ack message
                        try:
                            await redis_client.xack("observations_stream", "fusion", msg_id)
                        except Exception:
                            pass

                        # Trigger Sentinel simulation on confirmed smoke/ignition with location
                        try:
                            cls_lower = str(data.get("class", "")).lower()
                            is_smoke = cls_lower in {"smoke", "fire.smoke", "fire"}
                            conf_thr = float(os.getenv("SENTINEL_TRIGGER_CONF_THRESHOLD", "0.7"))
                            debounce_s = float(os.getenv("SENTINEL_TRIGGER_DEBOUNCE_S", "60"))
                            radius_m = float(os.getenv("SENTINEL_TRIGGER_DEBOUNCE_RADIUS_M", "200"))

                            if is_smoke and lat is not None and lon is not None and conf >= conf_thr:
                                # Debounce: skip if last call was very recent and nearby
                                from time import time as _now
                                global _last_simulation_ts, _last_simulation_point
                                now_s = _now()
                                should_call = True
                                if _last_simulation_ts is not None and _last_simulation_point is not None:
                                    dt = now_s - _last_simulation_ts
                                    if dt < debounce_s:
                                        # compute haversine distance
                                        from math import radians, sin, cos, sqrt, atan2
                                        R = 6371000.0
                                        lat1, lon1 = map(radians, [lat, lon])
                                        lat2, lon2 = map(radians, _last_simulation_point)
                                        dlat = lat2 - lat1
                                        dlon = lon2 - lon1
                                        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
                                        dist = 2*R*atan2(sqrt(a), sqrt(1-a))
                                        if dist < radius_m:
                                            should_call = False
                                if should_call:
                                    _last_simulation_ts = now_s
                                    _last_simulation_point = (lat, lon)
                                    # Build basic conditions from attributes if present
                                    cond = {
                                        "temperature_c": attributes.get("temperature") if isinstance(attributes, dict) else None,
                                        "relative_humidity": attributes.get("humidity") if isinstance(attributes, dict) else None,
                                        "wind_speed_mps": attributes.get("wind_speed") if isinstance(attributes, dict) else None,
                                        "wind_direction_deg": attributes.get("wind_direction") if isinstance(attributes, dict) else None,
                                        "elevation_m": attributes.get("elevation") if isinstance(attributes, dict) else None,
                                    }
                                    from sentinel_client import simulate_spread as _sentinel_sim
                                    asyncio.create_task(_sentinel_sim(lat, lon, cond))
                        except Exception as _e:
                            # Non-fatal; continue stream processing
                            pass
                    
                    except Exception as e:
                        print(f"Failed to process observation: {e}")
                        # DLQ: record bad payload for later inspection and ack to avoid poison pill
                        try:
                            if redis_client is not None:
                                bad_payload = payload if 'payload' in locals() else None
                                await redis_client.xadd(
                                    "observations_dlq",
                                    {
                                        "error": str(e),
                                        "payload": bad_payload or "",
                                        "ts": datetime.now(timezone.utc).isoformat(),
                                    },
                                )
                                try:
                                    await redis_client.xack("observations_stream", "fusion", msg_id)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        continue
        
        except Exception as e:
            print(f"Redis stream consumer error: {e}")
            try:
                if redis_client is not None:
                    await redis_client.xadd(
                        "observations_dlq",
                        {
                            "error": str(e),
                            "payload": "<stream_error>",
                            "ts": datetime.now(timezone.utc).isoformat(),
                        },
                    )
            except Exception:
                pass
            await asyncio.sleep(1)
