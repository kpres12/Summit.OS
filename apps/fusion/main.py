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
from fastapi import FastAPI
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
from .vision_inference import VisionInference

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
    from .tracking import SimpleTracker  # lightweight local tracker
except Exception:
    SimpleTracker = None  # type: ignore
tracker: Optional["SimpleTracker"] = None  # type: ignore


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
    Column("attributes", JSONB)
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
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.get("/observations", response_model=List[Observation])
async def list_observations(cls: Optional[str] = None, limit: int = 50):
    assert SessionLocal is not None
    async with SessionLocal() as session:
        if cls:
            rows = (await session.execute(
                text("SELECT id, class as cls, lat, lon, confidence, ts, source, attributes FROM observations WHERE class = :cls ORDER BY id DESC LIMIT :lim"),
                {"cls": cls, "lim": limit}
            )).all()
        else:
            rows = (await session.execute(
                text("SELECT id, class as cls, lat, lon, confidence, ts, source, attributes FROM observations ORDER BY id DESC LIMIT :lim"),
                {"lim": limit}
            )).all()
        return [Observation(**dict(r._mapping)) for r in rows]


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
                    class=cls,
                    lat=lat,
                    lon=lon,
                    confidence=conf,
                    ts=ts,
                    source=str(meta.get("device_id") or "vision"),
                    attributes=attrs,
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


async def _redis_stream_consumer():
    """Consume observations from Fabric's Redis Stream."""
    assert SessionLocal is not None
    assert redis_client is not None
    
    last_id = "$"  # Start from latest
    
    while True:
        try:
            # Block for 1 second waiting for new messages
            messages = await redis_client.xread({"observations_stream": last_id}, block=1000, count=10)
            
            if not messages:
                continue
            
            for stream_name, stream_messages in messages:
                for msg_id, fields in stream_messages:
                    last_id = msg_id
                    
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
                        
                        # Persist
                        async with SessionLocal() as session:
                            await session.execute(
                                observations.insert().values(
                                    **{"class": data["class"], "lat": lat, "lon": lon, "confidence": conf, "ts": ts, "source": source, "attributes": attributes}
                                )
                            )
                            await session.commit()
                    
                    except Exception as e:
                        print(f"Failed to process observation: {e}")
                        continue
        
        except Exception as e:
            print(f"Redis stream consumer error: {e}")
            await asyncio.sleep(1)
