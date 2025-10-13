import os
import json
import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
import redis.asyncio as aioredis
from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Optional XGBoost
try:
    import xgboost as xgb  # type: ignore
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# Globals
engine: Optional[AsyncEngine] = None
SessionLocal: Optional[sessionmaker] = None
redis_client: Optional[aioredis.Redis] = None

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
    Column("ts", DateTime(timezone=True))
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
    pg_url = _to_asyncpg_url(os.getenv("POSTGRES_URL", "postgresql://summit:summit_password@localhost:5432/summit_os"))
    engine = create_async_engine(pg_url, echo=False, future=True)
    SessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    
    # Redis setup (consume observations from Fusion)
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = await aioredis.from_url(redis_url, decode_responses=True)

    # Optional: load XGBoost model
    if INTELLIGENCE_ENABLE_XGB and XGB_AVAILABLE:
        model_path = RISK_MODEL_PATH or os.path.join(MODEL_REGISTRY, "risk_model.json")
        try:
            globals()['risk_model'] = xgb.Booster()  # type: ignore
            risk_model.load_model(model_path)  # type: ignore
        except Exception:
            globals()['risk_model'] = None
    
    # Start background risk scoring processor
    task = asyncio.create_task(_risk_scoring_processor())
    
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        if redis_client:
            await redis_client.close()
        if engine:
            await engine.dispose()

app = FastAPI(title="Summit Intelligence", version="0.2.0", lifespan=lifespan)

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

@app.get("/advisories", response_model=List[Advisory])
async def list_advisories(risk_level: Optional[str] = None, limit: int = 50):
    """List recent advisories, optionally filtered by risk level."""
    assert SessionLocal is not None
    
    async with SessionLocal() as session:
        if risk_level:
            rows = (await session.execute(
                text("SELECT advisory_id, observation_id, risk_level, message, confidence, ts FROM advisories WHERE risk_level = :rl ORDER BY id DESC LIMIT :lim"),
                {"rl": risk_level, "lim": limit}
            )).all()
        else:
            rows = (await session.execute(
                text("SELECT advisory_id, observation_id, risk_level, message, confidence, ts FROM advisories ORDER BY id DESC LIMIT :lim"),
                {"lim": limit}
            )).all()
        
        return [Advisory(**dict(r._mapping)) for r in rows]

async def _risk_scoring_processor():
    """Background processor that consumes observations and generates risk-scored advisories."""
    assert SessionLocal is not None
    assert redis_client is not None
    
    # Subscribe to observations stream (published by Fusion after persistence)
    last_id = "$"
    
    while True:
        try:
            # Read from observations_stream
            messages = await redis_client.xread({"observations_stream": last_id}, block=1000, count=10)
            
            if not messages:
                continue
            
            for stream_name, stream_messages in messages:
                for msg_id, fields in stream_messages:
                    last_id = msg_id
                    
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
                                    ts=datetime.now(timezone.utc)
                                )
                            )
                            await session.commit()
                        
                        # Optionally publish to MQTT for real-time alerts
                        # await mqtt_client.publish(f"advisories/{data.get('class')}", ...)
                    
                    except Exception as e:
                        print(f"Failed to process observation for risk scoring: {e}")
                        continue
        
        except Exception as e:
            print(f"Risk scoring processor error: {e}")
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
