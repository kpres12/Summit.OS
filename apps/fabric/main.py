"""
Summit.OS Data Fabric Service

Real-time message bus and synchronization layer for Summit.OS.
Handles MQTT, Redis Streams, and gRPC streaming for distributed intelligence.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import structlog
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as redis
import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime, timezone

from .config import Settings
from .mqtt_client import MQTTClient
from .redis_client import RedisClient
from .websocket_manager import WebSocketManager
from .models import TelemetryMessage, AlertMessage, MissionUpdate

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    global mqtt_client, redis_client, websocket_manager
    
    settings = Settings()
    
    # Initialize services
    logger.info("Starting Summit.OS Data Fabric Service")
    
    # Redis connection
    redis_client = RedisClient(settings.redis_url)
    await redis_client.connect()
    logger.info("Connected to Redis")
    
    # MQTT client
    mqtt_client = MQTTClient(
        broker=settings.mqtt_broker,
        port=settings.mqtt_port,
        username=settings.mqtt_username,
        password=settings.mqtt_password
    )
    await mqtt_client.connect()
    logger.info("Connected to MQTT broker")
    
    # WebSocket manager
    websocket_manager = WebSocketManager()
    
    # Subscribe to observations topics
    await mqtt_client.subscribe("observations/#", _handle_observation)
    await mqtt_client.subscribe("detections/#", _handle_observation)  # legacy
    await mqtt_client.subscribe("missions/#", _handle_mission)
    
    # Start background tasks
    asyncio.create_task(telemetry_processor())
    asyncio.create_task(alert_processor())
    
    yield
    
    # Cleanup
    if mqtt_client:
        await mqtt_client.disconnect()
    if redis_client:
        await redis_client.disconnect()
    logger.info("Shutting down Summit.OS Data Fabric Service")

app = FastAPI(
    title="Summit.OS Data Fabric",
    description="Real-time message bus and synchronization layer",
    version="1.0.0",
    lifespan=lifespan
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
    assets: list[str]
    objectives: list[str]

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "data-fabric",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Get service metrics."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not connected")
    
    metrics = await redis_client.get_metrics()
    return metrics

@app.post("/telemetry")
async def publish_telemetry(telemetry: TelemetryData):
    """Publish telemetry data to the fabric."""
    if not mqtt_client or not redis_client:
        raise HTTPException(status_code=503, detail="Services not connected")
    
    try:
        # Publish to MQTT
        topic = f"telemetry/{telemetry.device_id}"
        message = telemetry.model_dump_json()
        await mqtt_client.publish(topic, message)
        
        # Store in Redis Streams
        await redis_client.add_telemetry(telemetry)
        
        # Broadcast to WebSocket clients
        await websocket_manager.broadcast_telemetry(telemetry)
        
        logger.info("Published telemetry", device_id=telemetry.device_id)
        return {"status": "published", "device_id": telemetry.device_id}
        
    except Exception as e:
        logger.error("Failed to publish telemetry", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to publish telemetry")

@app.post("/alerts")
async def publish_alert(alert: AlertData):
    """Publish alert data to the fabric."""
    if not mqtt_client or not redis_client:
        raise HTTPException(status_code=503, detail="Services not connected")
    
    try:
        # Publish to MQTT
        topic = f"alerts/{alert.alert_id}"
        message = alert.model_dump_json()
        await mqtt_client.publish(topic, message)
        
        # Store in Redis Streams
        await redis_client.add_alert(alert)
        
        # Broadcast to WebSocket clients
        await websocket_manager.broadcast_alert(alert)
        
        logger.info("Published alert", alert_id=alert.alert_id, severity=alert.severity)
        return {"status": "published", "alert_id": alert.alert_id}
        
    except Exception as e:
        logger.error("Failed to publish alert", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to publish alert")

@app.post("/missions")
async def publish_mission_update(mission: MissionData):
    """Publish mission update to the fabric."""
    if not mqtt_client or not redis_client:
        raise HTTPException(status_code=503, detail="Services not connected")
    
    try:
        # Publish to MQTT
        topic = f"missions/{mission.mission_id}"
        message = mission.model_dump_json()
        await mqtt_client.publish(topic, message)
        
        # Store in Redis Streams
        await redis_client.add_mission_update(mission)
        
        # Broadcast to WebSocket clients
        await websocket_manager.broadcast_mission_update(mission)
        
        logger.info("Published mission update", mission_id=mission.mission_id)
        return {"status": "published", "mission_id": mission.mission_id}
        
    except Exception as e:
        logger.error("Failed to publish mission update", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to publish mission update")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data streams."""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back for now - could handle commands here
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

# MQTT message handlers
async def _handle_observation(topic: str, data: Dict[str, Any]):
    """Handle incoming observation from MQTT and publish to Redis Stream."""
    try:
        # Derive class from topic if not present
        if "class" not in data:
            if topic.startswith("observations/"):
                data["class"] = topic.split("/", 1)[1]
            elif topic.startswith("detections/"):
                data["class"] = topic.split("/", 1)[1]
        
        # Add to Redis Stream for Fusion to consume
        if redis_client and redis_client.redis:
            stream_data = {
                "topic": topic,
                "payload": json.dumps(data),
                "ts": datetime.now(timezone.utc).isoformat()
            }
            await redis_client.redis.xadd("observations_stream", stream_data)
            logger.info(f"Forwarded observation to stream", topic=topic, cls=data.get("class"))
    except Exception as e:
        logger.error(f"Failed to handle observation: {e}")


async def _handle_mission(topic: str, data: Dict[str, Any]):
    """Handle incoming mission events from MQTT and broadcast via WebSocket and Redis."""
    try:
        # Derive mission_id if not present
        mission_id = data.get("mission_id")
        if not mission_id and topic.startswith("missions/"):
            parts = topic.split("/", 1)
            if len(parts) == 2:
                mission_id = parts[1]
        if not mission_id:
            mission_id = "unknown"
        # Normalize to MissionData
        mission = MissionData(
            mission_id=mission_id,
            timestamp=datetime.now(timezone.utc),
            status=str(data.get("status") or data.get("event") or "UPDATE"),
            assets=data.get("assets") or [],
            objectives=data.get("objectives") or [],
        )
        # Store in Redis Streams for consumers
        if redis_client:
            await redis_client.add_mission_update(mission)
        # Broadcast to UI subscribers
        if websocket_manager:
            await websocket_manager.broadcast_mission_update(mission)
        logger.info("Forwarded mission to ws/redis", mission_id=mission.mission_id, status=mission.status)
    except Exception as e:
        logger.error(f"Failed to handle mission: {e}")

# Background processors
async def telemetry_processor():
    """Background processor for telemetry data."""
    while True:
        try:
            if redis_client:
                # Process telemetry from Redis Streams
                await redis_client.process_telemetry_stream()
            await asyncio.sleep(1)
        except Exception as e:
            logger.error("Telemetry processor error", error=str(e))
            await asyncio.sleep(5)

async def alert_processor():
    """Background processor for alert data."""
    while True:
        try:
            if redis_client:
                # Process alerts from Redis Streams
                await redis_client.process_alert_stream()
            await asyncio.sleep(1)
        except Exception as e:
            logger.error("Alert processor error", error=str(e))
            await asyncio.sleep(5)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
