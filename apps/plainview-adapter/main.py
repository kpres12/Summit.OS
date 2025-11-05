"""
Plainview Intelligence Adapter

Bridges Summit.OS Intelligence service to Plainview's domain modules:
- Consumes advisories from Intelligence service
- Enriches with Plainview domain context
- Forwards to Plainview API via HTTP/WebSocket
- Bidirectional: receives Plainview events and publishes to MQTT
"""

import os
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Config
INTELLIGENCE_URL = os.getenv("INTELLIGENCE_URL", "http://intelligence:8003")
PLAINVIEW_API_URL = os.getenv("PLAINVIEW_API_URL", "http://host.docker.internal:4000")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt://mqtt:1883")

# Globals
redis_client: Optional[aioredis.Redis] = None
http_client: Optional[httpx.AsyncClient] = None
plainview_ws_connections: set[WebSocket] = set()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plainview-adapter")


class Advisory(BaseModel):
    advisory_id: str
    observation_id: Optional[int]
    risk_level: str
    message: str
    confidence: float
    ts: str


class PlainviewInsight(BaseModel):
    """Enriched insight for Plainview domain."""
    type: str  # "flow_anomaly", "valve_health", "pipeline_integrity"
    severity: str  # "low", "medium", "high", "critical"
    title: str
    description: str
    confidence: float
    asset_id: Optional[str]
    recommendations: list[str]
    timestamp: str
    source_advisory_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, http_client
    
    # Redis setup
    redis_client = await aioredis.from_url(REDIS_URL, decode_responses=True)
    logger.info(f"Connected to Redis: {REDIS_URL}")
    
    # HTTP client for Intelligence and Plainview APIs
    http_client = httpx.AsyncClient(timeout=10.0)
    logger.info("HTTP client initialized")
    
    # Start background processors
    advisory_task = asyncio.create_task(_advisory_processor())
    plainview_event_task = asyncio.create_task(_plainview_event_forwarder())
    
    logger.info("Plainview Intelligence Adapter started")
    
    try:
        yield
    finally:
        advisory_task.cancel()
        plainview_event_task.cancel()
        try:
            await advisory_task
        except asyncio.CancelledError:
            pass
        try:
            await plainview_event_task
        except asyncio.CancelledError:
            pass
        
        if redis_client:
            await redis_client.close()
        if http_client:
            await http_client.aclose()
        logger.info("Plainview Intelligence Adapter shutdown complete")


app = FastAPI(
    title="Plainview Intelligence Adapter",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "plainview-adapter",
        "connections": {
            "redis": redis_client is not None,
            "http_client": http_client is not None,
            "plainview_ws": len(plainview_ws_connections)
        }
    }


@app.get("/readyz")
async def readyz():
    try:
        if redis_client:
            await redis_client.ping()
        if http_client:
            resp = await http_client.get(f"{INTELLIGENCE_URL}/health")
            resp.raise_for_status()
        return {"status": "ready"}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"Not ready: {e}")


@app.websocket("/ws/plainview")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for Plainview dashboard to receive real-time insights."""
    await websocket.accept()
    plainview_ws_connections.add(websocket)
    logger.info(f"Plainview WebSocket connected. Total: {len(plainview_ws_connections)}")
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "features": ["intelligence", "advisories", "insights"]
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            logger.info(f"Received from Plainview: {msg}")
            
            # Handle commands from Plainview
            if msg.get("type") == "request_advisory":
                await _send_latest_advisories(websocket)
    
    except WebSocketDisconnect:
        plainview_ws_connections.discard(websocket)
        logger.info(f"Plainview WebSocket disconnected. Remaining: {len(plainview_ws_connections)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        plainview_ws_connections.discard(websocket)


async def _send_latest_advisories(websocket: WebSocket):
    """Fetch and send latest advisories from Intelligence service."""
    try:
        if not http_client:
            return
        
        resp = await http_client.get(f"{INTELLIGENCE_URL}/advisories", params={"limit": 10})
        resp.raise_for_status()
        advisories = resp.json()
        
        for advisory_data in advisories:
            advisory = Advisory(**advisory_data)
            insight = _enrich_advisory(advisory)
            await websocket.send_json({
                "type": "insight",
                "data": insight.model_dump()
            })
    except Exception as e:
        logger.error(f"Failed to fetch advisories: {e}")


async def _advisory_processor():
    """
    Background processor that:
    1. Polls Intelligence service for new advisories
    2. Enriches them for Plainview domain
    3. Forwards to Plainview API and WebSocket connections
    """
    logger.info("Advisory processor started")
    seen_advisory_ids: set[str] = set()
    
    while True:
        try:
            if not http_client:
                await asyncio.sleep(1)
                continue
            
            # Poll Intelligence service
            resp = await http_client.get(f"{INTELLIGENCE_URL}/advisories", params={"limit": 50})
            resp.raise_for_status()
            advisories_data = resp.json()
            
            for advisory_data in advisories_data:
                advisory = Advisory(**advisory_data)
                
                # Skip if already processed
                if advisory.advisory_id in seen_advisory_ids:
                    continue
                
                seen_advisory_ids.add(advisory.advisory_id)
                logger.info(f"Processing advisory: {advisory.advisory_id}")
                
                # Enrich for Plainview domain
                insight = _enrich_advisory(advisory)
                
                # Forward to Plainview API
                await _forward_to_plainview(insight)
                
                # Broadcast to WebSocket connections
                await _broadcast_insight(insight)
            
            # Cleanup old IDs (keep last 1000)
            if len(seen_advisory_ids) > 1000:
                seen_advisory_ids = set(list(seen_advisory_ids)[-1000:])
            
            await asyncio.sleep(5)  # Poll every 5 seconds
        
        except Exception as e:
            logger.error(f"Advisory processor error: {e}")
            await asyncio.sleep(5)


def _enrich_advisory(advisory: Advisory) -> PlainviewInsight:
    """
    Enrich Summit.OS advisory with Plainview domain context.
    Maps risk levels to Plainview modules (FlowIQ, ValveOps, PipelineGuard).
    """
    # Parse advisory message to extract domain context
    msg_lower = advisory.message.lower()
    
    # Determine type and asset based on message content
    if "flow" in msg_lower or "pressure" in msg_lower or "temperature" in msg_lower:
        insight_type = "flow_anomaly"
        asset_id = "flow-system"
        recommendations = [
            "Monitor flow metrics for next 30 minutes",
            "Check pressure sensors for calibration",
            "Review recent maintenance logs"
        ]
    elif "valve" in msg_lower:
        insight_type = "valve_health"
        asset_id = "v-101"  # Extract from message if possible
        recommendations = [
            "Schedule valve inspection within 24 hours",
            "Test actuator response time",
            "Review torque readings"
        ]
    elif "pipeline" in msg_lower or "leak" in msg_lower:
        insight_type = "pipeline_integrity"
        asset_id = "pipeline-a"
        recommendations = [
            "Deploy inspection drone to segment",
            "Increase monitoring frequency",
            "Prepare repair crew standby"
        ]
    else:
        insight_type = "general"
        asset_id = None
        recommendations = ["Review system logs", "Consult operations team"]
    
    # Map risk level to severity
    severity_map = {
        "LOW": "low",
        "MEDIUM": "medium",
        "HIGH": "high",
        "CRITICAL": "critical"
    }
    severity = severity_map.get(advisory.risk_level, "medium")
    
    # Generate title
    title = f"{advisory.risk_level} Risk: {insight_type.replace('_', ' ').title()}"
    
    return PlainviewInsight(
        type=insight_type,
        severity=severity,
        title=title,
        description=advisory.message,
        confidence=advisory.confidence,
        asset_id=asset_id,
        recommendations=recommendations,
        timestamp=advisory.ts,
        source_advisory_id=advisory.advisory_id
    )


async def _forward_to_plainview(insight: PlainviewInsight):
    """Forward enriched insight to Plainview API."""
    try:
        if not http_client:
            return
        
        # Map to Plainview's expected format
        payload = {
            "type": "intelligence.insight",
            "insight_type": insight.type,
            "severity": insight.severity,
            "title": insight.title,
            "description": insight.description,
            "confidence": insight.confidence,
            "asset_id": insight.asset_id,
            "recommendations": insight.recommendations,
            "timestamp": insight.timestamp,
            "source": "summit-os-intelligence"
        }
        
        # Send to Plainview event endpoint (assuming it exists or will be added)
        resp = await http_client.post(
            f"{PLAINVIEW_API_URL}/intelligence/insights",
            json=payload,
            timeout=5.0
        )
        
        if resp.status_code == 200:
            logger.info(f"Forwarded insight to Plainview: {insight.type}")
        else:
            logger.warning(f"Plainview API returned {resp.status_code}")
    
    except httpx.ConnectError:
        logger.warning("Plainview API not reachable (expected if not running)")
    except Exception as e:
        logger.error(f"Failed to forward to Plainview: {e}")


async def _broadcast_insight(insight: PlainviewInsight):
    """Broadcast insight to all connected Plainview WebSocket clients."""
    if not plainview_ws_connections:
        return
    
    message = {
        "type": "insight",
        "data": insight.model_dump()
    }
    
    disconnected = set()
    for ws in plainview_ws_connections:
        try:
            await ws.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send to WebSocket: {e}")
            disconnected.add(ws)
    
    # Clean up disconnected clients
    plainview_ws_connections.difference_update(disconnected)


async def _plainview_event_forwarder():
    """
    Forward Plainview events back to Summit.OS via MQTT/Redis.
    This creates bidirectional integration.
    """
    logger.info("Plainview event forwarder started")
    
    while True:
        try:
            # This is a placeholder - would connect to Plainview's SSE /events endpoint
            # and forward relevant events to Summit.OS MQTT topics
            
            # For now, just log and sleep
            await asyncio.sleep(10)
        
        except Exception as e:
            logger.error(f"Event forwarder error: {e}")
            await asyncio.sleep(5)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
