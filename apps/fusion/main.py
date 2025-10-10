"""
Summit.OS Sensor Fusion Service

Normalizes and fuses multi-modal data (video, weather, IR, lightning, soil) 
into a unified world model for situational awareness.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import numpy as np
import structlog
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json
from datetime import datetime, timezone
import cv2
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, Polygon
import geopandas as gpd

from .config import Settings
from .fusion_engine import FusionEngine
from .world_model import WorldModel
from .detection_service import DetectionService
from .triangulation import TriangulationService

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
fusion_engine: Optional[FusionEngine] = None
world_model: Optional[WorldModel] = None
detection_service: Optional[DetectionService] = None
triangulation_service: Optional[TriangulationService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    global fusion_engine, world_model, detection_service, triangulation_service
    
    settings = Settings()
    
    # Initialize services
    logger.info("Starting Summit.OS Sensor Fusion Service")
    
    # World model
    world_model = WorldModel(settings)
    await world_model.initialize()
    logger.info("World model initialized")
    
    # Detection service
    detection_service = DetectionService(settings)
    await detection_service.initialize()
    logger.info("Detection service initialized")
    
    # Triangulation service
    triangulation_service = TriangulationService(settings)
    await triangulation_service.initialize()
    logger.info("Triangulation service initialized")
    
    # Fusion engine
    fusion_engine = FusionEngine(
        world_model=world_model,
        detection_service=detection_service,
        triangulation_service=triangulation_service,
        settings=settings
    )
    await fusion_engine.initialize()
    logger.info("Fusion engine initialized")
    
    # Start background tasks
    asyncio.create_task(fusion_processor())
    asyncio.create_task(world_model_updater())
    
    yield
    
    # Cleanup
    if fusion_engine:
        await fusion_engine.cleanup()
    if world_model:
        await world_model.cleanup()
    if detection_service:
        await detection_service.cleanup()
    if triangulation_service:
        await triangulation_service.cleanup()
    logger.info("Shutting down Summit.OS Sensor Fusion Service")

app = FastAPI(
    title="Summit.OS Sensor Fusion",
    description="Multi-modal sensor data fusion and world model generation",
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

# Pydantic models
class SensorData(BaseModel):
    device_id: str
    timestamp: datetime
    sensor_type: str  # camera, lidar, thermal, weather, etc.
    location: Dict[str, float]  # lat, lon, alt
    data: Dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)

class DetectionResult(BaseModel):
    detection_id: str
    timestamp: datetime
    location: Dict[str, float]
    object_type: str  # fire, smoke, person, vehicle, etc.
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FusedData(BaseModel):
    fusion_id: str
    timestamp: datetime
    location: Dict[str, float]
    data_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WorldModelState(BaseModel):
    timestamp: datetime
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    confidence: float = Field(ge=0.0, le=1.0)

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "sensor-fusion",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/sensor-data")
async def process_sensor_data(sensor_data: SensorData):
    """Process incoming sensor data."""
    if not fusion_engine:
        raise HTTPException(status_code=503, detail="Fusion engine not initialized")
    
    try:
        # Process sensor data through fusion engine
        result = await fusion_engine.process_sensor_data(sensor_data)
        
        logger.info("Processed sensor data", 
                   device_id=sensor_data.device_id, 
                   sensor_type=sensor_data.sensor_type)
        
        return {"status": "processed", "fusion_id": result.get("fusion_id")}
        
    except Exception as e:
        logger.error("Failed to process sensor data", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to process sensor data")

@app.post("/detections")
async def process_detection(detection: DetectionResult):
    """Process detection results."""
    if not fusion_engine:
        raise HTTPException(status_code=503, detail="Fusion engine not initialized")
    
    try:
        # Process detection through fusion engine
        result = await fusion_engine.process_detection(detection)
        
        logger.info("Processed detection", 
                   detection_id=detection.detection_id, 
                   object_type=detection.object_type)
        
        return {"status": "processed", "fusion_id": result.get("fusion_id")}
        
    except Exception as e:
        logger.error("Failed to process detection", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to process detection")

@app.get("/world-model")
async def get_world_model():
    """Get current world model state."""
    if not world_model:
        raise HTTPException(status_code=503, detail="World model not initialized")
    
    try:
        state = await world_model.get_current_state()
        return state
        
    except Exception as e:
        logger.error("Failed to get world model", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get world model")

@app.get("/detections")
async def get_detections(
    object_type: Optional[str] = None,
    confidence_threshold: float = 0.5,
    time_range: Optional[int] = None
):
    """Get recent detections."""
    if not detection_service:
        raise HTTPException(status_code=503, detail="Detection service not initialized")
    
    try:
        detections = await detection_service.get_detections(
            object_type=object_type,
            confidence_threshold=confidence_threshold,
            time_range=time_range
        )
        return {"detections": detections}
        
    except Exception as e:
        logger.error("Failed to get detections", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get detections")

@app.post("/triangulate")
async def triangulate_sources(sources: List[Dict[str, Any]]):
    """Triangulate position from multiple sources."""
    if not triangulation_service:
        raise HTTPException(status_code=503, detail="Triangulation service not initialized")
    
    try:
        result = await triangulation_service.triangulate(sources)
        return result
        
    except Exception as e:
        logger.error("Failed to triangulate", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to triangulate")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time fusion updates."""
    await websocket.accept()
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back for now - could handle commands here
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        pass

# Background processors
async def fusion_processor():
    """Background processor for sensor fusion."""
    while True:
        try:
            if fusion_engine:
                await fusion_engine.process_pending_data()
            await asyncio.sleep(0.1)  # High frequency for real-time processing
        except Exception as e:
            logger.error("Fusion processor error", error=str(e))
            await asyncio.sleep(1)

async def world_model_updater():
    """Background processor for world model updates."""
    while True:
        try:
            if world_model:
                await world_model.update_entities()
                await world_model.update_relationships()
            await asyncio.sleep(1)  # Update every second
        except Exception as e:
            logger.error("World model updater error", error=str(e))
            await asyncio.sleep(5)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
