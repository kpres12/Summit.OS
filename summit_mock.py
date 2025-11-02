#!/usr/bin/env python3
"""
Summit.OS Mock Server - Single Integration Standard (v1.1)

One-file FastAPI mock + MQTT feeder that implements the exact contract
for Summit.OS ↔ Sentinel integration.

Usage:
    python summit_mock.py

Endpoints:
    - http://localhost:8000/api/v1/system/health
    - http://localhost:8000/docs (OpenAPI docs)
    - ws://localhost:1883 (MQTT WebSocket)

Environment Variables:
    HTTP_PORT=8000
    SUMMIT_API_KEY=dev_key_placeholder
    MQTT_WS_URL=ws://localhost:1883
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import random
import math

from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import paho.mqtt.client as mqtt
import uvicorn


# ============================================================================
# PYDANTIC MODELS (Canonical Schemas)
# ============================================================================

class Location(BaseModel):
    lat: float
    lon: float

class Alert(BaseModel):
    id: str
    ts: float
    severity: str = Field(..., pattern="^(LOW|MED|HIGH|CRITICAL)$")
    title: str
    message: str
    location: Location
    context: Dict[str, Any] = {}
    acknowledged: bool = False

class Task(BaseModel):
    id: str
    asset: str
    kind: str = Field(..., pattern="^(PATROL|SURVEY_SMOKE|BUILD_LINE|SUPPRESS|RECON)$")
    state: str = Field(..., pattern="^(QUEUED|ENROUTE|ACTIVE|PAUSED|DONE|FAILED)$")
    eta_min: int
    params: Dict[str, Any] = {}

class TaskRequest(BaseModel):
    kind: str = Field(..., pattern="^(PATROL|SURVEY_SMOKE|BUILD_LINE|SUPPRESS|RECON)$")
    target: Location
    params: Dict[str, Any] = {}

class Telemetry(BaseModel):
    device_id: str
    ts: float
    lat: float
    lon: float
    batt: float
    rssi: int
    speed: float
    sensors: Dict[str, Any] = {}

class ScenarioRequest(BaseModel):
    aoi: Dict[str, Any]  # GeoJSON Polygon
    wind_dir_deg: int
    wind_mps: float
    rh_pct: int
    fuel_model: str
    lines: List[Dict[str, Any]] = []

class HealthResponse(BaseModel):
    status: str
    uptime: float

class AckResponse(BaseModel):
    ok: bool
    id: str


# ============================================================================
# MOCK DATA STORE
# ============================================================================

class MockDataStore:
    def __init__(self):
        self.start_time = time.time()
        self.alerts: List[Alert] = []
        self.tasks: List[Task] = []
        self.telemetry: Dict[str, Telemetry] = {}
        self._generate_initial_data()
    
    def _generate_initial_data(self):
        """Generate initial mock data"""
        # Generate some alerts
        for i in range(3):
            alert = Alert(
                id=f"A-{i+1:03d}",
                ts=time.time() - random.uniform(0, 3600),
                severity=random.choice(["LOW", "MED", "HIGH", "CRITICAL"]),
                title=f"Alert {i+1}",
                message=f"Mock alert message {i+1}",
                location=Location(
                    lat=34.123 + random.uniform(-0.01, 0.01),
                    lon=-117.456 + random.uniform(-0.01, 0.01)
                ),
                context={
                    "risk": random.uniform(0.1, 0.9),
                    "slope_deg": random.randint(5, 45),
                    "fuel_model": random.choice(["SH1", "SH2", "SH3", "SH4", "SH5"])
                }
            )
            self.alerts.append(alert)
        
        # Generate some tasks
        for i in range(2):
            task = Task(
                id=f"T-{i+1:02d}",
                asset=f"UGV-{chr(65+i)}",  # UGV-A, UGV-B
                kind=random.choice(["PATROL", "SURVEY_SMOKE", "BUILD_LINE"]),
                state=random.choice(["QUEUED", "ENROUTE", "ACTIVE"]),
                eta_min=random.randint(5, 30)
            )
            self.tasks.append(task)
        
        # Generate some telemetry
        devices = ["UGV-Alpha", "UGV-Beta", "Drone-001", "Drone-002"]
        for device in devices:
            telemetry = Telemetry(
                device_id=device,
                ts=time.time(),
                lat=34.123 + random.uniform(-0.01, 0.01),
                lon=-117.456 + random.uniform(-0.01, 0.01),
                batt=random.uniform(20, 100),
                rssi=random.randint(-80, -40),
                speed=random.uniform(0, 15),
                sensors={
                    "temp_c": random.uniform(20, 40),
                    "wind_mps": random.uniform(0, 20),
                    "smoke_prob": random.uniform(0, 1)
                }
            )
            self.telemetry[device] = telemetry
    
    def get_uptime(self) -> float:
        return time.time() - self.start_time
    
    def get_alerts_since(self, since_ts: Optional[float] = None) -> List[Alert]:
        if since_ts is None:
            return self.alerts
        return [alert for alert in self.alerts if alert.ts >= since_ts]
    
    def ack_alert(self, alert_id: str) -> bool:
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def get_active_tasks(self) -> List[Task]:
        return [task for task in self.tasks if task.state not in ["DONE", "FAILED"]]
    
    def assign_task(self, task_request: TaskRequest) -> Task:
        task = Task(
            id=f"T-{len(self.tasks)+1:02d}",
            asset="UGV-Alpha",  # Default assignment
            kind=task_request.kind,
            state="QUEUED",
            eta_min=random.randint(5, 30),
            params=task_request.params
        )
        self.tasks.append(task)
        return task
    
    def update_telemetry(self, device_id: str, telemetry: Telemetry):
        self.telemetry[device_id] = telemetry


# ============================================================================
# MQTT PUBLISHER
# ============================================================================

class MQTTPublisher:
    def __init__(self, broker_url: str = "localhost", port: int = 1883):
        self.broker_url = broker_url
        self.port = port
        self.client = mqtt.Client()
        self.connected = False
    
    def connect(self):
        try:
            self.client.connect(self.broker_url, self.port, 60)
            self.client.loop_start()
            self.connected = True
            print(f"MQTT connected to {self.broker_url}:{self.port}")
        except Exception as e:
            print(f"MQTT connection failed: {e}")
            self.connected = False
    
    def publish_alert(self, alert: Alert):
        if not self.connected:
            return
        
        topic = f"alerts/{alert.id}"
        payload = alert.json()
        self.client.publish(topic, payload, qos=2, retain=True)
        print(f"Published alert to {topic}")
    
    def publish_telemetry(self, telemetry: Telemetry):
        if not self.connected:
            return
        
        topic = f"devices/{telemetry.device_id}/telemetry"
        payload = telemetry.json()
        self.client.publish(topic, payload, qos=1, retain=True)
        print(f"Published telemetry to {topic}")
    
    def publish_task_update(self, task: Task):
        if not self.connected:
            return
        
        topic = "missions/updates"
        payload = task.json()
        self.client.publish(topic, payload, qos=2)
        print(f"Published task update to {topic}")
    
    def publish_fusion_event(self, event: Dict[str, Any]):
        if not self.connected:
            return
        
        topic = "fusion/events"
        payload = json.dumps(event)
        self.client.publish(topic, payload, qos=1)
        print(f"Published fusion event to {topic}")


# ============================================================================
# FASTAPI APP
# ============================================================================

# Initialize data store and MQTT
data_store = MockDataStore()
mqtt_publisher = MQTTPublisher()

# Create FastAPI app
app = FastAPI(
    title="Summit.OS Mock API",
description="Single Integration Standard (v1.1) for Summit.OS ↔ Sentinel",
    version="1.1.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth dependency
def get_api_key(authorization: Optional[str] = Header(None), 
                x_api_key: Optional[str] = Header(None)):
    """Extract API key from Authorization header or x-api-key header"""
    expected_key = os.getenv("SUMMIT_API_KEY", "dev_key_placeholder")
    
    if x_api_key:
        if x_api_key != expected_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return x_api_key
    
    if authorization and authorization.startswith("Bearer "):
        key = authorization[7:]
        if key != expected_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return key
    
    raise HTTPException(status_code=401, detail="Missing or invalid API key")


# ============================================================================
# REST ENDPOINTS
# ============================================================================

@app.get("/api/v1/system/health", response_model=HealthResponse)
async def get_health():
    """System health check"""
    return HealthResponse(
        status="ok",
        uptime=data_store.get_uptime()
    )

@app.get("/api/v1/intelligence/alerts", response_model=List[Alert])
async def get_alerts(
    since: Optional[float] = Query(None, description="Unix timestamp to filter alerts"),
    api_key: str = Depends(get_api_key)
):
    """Get intelligence alerts"""
    return data_store.get_alerts_since(since)

@app.post("/api/v1/alerts/{alert_id}/ack", response_model=AckResponse)
async def ack_alert(
    alert_id: str,
    api_key: str = Depends(get_api_key)
):
    """Acknowledge an alert"""
    if data_store.ack_alert(alert_id):
        return AckResponse(ok=True, id=alert_id)
    else:
        raise HTTPException(status_code=404, detail="Alert not found")

@app.get("/api/v1/tasks/active", response_model=List[Task])
async def get_active_tasks(api_key: str = Depends(get_api_key)):
    """Get active tasks"""
    return data_store.get_active_tasks()

@app.post("/api/v1/task/assign", response_model=Task)
async def assign_task(
    task_request: TaskRequest,
    api_key: str = Depends(get_api_key)
):
    """Assign a new task"""
    task = data_store.assign_task(task_request)
    
    # Publish task update via MQTT
    mqtt_publisher.publish_task_update(task)
    
    return task

@app.post("/api/v1/predict/scenario")
async def predict_scenario(
    scenario: ScenarioRequest,
    api_key: str = Depends(get_api_key)
):
    """Predict fire scenario and return GeoJSON FeatureCollection"""
    
    # Mock fire spread prediction
    # This would normally use AI models to predict fire spread
    features = []
    
    # Generate mock fire perimeter based on scenario
    center_lat = 34.13  # Center of AOI
    center_lon = -117.45
    
    # Create fire spread prediction
    for i in range(5):  # 5 time steps
        radius = (i + 1) * 0.01  # Growing radius
        time_minutes = (i + 1) * 15  # 15-minute intervals
        
        # Create circular fire perimeter
        perimeter_coords = []
        for angle in range(0, 360, 10):
            rad = math.radians(angle)
            lat = center_lat + radius * math.cos(rad)
            lon = center_lon + radius * math.sin(rad)
            perimeter_coords.append([lon, lat])
        
        # Close the polygon
        perimeter_coords.append(perimeter_coords[0])
        
        feature = {
            "type": "Feature",
            "properties": {
                "time_minutes": time_minutes,
                "confidence": 0.8 - (i * 0.1),
                "intensity": "high" if i < 2 else "medium"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [perimeter_coords]
            }
        }
        features.append(feature)
    
    # Return GeoJSON FeatureCollection
    return {
        "type": "FeatureCollection",
        "features": features
    }


# ============================================================================
# MQTT TELEMETRY FEEDER
# ============================================================================

async def mqtt_telemetry_feeder():
    """Continuously publish telemetry data via MQTT"""
    while True:
        try:
            # Update telemetry for all devices
            for device_id in data_store.telemetry:
                telemetry = data_store.telemetry[device_id]
                
                # Update with new random values
                telemetry.ts = time.time()
                telemetry.lat += random.uniform(-0.001, 0.001)
                telemetry.lon += random.uniform(-0.001, 0.001)
                telemetry.batt = max(0, telemetry.batt + random.uniform(-2, 1))
                telemetry.rssi = random.randint(-80, -40)
                telemetry.speed = random.uniform(0, 15)
                telemetry.sensors.update({
                    "temp_c": random.uniform(20, 40),
                    "wind_mps": random.uniform(0, 20),
                    "smoke_prob": random.uniform(0, 1)
                })
                
                # Publish telemetry
                mqtt_publisher.publish_telemetry(telemetry)
            
            # Occasionally publish fusion events
            if random.random() < 0.1:  # 10% chance
                event = {
                    "ts": time.time(),
                    "type": "sensor_fusion",
                    "confidence": random.uniform(0.7, 0.95),
                    "location": {
                        "lat": 34.123 + random.uniform(-0.01, 0.01),
                        "lon": -117.456 + random.uniform(-0.01, 0.01)
                    },
                    "details": {
                        "fusion_score": random.uniform(0.8, 0.99),
                        "sensors_used": ["thermal", "visual", "lidar"]
                    }
                }
                mqtt_publisher.publish_fusion_event(event)
            
            await asyncio.sleep(5)  # Publish every 5 seconds
            
        except Exception as e:
            print(f"MQTT feeder error: {e}")
            await asyncio.sleep(10)


# ============================================================================
# STARTUP AND SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize MQTT connection and start telemetry feeder"""
    print("Starting Summit.OS Mock Server...")
    
    # Connect to MQTT
    mqtt_publisher.connect()
    
    # Start telemetry feeder
    asyncio.create_task(mqtt_telemetry_feeder())
    
    print("Summit.OS Mock Server started!")
    print(f"API: http://localhost:{os.getenv('HTTP_PORT', '8000')}/api/v1")
    print(f"Docs: http://localhost:{os.getenv('HTTP_PORT', '8000')}/docs")
    print(f"MQTT: ws://localhost:1883")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down Summit.OS Mock Server...")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Get configuration from environment
    http_port = int(os.getenv("HTTP_PORT", "8000"))
    api_key = os.getenv("SUMMIT_API_KEY", "dev_key_placeholder")
    mqtt_url = os.getenv("MQTT_WS_URL", "ws://localhost:1883")
    
    print("=" * 60)
    print("Summit.OS Mock Server - Single Integration Standard (v1.1)")
    print("=" * 60)
    print(f"HTTP_PORT: {http_port}")
    print(f"SUMMIT_API_KEY: {api_key}")
    print(f"MQTT_WS_URL: {mqtt_url}")
    print("=" * 60)
    
    # Run the server
    uvicorn.run(
        "summit_mock:app",
        host="0.0.0.0",
        port=http_port,
        reload=False,
        log_level="info"
    )
