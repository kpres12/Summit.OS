"""
Summit.OS AI Model Showcase

Demonstrates the AI capabilities of Summit.OS with interactive examples
of sensor fusion, risk assessment, and autonomous decision-making.
"""

import asyncio
import json
import numpy as np
import cv2
from datetime import datetime, timezone
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# Import AI models
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fusion.ai_models import MultimodalFusionNetwork, FireSpreadPredictor, AnomalyDetector, EdgeInferenceEngine
from intelligence.ai_reasoning import RiskAssessmentEngine, AdvisoryGenerator

app = FastAPI(
    title="Summit.OS AI Showcase",
    description="Interactive demonstration of Summit.OS AI capabilities",
    version="1.0.0"
)

# CORS middleware
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global AI models
fusion_network = None
fire_predictor = None
anomaly_detector = None
risk_engine = None
advisory_generator = None

@app.on_event("startup")
async def startup_event():
    """Initialize AI models on startup."""
    global fusion_network, fire_predictor, anomaly_detector, risk_engine, advisory_generator
    
    print("🧠 Initializing Summit.OS AI Models...")
    
    # Initialize AI models
    fusion_network = MultimodalFusionNetwork(
        input_dims={'weather': 5, 'lidar': 1, 'thermal': 1, 'visual': 3},
        hidden_dim=512
    )
    
    fire_predictor = FireSpreadPredictor()
    anomaly_detector = AnomalyDetector()
    risk_engine = RiskAssessmentEngine()
    advisory_generator = AdvisoryGenerator()
    
    print("✅ AI Models initialized successfully!")

@app.get("/")
async def root():
    """Root endpoint with AI showcase interface."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Summit.OS AI Showcase</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: #fff; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 40px; }
            .demo-section { margin: 30px 0; padding: 20px; border: 1px solid #333; border-radius: 8px; }
            .demo-button { background: #ff4444; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
            .demo-button:hover { background: #ff6666; }
            .result { background: #2a2a2a; padding: 15px; margin: 10px 0; border-radius: 4px; }
            .ai-capability { display: inline-block; margin: 10px; padding: 15px; background: #333; border-radius: 8px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔥 Summit.OS AI Showcase</h1>
                <p>Distributed Intelligence Fabric for Autonomous Robotics</p>
            </div>
            
            <div class="demo-section">
                <h2>🧠 AI Capabilities</h2>
                <div class="ai-capability">
                    <h3>Sensor Fusion</h3>
                    <p>Multimodal data fusion from weather, LiDAR, thermal, and visual sensors</p>
                    <button class="demo-button" onclick="demoSensorFusion()">Demo</button>
                </div>
                
                <div class="ai-capability">
                    <h3>Fire Spread Prediction</h3>
                    <p>Physics + ML hybrid models for fire behavior prediction</p>
                    <button class="demo-button" onclick="demoFirePrediction()">Demo</button>
                </div>
                
                <div class="ai-capability">
                    <h3>Anomaly Detection</h3>
                    <p>Real-time anomaly detection in sensor data streams</p>
                    <button class="demo-button" onclick="demoAnomalyDetection()">Demo</button>
                </div>
                
                <div class="ai-capability">
                    <h3>Risk Assessment</h3>
                    <p>AI-powered risk evaluation and contextual recommendations</p>
                    <button class="demo-button" onclick="demoRiskAssessment()">Demo</button>
                </div>
                
                <div class="ai-capability">
                    <h3>Advisory Generation</h3>
                    <p>Human-readable intelligence reports and recommendations</p>
                    <button class="demo-button" onclick="demoAdvisoryGeneration()">Demo</button>
                </div>
            </div>
            
            <div class="demo-section">
                <h2>📊 Results</h2>
                <div id="results" class="result">
                    <p>Click a demo button above to see AI capabilities in action...</p>
                </div>
            </div>
        </div>
        
        <script>
            async function demoSensorFusion() {
                const response = await fetch('/demo/sensor-fusion');
                const result = await response.json();
                document.getElementById('results').innerHTML = '<h3>Sensor Fusion Demo</h3><pre>' + JSON.stringify(result, null, 2) + '</pre>';
            }
            
            async function demoFirePrediction() {
                const response = await fetch('/demo/fire-prediction');
                const result = await response.json();
                document.getElementById('results').innerHTML = '<h3>Fire Prediction Demo</h3><pre>' + JSON.stringify(result, null, 2) + '</pre>';
            }
            
            async function demoAnomalyDetection() {
                const response = await fetch('/demo/anomaly-detection');
                const result = await response.json();
                document.getElementById('results').innerHTML = '<h3>Anomaly Detection Demo</h3><pre>' + JSON.stringify(result, null, 2) + '</pre>';
            }
            
            async function demoRiskAssessment() {
                const response = await fetch('/demo/risk-assessment');
                const result = await response.json();
                document.getElementById('results').innerHTML = '<h3>Risk Assessment Demo</h3><pre>' + JSON.stringify(result, null, 2) + '</pre>';
            }
            
            async function demoAdvisoryGeneration() {
                const response = await fetch('/demo/advisory-generation');
                const result = await response.json();
                document.getElementById('results').innerHTML = '<h3>Advisory Generation Demo</h3><pre>' + JSON.stringify(result, null, 2) + '</pre>';
            }
        </script>
    </body>
    </html>
    """)

@app.get("/demo/sensor-fusion")
async def demo_sensor_fusion():
    """Demo sensor fusion capabilities."""
    # Generate mock sensor data
    weather_data = {
        'temperature': 35.0,
        'humidity': 25.0,
        'wind_speed': 15.0,
        'wind_direction': 180.0,
        'pressure': 1005.0
    }
    
    # Mock sensor readings
    lidar_data = np.random.rand(64, 64) * 100  # Mock LiDAR point cloud
    thermal_data = np.random.rand(64, 64) * 1000  # Mock thermal image
    visual_data = np.random.rand(64, 64, 3) * 255  # Mock RGB image
    
    # Simulate sensor fusion
    fusion_result = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'sensor_data': {
            'weather': weather_data,
            'lidar_points': len(lidar_data.flatten()),
            'thermal_range': [thermal_data.min(), thermal_data.max()],
            'visual_resolution': visual_data.shape
        },
        'fusion_results': {
            'fire_detected': True,
            'fire_confidence': 0.87,
            'smoke_detected': True,
            'smoke_confidence': 0.73,
            'terrain_classification': 'brush',
            'terrain_confidence': 0.92,
            'anomaly_score': 0.65
        },
        'world_model_update': {
            'entities': [
                {
                    'type': 'fire',
                    'location': {'lat': 37.7749, 'lon': -122.4194},
                    'confidence': 0.87,
                    'size': 50.0,
                    'temperature': 850.0
                }
            ],
            'relationships': [
                {
                    'source': 'fire',
                    'target': 'weather',
                    'relationship': 'influenced_by',
                    'confidence': 0.8
                }
            ]
        }
    }
    
    return fusion_result

@app.get("/demo/fire-prediction")
async def demo_fire_prediction():
    """Demo fire spread prediction capabilities."""
    # Mock fire location and conditions
    fire_location = (37.7749, -122.4194)
    
    weather_data = {
        'temperature': 35.0,
        'humidity': 25.0,
        'wind_speed': 15.0,
        'wind_direction': 180.0,
        'pressure': 1005.0
    }
    
    terrain_data = {
        'slope': 15.0,
        'elevation': 200.0,
        'vegetation_density': 0.8,
        'fuel_load': 2.5
    }
    
    # Generate fire spread prediction
    prediction = fire_predictor.predict_spread(fire_location, weather_data, terrain_data)
    
    # Add visualization data
    prediction['visualization'] = {
        'spread_pattern': generate_spread_pattern(fire_location, prediction['spread_rate']),
        'risk_zones': generate_risk_zones(fire_location, prediction['spread_rate']),
        'timeline': generate_spread_timeline(prediction['spread_rate'])
    }
    
    return prediction

@app.get("/demo/anomaly-detection")
async def demo_anomaly_detection():
    """Demo anomaly detection capabilities."""
    # Generate mock sensor data with anomalies
    sensor_data = {
        'temperature': 45.0,  # Unusually high
        'humidity': 15.0,    # Unusually low
        'wind_speed': 25.0,  # High wind
        'pressure': 980.0,   # Low pressure
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    # Detect anomalies
    anomaly_result = anomaly_detector.detect_anomalies(sensor_data)
    
    # Add context
    anomaly_result['context'] = {
        'sensor_quality': 'good',
        'data_completeness': 1.0,
        'historical_baseline': {
            'temperature': 25.0,
            'humidity': 50.0,
            'wind_speed': 8.0,
            'pressure': 1013.0
        }
    }
    
    return anomaly_result

@app.get("/demo/risk-assessment")
async def demo_risk_assessment():
    """Demo risk assessment capabilities."""
    # Mock sensor data
    sensor_data = {
        'temperature': 35.0,
        'humidity': 25.0,
        'wind_speed': 15.0,
        'pressure': 1005.0,
        'visibility': 5000.0
    }
    
    # Mock world model
    world_model = {
        'spatial_data': {
            'cell_1': {
                'location': {'lat': 37.7749, 'lon': -122.4194},
                'predictions': {
                    'fire_detected': True,
                    'confidence': 0.87,
                    'temperature': 850.0
                },
                'confidence': 0.87
            }
        },
        'temporal_trends': {
            'temperature_trend': 'increasing',
            'fire_detection_trend': 'increasing',
            'data_points': 50
        }
    }
    
    # Assess risk
    risk_assessment = risk_engine.assess_risk(sensor_data, world_model)
    
    return risk_assessment

@app.get("/demo/advisory-generation")
async def demo_advisory_generation():
    """Demo advisory generation capabilities."""
    # Mock risk assessment
    risk_assessment = {
        'overall_risk': 0.85,
        'fire_risk': {'score': 0.9, 'confidence': 0.87},
        'environmental_risk': {'score': 0.7, 'confidence': 0.8},
        'operational_risk': {'score': 0.6, 'confidence': 0.75},
        'risk_level': 'high',
        'recommendations': [
            'Deploy additional fire monitoring assets',
            'Prepare suppression resources',
            'Issue fire weather warning'
        ]
    }
    
    # Mock world model
    world_model = {
        'spatial_data': {},
        'temporal_trends': {
            'fire_detection_trend': 'increasing',
            'temperature_trend': 'increasing'
        }
    }
    
    # Generate advisory
    advisory = advisory_generator.generate_advisory(risk_assessment, world_model)
    
    return advisory

def generate_spread_pattern(fire_location, spread_rate):
    """Generate mock fire spread pattern."""
    lat, lon = fire_location
    pattern = []
    
    # Generate points in a circular pattern around the fire
    for angle in range(0, 360, 30):
        distance = spread_rate * 1000  # Convert to meters
        new_lat = lat + (distance / 111000) * np.cos(np.radians(angle))
        new_lon = lon + (distance / 111000) * np.sin(np.radians(angle))
        pattern.append({'lat': new_lat, 'lon': new_lon, 'intensity': 0.8})
    
    return pattern

def generate_risk_zones(fire_location, spread_rate):
    """Generate risk zones around fire."""
    lat, lon = fire_location
    zones = []
    
    # High risk zone (immediate area)
    zones.append({
        'zone': 'high_risk',
        'radius': spread_rate * 500,
        'center': {'lat': lat, 'lon': lon},
        'risk_level': 0.9
    })
    
    # Medium risk zone (extended area)
    zones.append({
        'zone': 'medium_risk',
        'radius': spread_rate * 1000,
        'center': {'lat': lat, 'lon': lon},
        'risk_level': 0.6
    })
    
    # Low risk zone (monitoring area)
    zones.append({
        'zone': 'low_risk',
        'radius': spread_rate * 2000,
        'center': {'lat': lat, 'lon': lon},
        'risk_level': 0.3
    })
    
    return zones

def generate_spread_timeline(spread_rate):
    """Generate fire spread timeline."""
    timeline = []
    
    for hour in range(0, 24, 2):
        size = spread_rate * hour * 1000  # Size in meters
        timeline.append({
            'time': f"{hour:02d}:00",
            'size': size,
            'area': np.pi * (size / 1000) ** 2,  # Area in km²
            'risk_level': min(1.0, size / 10000)
        })
    
    return timeline

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time AI demonstrations."""
    await websocket.accept()
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            await websocket.send_text(f"AI Demo: {data}")
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8008,
        reload=True,
        log_level="info"
    )
