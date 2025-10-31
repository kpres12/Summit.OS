#!/usr/bin/env python3
"""
Summit.OS Mock Data Generator

Generates realistic mock data for development and testing.
Simulates wildfire detection scenario with multiple assets.
"""

import asyncio
import json
import random
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
import requests
import math

# Configuration
API_BASE_URL = "http://localhost:8000"
FABRIC_URL = "http://localhost:8001"
FUSION_URL = "http://localhost:8002"

# Mock devices and their capabilities
DEVICES = {
    "drone-001": {
        "type": "quadcopter",
        "capabilities": ["thermal_camera", "rgb_camera", "lidar", "weather_sensor"],
        "base_location": {"lat": 37.7749, "lon": -122.4194, "alt": 100.0},
        "operational_range": 5000,  # meters
        "battery_capacity": 100.0,
        "current_battery": 85.0
    },
    "drone-002": {
        "type": "fixed_wing",
        "capabilities": ["thermal_camera", "multispectral", "weather_sensor"],
        "base_location": {"lat": 37.7849, "lon": -122.4094, "alt": 150.0},
        "operational_range": 10000,
        "battery_capacity": 100.0,
        "current_battery": 92.0
    },
    "ugv-001": {
        "type": "ground_vehicle",
        "capabilities": ["thermal_camera", "gas_sensor", "weather_station"],
        "base_location": {"lat": 37.7649, "lon": -122.4294, "alt": 0.0},
        "operational_range": 2000,
        "battery_capacity": 100.0,
        "current_battery": 78.0
    },
    "weather-station-001": {
        "type": "weather_station",
        "capabilities": ["temperature", "humidity", "wind", "pressure", "rain"],
        "base_location": {"lat": 37.7549, "lon": -122.4394, "alt": 10.0},
        "operational_range": 0,
        "battery_capacity": 100.0,
        "current_battery": 95.0
    }
}

# Fire simulation parameters
FIRE_LOCATIONS = [
    {"lat": 37.7749, "lon": -122.4194, "intensity": 0.8, "size": 50.0},
    {"lat": 37.7849, "lon": -122.4094, "intensity": 0.6, "size": 30.0},
    {"lat": 37.7649, "lon": -122.4294, "intensity": 0.4, "size": 20.0}
]

class MockDataGenerator:
    def __init__(self):
        self.running = False
        self.device_positions = {}
        self.fire_detections = []
        self.mission_active = False
        
    async def start(self):
        """Start generating mock data."""
        print("🔥 Starting Summit.OS Mock Data Generator")
        print("=" * 50)
        
        self.running = True
        
        # Initialize device positions
        for device_id, device in DEVICES.items():
            self.device_positions[device_id] = {
                "lat": device["base_location"]["lat"],
                "lon": device["base_location"]["lon"],
                "alt": device["base_location"]["alt"],
                "heading": random.uniform(0, 360),
                "speed": 0.0
            }
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self.generate_telemetry()),
            asyncio.create_task(self.generate_weather_data()),
            asyncio.create_task(self.simulate_fire_detection()),
            asyncio.create_task(self.generate_alerts()),
            asyncio.create_task(self.simulate_mission())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("\n🛑 Stopping mock data generator...")
            self.running = False
    
    async def generate_telemetry(self):
        """Generate telemetry data for all devices."""
        while self.running:
            for device_id, device in DEVICES.items():
                if not self.running:
                    break
                    
                # Update position (simulate movement)
                await self.update_device_position(device_id, device)
                
                # Generate sensor data
                sensor_data = await self.generate_sensor_data(device_id, device)
                
                # Create telemetry message
                telemetry = {
                    "device_id": device_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "location": {
                        "latitude": self.device_positions[device_id]["lat"],
                        "longitude": self.device_positions[device_id]["lon"],
                        "altitude": self.device_positions[device_id]["alt"],
                        "heading": self.device_positions[device_id]["heading"],
                        "speed": self.device_positions[device_id]["speed"]
                    },
                    "sensors": sensor_data,
                    "status": "online",
                    "battery_level": device["current_battery"],
                    "signal_strength": random.uniform(-80, -40),
                    "metadata": {
                        "device_type": device["type"],
                        "capabilities": device["capabilities"]
                    }
                }
                
                # Send to fabric service
                try:
                    response = requests.post(
                        f"{FABRIC_URL}/telemetry",
                        json=telemetry,
                        timeout=5
                    )
                    if response.status_code == 200:
                        print(f"📡 Telemetry: {device_id} at {telemetry['location']['latitude']:.4f}, {telemetry['location']['longitude']:.4f}")
                except Exception as e:
                    print(f"❌ Failed to send telemetry for {device_id}: {e}")
            
            await asyncio.sleep(2)  # Send telemetry every 2 seconds
    
    async def update_device_position(self, device_id: str, device: Dict[str, Any]):
        """Update device position based on mission or patrol pattern."""
        pos = self.device_positions[device_id]
        
        if device["type"] in ["quadcopter", "fixed_wing"]:
            # Simulate patrol pattern
            if self.mission_active:
                # Move towards fire locations
                target_fire = random.choice(FIRE_LOCATIONS)
                dx = target_fire["lat"] - pos["lat"]
                dy = target_fire["lon"] - pos["lon"]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > 0.001:  # 100m threshold
                    # Move towards target
                    speed = 0.0001  # degrees per update
                    pos["lat"] += (dx / distance) * speed
                    pos["lon"] += (dy / distance) * speed
                    pos["speed"] = 15.0  # m/s
                else:
                    pos["speed"] = 0.0
            else:
                # Patrol pattern
                pos["heading"] += random.uniform(-5, 5)
                pos["lat"] += math.cos(math.radians(pos["heading"])) * 0.0001
                pos["lon"] += math.sin(math.radians(pos["heading"])) * 0.0001
                pos["speed"] = 10.0
        else:
            # Ground vehicles move slower
            pos["heading"] += random.uniform(-2, 2)
            pos["lat"] += math.cos(math.radians(pos["heading"])) * 0.00005
            pos["lon"] += math.sin(math.radians(pos["heading"])) * 0.00005
            pos["speed"] = 5.0
    
    async def generate_sensor_data(self, device_id: str, device: Dict[str, Any]) -> Dict[str, Any]:
        """Generate realistic sensor data based on device capabilities."""
        sensors = {}
        pos = self.device_positions[device_id]
        
        # Check for nearby fires
        nearby_fire = None
        for fire in FIRE_LOCATIONS:
            distance = math.sqrt(
                (fire["lat"] - pos["lat"])**2 + (fire["lon"] - pos["lon"])**2
            )
            if distance < 0.01:  # Within 1km
                nearby_fire = fire
                break
        
        # Temperature sensor
        if "temperature" in device["capabilities"] or "thermal_camera" in device["capabilities"]:
            base_temp = 25.0
            if nearby_fire:
                # Higher temperature near fire
                fire_effect = nearby_fire["intensity"] * 50.0 * (1.0 - distance * 100)
                sensors["temperature"] = base_temp + fire_effect + random.uniform(-2, 2)
            else:
                sensors["temperature"] = base_temp + random.uniform(-3, 3)
        
        # Humidity sensor
        if "humidity" in device["capabilities"]:
            base_humidity = 45.0
            if nearby_fire:
                # Lower humidity near fire
                fire_effect = nearby_fire["intensity"] * 20.0 * (1.0 - distance * 100)
                sensors["humidity"] = max(10.0, base_humidity - fire_effect + random.uniform(-5, 5))
            else:
                sensors["humidity"] = base_humidity + random.uniform(-10, 10)
        
        # Wind sensor
        if "wind" in device["capabilities"]:
            sensors["wind_speed"] = random.uniform(5, 20)
            sensors["wind_direction"] = random.uniform(0, 360)
        
        # Gas sensors
        if "gas_sensor" in device["capabilities"]:
            if nearby_fire:
                sensors["co_level"] = nearby_fire["intensity"] * 100.0 + random.uniform(0, 20)
                sensors["smoke_density"] = nearby_fire["intensity"] * 0.8 + random.uniform(0, 0.2)
            else:
                sensors["co_level"] = random.uniform(0, 10)
                sensors["smoke_density"] = random.uniform(0, 0.1)
        
        # Thermal camera data
        if "thermal_camera" in device["capabilities"]:
            if nearby_fire:
                # Simulate thermal detection
                fire_confidence = nearby_fire["intensity"] * (1.0 - distance * 50)
                if fire_confidence > 0.3:
                    sensors["thermal_detection"] = {
                        "fire_detected": True,
                        "confidence": fire_confidence,
                        "temperature": 800.0 + random.uniform(-100, 100),
                        "size_estimate": nearby_fire["size"] + random.uniform(-10, 10)
                    }
                else:
                    sensors["thermal_detection"] = {
                        "fire_detected": False,
                        "confidence": 0.0
                    }
            else:
                sensors["thermal_detection"] = {
                    "fire_detected": False,
                    "confidence": 0.0
                }
        
        return sensors
    
    async def generate_weather_data(self):
        """Generate weather station data."""
        while self.running:
            device_id = "weather-station-001"
            device = DEVICES[device_id]
            pos = self.device_positions[device_id]
            
            # Generate weather data
            weather_data = {
                "temperature": 25.0 + random.uniform(-5, 5),
                "humidity": 45.0 + random.uniform(-15, 15),
                "wind_speed": random.uniform(5, 25),
                "wind_direction": random.uniform(0, 360),
                "pressure": 1013.25 + random.uniform(-10, 10),
                "rain_rate": random.uniform(0, 5),
                "visibility": random.uniform(1000, 10000)
            }
            
            # Create telemetry with weather data
            telemetry = {
                "device_id": device_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "location": {
                    "latitude": pos["lat"],
                    "longitude": pos["lon"],
                    "altitude": pos["alt"]
                },
                "sensors": weather_data,
                "status": "online",
                "battery_level": device["current_battery"],
                "signal_strength": random.uniform(-60, -30),
                "metadata": {
                    "device_type": device["type"],
                    "station_id": "WS001"
                }
            }
            
            try:
                response = requests.post(
                    f"{FABRIC_URL}/telemetry",
                    json=telemetry,
                    timeout=5
                )
                if response.status_code == 200:
                    print(f"🌤️  Weather: {weather_data['temperature']:.1f}°C, {weather_data['humidity']:.1f}% RH, {weather_data['wind_speed']:.1f} m/s")
            except Exception as e:
                print(f"❌ Failed to send weather data: {e}")
            
            await asyncio.sleep(30)  # Weather updates every 30 seconds
    
    async def simulate_fire_detection(self):
        """Simulate fire detection events."""
        while self.running:
            # Randomly detect fires based on device proximity
            for device_id, device in DEVICES.items():
                if "thermal_camera" not in device["capabilities"]:
                    continue
                
                pos = self.device_positions[device_id]
                
                for fire in FIRE_LOCATIONS:
                    distance = math.sqrt(
                        (fire["lat"] - pos["lat"])**2 + (fire["lon"] - pos["lon"])**2
                    )
                    
                    # Detection probability based on distance and fire intensity
                    detection_prob = fire["intensity"] * (1.0 - distance * 100)
                    
                    if random.random() < detection_prob * 0.1:  # 10% chance per check
                        # Generate detection
                        detection = {
                            "detection_id": f"det-{int(time.time())}",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "location": {
                                "latitude": fire["lat"] + random.uniform(-0.001, 0.001),
                                "longitude": fire["lon"] + random.uniform(-0.001, 0.001),
                                "altitude": 0.0
                            },
                            "object_type": "fire",
                            "confidence": fire["intensity"] + random.uniform(-0.2, 0.2),
                            "bounding_box": {
                                "x": random.randint(100, 400),
                                "y": random.randint(100, 300),
                                "width": random.randint(50, 150),
                                "height": random.randint(50, 150)
                            },
                            "metadata": {
                                "temperature": 800.0 + random.uniform(-100, 100),
                                "size_estimate": fire["size"] + random.uniform(-10, 10),
                                "detection_method": "thermal_camera",
                                "device_id": device_id
                            }
                        }
                        
                        try:
                            response = requests.post(
                                f"{FUSION_URL}/detections",
                                json=detection,
                                timeout=5
                            )
                            if response.status_code == 200:
                                print(f"🔥 Fire Detection: {detection['confidence']:.2f} confidence at {detection['location']['latitude']:.4f}, {detection['location']['longitude']:.4f}")
                        except Exception as e:
                            print(f"❌ Failed to send detection: {e}")
            
            await asyncio.sleep(10)  # Check for detections every 10 seconds
    
    async def generate_alerts(self):
        """Generate contextual alerts based on conditions."""
        while self.running:
            # Fire risk alerts
            if random.random() < 0.05:  # 5% chance per check
                alert = {
                    "alert_id": f"alert-{int(time.time())}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "severity": random.choice(["medium", "high", "critical"]),
                    "location": {
                        "latitude": random.choice(FIRE_LOCATIONS)["lat"],
                        "longitude": random.choice(FIRE_LOCATIONS)["lon"],
                        "altitude": 0.0
                    },
                    "title": "Fire Risk Alert",
                    "description": "High fire risk detected in the area",
                    "source": "intelligence_engine",
                    "category": "fire",
                    "tags": ["fire_risk", "weather", "safety"],
                    "metadata": {
                        "risk_factors": ["high_temperature", "low_humidity", "wind_conditions"],
                        "affected_area": random.uniform(1.0, 5.0)
                    }
                }
                
                try:
                    response = requests.post(
                        f"{FABRIC_URL}/alerts",
                        json=alert,
                        timeout=5
                    )
                    if response.status_code == 200:
                        print(f"🚨 Alert: {alert['severity'].upper()} - {alert['description']}")
                except Exception as e:
                    print(f"❌ Failed to send alert: {e}")
            
            await asyncio.sleep(60)  # Check for alerts every minute
    
    async def simulate_mission(self):
        """Simulate mission execution."""
        while self.running:
            if not self.mission_active and random.random() < 0.1:  # 10% chance to start mission
                # Start new mission
                self.mission_active = True
                mission = {
                    "mission_id": f"mission-{int(time.time())}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "active",
                    "assets": ["drone-001", "ugv-001"],
                    "objectives": ["patrol_sector_7", "detect_fire_hazards", "report_conditions"],
                    "progress": 0.0,
                    "estimated_completion": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
                    "metadata": {
                        "mission_type": "patrol",
                        "priority": "high",
                        "sector": "7"
                    }
                }
                
                try:
                    response = requests.post(
                        f"{FABRIC_URL}/missions",
                        json=mission,
                        timeout=5
                    )
                    if response.status_code == 200:
                        print(f"🎯 Mission Started: {mission['mission_id']}")
                except Exception as e:
                    print(f"❌ Failed to start mission: {e}")
            
            elif self.mission_active:
                # Update mission progress
                mission_update = {
                    "mission_id": f"mission-{int(time.time())}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "active",
                    "assets": ["drone-001", "ugv-001"],
                    "objectives": ["patrol_sector_7", "detect_fire_hazards", "report_conditions"],
                    "progress": min(1.0, random.uniform(0.1, 0.3)),
                    "metadata": {
                        "completed_objectives": ["patrol_sector_7"],
                        "active_objectives": ["detect_fire_hazards", "report_conditions"]
                    }
                }
                
                try:
                    response = requests.post(
                        f"{FABRIC_URL}/missions",
                        json=mission_update,
                        timeout=5
                    )
                    if response.status_code == 200:
                        print(f"📊 Mission Update: {mission_update['progress']:.1%} complete")
                except Exception as e:
                    print(f"❌ Failed to update mission: {e}")
                
                # Randomly complete mission
                if random.random() < 0.1:  # 10% chance to complete
                    self.mission_active = False
                    print("✅ Mission Completed")
            
            await asyncio.sleep(30)  # Check missions every 30 seconds

async def main():
    """Main entry point."""
    generator = MockDataGenerator()
    await generator.start()

if __name__ == "__main__":
    asyncio.run(main())
