#!/usr/bin/env python3
"""
Summit.OS Demo Mission

Demonstrates a complete wildfire detection and response scenario:
1. Fire detection via thermal camera
2. Alert generation and escalation
3. Mission planning and asset deployment
4. Real-time coordination and monitoring
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
import requests

# Configuration
API_BASE_URL = "http://localhost:8000"
FABRIC_URL = "http://localhost:8001"
FUSION_URL = "http://localhost:8002"
INTELLIGENCE_URL = "http://localhost:8003"
TASKING_URL = "http://localhost:8004"

class DemoMission:
    def __init__(self):
        self.mission_id = f"demo-mission-{int(time.time())}"
        self.fire_location = {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "altitude": 0.0
        }
        self.assets = ["drone-001", "drone-002", "ugv-001"]
        self.step = 0
        
    async def run_demo(self):
        """Run the complete demo mission."""
        print("🔥 Summit.OS Demo Mission: Wildfire Detection & Response")
        print("=" * 60)
        print(f"Mission ID: {self.mission_id}")
        print(f"Fire Location: {self.fire_location['latitude']:.4f}, {self.fire_location['longitude']:.4f}")
        print()
        
        try:
            # Step 1: Initial fire detection
            await self.step_1_fire_detection()
            await asyncio.sleep(3)
            
            # Step 2: Alert generation
            await self.step_2_alert_generation()
            await asyncio.sleep(3)
            
            # Step 3: Mission planning
            await self.step_3_mission_planning()
            await asyncio.sleep(3)
            
            # Step 4: Asset deployment
            await self.step_4_asset_deployment()
            await asyncio.sleep(5)
            
            # Step 5: Real-time monitoring
            await self.step_5_real_time_monitoring()
            await asyncio.sleep(5)
            
            # Step 6: Mission completion
            await self.step_6_mission_completion()
            
            print("\n✅ Demo Mission Completed Successfully!")
            
        except Exception as e:
            print(f"\n❌ Demo Mission Failed: {e}")
    
    async def step_1_fire_detection(self):
        """Step 1: Simulate fire detection via thermal camera."""
        print("🔍 Step 1: Fire Detection")
        print("-" * 30)
        
        # Simulate thermal camera detection
        detection = {
            "detection_id": f"det-{int(time.time())}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "location": self.fire_location,
            "object_type": "fire",
            "confidence": 0.95,
            "bounding_box": {
                "x": 200,
                "y": 150,
                "width": 100,
                "height": 120
            },
            "metadata": {
                "temperature": 850.0,
                "size_estimate": 50.0,
                "detection_method": "thermal_camera",
                "device_id": "thermal-camera-001",
                "sector": "7"
            }
        }
        
        try:
            response = requests.post(f"{FUSION_URL}/detections", json=detection, timeout=5)
            if response.status_code == 200:
                print(f"✅ Fire detected with {detection['confidence']:.1%} confidence")
                print(f"   Location: {detection['location']['latitude']:.4f}, {detection['location']['longitude']:.4f}")
                print(f"   Temperature: {detection['metadata']['temperature']}°C")
                print(f"   Size: {detection['metadata']['size_estimate']}m")
            else:
                print(f"❌ Failed to process detection: {response.status_code}")
        except Exception as e:
            print(f"❌ Error sending detection: {e}")
    
    async def step_2_alert_generation(self):
        """Step 2: Generate contextual alert."""
        print("\n🚨 Step 2: Alert Generation")
        print("-" * 30)
        
        alert = {
            "alert_id": f"alert-{int(time.time())}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": "critical",
            "location": self.fire_location,
            "title": "Critical Fire Alert - Sector 7",
            "description": "Large fire detected in sector 7. Immediate response required.",
            "source": "thermal-camera-001",
            "category": "fire",
            "tags": ["wildfire", "critical", "sector-7", "thermal-detection"],
            "metadata": {
                "fire_size": "large",
                "temperature": 850.0,
                "wind_conditions": "favorable",
                "evacuation_required": True,
                "affected_area": 2.5
            }
        }
        
        try:
            response = requests.post(f"{FABRIC_URL}/alerts", json=alert, timeout=5)
            if response.status_code == 200:
                print(f"✅ Critical alert generated: {alert['title']}")
                print(f"   Severity: {alert['severity'].upper()}")
                print(f"   Affected Area: {alert['metadata']['affected_area']} km²")
                print(f"   Evacuation Required: {alert['metadata']['evacuation_required']}")
            else:
                print(f"❌ Failed to generate alert: {response.status_code}")
        except Exception as e:
            print(f"❌ Error sending alert: {e}")
    
    async def step_3_mission_planning(self):
        """Step 3: Create mission plan."""
        print("\n🎯 Step 3: Mission Planning")
        print("-" * 30)
        
        mission = {
            "mission_id": self.mission_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "planned",
            "mission_type": "fire_response",
            "priority": "critical",
            "objectives": [
                "assess_fire_conditions",
                "monitor_fire_spread",
                "coordinate_suppression",
                "ensure_safety"
            ],
            "assets": self.assets,
            "constraints": {
                "max_duration": 3600,
                "weather_limits": {
                    "max_wind_speed": 30.0,
                    "min_visibility": 500.0
                },
                "safety_radius": 1000.0
            },
            "start_time": datetime.now(timezone.utc).isoformat(),
            "estimated_duration": 1800,
            "metadata": {
                "fire_location": self.fire_location,
                "sector": "7",
                "response_level": "emergency"
            }
        }
        
        try:
            response = requests.post(f"{TASKING_URL}/missions", json=mission, timeout=5)
            if response.status_code == 200:
                print(f"✅ Mission planned: {self.mission_id}")
                print(f"   Objectives: {len(mission['objectives'])}")
                print(f"   Assets: {', '.join(mission['assets'])}")
                print(f"   Duration: {mission['estimated_duration']}s")
            else:
                print(f"❌ Failed to create mission: {response.status_code}")
        except Exception as e:
            print(f"❌ Error creating mission: {e}")
    
    async def step_4_asset_deployment(self):
        """Step 4: Deploy assets to mission."""
        print("\n🚁 Step 4: Asset Deployment")
        print("-" * 30)
        
        # Update mission status to active
        mission_update = {
            "mission_id": self.mission_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "active",
            "assets": self.assets,
            "objectives": [
                "assess_fire_conditions",
                "monitor_fire_spread", 
                "coordinate_suppression",
                "ensure_safety"
            ],
            "progress": 0.1,
            "metadata": {
                "deployment_status": "in_progress",
                "assets_deployed": self.assets,
                "estimated_completion": (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
            }
        }
        
        try:
            response = requests.post(f"{FABRIC_URL}/missions", json=mission_update, timeout=5)
            if response.status_code == 200:
                print(f"✅ Mission activated: {self.mission_id}")
                print(f"   Assets deployed: {', '.join(self.assets)}")
                print(f"   Progress: {mission_update['progress']:.1%}")
            else:
                print(f"❌ Failed to activate mission: {response.status_code}")
        except Exception as e:
            print(f"❌ Error activating mission: {e}")
        
        # Simulate asset telemetry
        for i, asset in enumerate(self.assets):
            await asyncio.sleep(1)
            
            # Calculate asset position (spread around fire)
            angle = (i * 120) % 360
            distance = 0.001  # ~100m
            lat_offset = distance * math.cos(math.radians(angle))
            lon_offset = distance * math.sin(math.radians(angle))
            
            asset_location = {
                "latitude": self.fire_location["latitude"] + lat_offset,
                "longitude": self.fire_location["longitude"] + lon_offset,
                "altitude": 100.0 if "drone" in asset else 0.0
            }
            
            telemetry = {
                "device_id": asset,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "location": asset_location,
                "sensors": {
                    "temperature": 25.0 + random.uniform(-5, 5),
                    "humidity": 45.0 + random.uniform(-10, 10),
                    "wind_speed": random.uniform(5, 15),
                    "visibility": random.uniform(1000, 10000)
                },
                "status": "online",
                "battery_level": 85.0 - (i * 5),
                "signal_strength": -65.0,
                "metadata": {
                    "mission_id": self.mission_id,
                    "task": "fire_monitoring",
                    "sector": "7"
                }
            }
            
            try:
                response = requests.post(f"{FABRIC_URL}/telemetry", json=telemetry, timeout=5)
                if response.status_code == 200:
                    print(f"   📡 {asset}: Deployed at {asset_location['latitude']:.4f}, {asset_location['longitude']:.4f}")
            except Exception as e:
                print(f"   ❌ Failed to send telemetry for {asset}: {e}")
    
    async def step_5_real_time_monitoring(self):
        """Step 5: Simulate real-time monitoring."""
        print("\n📊 Step 5: Real-time Monitoring")
        print("-" * 30)
        
        # Simulate monitoring updates
        for update in range(3):
            await asyncio.sleep(2)
            
            # Update mission progress
            progress = 0.2 + (update * 0.2)
            mission_update = {
                "mission_id": self.mission_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "active",
                "assets": self.assets,
                "objectives": [
                    "assess_fire_conditions",
                    "monitor_fire_spread",
                    "coordinate_suppression", 
                    "ensure_safety"
                ],
                "progress": progress,
                "metadata": {
                    "monitoring_update": update + 1,
                    "fire_status": "contained" if progress > 0.5 else "spreading",
                    "suppression_effectiveness": min(1.0, progress * 2),
                    "safety_status": "all_clear"
                }
            }
            
            try:
                response = requests.post(f"{FABRIC_URL}/missions", json=mission_update, timeout=5)
                if response.status_code == 200:
                    print(f"   📈 Progress Update: {progress:.1%} complete")
                    print(f"   🔥 Fire Status: {mission_update['metadata']['fire_status']}")
                    print(f"   🛡️  Suppression: {mission_update['metadata']['suppression_effectiveness']:.1%}")
            except Exception as e:
                print(f"   ❌ Failed to update mission: {e}")
            
            # Simulate additional detections
            if update == 1:  # Second update
                additional_detection = {
                    "detection_id": f"det-{int(time.time())}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "location": {
                        "latitude": self.fire_location["latitude"] + 0.0005,
                        "longitude": self.fire_location["longitude"] + 0.0005,
                        "altitude": 0.0
                    },
                    "object_type": "smoke",
                    "confidence": 0.85,
                    "metadata": {
                        "detection_method": "visual",
                        "device_id": "drone-001",
                        "smoke_density": "moderate"
                    }
                }
                
                try:
                    response = requests.post(f"{FUSION_URL}/detections", json=additional_detection, timeout=5)
                    if response.status_code == 200:
                        print(f"   🔍 Additional Detection: Smoke at {additional_detection['location']['latitude']:.4f}, {additional_detection['location']['longitude']:.4f}")
                except Exception as e:
                    print(f"   ❌ Failed to send additional detection: {e}")
    
    async def step_6_mission_completion(self):
        """Step 6: Complete mission."""
        print("\n✅ Step 6: Mission Completion")
        print("-" * 30)
        
        # Final mission update
        mission_completion = {
            "mission_id": self.mission_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "assets": self.assets,
            "objectives": [
                "assess_fire_conditions",
                "monitor_fire_spread",
                "coordinate_suppression",
                "ensure_safety"
            ],
            "progress": 1.0,
            "metadata": {
                "completion_status": "successful",
                "fire_contained": True,
                "suppression_effective": True,
                "no_casualties": True,
                "assets_returned": True,
                "final_report": {
                    "fire_size": "50m",
                    "containment_time": "25 minutes",
                    "resources_used": ["drone-001", "drone-002", "ugv-001"],
                    "effectiveness": "high"
                }
            }
        }
        
        try:
            response = requests.post(f"{FABRIC_URL}/missions", json=mission_completion, timeout=5)
            if response.status_code == 200:
                print(f"✅ Mission completed: {self.mission_id}")
                print(f"   Status: {mission_completion['status']}")
                print(f"   Fire Contained: {mission_completion['metadata']['fire_contained']}")
                print(f"   Suppression Effective: {mission_completion['metadata']['suppression_effective']}")
                print(f"   No Casualties: {mission_completion['metadata']['no_casualties']}")
            else:
                print(f"❌ Failed to complete mission: {response.status_code}")
        except Exception as e:
            print(f"❌ Error completing mission: {e}")

async def main():
    """Main entry point."""
    import random
    import math
    
    demo = DemoMission()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main())
