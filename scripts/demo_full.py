#!/usr/bin/env python3
"""
Summit.OS Comprehensive Demo Mission

Demonstrates the full intelligence fabric flow:
1. Publishes realistic observations (smoke/fire) via MQTT
2. Shows Fabric → Fusion → Intelligence → Tasking pipeline
3. Demonstrates policy gate for high-risk tasks
4. Queries all services to show data flow
5. Displays real-time statistics
"""

import asyncio
import json
import random
import sys
import time
from datetime import datetime, timezone
from typing import List, Dict, Any

import httpx
import paho.mqtt.client as mqtt

# Configuration
API_GATEWAY_URL = "http://localhost:8000"
FUSION_URL = "http://localhost:8002"
INTELLIGENCE_URL = "http://localhost:8003"
TASKING_URL = "http://localhost:8004"
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# Demo scenario: Wildfire detection mission
DEMO_REGION = {
    "name": "Sierra Nevada Test Zone",
    "center": {"lat": 37.7749, "lon": -119.4194},
    "radius_km": 10
}

# Simulated assets (drones/cameras)
ASSETS = [
    {"id": "drone-alpha", "type": "quadcopter", "capabilities": ["thermal", "rgb"]},
    {"id": "drone-bravo", "type": "fixed-wing", "capabilities": ["thermal", "multispectral"]},
    {"id": "camera-tower-01", "type": "tower", "capabilities": ["rgb", "ir"]},
    {"id": "camera-tower-02", "type": "tower", "capabilities": ["thermal"]},
]

# Observation scenarios (escalating severity)
SCENARIOS = [
    {
        "class": "smoke",
        "confidence": 0.55,
        "severity": "LOW",
        "message": "Possible smoke plume detected",
        "temp_range": (15, 45)
    },
    {
        "class": "smoke",
        "confidence": 0.75,
        "severity": "MEDIUM",
        "message": "Smoke plume confirmed",
        "temp_range": (45, 150)
    },
    {
        "class": "ignition_point",
        "confidence": 0.85,
        "severity": "HIGH",
        "message": "Active ignition point detected",
        "temp_range": (300, 500)
    },
    {
        "class": "active_fire",
        "confidence": 0.92,
        "severity": "HIGH",
        "message": "Active fire spreading",
        "temp_range": (500, 800)
    },
    {
        "class": "active_fire",
        "confidence": 0.95,
        "severity": "CRITICAL",
        "message": "Large wildfire - immediate response required",
        "temp_range": (800, 1200)
    }
]


class DemoOrchestrator:
    """Orchestrates the Summit.OS demo mission."""
    
    def __init__(self):
        self.mqtt_client = mqtt.Client(client_id="demo_mission")
        self.mqtt_connected = False
        self.published_count = 0
        self.observation_ids = []
    
    def on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            self.mqtt_connected = True
            print("✅ Connected to MQTT broker")
        else:
            print(f"❌ Failed to connect to MQTT broker: {rc}")
    
    async def connect_mqtt(self):
        """Connect to MQTT broker."""
        print(f"\n📡 Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
        self.mqtt_client.on_connect = self.on_connect
        
        try:
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
            
            # Wait for connection
            for _ in range(10):
                if self.mqtt_connected:
                    return True
                await asyncio.sleep(0.5)
            
            print("❌ MQTT connection timeout")
            return False
        except Exception as e:
            print(f"❌ MQTT connection error: {e}")
            return False
    
    def generate_observation(self, scenario: Dict[str, Any], asset: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a realistic observation."""
        # Random location within demo region
        lat_offset = random.uniform(-0.1, 0.1)
        lon_offset = random.uniform(-0.1, 0.1)
        
        observation = {
            "class": scenario["class"],
            "confidence": scenario["confidence"] + random.uniform(-0.05, 0.05),
            "lat": DEMO_REGION["center"]["lat"] + lat_offset,
            "lon": DEMO_REGION["center"]["lon"] + lon_offset,
            "ts_iso": datetime.now(timezone.utc).isoformat(),
            "source": asset["id"],
            "attributes": {
                "asset_type": asset["type"],
                "capabilities": asset["capabilities"],
                "severity": scenario["severity"],
                "message": scenario["message"],
                "temperature_c": random.uniform(*scenario["temp_range"])
            }
        }
        
        # Ensure confidence is in valid range
        observation["confidence"] = max(0.0, min(1.0, observation["confidence"]))
        
        return observation
    
    def publish_observation(self, observation: Dict[str, Any]) -> bool:
        """Publish observation to MQTT."""
        topic = f"observations/{observation['class']}"
        payload = json.dumps(observation)
        
        try:
            result = self.mqtt_client.publish(topic, payload, qos=1)
            result.wait_for_publish()
            
            if result.is_published():
                self.published_count += 1
                return True
            return False
        except Exception as e:
            print(f"❌ Failed to publish observation: {e}")
            return False
    
    async def check_api_health(self) -> bool:
        """Check if API Gateway is healthy."""
        print("\n🏥 Checking API Gateway health...")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{API_GATEWAY_URL}/health")
                if response.status_code == 200:
                    print("✅ API Gateway is healthy")
                    return True
                else:
                    print(f"❌ API Gateway returned {response.status_code}")
                    return False
        except Exception as e:
            print(f"❌ Cannot reach API Gateway: {e}")
            print(f"   Make sure services are running: make dev")
            return False
    
    async def query_observations(self) -> List[Dict[str, Any]]:
        """Query observations from Fusion service."""
        print("\n🔍 Querying observations from Fusion...")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{FUSION_URL}/observations?limit=20")
                if response.status_code == 200:
                    observations = response.json()
                    print(f"✅ Retrieved {len(observations)} observations")
                    return observations
                else:
                    print(f"⚠️  Fusion returned {response.status_code}")
                    return []
        except Exception as e:
            print(f"❌ Failed to query observations: {e}")
            return []
    
    async def query_advisories(self) -> List[Dict[str, Any]]:
        """Query advisories from Intelligence service."""
        print("\n🧠 Querying advisories from Intelligence...")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{INTELLIGENCE_URL}/advisories?limit=20")
                if response.status_code == 200:
                    advisories = response.json()
                    print(f"✅ Retrieved {len(advisories)} advisories")
                    return advisories
                else:
                    print(f"⚠️  Intelligence returned {response.status_code}")
                    return []
        except Exception as e:
            print(f"❌ Failed to query advisories: {e}")
            return []
    
    async def query_tasks(self) -> List[Dict[str, Any]]:
        """Query tasks from Tasking service."""
        print("\n📋 Querying tasks from Tasking...")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{TASKING_URL}/tasks?limit=20")
                if response.status_code == 200:
                    tasks = response.json()
                    print(f"✅ Retrieved {len(tasks)} tasks")
                    return tasks
                else:
                    print(f"⚠️  Tasking returned {response.status_code}")
                    return []
        except Exception as e:
            print(f"❌ Failed to query tasks: {e}")
            return []
    
    async def submit_task(self, asset_id: str, action: str, risk_level: str, waypoints: List = None) -> Dict[str, Any]:
        """Submit a task through API Gateway."""
        print(f"\n🎯 Submitting {risk_level} risk task: {action}")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{API_GATEWAY_URL}/v1/tasks",
                    json={
                        "asset_id": asset_id,
                        "action": action,
                        "risk_level": risk_level,
                        "waypoints": waypoints or []
                    }
                )
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Task submitted: {result['task_id']} - Status: {result['status']}")
                    return result
                else:
                    print(f"❌ Failed to submit task: {response.status_code}")
                    return {}
        except Exception as e:
            print(f"❌ Task submission error: {e}")
            return {}
    
    async def query_pending_approvals(self) -> List[Dict[str, Any]]:
        """Query pending approvals."""
        print("\n⏳ Querying pending task approvals...")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{API_GATEWAY_URL}/v1/tasks/pending")
                if response.status_code == 200:
                    pending = response.json()
                    print(f"✅ {len(pending)} tasks pending approval")
                    return pending
                else:
                    print(f"⚠️  API returned {response.status_code}")
                    return []
        except Exception as e:
            print(f"❌ Failed to query pending approvals: {e}")
            return []
    
    async def approve_task(self, task_id: str, approved_by: str = "demo_operator") -> bool:
        """Approve a pending task."""
        print(f"\n✅ Approving task {task_id}...")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{API_GATEWAY_URL}/v1/tasks/{task_id}/approve",
                    json={"approved_by": approved_by}
                )
                if response.status_code == 200:
                    print(f"✅ Task approved and dispatched")
                    return True
                else:
                    print(f"❌ Failed to approve task: {response.status_code}")
                    return False
        except Exception as e:
            print(f"❌ Approval error: {e}")
            return False
    
    def display_summary(self, observations: List, advisories: List, tasks: List):
        """Display demo summary."""
        print("\n" + "="*80)
        print("📊 SUMMIT.OS DEMO SUMMARY")
        print("="*80)
        
        print(f"\n📡 MQTT Messages Published: {self.published_count}")
        
        print(f"\n🔍 OBSERVATIONS (Fusion): {len(observations)}")
        if observations:
            class_counts = {}
            for obs in observations[:10]:
                cls = obs.get("cls", "unknown")
                class_counts[cls] = class_counts.get(cls, 0) + 1
                conf = obs.get("confidence", 0)
                src = obs.get("source", "unknown")
                print(f"   • {cls} (conf: {conf:.2f}) from {src}")
            print(f"\n   Class distribution: {class_counts}")
        
        print(f"\n🧠 ADVISORIES (Intelligence): {len(advisories)}")
        if advisories:
            risk_counts = {}
            for adv in advisories[:10]:
                risk = adv.get("risk_level", "UNKNOWN")
                risk_counts[risk] = risk_counts.get(risk, 0) + 1
                msg = adv.get("message", "N/A")
                print(f"   • [{risk}] {msg}")
            print(f"\n   Risk distribution: {risk_counts}")
        
        print(f"\n📋 TASKS (Tasking): {len(tasks)}")
        if tasks:
            status_counts = {}
            for task in tasks[:10]:
                status = task.get("status", "UNKNOWN")
                status_counts[status] = status_counts.get(status, 0) + 1
                action = task.get("action", "N/A")
                asset = task.get("asset_id", "N/A")
                print(f"   • {action} → {asset} [{status}]")
            print(f"\n   Status distribution: {status_counts}")
        
        print("\n" + "="*80)
        print("🎉 Demo Complete!")
        print("="*80)
        print("\n💡 Next Steps:")
        print("   • Open Console UI: http://localhost:3000/observations")
        print("   • View Grafana: http://localhost:3001")
        print("   • Check logs: make logs")
        print("   • Run integration test: pytest tests/test_observation_flow.py -v")
        print()
    
    async def run_demo(self):
        """Run the complete demo sequence."""
        print("="*80)
        print("🚀 SUMMIT.OS COMPREHENSIVE DEMONSTRATION")
        print("="*80)
        print(f"\n📍 Region: {DEMO_REGION['name']}")
        print(f"🎯 Scenario: Wildfire Detection and Response")
        print(f"🤖 Assets: {len(ASSETS)} deployed")
        print(f"📦 Scenarios: {len(SCENARIOS)} (escalating severity)")
        
        # Check API health
        if not await self.check_api_health():
            print("\n⚠️  API Gateway not available. Run 'make dev' to start services.")
            return False
        
        # Connect to MQTT
        if not await self.connect_mqtt():
            print("\n⚠️  MQTT broker not available.")
            return False
        
        # Phase 1: Publish observations
        print("\n" + "="*80)
        print("📡 PHASE 1: Publishing Observations (Escalating Scenario)")
        print("="*80)
        
        for i, scenario in enumerate(SCENARIOS):
            asset = random.choice(ASSETS)
            observation = self.generate_observation(scenario, asset)
            
            print(f"\n[{i+1}/{len(SCENARIOS)}] Publishing {scenario['class']} observation...")
            print(f"   Source: {asset['id']} ({asset['type']})")
            print(f"   Confidence: {observation['confidence']:.2f}")
            print(f"   Severity: {scenario['severity']}")
            print(f"   Temperature: {observation['attributes']['temperature_c']:.1f}°C")
            print(f"   Location: ({observation['lat']:.4f}, {observation['lon']:.4f})")
            
            if self.publish_observation(observation):
                print(f"   ✅ Published to MQTT topic: observations/{scenario['class']}")
            else:
                print(f"   ❌ Failed to publish")
            
            # Small delay between observations for realism
            await asyncio.sleep(1.5)
        
        # Wait for processing
        print("\n⏳ Waiting for pipeline processing (Fabric→Fusion→Intelligence)...")
        await asyncio.sleep(5)
        
        # Phase 2: Query results
        print("\n" + "="*80)
        print("🔍 PHASE 2: Querying Pipeline Results")
        print("="*80)
        
        observations = await self.query_observations()
        advisories = await self.query_advisories()
        tasks_before = await self.query_tasks()
        
        # Phase 3: Task submission and approval
        print("\n" + "="*80)
        print("🎯 PHASE 3: Task Submission & Policy Gate")
        print("="*80)
        
        # Submit LOW-risk task (auto-approved)
        print("\n📌 Submitting LOW-risk reconnaissance task...")
        low_risk_result = await self.submit_task(
            asset_id="drone-alpha",
            action="INVESTIGATE_SMOKE_PLUME",
            risk_level="LOW"
        )
        
        await asyncio.sleep(1)
        
        # Submit MEDIUM-risk task (auto-approved)
        print("\n📌 Submitting MEDIUM-risk monitoring task...")
        med_risk_result = await self.submit_task(
            asset_id="camera-tower-01",
            action="CONTINUOUS_MONITORING",
            risk_level="MEDIUM"
        )
        
        await asyncio.sleep(1)
        
        # Submit HIGH-risk task (requires approval)
        print("\n📌 Submitting HIGH-risk fire zone entry task...")
        high_risk_result = await self.submit_task(
            asset_id="drone-bravo",
            action="ENTER_ACTIVE_FIRE_ZONE",
            risk_level="HIGH"
        )
        
        await asyncio.sleep(1)
        
        # Submit CRITICAL-risk task (requires approval)
        print("\n📌 Submitting CRITICAL-risk suppression task...")
        critical_risk_result = await self.submit_task(
            asset_id="drone-alpha",
            action="DEPLOY_FIRE_RETARDANT",
            risk_level="CRITICAL"
        )
        
        await asyncio.sleep(2)
        
        # Query pending approvals
        pending = await self.query_pending_approvals()
        
        # Approve tasks requiring human approval
        if pending:
            print(f"\n✋ Found {len(pending)} tasks requiring approval")
            for task in pending[:2]:  # Approve first 2
                print(f"\n   Reviewing task {task['task_id']}:")
                print(f"   • Asset: {task['asset_id']}")
                print(f"   • Action: {task['action']}")
                print(f"   • Risk: {task['risk_level']}")
                await self.approve_task(task['task_id'], approved_by="demo_operator")
                await asyncio.sleep(1)
        
        # Wait for task dispatch
        print("\n⏳ Waiting for task dispatch...")
        await asyncio.sleep(3)
        
        # Final query
        tasks_after = await self.query_tasks()
        
        # Display summary
        self.display_summary(observations, advisories, tasks_after)
        
        # Cleanup
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        
        return True


async def main():
    """Main entry point."""
    orchestrator = DemoOrchestrator()
    
    try:
        success = await orchestrator.run_demo()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
