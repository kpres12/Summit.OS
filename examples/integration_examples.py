#!/usr/bin/env python3
"""
Summit.OS Integration Examples

Practical examples showing how to integrate Summit.OS with various
platforms, robots, and systems.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any
import requests
import paho.mqtt.client as mqtt
import websocket
import numpy as np

# Summit.OS SDK (mock implementation)
class SummitClient:
    """Summit.OS Python SDK client."""
    
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    async def publish_telemetry(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Publish telemetry data to Summit.OS."""
        response = self.session.post(
            f"{self.base_url}/api/v1/telemetry",
            json=telemetry
        )
        return response.json()
    
    async def publish_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Publish alert to Summit.OS."""
        response = self.session.post(
            f"{self.base_url}/api/v1/alerts",
            json=alert
        )
        return response.json()
    
    async def get_alerts(self, severity: str = None) -> List[Dict[str, Any]]:
        """Get alerts from Summit.OS."""
        params = {'severity': severity} if severity else {}
        response = self.session.get(
            f"{self.base_url}/api/v1/alerts",
            params=params
        )
        return response.json()
    
    async def create_mission(self, mission: Dict[str, Any]) -> Dict[str, Any]:
        """Create mission in Summit.OS."""
        response = self.session.post(
            f"{self.base_url}/api/v1/missions",
            json=mission
        )
        return response.json()


class SummitMQTTClient:
    """Summit.OS MQTT client for IoT devices."""
    
    def __init__(self, broker: str = "localhost", port: int = 1883, 
                 device_id: str = "device-001"):
        self.device_id = device_id
        self.client = mqtt.Client(client_id=device_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(broker, port, 60)
        self.client.loop_start()
        
        self.command_handlers = {}
    
    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected to Summit.OS MQTT: {rc}")
        # Subscribe to commands for this device
        client.subscribe(f"summit-os/devices/{self.device_id}/commands/+")
    
    def on_message(self, client, userdata, msg):
        """Handle incoming commands."""
        try:
            command = json.loads(msg.payload.decode())
            self.handle_command(command)
        except Exception as e:
            print(f"Error handling command: {e}")
    
    def handle_command(self, command: Dict[str, Any]):
        """Handle command from Summit.OS."""
        command_type = command.get('type')
        if command_type in self.command_handlers:
            self.command_handlers[command_type](command)
        else:
            print(f"Unknown command type: {command_type}")
    
    def register_command_handler(self, command_type: str, handler):
        """Register command handler."""
        self.command_handlers[command_type] = handler
    
    def publish_telemetry(self, telemetry: Dict[str, Any]):
        """Publish telemetry to Summit.OS."""
        topic = f"summit-os/devices/{self.device_id}/telemetry"
        self.client.publish(topic, json.dumps(telemetry))
    
    def publish_alert(self, alert: Dict[str, Any]):
        """Publish alert to Summit.OS."""
        topic = f"summit-os/alerts/{alert.get('category', 'general')}"
        self.client.publish(topic, json.dumps(alert))


class SummitWebSocketClient:
    """Summit.OS WebSocket client for real-time applications."""
    
    def __init__(self, url: str = "ws://localhost:8001/ws", api_key: str = "your-key"):
        self.url = url
        self.api_key = api_key
        self.ws = None
        self.message_handlers = {}
    
    def connect(self):
        """Connect to Summit.OS WebSocket."""
        self.ws = websocket.WebSocketApp(
            self.url,
            header=[f"Authorization: Bearer {self.api_key}"],
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws.run_forever()
    
    def on_open(self, ws):
        print("Connected to Summit.OS WebSocket")
    
    def on_message(self, ws, message):
        """Handle incoming messages."""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            if message_type in self.message_handlers:
                self.message_handlers[message_type](data)
        except Exception as e:
            print(f"Error handling message: {e}")
    
    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        print("Disconnected from Summit.OS WebSocket")
    
    def register_message_handler(self, message_type: str, handler):
        """Register message handler."""
        self.message_handlers[message_type] = handler
    
    def send_message(self, message: Dict[str, Any]):
        """Send message to Summit.OS."""
        if self.ws:
            self.ws.send(json.dumps(message))


# Example 1: Basic API Integration
async def example_basic_api_integration():
    """Example: Basic API integration with Summit.OS."""
    print("🔌 Example 1: Basic API Integration")
    
    # Initialize Summit.OS client
    client = SummitClient(api_key="your-api-key")
    
    # Publish telemetry
    telemetry = {
        "device_id": "drone-001",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "location": {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "altitude": 100.0
        },
        "sensors": {
            "temperature": 25.5,
            "humidity": 45.2,
            "battery_level": 85.0
        },
        "status": "online"
    }
    
    result = await client.publish_telemetry(telemetry)
    print(f"Telemetry published: {result}")
    
    # Get alerts
    alerts = await client.get_alerts(severity="high")
    print(f"High severity alerts: {len(alerts)}")
    
    # Create mission
    mission = {
        "mission_id": "mission-001",
        "objectives": ["patrol", "detect", "report"],
        "assets": ["drone-001"],
        "priority": "medium"
    }
    
    mission_result = await client.create_mission(mission)
    print(f"Mission created: {mission_result}")


# Example 2: MQTT Integration for IoT Devices
def example_mqtt_integration():
    """Example: MQTT integration for IoT sensors."""
    print("📡 Example 2: MQTT Integration for IoT Devices")
    
    # Initialize MQTT client
    mqtt_client = SummitMQTTClient(device_id="weather-station-001")
    
    # Register command handlers
    def handle_navigate_command(command):
        print(f"Navigate command: {command}")
        # Implement navigation logic here
    
    def handle_return_home_command(command):
        print(f"Return home command: {command}")
        # Implement return home logic here
    
    mqtt_client.register_command_handler("navigate", handle_navigate_command)
    mqtt_client.register_command_handler("return_home", handle_return_home_command)
    
    # Simulate sensor readings
    def simulate_sensor_readings():
        while True:
            # Read sensors (mock data)
            temperature = 25.0 + np.random.normal(0, 2)
            humidity = 50.0 + np.random.normal(0, 5)
            pressure = 1013.25 + np.random.normal(0, 1)
            
            # Publish telemetry
            telemetry = {
                "device_id": "weather-station-001",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sensors": {
                    "temperature": temperature,
                    "humidity": humidity,
                    "pressure": pressure
                },
                "location": {
                    "latitude": 37.7749,
                    "longitude": -122.4194
                }
            }
            
            mqtt_client.publish_telemetry(telemetry)
            print(f"Published telemetry: {telemetry}")
            
            time.sleep(30)  # Send every 30 seconds
    
    # Start sensor simulation
    simulate_sensor_readings()


# Example 3: WebSocket Integration for Real-time Apps
def example_websocket_integration():
    """Example: WebSocket integration for real-time applications."""
    print("🌐 Example 3: WebSocket Integration for Real-time Apps")
    
    # Initialize WebSocket client
    ws_client = SummitWebSocketClient()
    
    # Register message handlers
    def handle_telemetry(data):
        print(f"Received telemetry: {data}")
        # Update dashboard with telemetry data
    
    def handle_alert(data):
        print(f"Received alert: {data}")
        # Show alert in UI
    
    def handle_mission(data):
        print(f"Received mission update: {data}")
        # Update mission status in UI
    
    ws_client.register_message_handler("telemetry", handle_telemetry)
    ws_client.register_message_handler("alert", handle_alert)
    ws_client.register_message_handler("mission", handle_mission)
    
    # Connect to Summit.OS
    ws_client.connect()


# Example 4: ROS 2 Integration
class ROS2SummitIntegration:
    """Example: ROS 2 integration with Summit.OS."""
    
    def __init__(self, robot_id: str = "turtlebot-001"):
        self.robot_id = robot_id
        self.summit_client = SummitClient(api_key="your-api-key")
        self.last_location = None
        
    async def gps_callback(self, gps_msg):
        """Handle GPS messages from ROS 2."""
        # Convert ROS 2 GPS message to Summit.OS format
        telemetry = {
            "device_id": self.robot_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "location": {
                "latitude": gps_msg.latitude,
                "longitude": gps_msg.longitude,
                "altitude": gps_msg.altitude
            },
            "sensors": {
                "gps_quality": gps_msg.status.status
            }
        }
        
        # Send to Summit.OS
        await self.summit_client.publish_telemetry(telemetry)
        self.last_location = (gps_msg.latitude, gps_msg.longitude)
    
    async def camera_callback(self, image_msg):
        """Handle camera messages from ROS 2."""
        # Process image for fire detection (mock)
        fire_detected = self.detect_fire_in_image(image_msg)
        
        if fire_detected and self.last_location:
            alert = {
                "alert_id": f"fire-{int(time.time())}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "severity": "high",
                "location": {
                    "latitude": self.last_location[0],
                    "longitude": self.last_location[1]
                },
                "description": "Fire detected in camera view",
                "source": self.robot_id
            }
            
            await self.summit_client.publish_alert(alert)
    
    def detect_fire_in_image(self, image_msg):
        """Mock fire detection in image."""
        # In real implementation, this would use computer vision
        return np.random.random() > 0.9  # 10% chance of fire detection
    
    async def handle_summit_command(self, command):
        """Handle commands from Summit.OS."""
        if command['type'] == 'navigate':
            target_location = command['target_location']
            # Convert Summit.OS location to ROS 2 navigation goal
            # This would publish to ROS 2 navigation stack
            print(f"Navigating to: {target_location}")
        
        elif command['type'] == 'return_home':
            # Return to home position
            print("Returning home")
        
        elif command['type'] == 'emergency_stop':
            # Emergency stop
            print("Emergency stop activated")


# Example 5: DJI Drone Integration
class DJISummitIntegration:
    """Example: DJI drone integration with Summit.OS."""
    
    def __init__(self, drone_id: str = "dji-001"):
        self.drone_id = drone_id
        self.summit_client = SummitClient(api_key="your-api-key")
        # In real implementation, this would use DJI SDK
        self.drone_connected = False
    
    async def connect_drone(self):
        """Connect to DJI drone."""
        # Mock drone connection
        self.drone_connected = True
        print(f"Connected to DJI drone: {self.drone_id}")
    
    async def takeoff(self):
        """Take off drone."""
        if self.drone_connected:
            print("Drone taking off...")
            # In real implementation, this would send takeoff command to DJI SDK
            await asyncio.sleep(2)
            print("Drone airborne")
    
    async def navigate_to_waypoint(self, waypoint):
        """Navigate to waypoint."""
        if self.drone_connected:
            print(f"Navigating to waypoint: {waypoint}")
            # In real implementation, this would send navigation command to DJI SDK
            await asyncio.sleep(5)
            print("Reached waypoint")
    
    async def capture_image(self):
        """Capture image with drone camera."""
        if self.drone_connected:
            print("Capturing image...")
            # In real implementation, this would capture image from DJI camera
            # Mock image data
            image_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            return image_data
    
    async def analyze_image(self, image):
        """Analyze image for fire detection."""
        # Mock fire detection analysis
        fire_confidence = np.random.random()
        return {
            "fire_detected": fire_confidence > 0.7,
            "confidence": fire_confidence,
            "analysis_time": datetime.now(timezone.utc).isoformat()
        }
    
    async def execute_mission(self, mission):
        """Execute Summit.OS mission with DJI drone."""
        print(f"Executing mission: {mission['mission_id']}")
        
        # Connect to drone
        await self.connect_drone()
        
        # Take off
        await self.takeoff()
        
        # Execute waypoints
        for waypoint in mission.get('waypoints', []):
            await self.navigate_to_waypoint(waypoint)
            
            # Capture and analyze image
            image = await self.capture_image()
            analysis = await self.analyze_image(image)
            
            # Send analysis to Summit.OS
            if analysis['fire_detected']:
                alert = {
                    "alert_id": f"fire-{int(time.time())}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "severity": "high",
                    "location": waypoint,
                    "description": f"Fire detected with {analysis['confidence']:.2f} confidence",
                    "source": self.drone_id
                }
                await self.summit_client.publish_alert(alert)
        
        # Return home
        print("Returning home...")
        await self.navigate_to_waypoint(mission.get('home_location', {"lat": 37.7749, "lon": -122.4194}))
        
        # Land
        print("Landing...")
        print("Mission completed")


# Example 6: Edge Device Integration
class EdgeSummitIntegration:
    """Example: Edge device integration with Summit.OS."""
    
    def __init__(self, device_id: str = "edge-device-001"):
        self.device_id = device_id
        self.summit_client = SummitClient(api_key="your-api-key")
        self.data_buffer = []
        self.connected = False
        
        # Mock AI models for edge inference
        self.fire_detector = MockFireDetector()
        self.smoke_detector = MockSmokeDetector()
    
    async def process_sensor_data(self, sensor_data):
        """Process sensor data with edge AI."""
        print(f"Processing sensor data: {sensor_data['sensor_type']}")
        
        # Edge AI inference
        if sensor_data['sensor_type'] == 'thermal_camera':
            fire_result = self.fire_detector.detect(sensor_data['data'])
            if fire_result['confidence'] > 0.8:
                # Critical detection - send immediately
                alert = {
                    "alert_id": f"fire-{int(time.time())}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "severity": "critical",
                    "location": sensor_data['location'],
                    "description": f"Fire detected with {fire_result['confidence']:.2f} confidence",
                    "source": self.device_id
                }
                await self.summit_client.publish_alert(alert)
        
        elif sensor_data['sensor_type'] == 'rgb_camera':
            smoke_result = self.smoke_detector.detect(sensor_data['data'])
            if smoke_result['confidence'] > 0.6:
                alert = {
                    "alert_id": f"smoke-{int(time.time())}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "severity": "medium",
                    "location": sensor_data['location'],
                    "description": f"Smoke detected with {smoke_result['confidence']:.2f} confidence",
                    "source": self.device_id
                }
                await self.summit_client.publish_alert(alert)
        
        # Buffer telemetry for batch sending
        self.data_buffer.append(sensor_data)
        
        # Send batch when buffer is full or connection is available
        if len(self.data_buffer) >= 10 or self.connected:
            await self.send_telemetry_batch()
    
    async def send_telemetry_batch(self):
        """Send buffered telemetry data."""
        if self.data_buffer:
            telemetry_batch = {
                "device_id": self.device_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "batch_data": self.data_buffer
            }
            
            try:
                await self.summit_client.publish_telemetry(telemetry_batch)
                print(f"Sent telemetry batch: {len(self.data_buffer)} items")
                self.data_buffer = []
            except Exception as e:
                print(f"Failed to send telemetry batch: {e}")
                # Keep data in buffer for retry


class MockFireDetector:
    """Mock fire detector for edge inference."""
    
    def detect(self, image_data):
        # Mock fire detection
        confidence = np.random.random()
        return {
            "fire_detected": confidence > 0.7,
            "confidence": confidence,
            "bounding_box": [100, 100, 200, 200] if confidence > 0.7 else None
        }


class MockSmokeDetector:
    """Mock smoke detector for edge inference."""
    
    def detect(self, image_data):
        # Mock smoke detection
        confidence = np.random.random()
        return {
            "smoke_detected": confidence > 0.5,
            "confidence": confidence,
            "bounding_box": [150, 150, 250, 250] if confidence > 0.5 else None
        }


# Example 7: Multi-Robot Coordination
class MultiRobotSummitIntegration:
    """Example: Multi-robot coordination with Summit.OS."""
    
    def __init__(self):
        self.robots = {
            "drone-001": SummitClient(api_key="your-key"),
            "drone-002": SummitClient(api_key="your-key"),
            "ugv-001": SummitClient(api_key="your-key")
        }
        self.mission_coordinator = MissionCoordinator()
    
    async def coordinate_mission(self, mission):
        """Coordinate multiple robots for a mission."""
        print(f"Coordinating mission: {mission['mission_id']}")
        
        # Assign tasks to robots
        task_assignments = self.mission_coordinator.assign_tasks(mission, self.robots.keys())
        
        # Execute tasks in parallel
        tasks = []
        for robot_id, tasks in task_assignments.items():
            for task in tasks:
                task_coroutine = self.execute_robot_task(robot_id, task)
                tasks.append(task_coroutine)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Aggregate results
        mission_result = {
            "mission_id": mission['mission_id'],
            "status": "completed",
            "results": results,
            "completion_time": datetime.now(timezone.utc).isoformat()
        }
        
        return mission_result
    
    async def execute_robot_task(self, robot_id, task):
        """Execute a task with a specific robot."""
        print(f"Robot {robot_id} executing task: {task['type']}")
        
        # Mock task execution
        await asyncio.sleep(2)
        
        return {
            "robot_id": robot_id,
            "task_id": task['task_id'],
            "status": "completed",
            "result": f"Task {task['type']} completed by {robot_id}"
        }


class MissionCoordinator:
    """Coordinates missions across multiple robots."""
    
    def assign_tasks(self, mission, robot_ids):
        """Assign tasks to robots based on capabilities."""
        task_assignments = {robot_id: [] for robot_id in robot_ids}
        
        # Simple task assignment logic
        for i, objective in enumerate(mission['objectives']):
            robot_id = list(robot_ids)[i % len(robot_ids)]
            task = {
                "task_id": f"task-{i}",
                "type": objective,
                "priority": "medium"
            }
            task_assignments[robot_id].append(task)
        
        return task_assignments


# Main execution
async def main():
    """Run all integration examples."""
    print("🚀 Summit.OS Integration Examples")
    print("=" * 50)
    
    # Example 1: Basic API Integration
    await example_basic_api_integration()
    print()
    
    # Example 2: MQTT Integration
    print("Starting MQTT integration example...")
    # Note: This would run in a separate thread in practice
    # example_mqtt_integration()
    print()
    
    # Example 3: WebSocket Integration
    print("Starting WebSocket integration example...")
    # Note: This would run in a separate thread in practice
    # example_websocket_integration()
    print()
    
    # Example 4: ROS 2 Integration
    print("🔧 Example 4: ROS 2 Integration")
    ros2_integration = ROS2SummitIntegration("turtlebot-001")
    print("ROS 2 integration initialized")
    print()
    
    # Example 5: DJI Drone Integration
    print("🚁 Example 5: DJI Drone Integration")
    dji_integration = DJISummitIntegration("dji-001")
    
    # Mock mission
    mission = {
        "mission_id": "patrol-mission-001",
        "waypoints": [
            {"lat": 37.7749, "lon": -122.4194},
            {"lat": 37.7849, "lon": -122.4094},
            {"lat": 37.7649, "lon": -122.4294}
        ],
        "home_location": {"lat": 37.7749, "lon": -122.4194}
    }
    
    await dji_integration.execute_mission(mission)
    print()
    
    # Example 6: Edge Device Integration
    print("🔧 Example 6: Edge Device Integration")
    edge_integration = EdgeSummitIntegration("edge-device-001")
    
    # Mock sensor data
    sensor_data = {
        "sensor_type": "thermal_camera",
        "location": {"lat": 37.7749, "lon": -122.4194},
        "data": np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    }
    
    await edge_integration.process_sensor_data(sensor_data)
    print()
    
    # Example 7: Multi-Robot Coordination
    print("🤖 Example 7: Multi-Robot Coordination")
    multi_robot = MultiRobotSummitIntegration()
    
    mission = {
        "mission_id": "multi-robot-mission-001",
        "objectives": ["patrol", "detect", "suppress", "verify"],
        "assets": ["drone-001", "drone-002", "ugv-001"]
    }
    
    result = await multi_robot.coordinate_mission(mission)
    print(f"Multi-robot mission result: {result}")
    
    print("\n✅ All integration examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
