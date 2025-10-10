# Summit.OS Integration Guide

## Overview

Summit.OS is designed as a **distributed intelligence fabric** that can integrate with any platform, robot, or system. This guide covers all integration patterns, from simple sensor connections to complex multi-robot coordination.

## 🏗️ Integration Architecture

### Core Integration Principles

1. **API-First Design** - All capabilities exposed via REST/gRPC/WebSocket APIs
2. **Edge-to-Cloud** - Seamless operation from edge devices to cloud infrastructure
3. **Protocol Agnostic** - Support for MQTT, gRPC, HTTP, WebSocket, ROS 2
4. **Language Agnostic** - SDKs for Python, JavaScript, C++, Go, Rust
5. **Real-time Streaming** - Live data flows and event-driven architecture

### Integration Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                         │
│              (FireLine, DitchBot, OilfieldBot, etc.)       │
└─────────────────────▲───────────────────────────────────────┘
                      │ REST/gRPC/WebSocket APIs
┌─────────────────────┴───────────────────────────────────────┐
│                 Summit.OS Core                              │
│ ┌───────────────┬─────────────────────────────────────────┐ │
│ │ API Gateway   │ Unified REST/gRPC interface              │ │
│ │ Data Fabric   │ MQTT + Redis Streams + WebSocket        │ │
│ │ AI Services   │ Fusion, Intelligence, Tasking, Predict  │ │
│ │ Edge Runtime  │ ONNX inference + store-and-forward      │ │
│ └───────────────┴─────────────────────────────────────────┘ │
└─────────────────────▲───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│              Integration Adapters                           │
│ ┌───────────────┬─────────────────────────────────────────┐ │
│ │ ROS 2 Bridge  │ ROS 2 ↔ Summit.OS translation          │ │
│ │ IoT Gateway   │ MQTT/CoAP/HTTP sensor integration      │ │
│ │ Cloud Bridge  │ AWS/GCP/Azure cloud integration        │ │
│ │ Legacy Bridge │ Legacy system integration               │ │
│ └───────────────┴─────────────────────────────────────────┘ │
└─────────────────────▲───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│              Your Robots & Systems                         │
│ ┌───────────────┬─────────────────────────────────────────┐ │
│ │ Drones/UAVs   │ DJI, ArduPilot, PX4, custom            │ │
│ │ Ground Robots │ ROS 2, custom platforms                │ │
│ │ IoT Sensors   │ Weather stations, cameras, lidars     │ │
│ │ Legacy Systems│ Existing infrastructure                 │ │
│ └───────────────┴─────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🔌 Integration Patterns

### 1. Direct API Integration

**Best for**: New applications, web services, mobile apps

```python
from summit_os import SummitClient

# Initialize client
client = SummitClient(
    api_key="your-api-key",
    base_url="https://api.summit-os.bigmt.ai"
)

# Publish telemetry
await client.fabric.publish_telemetry({
    "device_id": "drone-001",
    "location": {"latitude": 37.7749, "longitude": -122.4194},
    "sensors": {"temperature": 25.5, "humidity": 45.2}
})

# Get intelligence alerts
alerts = await client.intelligence.get_alerts(severity="high")

# Create mission
mission = await client.tasking.create_mission({
    "objectives": ["patrol", "detect", "suppress"],
    "assets": ["drone-001", "ugv-002"]
})
```

### 2. ROS 2 Integration

**Best for**: ROS 2 robots, existing robotics systems

```python
# ROS 2 Summit.OS Bridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Image, Temperature
from summit_os_ros2 import SummitOSBridge

class RobotNode(Node):
    def __init__(self):
        super().__init__('robot_node')
        
        # Initialize Summit.OS bridge
        self.summit_bridge = SummitOSBridge(
            robot_id="ugv-001",
            summit_api_url="http://localhost:8000"
        )
        
        # ROS 2 subscribers
        self.gps_sub = self.create_subscription(
            NavSatFix, '/gps/fix', self.gps_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, '/camera/image', self.camera_callback, 10)
        self.temp_sub = self.create_subscription(
            Temperature, '/sensors/temperature', self.temp_callback, 10)
    
    def gps_callback(self, msg):
        # Convert ROS 2 message to Summit.OS format
        telemetry = {
            "device_id": "ugv-001",
            "location": {
                "latitude": msg.latitude,
                "longitude": msg.longitude,
                "altitude": msg.altitude
            },
            "sensors": {"gps_quality": msg.status.status}
        }
        
        # Send to Summit.OS
        asyncio.create_task(
            self.summit_bridge.publish_telemetry(telemetry)
        )
    
    def camera_callback(self, msg):
        # Process image for fire detection
        if self.summit_bridge.detect_fire(msg):
            alert = {
                "alert_id": f"fire-{int(time.time())}",
                "severity": "high",
                "location": self.last_gps_location,
                "description": "Fire detected in camera view"
            }
            asyncio.create_task(
                self.summit_bridge.publish_alert(alert)
            )
```

### 3. MQTT Integration

**Best for**: IoT devices, sensors, lightweight systems

```python
import paho.mqtt.client as mqtt
import json

class SummitMQTTClient:
    def __init__(self, broker="mqtt.summit-os.bigmt.ai", port=1883):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(broker, port, 60)
        self.client.loop_start()
    
    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected to Summit.OS MQTT: {rc}")
        # Subscribe to commands
        client.subscribe("summit-os/devices/+/commands/+")
    
    def on_message(self, client, userdata, msg):
        # Handle commands from Summit.OS
        command = json.loads(msg.payload.decode())
        self.handle_command(command)
    
    def publish_telemetry(self, device_id, telemetry):
        topic = f"summit-os/devices/{device_id}/telemetry"
        self.client.publish(topic, json.dumps(telemetry))
    
    def publish_alert(self, device_id, alert):
        topic = f"summit-os/alerts/{alert['category']}"
        self.client.publish(topic, json.dumps(alert))

# Usage
mqtt_client = SummitMQTTClient()
mqtt_client.publish_telemetry("sensor-001", {
    "temperature": 25.5,
    "humidity": 45.2,
    "location": {"lat": 37.7749, "lon": -122.4194}
})
```

### 4. WebSocket Integration

**Best for**: Real-time applications, dashboards, live monitoring

```javascript
// JavaScript/TypeScript integration
import { SummitWebSocketClient } from '@bigmt/summit-os-sdk';

const summit = new SummitWebSocketClient({
  url: 'wss://ws.summit-os.bigmt.ai',
  apiKey: 'your-api-key'
});

// Connect to Summit.OS
await summit.connect();

// Subscribe to real-time data
summit.on('telemetry', (telemetry) => {
  console.log('New telemetry:', telemetry);
  updateDashboard(telemetry);
});

summit.on('alert', (alert) => {
  console.log('New alert:', alert);
  showAlert(alert);
});

summit.on('mission', (mission) => {
  console.log('Mission update:', mission);
  updateMissionStatus(mission);
});

// Publish data
await summit.publishTelemetry({
  deviceId: 'drone-001',
  location: { latitude: 37.7749, longitude: -122.4194 },
  sensors: { temperature: 25.5, battery: 85.0 }
});
```

### 5. Edge Integration

**Best for**: Edge devices, offline operation, local processing

```python
# Edge Summit.OS Agent
import asyncio
from summit_os_edge import EdgeAgent, ONNXInferenceEngine

class CustomEdgeAgent(EdgeAgent):
    def __init__(self, device_id, device_type):
        super().__init__(device_id, device_type)
        
        # Initialize local AI models
        self.fire_detector = ONNXInferenceEngine("models/fire_detection.onnx")
        self.smoke_detector = ONNXInferenceEngine("models/smoke_detection.onnx")
        
        # Local data buffer for offline operation
        self.data_buffer = []
    
    async def process_sensor_data(self, sensor_data):
        """Process sensor data locally before sending to Summit.OS"""
        
        # Local AI inference
        if 'thermal_image' in sensor_data:
            fire_result = self.fire_detector.infer(sensor_data['thermal_image'])
            if fire_result['confidence'] > 0.8:
                # Critical detection - send immediately
                await self.send_alert({
                    "alert_id": f"fire-{int(time.time())}",
                    "severity": "critical",
                    "confidence": fire_result['confidence'],
                    "location": sensor_data['location']
                })
        
        # Buffer for batch sending
        self.data_buffer.append(sensor_data)
        
        # Send batch when buffer is full or connection is available
        if len(self.data_buffer) >= 10 or self.is_connected():
            await self.send_telemetry_batch(self.data_buffer)
            self.data_buffer = []
    
    async def handle_command(self, command):
        """Handle commands from Summit.OS"""
        if command['type'] == 'navigate':
            await self.navigate_to(command['target_location'])
        elif command['type'] == 'return_home':
            await self.return_home()
        elif command['type'] == 'emergency_land':
            await self.emergency_land()

# Initialize edge agent
agent = CustomEdgeAgent("drone-001", "quadcopter")
await agent.start()
```

## 🤖 Platform-Specific Integrations

### DJI Drones

```python
# DJI Tello integration
from djitellopy import Tello
from summit_os_dji import DJISummitBridge

class DJISummitIntegration:
    def __init__(self):
        self.tello = Tello()
        self.summit_bridge = DJISummitBridge("tello-001")
        
    async def start_mission(self, mission):
        """Execute Summit.OS mission with DJI drone"""
        
        # Take off
        self.tello.takeoff()
        
        # Navigate to waypoints
        for waypoint in mission['waypoints']:
            await self.navigate_to_waypoint(waypoint)
            
            # Capture and analyze image
            image = self.tello.get_frame_read().frame
            analysis = await self.summit_bridge.analyze_image(image)
            
            # Send results to Summit.OS
            await self.summit_bridge.publish_analysis(analysis)
        
        # Return home
        self.tello.land()
```

### ArduPilot/PX4 Integration

```python
# ArduPilot MAVLink integration
from pymavlink import mavutil
from summit_os_mavlink import MAVLinkSummitBridge

class ArduPilotSummitIntegration:
    def __init__(self):
        self.connection = mavutil.mavlink_connection('udp:localhost:14550')
        self.summit_bridge = MAVLinkSummitBridge("ardupilot-001")
        
    async def start_autonomous_mission(self, mission):
        """Execute Summit.OS mission with ArduPilot"""
        
        # Upload mission to ArduPilot
        waypoints = self.convert_mission_to_waypoints(mission)
        self.connection.waypoint_clear_all_send()
        self.connection.waypoint_count_send(len(waypoints))
        
        for i, waypoint in enumerate(waypoints):
            self.connection.mav.mission_item_send(
                mission.target_system,
                mission.target_component,
                i,
                waypoint.frame,
                waypoint.command,
                waypoint.current,
                waypoint.autocontinue,
                waypoint.param1, waypoint.param2, waypoint.param3,
                waypoint.param4, waypoint.x, waypoint.y, waypoint.z
            )
        
        # Start mission
        self.connection.mav.command_long_send(
            mission.target_system, mission.target_component,
            mavutil.mavlink.MAV_CMD_MISSION_START, 0, 0, 0, 0, 0, 0, 0, 0
        )
```

### ROS 2 Robots

```python
# ROS 2 TurtleBot integration
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from summit_os_ros2 import ROS2SummitBridge

class TurtleBotSummitNode(Node):
    def __init__(self):
        super().__init__('turtlebot_summit_node')
        
        self.summit_bridge = ROS2SummitBridge("turtlebot-001")
        
        # ROS 2 publishers/subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        # Summit.OS command handler
        self.summit_bridge.on_command(self.handle_summit_command)
    
    def handle_summit_command(self, command):
        """Handle commands from Summit.OS"""
        if command['type'] == 'navigate':
            # Convert Summit.OS location to ROS 2 pose
            pose = self.summit_to_ros_pose(command['target_location'])
            self.goal_pub.publish(pose)
            
        elif command['type'] == 'patrol':
            # Start patrol behavior
            self.start_patrol_behavior(command['patrol_route'])
```

### IoT Sensors

```python
# Raspberry Pi sensor integration
import board
import adafruit_dht
from summit_os_iot import IoTSummitBridge

class SensorNode:
    def __init__(self):
        self.dht = adafruit_dht.DHT22(board.D4)
        self.summit_bridge = IoTSummitBridge("weather-station-001")
        
    async def read_sensors(self):
        """Read sensors and send to Summit.OS"""
        try:
            temperature = self.dht.temperature
            humidity = self.dht.humidity
            
            if temperature is not None and humidity is not None:
                await self.summit_bridge.publish_telemetry({
                    "device_id": "weather-station-001",
                    "sensors": {
                        "temperature": temperature,
                        "humidity": humidity
                    },
                    "location": {"lat": 37.7749, "lon": -122.4194}
                })
                
        except RuntimeError as e:
            print(f"Sensor error: {e}")
```

## 🌐 Cloud Platform Integrations

### AWS Integration

```python
# AWS IoT Core integration
import boto3
from summit_os_aws import AWSSummitBridge

class AWSSummitIntegration:
    def __init__(self):
        self.iot_client = boto3.client('iot-data')
        self.summit_bridge = AWSSummitBridge()
        
    async def publish_to_iot_core(self, topic, payload):
        """Publish data to AWS IoT Core"""
        response = self.iot_client.publish(
            topic=topic,
            payload=json.dumps(payload)
        )
        return response
    
    async def subscribe_to_iot_core(self, topic):
        """Subscribe to AWS IoT Core topics"""
        # This would use AWS IoT SDK for Python
        pass
```

### Azure Integration

```python
# Azure IoT Hub integration
from azure.iot.device import IoTHubDeviceClient
from summit_os_azure import AzureSummitBridge

class AzureSummitIntegration:
    def __init__(self, connection_string):
        self.client = IoTHubDeviceClient.create_from_connection_string(connection_string)
        self.summit_bridge = AzureSummitBridge()
        
    async def send_telemetry(self, telemetry):
        """Send telemetry to Azure IoT Hub"""
        await self.client.send_message(json.dumps(telemetry))
```

### Google Cloud Integration

```python
# Google Cloud IoT integration
from google.cloud import iot_v1
from summit_os_gcp import GCPSummitBridge

class GCPSummitIntegration:
    def __init__(self, project_id, registry_id, device_id):
        self.client = iot_v1.DeviceManagerClient()
        self.summit_bridge = GCPSummitBridge(project_id, registry_id, device_id)
        
    async def send_telemetry(self, telemetry):
        """Send telemetry to Google Cloud IoT"""
        # Implementation would use Google Cloud IoT SDK
        pass
```

## 🔧 Integration Tools & SDKs

### Python SDK

```bash
pip install summit-os-sdk
```

```python
from summit_os import SummitClient, EdgeAgent, ROS2Bridge

# Full-featured Python SDK
client = SummitClient(api_key="your-key")
agent = EdgeAgent(device_id="robot-001")
bridge = ROS2Bridge()
```

### JavaScript/TypeScript SDK

```bash
npm install @bigmt/summit-os-sdk
```

```javascript
import { SummitClient, WebSocketClient } from '@bigmt/summit-os-sdk';

const client = new SummitClient({ apiKey: 'your-key' });
const ws = new WebSocketClient({ url: 'wss://ws.summit-os.bigmt.ai' });
```

### C++ SDK

```cpp
#include <summit_os_cpp/summit_client.h>

SummitClient client("your-api-key");
client.publishTelemetry({
    {"device_id", "robot-001"},
    {"location", {{"lat", 37.7749}, {"lon", -122.4194}}}
});
```

### ROS 2 Package

```bash
# Install ROS 2 Summit.OS package
sudo apt install ros-humble-summit-os

# Use in your ROS 2 package
<depend>summit_os</depend>
```

## 📋 Integration Checklist

### Pre-Integration Setup

- [ ] **Obtain API Keys** - Get Summit.OS API credentials
- [ ] **Choose Integration Pattern** - Direct API, ROS 2, MQTT, WebSocket, Edge
- [ ] **Install SDK** - Install appropriate Summit.OS SDK
- [ ] **Configure Network** - Ensure network connectivity to Summit.OS
- [ ] **Test Connection** - Verify connection to Summit.OS services

### Device Registration

- [ ] **Register Device** - Register your robot/device with Summit.OS
- [ ] **Configure Capabilities** - Define device capabilities and constraints
- [ ] **Set Permissions** - Configure device permissions and access levels
- [ ] **Test Authentication** - Verify device authentication works

### Data Integration

- [ ] **Telemetry Stream** - Implement telemetry data publishing
- [ ] **Command Handling** - Implement command reception and execution
- [ ] **Alert Publishing** - Implement alert/event publishing
- [ ] **Mission Integration** - Implement mission execution capabilities

### AI Integration

- [ ] **Edge Inference** - Deploy ONNX models to edge devices
- [ ] **Sensor Fusion** - Implement multimodal sensor data fusion
- [ ] **Anomaly Detection** - Implement real-time anomaly detection
- [ ] **Risk Assessment** - Integrate risk assessment capabilities

### Testing & Validation

- [ ] **Unit Tests** - Test individual integration components
- [ ] **Integration Tests** - Test end-to-end integration
- [ ] **Performance Tests** - Validate performance under load
- [ ] **Offline Tests** - Test offline operation and store-and-forward

## 🚀 Getting Started

### Quick Start Example

```python
# 1. Install SDK
pip install summit-os-sdk

# 2. Basic integration
from summit_os import SummitClient

client = SummitClient(api_key="your-key")

# 3. Publish telemetry
await client.fabric.publish_telemetry({
    "device_id": "my-robot-001",
    "location": {"latitude": 37.7749, "longitude": -122.4194},
    "sensors": {"temperature": 25.5, "battery": 85.0}
})

# 4. Get intelligence
alerts = await client.intelligence.get_alerts()
print(f"Active alerts: {len(alerts)}")

# 5. Create mission
mission = await client.tasking.create_mission({
    "objectives": ["patrol", "detect", "report"],
    "assets": ["my-robot-001"]
})
```

### Next Steps

1. **Choose your integration pattern** based on your platform
2. **Install the appropriate SDK** for your language/platform
3. **Register your device** with Summit.OS
4. **Implement telemetry publishing** for your sensors
5. **Implement command handling** for mission execution
6. **Test the integration** with the Summit.OS development environment

## 📞 Support

- **Documentation**: https://docs.summit-os.bigmt.ai
- **SDK Downloads**: https://github.com/bigmt/summit-os-sdk
- **Integration Examples**: https://github.com/bigmt/summit-os-examples
- **Support**: support@bigmt.ai
- **Community**: https://community.summit-os.bigmt.ai
