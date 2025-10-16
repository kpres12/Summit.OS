# Summit.OS Edge Protocol

## Overview

The Summit.OS Edge Protocol defines the communication standards and data formats for edge devices (robots, drones, sensors) to integrate with the Summit.OS distributed intelligence fabric. This protocol ensures reliable, secure, and efficient communication between edge devices and the central intelligence system.

## Protocol Architecture

### Communication Layers

```
┌─────────────────────────────────────────┐
│           Application Layer             │
│     (Mission Commands, Telemetry)       │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│            Transport Layer              │
│        (MQTT, gRPC, WebSocket)          │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│            Security Layer               │
│           (mTLS, JWT, OAuth)            │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│          Multi-Link Layer               │
│    (Link Selection, Failover, QoS)      │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│            Network Layer                │
│ (Radio Mesh, Cellular, Satellite, WiFi) │
└─────────────────────────────────────────┘
```

## Message Formats

### Telemetry Message Format

```json
{
  "header": {
    "message_id": "uuid",
    "timestamp": "2024-01-15T10:30:00Z",
    "device_id": "drone-001",
    "message_type": "telemetry",
    "version": "1.0",
    "sequence_number": 12345
  },
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "altitude": 100.0,
    "accuracy": 5.0,
    "heading": 180.0,
    "speed": 15.0
  },
  "sensors": {
    "temperature": 25.5,
    "humidity": 45.2,
    "pressure": 1013.25,
    "wind_speed": 12.3,
    "wind_direction": 180.0,
    "visibility": 10000.0
  },
  "status": {
    "battery_level": 85.0,
    "signal_strength": -65.0,
    "operational_status": "online",
    "error_codes": []
  },
  "metadata": {
    "firmware_version": "1.2.3",
    "hardware_id": "HW-001",
    "mission_id": "mission-123",
    "custom_fields": {}
  }
}
```

### Command Message Format

```json
{
  "header": {
    "message_id": "uuid",
    "timestamp": "2024-01-15T10:30:00Z",
    "device_id": "drone-001",
    "message_type": "command",
    "version": "1.0",
    "sequence_number": 12346
  },
  "command": {
    "command_type": "navigate",
    "priority": "high",
    "timeout": 300,
    "parameters": {
      "target_location": {
        "latitude": 37.7849,
        "longitude": -122.4094,
        "altitude": 150.0
      },
      "speed": 20.0,
      "altitude_mode": "absolute"
    }
  },
  "constraints": {
    "max_duration": 1800,
    "weather_limits": {
      "max_wind_speed": 25.0,
      "min_visibility": 1000.0
    },
    "safety_radius": 500.0
  },
  "metadata": {
    "mission_id": "mission-123",
    "operator_id": "op-001",
    "command_source": "autonomous"
  }
}
```

### Alert Message Format

```json
{
  "header": {
    "message_id": "uuid",
    "timestamp": "2024-01-15T10:30:00Z",
    "device_id": "thermal-camera-001",
    "message_type": "alert",
    "version": "1.0",
    "sequence_number": 12347
  },
  "alert": {
    "alert_id": "alert-001",
    "severity": "critical",
    "category": "fire",
    "title": "Fire Detected",
    "description": "Large fire detected in sector 7",
    "location": {
      "latitude": 37.7749,
      "longitude": -122.4194,
      "altitude": 0.0
    },
    "confidence": 0.95,
    "detection_method": "thermal_camera"
  },
  "data": {
    "temperature": 850.0,
    "size_estimate": 50.0,
    "spread_rate": 2.5,
    "wind_effect": 1.2
  },
  "metadata": {
    "sensor_calibration": "2024-01-01T00:00:00Z",
    "detection_algorithm": "fire_detection_v2.1",
    "environmental_conditions": {
      "ambient_temperature": 25.0,
      "humidity": 45.0,
      "wind_speed": 12.3
    }
  }
}
```

## Communication Protocols

### MQTT Protocol

#### Connection Parameters
- **Broker**: `mqtt.summit-os.bigmt.ai:1883`
- **Client ID**: `{device_type}-{device_id}`
- **Username**: Device authentication token
- **Password**: Device secret key
- **Keep Alive**: 60 seconds
- **QoS**: 1 (At least once delivery)

#### Topic Structure
```
summit-os/
├── devices/
│   ├── {device_id}/
│   │   ├── telemetry/
│   │   ├── status/
│   │   ├── alerts/
│   │   └── commands/
├── missions/
│   ├── {mission_id}/
│   │   ├── commands/
│   │   ├── updates/
│   │   └── status/
├── alerts/
│   ├── fire/
│   ├── weather/
│   └── system/
└── system/
    ├── health/
    ├── config/
    └── updates/
```

#### Message Examples

**Publish Telemetry**
```bash
Topic: summit-os/devices/drone-001/telemetry
QoS: 1
Payload: {telemetry_message_json}
```

**Subscribe to Commands**
```bash
Topic: summit-os/devices/drone-001/commands/+
QoS: 1
```

**Publish Alert**
```bash
Topic: summit-os/alerts/fire
QoS: 1
Payload: {alert_message_json}
```

### gRPC Protocol

#### Service Definition
```protobuf
syntax = "proto3";

package summit.os.edge;

service EdgeService {
  rpc StreamTelemetry(stream TelemetryRequest) returns (TelemetryResponse);
  rpc StreamCommands(CommandRequest) returns (stream CommandResponse);
  rpc SendAlert(AlertRequest) returns (AlertResponse);
  rpc GetStatus(StatusRequest) returns (StatusResponse);
  rpc UpdateConfiguration(ConfigRequest) returns (ConfigResponse);
}

message TelemetryRequest {
  string device_id = 1;
  string timestamp = 2;
  Location location = 3;
  map<string, double> sensors = 4;
  DeviceStatus status = 5;
}

message CommandResponse {
  string command_id = 1;
  string command_type = 2;
  map<string, string> parameters = 3;
  int64 timeout = 4;
}
```

#### Connection Parameters
- **Endpoint**: `grpc.summit-os.bigmt.ai:443`
- **TLS**: Required
- **Authentication**: mTLS certificates
- **Compression**: gzip
- **Keep Alive**: 30 seconds

### WebSocket Protocol

#### Connection
```javascript
const ws = new WebSocket('wss://ws.summit-os.bigmt.ai/edge', {
  protocols: ['summit-os-v1'],
  headers: {
    'Authorization': 'Bearer <device_token>',
    'Device-ID': 'drone-001',
    'Device-Type': 'quadcopter'
  }
});
```

#### Message Format
```json
{
  "type": "telemetry|command|alert|status",
  "id": "message-uuid",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": { /* message-specific data */ }
}
```

## Security

### Authentication

#### Device Registration
1. **Pre-shared Keys**: Initial device registration
2. **Certificate-based**: mTLS for secure communication
3. **JWT Tokens**: Short-lived access tokens
4. **OAuth 2.0**: For third-party integrations

#### Certificate Management
```bash
# Generate device certificate
openssl req -new -x509 -key device.key -out device.crt -days 365 \
  -subj "/CN=drone-001/O=BigMT/C=US"

# Install certificate on device
scp device.crt device.key root@drone-001:/etc/summit-os/certs/
```

### Encryption

#### Data in Transit
- **MQTT**: TLS 1.3 encryption
- **gRPC**: TLS 1.3 with mTLS
- **WebSocket**: WSS (WebSocket Secure)

#### Data at Rest
- **Local Storage**: AES-256 encryption
- **Database**: Transparent Data Encryption (TDE)
- **Backup**: Encrypted backups with key rotation

### Access Control

#### Role-Based Permissions
```yaml
device_roles:
  drone:
    permissions:
      - telemetry:write
      - commands:read
      - alerts:write
      - status:read
  sensor:
    permissions:
      - telemetry:write
      - alerts:write
      - status:read
  ground_vehicle:
    permissions:
      - telemetry:write
      - commands:read
      - alerts:write
      - status:read
```

## Data Synchronization

### Store-and-Forward

#### Local Buffer
```json
{
  "buffer_config": {
    "max_size_mb": 100,
    "max_age_hours": 24,
    "compression": "gzip",
    "encryption": "aes256"
  },
  "sync_policy": {
    "batch_size": 100,
    "sync_interval": 300,
    "retry_attempts": 3,
    "retry_delay": 60
  }
}
```

#### Sync Protocol
1. **Connection Check**: Verify network connectivity
2. **Batch Upload**: Send buffered messages in batches
3. **Acknowledgment**: Confirm successful delivery
4. **Cleanup**: Remove successfully sent messages
5. **Retry Logic**: Handle failed transmissions

### Conflict Resolution

#### Timestamp-based Resolution
```python
def resolve_conflict(local_msg, remote_msg):
    if local_msg.timestamp > remote_msg.timestamp:
        return local_msg
    elif remote_msg.timestamp > local_msg.timestamp:
        return remote_msg
    else:
        # Same timestamp, use sequence number
        return local_msg if local_msg.sequence > remote_msg.sequence else remote_msg
```

## Device Management

### Device Registration

#### Registration Process
1. **Device Discovery**: Automatic device detection
2. **Capability Exchange**: Device capabilities and constraints
3. **Authentication**: Secure device authentication
4. **Configuration**: Device-specific configuration
5. **Health Monitoring**: Continuous health checks

#### Device Capabilities
```json
{
  "device_id": "drone-001",
  "device_type": "quadcopter",
  "capabilities": {
    "sensors": ["thermal_camera", "rgb_camera", "lidar", "gps"],
    "actuators": ["propellers", "gimbal", "landing_gear"],
    "communication": {
      "radio_mesh": {
        "enabled": true,
        "frequency_bands": ["2.4GHz", "5GHz", "900MHz"],
        "mesh_protocol": "802.11s",
        "max_hops": 5,
        "power_level": "high"
      },
      "cellular": {
        "enabled": true,
        "networks": ["LTE", "5G"],
        "carriers": ["primary", "secondary"],
        "data_limit_mb": 10000
      },
      "satellite": {
        "enabled": false,
        "provider": "starlink",
        "dish_type": "mobile",
        "backup_only": true
      },
      "wifi": {
        "enabled": true,
        "bands": ["2.4GHz", "5GHz"],
        "infrastructure_only": false
      }
    },
    "processing": {
      "cpu": "ARM Cortex-A78",
      "memory": "8GB",
      "storage": "256GB",
      "ai_acceleration": "NPU"
    }
  },
  "constraints": {
    "max_altitude": 400.0,
    "max_speed": 50.0,
    "operational_range": 5000.0,
    "weather_limits": {
      "max_wind_speed": 25.0,
      "min_visibility": 1000.0
    }
  }
}
```

### Configuration Management

#### Configuration Schema
```json
{
  "device_config": {
    "telemetry": {
      "frequency": 1.0,
      "sensors": ["all"],
      "compression": true
    },
    "communication": {
      "multi_link": {
        "enabled": true,
        "primary_link": "radio_mesh",
        "failover_order": ["radio_mesh", "cellular", "satellite", "wifi"],
        "link_selection_criteria": {
          "latency_weight": 0.4,
          "bandwidth_weight": 0.3,
          "reliability_weight": 0.3
        },
        "autonomous_mode": {
          "enabled": true,
          "sync_interval": 300,
          "buffer_size_mb": 100
        }
      },
      "radio_mesh": {
        "frequency": "900MHz",
        "power_level": "high",
        "mesh_id": "summit-mesh",
        "encryption": "wpa3"
      },
      "cellular": {
        "apn": "iot.carrier.com",
        "bands": ["B1", "B3", "B7", "B20"],
        "fallback_2g": false
      },
      "satellite": {
        "provider": "starlink",
        "backup_only": true,
        "data_priority": "critical_only"
      },
      "mqtt": {
        "broker": "mqtt.summit-os.bigmt.ai",
        "port": 1883,
        "qos": 1
      },
      "grpc": {
        "endpoint": "grpc.summit-os.bigmt.ai:443",
        "tls": true
      }
    },
    "safety": {
      "geofence": {
        "enabled": true,
        "bounds": {
          "north": 37.8,
          "south": 37.7,
          "east": -122.4,
          "west": -122.5
        }
      },
      "emergency_landing": {
        "enabled": true,
        "battery_threshold": 20.0
      }
    }
  }
}
```

## Performance Optimization

### Message Compression

#### Compression Algorithms
- **JSON**: gzip compression (default)
- **Binary**: LZ4 compression for high-frequency data
- **Images**: JPEG compression with quality settings

#### Compression Configuration
```json
{
  "compression": {
    "algorithm": "gzip",
    "level": 6,
    "threshold": 1024,
    "types": ["telemetry", "sensor_data"]
  }
}
```

### Bandwidth Management

#### Adaptive Quality
```python
def adjust_quality(connection_quality):
    if connection_quality > 0.8:
        return "high"  # Full resolution, high frequency
    elif connection_quality > 0.5:
        return "medium"  # Reduced resolution, medium frequency
    else:
        return "low"  # Minimal data, low frequency
```

#### Data Prioritization
1. **Critical Alerts**: Highest priority
2. **Commands**: High priority
3. **Telemetry**: Medium priority
4. **Status Updates**: Low priority

## Error Handling

### Error Codes

#### System Errors
- `E001`: Network connection lost
- `E002`: Authentication failed
- `E003`: Message format invalid
- `E004`: Device not registered
- `E005`: Command timeout

#### Device Errors
- `D001`: Battery low
- `D002`: GPS signal lost
- `D003`: Sensor malfunction
- `D004`: Communication error
- `D005`: System overload

### Retry Logic

#### Exponential Backoff
```python
def calculate_retry_delay(attempt):
    base_delay = 1.0
    max_delay = 60.0
    delay = base_delay * (2 ** attempt)
    return min(delay, max_delay)
```

#### Circuit Breaker
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
```

## Testing and Validation

### Protocol Testing

#### Unit Tests
```python
def test_telemetry_message_format():
    message = create_telemetry_message(device_id="drone-001")
    assert validate_message_schema(message)
    assert message["header"]["message_type"] == "telemetry"
```

#### Integration Tests
```python
def test_mqtt_connection():
    client = MQTTClient(broker="localhost", port=1883)
    assert client.connect() == True
    assert client.publish("test/topic", "test message") == True
```

#### Performance Tests
```python
def test_message_throughput():
    start_time = time.time()
    for i in range(1000):
        send_telemetry_message(f"device-{i}")
    duration = time.time() - start_time
    throughput = 1000 / duration
    assert throughput > 100  # messages per second
```

## Implementation Examples

### Python Edge Agent

```python
import asyncio
import json
import paho.mqtt.client as mqtt
from datetime import datetime, timezone

class SummitEdgeAgent:
    def __init__(self, device_id, device_type):
        self.device_id = device_id
        self.device_type = device_type
        self.mqtt_client = None
        self.connected = False
        
    async def connect(self):
        self.mqtt_client = mqtt.Client(client_id=self.device_id)
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        
        self.mqtt_client.connect("mqtt.summit-os.bigmt.ai", 1883, 60)
        self.mqtt_client.loop_start()
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            print(f"Connected to Summit.OS as {self.device_id}")
            # Subscribe to commands
            client.subscribe(f"summit-os/devices/{self.device_id}/commands/+")
        else:
            print(f"Failed to connect: {rc}")
            
    def on_message(self, client, userdata, msg):
        try:
            command = json.loads(msg.payload.decode())
            asyncio.create_task(self.handle_command(command))
        except Exception as e:
            print(f"Error processing command: {e}")
            
    async def handle_command(self, command):
        print(f"Received command: {command['command']['command_type']}")
        # Process command based on type
        # Send acknowledgment
        await self.send_command_ack(command["header"]["message_id"])
        
    async def send_telemetry(self, location, sensors, status):
        message = {
            "header": {
                "message_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "device_id": self.device_id,
                "message_type": "telemetry",
                "version": "1.0"
            },
            "location": location,
            "sensors": sensors,
            "status": status
        }
        
        topic = f"summit-os/devices/{self.device_id}/telemetry"
        self.mqtt_client.publish(topic, json.dumps(message))
        
    async def send_alert(self, alert_type, location, data):
        message = {
            "header": {
                "message_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "device_id": self.device_id,
                "message_type": "alert",
                "version": "1.0"
            },
            "alert": {
                "alert_id": str(uuid.uuid4()),
                "severity": "high",
                "category": alert_type,
                "location": location
            },
            "data": data
        }
        
        topic = f"summit-os/alerts/{alert_type}"
        self.mqtt_client.publish(topic, json.dumps(message))
```

### ROS 2 Integration

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, Temperature
from std_msgs.msg import String

class SummitROS2Bridge(Node):
    def __init__(self):
        super().__init__('summit_ros2_bridge')
        self.summit_agent = SummitEdgeAgent("ugv-001", "ground_vehicle")
        
        # ROS 2 subscribers
        self.gps_sub = self.create_subscription(
            NavSatFix, '/gps/fix', self.gps_callback, 10)
        self.temp_sub = self.create_subscription(
            Temperature, '/sensors/temperature', self.temp_callback, 10)
            
        # ROS 2 publishers
        self.command_pub = self.create_publisher(
            String, '/summit/commands', 10)
            
    def gps_callback(self, msg):
        location = {
            "latitude": msg.latitude,
            "longitude": msg.longitude,
            "altitude": msg.altitude,
            "accuracy": msg.position_covariance[0]
        }
        
        sensors = {"gps_quality": msg.status.status}
        status = {"operational_status": "online"}
        
        asyncio.create_task(
            self.summit_agent.send_telemetry(location, sensors, status)
        )
        
    def temp_callback(self, msg):
        if msg.temperature > 100.0:  # Fire detection threshold
            location = {
                "latitude": self.last_gps.latitude,
                "longitude": self.last_gps.longitude,
                "altitude": self.last_gps.altitude
            }
            
            data = {
                "temperature": msg.temperature,
                "detection_method": "temperature_sensor"
            }
            
            asyncio.create_task(
                self.summit_agent.send_alert("fire", location, data)
            )
```

## Conclusion

The Summit.OS Edge Protocol provides a comprehensive framework for integrating edge devices with the distributed intelligence fabric. It ensures reliable, secure, and efficient communication while supporting offline operation and automatic synchronization. The protocol is designed to be extensible and adaptable to various device types and communication scenarios.

For more information, see:
- [API Documentation](./API.md)
- [Architecture Overview](./ARCHITECTURE.md)
- [Development Guide](./DEVELOPMENT.md)
