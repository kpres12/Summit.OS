# Summit.OS API Documentation

## Overview

Summit.OS provides a comprehensive REST API and real-time WebSocket interface for integrating with the distributed intelligence fabric. This document covers the API endpoints, data schemas, and integration patterns.

## Base URLs

- **Development**: `http://localhost:8000`
- **Production**: `https://api.summit-os.bigmt.ai`
- **WebSocket**: `ws://localhost:8001/ws`

## Authentication

All API requests require authentication via JWT tokens or API keys.

```bash
# JWT Token
Authorization: Bearer <jwt_token>

# API Key
X-API-Key: <api_key>
```

## Core Services

### Data Fabric Service (`:8001`)

Real-time message bus and synchronization layer.

#### Publish Telemetry
```http
POST /telemetry
Content-Type: application/json

{
  "device_id": "drone-001",
  "timestamp": "2024-01-15T10:30:00Z",
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "altitude": 100.0,
    "accuracy": 5.0
  },
  "sensors": {
    "temperature": 25.5,
    "humidity": 45.2,
    "wind_speed": 12.3,
    "wind_direction": 180.0
  },
  "status": "online",
  "battery_level": 85.0,
  "signal_strength": -65.0
}
```

#### Publish Alert
```http
POST /alerts
Content-Type: application/json

{
  "alert_id": "alert-001",
  "timestamp": "2024-01-15T10:30:00Z",
  "severity": "high",
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "altitude": 0.0
  },
  "description": "Fire detected in sector 7",
  "source": "thermal-camera-001",
  "category": "fire",
  "tags": ["wildfire", "thermal", "sector-7"]
}
```

#### Publish Mission Update
```http
POST /missions
Content-Type: application/json

{
  "mission_id": "mission-001",
  "timestamp": "2024-01-15T10:30:00Z",
  "status": "active",
  "assets": ["drone-001", "ugv-002"],
  "objectives": ["patrol", "detect", "suppress"],
  "progress": 0.3,
  "estimated_completion": "2024-01-15T12:00:00Z"
}
```

### Sensor Fusion Service (`:8002`)

Multi-modal sensor data fusion and world model generation.

#### Process Sensor Data
```http
POST /sensor-data
Content-Type: application/json

{
  "device_id": "thermal-camera-001",
  "timestamp": "2024-01-15T10:30:00Z",
  "sensor_type": "thermal",
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "altitude": 50.0
  },
  "data": {
    "image_data": "base64_encoded_image",
    "temperature_matrix": [[25.5, 26.1], [24.8, 25.9]],
    "detection_confidence": 0.95
  },
  "confidence": 0.95
}
```

#### Process Detection
```http
POST /detections
Content-Type: application/json

{
  "detection_id": "det-001",
  "timestamp": "2024-01-15T10:30:00Z",
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "altitude": 0.0
  },
  "object_type": "fire",
  "confidence": 0.95,
  "bounding_box": {
    "x": 100,
    "y": 150,
    "width": 200,
    "height": 300
  },
  "metadata": {
    "temperature": 850.0,
    "size_estimate": "large"
  }
}
```

#### Get World Model
```http
GET /world-model
```

Response:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "entities": [
    {
      "entity_id": "fire-001",
      "entity_type": "fire",
      "location": {
        "latitude": 37.7749,
        "longitude": -122.4194
      },
      "properties": {
        "temperature": 850.0,
        "size": "large",
        "confidence": 0.95
      },
      "last_updated": "2024-01-15T10:30:00Z"
    }
  ],
  "relationships": [
    {
      "source": "fire-001",
      "target": "drone-001",
      "relationship_type": "monitoring",
      "confidence": 0.9
    }
  ],
  "confidence": 0.92
}
```

### Intelligence Service (`:8003`)

AI reasoning and contextual intelligence generation.

#### Get Intelligence Alerts
```http
GET /intelligence/alerts?severity=high&limit=10
```

Response:
```json
{
  "alerts": [
    {
      "alert_id": "intel-001",
      "timestamp": "2024-01-15T10:30:00Z",
      "severity": "high",
      "category": "fire_risk",
      "title": "High Fire Risk Detected",
      "description": "Multiple indicators suggest high fire risk in sector 7",
      "location": {
        "latitude": 37.7749,
        "longitude": -122.4194
      },
      "confidence": 0.92,
      "recommendations": [
        "Deploy additional monitoring assets",
        "Prepare suppression resources",
        "Issue public safety alert"
      ],
      "metadata": {
        "risk_factors": ["high_temperature", "low_humidity", "wind_conditions"],
        "affected_area": 2.5
      }
    }
  ]
}
```

#### Get Risk Assessment
```http
GET /intelligence/risk?location=37.7749,-122.4194&radius=5000
```

Response:
```json
{
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194
  },
  "radius": 5000,
  "risk_score": 0.85,
  "risk_level": "high",
  "factors": {
    "weather": 0.8,
    "vegetation": 0.9,
    "topography": 0.7,
    "human_activity": 0.6
  },
  "recommendations": [
    "Increase monitoring frequency",
    "Prepare evacuation routes",
    "Alert emergency services"
  ],
  "valid_until": "2024-01-15T12:00:00Z"
}
```

### Mission Tasking Service (`:8004`)

Mission planning and autonomous coordination.

#### Create Mission
```http
POST /missions
Content-Type: application/json

{
  "mission_id": "mission-001",
  "mission_type": "patrol",
  "priority": "high",
  "objectives": [
    "patrol_sector_7",
    "detect_fire_hazards",
    "report_conditions"
  ],
  "assets": ["drone-001", "ugv-002"],
  "constraints": {
    "max_duration": 3600,
    "weather_limits": {
      "max_wind_speed": 25.0,
      "min_visibility": 1000.0
    }
  },
  "start_time": "2024-01-15T11:00:00Z",
  "estimated_duration": 1800
}
```

#### Get Mission Status
```http
GET /missions/mission-001
```

Response:
```json
{
  "mission_id": "mission-001",
  "status": "active",
  "progress": 0.45,
  "assets": [
    {
      "asset_id": "drone-001",
      "status": "active",
      "current_task": "patrol_sector_7",
      "location": {
        "latitude": 37.7749,
        "longitude": -122.4194,
        "altitude": 100.0
      },
      "battery_level": 75.0,
      "estimated_completion": "2024-01-15T11:30:00Z"
    }
  ],
  "objectives": [
    {
      "objective": "patrol_sector_7",
      "status": "in_progress",
      "progress": 0.6
    },
    {
      "objective": "detect_fire_hazards",
      "status": "pending",
      "progress": 0.0
    }
  ],
  "estimated_completion": "2024-01-15T11:30:00Z"
}
```

## WebSocket API

### Connection
```javascript
const socket = io('ws://localhost:8001/ws');

socket.on('connect', () => {
  console.log('Connected to Summit.OS');
});

socket.on('telemetry', (data) => {
  console.log('Telemetry update:', data);
});

socket.on('alert', (data) => {
  console.log('Alert:', data);
});

socket.on('mission', (data) => {
  console.log('Mission update:', data);
});
```

### Message Types

#### Telemetry Message
```json
{
  "type": "telemetry",
  "data": {
    "device_id": "drone-001",
    "timestamp": "2024-01-15T10:30:00Z",
    "location": {
      "latitude": 37.7749,
      "longitude": -122.4194,
      "altitude": 100.0
    },
    "sensors": {
      "temperature": 25.5,
      "humidity": 45.2
    },
    "status": "online",
    "battery_level": 85.0
  }
}
```

#### Alert Message
```json
{
  "type": "alert",
  "data": {
    "alert_id": "alert-001",
    "timestamp": "2024-01-15T10:30:00Z",
    "severity": "high",
    "location": {
      "latitude": 37.7749,
      "longitude": -122.4194
    },
    "description": "Fire detected in sector 7",
    "source": "thermal-camera-001",
    "category": "fire"
  }
}
```

#### Mission Message
```json
{
  "type": "mission",
  "data": {
    "mission_id": "mission-001",
    "timestamp": "2024-01-15T10:30:00Z",
    "status": "active",
    "assets": ["drone-001", "ugv-002"],
    "objectives": ["patrol", "detect", "suppress"],
    "progress": 0.3
  }
}
```

## Data Schemas

### Location Schema
```json
{
  "latitude": 37.7749,
  "longitude": -122.4194,
  "altitude": 100.0,
  "accuracy": 5.0,
  "heading": 180.0,
  "speed": 15.0
}
```

### Telemetry Schema
```json
{
  "device_id": "string",
  "timestamp": "2024-01-15T10:30:00Z",
  "location": { /* Location Schema */ },
  "status": "online|offline|error|maintenance",
  "battery_level": 85.0,
  "signal_strength": -65.0,
  "sensors": {
    "temperature": 25.5,
    "humidity": 45.2,
    "wind_speed": 12.3
  },
  "metadata": {}
}
```

### Alert Schema
```json
{
  "alert_id": "string",
  "timestamp": "2024-01-15T10:30:00Z",
  "severity": "low|medium|high|critical",
  "category": "fire|smoke|weather|equipment|security",
  "status": "active|acknowledged|resolved|escalated",
  "location": { /* Location Schema */ },
  "title": "string",
  "description": "string",
  "source": "string",
  "tags": ["string"],
  "metadata": {},
  "acknowledged": false,
  "acknowledged_by": "string",
  "acknowledged_at": "2024-01-15T10:30:00Z"
}
```

## Error Handling

### HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error
- `503` - Service Unavailable

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request data",
    "details": {
      "field": "location.latitude",
      "issue": "Value must be between -90 and 90"
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req-123456"
  }
}
```

## Rate Limiting

- **Standard**: 1000 requests per hour
- **Premium**: 10000 requests per hour
- **Enterprise**: Unlimited

Rate limit headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248600
```

## SDKs and Libraries

### Python SDK
```python
from summit_os import SummitClient

client = SummitClient(
    api_key="your-api-key",
    base_url="http://localhost:8000"
)

# Publish telemetry
client.fabric.publish_telemetry({
    "device_id": "drone-001",
    "location": {"latitude": 37.7749, "longitude": -122.4194},
    "sensors": {"temperature": 25.5}
})

# Get intelligence alerts
alerts = client.intelligence.get_alerts(severity="high")
```

### JavaScript SDK
```javascript
import { SummitClient } from '@bigmt/summit-os-sdk';

const client = new SummitClient({
  apiKey: 'your-api-key',
  baseUrl: 'http://localhost:8000'
});

// Publish telemetry
await client.fabric.publishTelemetry({
  device_id: 'drone-001',
  location: { latitude: 37.7749, longitude: -122.4194 },
  sensors: { temperature: 25.5 }
});

// Get intelligence alerts
const alerts = await client.intelligence.getAlerts({ severity: 'high' });
```

## Integration Examples

### FireLine Console Integration
```javascript
// Connect to Summit.OS
const socket = io('ws://localhost:8001/ws');

// Subscribe to real-time updates
socket.on('alert', (alert) => {
  // Display alert in FireLine Console
  displayAlert(alert);
});

socket.on('telemetry', (telemetry) => {
  // Update device status on map
  updateDeviceStatus(telemetry);
});

// Query intelligence data
const response = await fetch('http://localhost:8000/api/v1/intelligence/alerts');
const alerts = await response.json();
```

### Edge Device Integration
```python
import requests
import json

# Publish telemetry from edge device
telemetry_data = {
    "device_id": "thermal-camera-001",
    "location": {
        "latitude": 37.7749,
        "longitude": -122.4194,
        "altitude": 50.0
    },
    "sensors": {
        "temperature": 850.0,
        "detection_confidence": 0.95
    }
}

response = requests.post(
    'http://localhost:8001/telemetry',
    json=telemetry_data,
    headers={'Authorization': 'Bearer your-token'}
)
```

## Support and Resources

- **Documentation**: https://docs.summit-os.bigmt.ai
- **API Reference**: https://api-docs.summit-os.bigmt.ai
- **SDK Downloads**: https://github.com/bigmt/summit-os-sdk
- **Support**: support@bigmt.ai
- **Status Page**: https://status.summit-os.bigmt.ai
