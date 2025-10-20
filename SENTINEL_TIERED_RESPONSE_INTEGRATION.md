# Sentinel Integration: FireFly + EmberWing Tiered Response

## Overview

Summit.OS now supports **tiered drone response architecture** with FireFly (recon) + EmberWing (intervention) doctrine. This integration prompt will help you implement the Sentinel-side components to create a complete autonomous suppression network.

## 🎯 What Summit.OS Now Provides

### 1. **Tiered Response Data Models**
- `DroneType`: FireFly, EmberWing, Generic
- `MissionTier`: Tier-1 Verify, Tier-2 Suppress, Tier-3 Contain, Tier-4 Escalate
- `PayloadConfig`: Suppressant capsules, retardant gel, water pods, beacons
- `SwarmCoordination`: Containment rings, anti-collision, formation patterns
- `FireThreshold`: Size/spread thresholds for escalation decisions

### 2. **Tiered Mission Planning**
- **Tier 1**: FireFly dispatched for rapid verification (< 60s response)
- **Tier 2**: EmberWing deployed with suppressant payload for intervention  
- **Tier 3**: Multi-drone containment ring formation
- **Automatic escalation** based on fire size, temperature, spread rate

### 3. **New API Endpoints**
- `POST /api/v1/tiered-missions` - Create tiered response mission
- `POST /api/v1/tiered-missions/{id}/escalate` - Escalate to next tier
- `GET /api/v1/tiered-missions/{id}` - Get mission status
- `GET /api/v1/tiered-missions` - List tiered missions

### 4. **Asset Management Extensions**
Extended asset model with:
- Drone capabilities (speed, endurance, payload capacity)
- Tiered role assignments
- Deployment box groupings
- Performance characteristics

## 🚀 Sentinel Integration Tasks

### Phase 1: Sentinel Detection Integration

**Goal**: Integrate Sentinel's smoke detection with Summit.OS tiered response.

#### 1.1 Update Sentinel Detection Pipeline
```typescript
// In apps/console or detection service
interface FireDetection {
  detection_id: string;
  timestamp: string;
  location: { lat: number; lon: number };
  confidence: number;
  fire_characteristics?: {
    size_m2?: number;
    temperature?: number;
    smoke_density?: number;
  };
  sensor_data: SensorReading[];
}
```

#### 1.2 Add Summit.OS Integration Service
Create `apps/summit-integration/src/tiered_response.py`:

```python
import asyncio
import aiohttp
from typing import Dict, Any, Optional

class SummitTieredResponse:
    def __init__(self, summit_base_url: str, api_key: str):
        self.summit_url = summit_base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    async def trigger_tiered_response(self, detection: Dict[str, Any]) -> str:
        """Trigger tiered response mission in Summit.OS"""
        
        # Create fire threshold based on detection confidence
        threshold = self._create_fire_threshold(detection)
        
        payload = {
            "alert_id": detection["detection_id"],
            "initial_location": detection["location"],
            "verification_required": True,
            "intervention_threshold": threshold,
            "max_tier": "tier_3_contain",
            "weather_data": await self._get_weather_data(detection["location"]),
            "terrain_data": await self._get_terrain_data(detection["location"])
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.summit_url}/api/v1/tiered-missions",
                json=payload,
                headers=self.headers
            ) as resp:
                result = await resp.json()
                return result["mission_id"]
    
    async def escalate_mission(self, mission_id: str, verification_data: Dict[str, Any]):
        """Send verification data to trigger escalation"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.summit_url}/api/v1/tiered-missions/{mission_id}/escalate",
                json=verification_data,
                headers=self.headers
            ) as resp:
                return await resp.json()
    
    def _create_fire_threshold(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """Create escalation thresholds based on detection characteristics"""
        base_confidence = detection.get("confidence", 0.5)
        
        # Lower thresholds for higher confidence detections
        size_threshold = 25.0 if base_confidence > 0.8 else 50.0
        temp_threshold = 150.0 if base_confidence > 0.8 else 200.0
        
        return {
            "size_m2": size_threshold,
            "temperature": temp_threshold,
            "smoke_density": 0.3,
            "wind_factor": 1.2,
            "terrain_factor": 1.1
        }
```

#### 1.3 Update Detection Flow
```python
# In your existing detection pipeline
async def handle_fire_detection(detection: FireDetection):
    # Existing Sentinel processing...
    await store_detection(detection)
    await notify_operators(detection)
    
    # NEW: Trigger Summit.OS tiered response
    summit_client = SummitTieredResponse(SUMMIT_URL, SUMMIT_API_KEY)
    mission_id = await summit_client.trigger_tiered_response(detection)
    
    # Store mission_id for tracking
    await update_detection_with_mission(detection.detection_id, mission_id)
```

### Phase 2: Verification Data Pipeline

**Goal**: Receive FireFly verification data and feed back to Summit.OS for escalation decisions.

#### 2.1 Add Verification Data Receiver
```python
# MQTT subscriber for drone verification data
import paho.mqtt.client as mqtt

class VerificationProcessor:
    def __init__(self, summit_client: SummitTieredResponse):
        self.summit = summit_client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_message = self.on_verification_message
    
    async def on_verification_message(self, client, userdata, msg):
        """Process verification data from FireFly drones"""
        try:
            data = json.loads(msg.payload.decode())
            
            if data.get("action") == "VERIFICATION_COMPLETE":
                verification_data = self._process_verification(data)
                mission_id = data.get("mission_id")
                
                if mission_id:
                    # Send back to Summit.OS for escalation decision
                    result = await self.summit.escalate_mission(mission_id, verification_data)
                    
                    if result.get("escalated"):
                        await self._notify_escalation(mission_id, result)
        
        except Exception as e:
            logger.error(f"Verification processing error: {e}")
    
    def _process_verification(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform drone telemetry into verification data"""
        return {
            "lat": raw_data.get("location", {}).get("lat"),
            "lon": raw_data.get("location", {}).get("lon"),
            "fire_confirmed": raw_data.get("fire_detected", False),
            "fire_size_m2": raw_data.get("estimated_fire_area", 0.0),
            "max_temperature": raw_data.get("thermal_max", 20.0),
            "smoke_density": raw_data.get("smoke_opacity", 0.0),
            "wind_speed": raw_data.get("wind_speed", 0.0),
            "wind_direction": raw_data.get("wind_direction", 0.0),
            "terrain_slope": raw_data.get("terrain_slope", 0.0),
            "verification_timestamp": raw_data.get("timestamp")
        }
```

### Phase 3: Operator Interface Updates

**Goal**: Update Sentinel console to show tiered response status and allow manual control.

#### 3.1 Add Tiered Response Components
```typescript
// apps/console/src/components/TieredResponsePanel.tsx
import React, { useState, useEffect } from 'react';

interface TieredMission {
  mission_id: string;
  alert_id: string;
  current_tier: 'tier_1_verify' | 'tier_2_suppress' | 'tier_3_contain';
  tier_1_status?: string;
  tier_2_status?: string;
  tier_3_status?: string;
  assets_deployed: string[];
  created_at: string;
  updated_at: string;
}

export const TieredResponsePanel: React.FC = () => {
  const [missions, setMissions] = useState<TieredMission[]>([]);
  const [selectedMission, setSelectedMission] = useState<TieredMission | null>(null);
  
  useEffect(() => {
    // Poll for tiered mission updates
    const interval = setInterval(async () => {
      const response = await fetch('/api/v1/tiered-missions');
      const data = await response.json();
      setMissions(data);
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);
  
  const handleManualEscalation = async (missionId: string, verificationData: any) => {
    await fetch(`/api/v1/tiered-missions/${missionId}/escalate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(verificationData)
    });
  };
  
  return (
    <div className="tiered-response-panel">
      <h2>🚁 Tiered Response Status</h2>
      
      {missions.map(mission => (
        <div key={mission.mission_id} className="mission-card">
          <div className="mission-header">
            <span className="mission-id">{mission.mission_id.slice(0, 8)}</span>
            <span className={`tier-badge tier-${mission.current_tier}`}>
              {mission.current_tier.replace('_', ' ').toUpperCase()}
            </span>
          </div>
          
          <div className="tier-status">
            <div className={`tier ${mission.tier_1_status?.toLowerCase()}`}>
              <span>🔍 Tier 1 Verify</span>
              <span>{mission.tier_1_status || 'PENDING'}</span>
            </div>
            
            {mission.tier_2_status && (
              <div className={`tier ${mission.tier_2_status.toLowerCase()}`}>
                <span>💧 Tier 2 Suppress</span>
                <span>{mission.tier_2_status}</span>
              </div>
            )}
            
            {mission.tier_3_status && (
              <div className={`tier ${mission.tier_3_status.toLowerCase()}`}>
                <span>🛡️ Tier 3 Contain</span>
                <span>{mission.tier_3_status}</span>
              </div>
            )}
          </div>
          
          <div className="assets-deployed">
            <span>Assets: {mission.assets_deployed.join(', ')}</span>
          </div>
          
          <div className="mission-actions">
            <button onClick={() => setSelectedMission(mission)}>
              View Details
            </button>
            {mission.current_tier === 'tier_1_verify' && (
              <button 
                onClick={() => handleManualEscalation(mission.mission_id, {
                  fire_confirmed: true,
                  fire_size_m2: 50,
                  max_temperature: 200
                })}
                className="escalate-btn"
              >
                Force Escalate
              </button>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};
```

#### 3.2 Update Map Integration
```typescript
// Add tiered response layer to your existing map
const TieredResponseLayer: React.FC<{ missions: TieredMission[] }> = ({ missions }) => {
  return (
    <>
      {missions.map(mission => (
        <Marker
          key={mission.mission_id}
          position={[mission.location.lat, mission.location.lon]}
          icon={getTierIcon(mission.current_tier)}
        >
          <Popup>
            <div>
              <h3>Tiered Response: {mission.current_tier}</h3>
              <p>Status: {getCurrentTierStatus(mission)}</p>
              <p>Assets: {mission.assets_deployed.length}</p>
            </div>
          </Popup>
        </Marker>
      ))}
    </>
  );
};

const getTierIcon = (tier: string) => {
  const icons = {
    'tier_1_verify': '🔍',
    'tier_2_suppress': '💧', 
    'tier_3_contain': '🛡️'
  };
  return new DivIcon({
    html: `<div class="tier-icon">${icons[tier]}</div>`,
    className: 'custom-tier-icon',
    iconSize: [30, 30]
  });
};
```

### Phase 4: Configuration & Deployment

#### 4.1 Environment Configuration
```bash
# In Sentinel .env
SUMMIT_OS_BASE_URL=http://localhost:8000
SUMMIT_OS_API_KEY=your_summit_api_key
TIERED_RESPONSE_ENABLED=true

# Drone box configurations
DRONE_BOX_01_LAT=37.422
DRONE_BOX_01_LON=-122.084
DRONE_BOX_01_FIREFLY=firefly-001
DRONE_BOX_01_EMBERWING=emberwing-001
```

#### 4.2 Docker Compose Integration
```yaml
# Add to Sentinel's docker-compose.yml
services:
  summit-integration:
    build: ./apps/summit-integration
    environment:
      - SUMMIT_OS_URL=${SUMMIT_OS_BASE_URL}
      - SUMMIT_API_KEY=${SUMMIT_OS_API_KEY}
      - MQTT_BROKER=mqtt
    depends_on:
      - mqtt
      - postgres
    networks:
      - sentinel-network
      - summit-network  # Connect to Summit.OS network
```

### Phase 5: Testing & Validation

#### 5.1 End-to-End Test Scenario
```python
async def test_tiered_response_flow():
    """Test complete FireFly + EmberWing response"""
    
    # 1. Simulate fire detection
    detection = {
        "detection_id": "test-fire-001",
        "location": {"lat": 37.422, "lon": -122.084},
        "confidence": 0.9,
        "timestamp": datetime.now().isoformat()
    }
    
    # 2. Trigger tiered response
    summit = SummitTieredResponse(SUMMIT_URL, API_KEY)
    mission_id = await summit.trigger_tiered_response(detection)
    
    # 3. Wait for Tier 1 completion
    await asyncio.sleep(120)  # 2 minutes for verification
    
    # 4. Send verification data
    verification_data = {
        "lat": 37.422,
        "lon": -122.084,
        "fire_confirmed": True,
        "fire_size_m2": 75.0,  # Exceeds threshold
        "max_temperature": 250.0,
        "smoke_density": 0.6
    }
    
    result = await summit.escalate_mission(mission_id, verification_data)
    
    # 5. Verify escalation to Tier 2
    assert result["escalated"] == True
    assert result["next_tier"] == "tier_2_suppress"
    
    print("✅ Tiered response test completed successfully")
```

## 🔧 Implementation Checklist

### Summit.OS Integration ✅ (Completed)
- [x] Tiered drone data models
- [x] Mission planning logic with escalation
- [x] Swarm coordination capabilities  
- [x] Fire threshold assessment
- [x] API endpoints for tiered response

### Sentinel Integration (Your Tasks)
- [ ] **Detection Pipeline**: Connect fire detection to Summit.OS tiered missions
- [ ] **Verification Processor**: Handle FireFly verification data and escalation
- [ ] **Operator Interface**: Update console with tiered response panel
- [ ] **Configuration**: Environment setup and deployment
- [ ] **Testing**: End-to-end validation of tiered response flow

## 📡 Key Integration Points

1. **Detection → Summit.OS**: `POST /api/v1/tiered-missions`
2. **Verification → Summit.OS**: `POST /api/v1/tiered-missions/{id}/escalate` 
3. **Status Updates**: `GET /api/v1/tiered-missions/{id}`
4. **MQTT Topics**: 
   - `tasks/{asset_id}/dispatch` (outbound to drones)
   - `telemetry/{asset_id}` (inbound from drones)
   - `verification/{mission_id}` (verification results)

## 🚁 Expected Behavior

**Normal Flow:**
1. **Detection**: Sentinel detects smoke/flame
2. **Tier 1**: FireFly dispatched, reaches target in < 60s
3. **Verification**: FireFly confirms fire, assesses size/threat
4. **Escalation**: If threshold exceeded → Tier 2 activated
5. **Tier 2**: EmberWing deploys suppressant at target
6. **Containment**: If fire spreads → Tier 3 multi-drone containment
7. **Success**: Fire suppressed before human responders arrive

**This gives you true "seconds to action" autonomous suppression capability.**

## 🎯 Success Metrics

- **Response Time**: FireFly on-target < 60 seconds
- **Escalation Time**: Tier 1 → Tier 2 < 3 minutes  
- **Suppression Rate**: >80% of small fires contained at Tier 2
- **Coordination**: Zero drone collisions in Tier 3 swarm operations
- **Reliability**: 99%+ mission dispatch success rate

---

**Ready to integrate? Start with Phase 1 and work through each phase systematically. The Summit.OS foundation is ready to support your tiered response implementation.**