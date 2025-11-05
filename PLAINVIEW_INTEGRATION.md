# Summit.OS + Plainview Integration

This document describes the complete AI integration between Summit.OS and Plainview.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Summit.OS                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   Fusion     │─────▶│Intelligence  │◀─────│   Models     │  │
│  │  (8002)      │      │   (8003)     │      │  /models/    │  │
│  └──────────────┘      └──────┬───────┘      └──────────────┘  │
│         │                     │                                  │
│         │                     │ Advisories                       │
│    Observations               │                                  │
│         │                     ▼                                  │
│         │              ┌──────────────┐                          │
│         └──────────────│ Redis Streams│                          │
│                        └──────────────┘                          │
│                               │                                  │
└───────────────────────────────┼──────────────────────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │ Plainview Adapter│
                        │     (8005)       │
                        └─────────┬────────┘
                                  │
                 Enriches with    │ HTTP POST
                 domain context   │ WebSocket
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                          Plainview                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      │
│  │  FlowIQ     │      │  ValveOps   │      │PipelineGuard│      │
│  │  Module     │      │   Module    │      │   Module    │      │
│  └─────────────┘      └─────────────┘      └─────────────┘      │
│         │                     │                     │             │
│         └─────────────────────┴─────────────────────┘             │
│                               │                                   │
│                               ▼                                   │
│                    ┌─────────────────────┐                        │
│                    │  Intelligence API   │                        │
│                    │  /intelligence/*    │                        │
│                    └──────────┬──────────┘                        │
│                               │                                   │
│                               ▼                                   │
│                    ┌─────────────────────┐                        │
│                    │   Event Bus (SSE)   │                        │
│                    │     /events         │                        │
│                    └──────────┬──────────┘                        │
│                               │                                   │
│                               ▼                                   │
│                    ┌─────────────────────┐                        │
│                    │    Dashboard        │                        │
│                    │  AIAssistant.tsx    │                        │
│                    └─────────────────────┘                        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Summit.OS Intelligence Service
**Location:** `apps/intelligence/`

**Purpose:** Core AI/ML reasoning service that generates risk advisories

**Features:**
- Consumes observations from Fusion via Redis streams
- Risk scoring (rule-based + optional XGBoost)
- **NEW:** Plainview-specific ML models for:
  - Flow anomaly detection
  - Valve health prediction
  - Pipeline integrity assessment

**Endpoints:**
- `GET /advisories` - List advisories
- `POST /plainview/flow/analyze` - Analyze flow metrics
- `POST /plainview/valve/health` - Predict valve health
- `POST /plainview/pipeline/assess` - Assess pipeline integrity

### 2. Plainview Adapter
**Location:** `apps/plainview-adapter/`

**Purpose:** Bridge between Summit.OS and Plainview domains

**Features:**
- Polls Intelligence service for new advisories
- Enriches advisories with Plainview domain context
- Forwards insights to Plainview via HTTP
- Broadcasts to Plainview dashboard via WebSocket
- **Bidirectional:** Can forward Plainview events back to Summit.OS

**Endpoints:**
- `GET /health` - Health check
- `GET /readyz` - Readiness check
- `WS /ws/plainview` - WebSocket for real-time insights

### 3. Plainview Intelligence Module
**Location:** `Plainview/services/api/app/modules/intelligence.py`

**Purpose:** Receive and store AI insights in Plainview

**Endpoints:**
- `POST /intelligence/insights` - Receive insight from adapter
- `GET /intelligence/insights` - List insights
- `GET /intelligence/insights/{id}` - Get specific insight

### 4. ML Models
**Location:** `models/`

**Models:**
- `flow_anomaly.onnx` - Flow metrics anomaly detection
- `valve_health.onnx` - Valve health + maintenance prediction
- `pipeline_integrity.onnx` - Pipeline integrity assessment

**Training:** `scripts/train_plainview_models.py`

## Setup Instructions

### Prerequisites
```bash
# Summit.OS side
cd Summit.OS
make install-deps

# Plainview side
cd Plainview
npm install
```

### Step 1: Train ML Models
```bash
cd Summit.OS

# Install training dependencies
pip install onnx scikit-learn skl2onnx

# Train models
python scripts/train_plainview_models.py

# Models saved to: models/flow_anomaly.onnx, valve_health.onnx, pipeline_integrity.onnx
```

### Step 2: Start Summit.OS Stack
```bash
cd Summit.OS

# Start full stack (includes plainview-adapter)
make dev

# Or start services individually:
make dev-services     # Redis, Postgres, MQTT, etc.
make dev-backend      # All Python microservices
```

**Services running:**
- Intelligence: http://localhost:8003
- Plainview Adapter: http://localhost:8005
- Redis: localhost:6379
- Postgres: localhost:5433
- MQTT: localhost:1883

### Step 3: Start Plainview
```bash
cd Plainview

# Start Plainview API
npm run dev -w @plainview/api

# In another terminal, start dashboard
npm run dev -w @plainview/dashboard
```

**Services running:**
- Plainview API: http://localhost:4000
- Plainview Dashboard: http://localhost:5173

### Step 4: Verify Integration

#### Check adapter connection:
```bash
curl http://localhost:8005/health
# Should return: {"status": "ok", "connections": {...}}
```

#### Check Intelligence endpoints:
```bash
# List advisories
curl http://localhost:8003/advisories

# Test flow analysis
curl -X POST http://localhost:8003/plainview/flow/analyze \
  -H "Content-Type: application/json" \
  -d '{"flow_rate_lpm": 180, "pressure_pa": 2600000, "temperature_c": 50}'
```

#### Check Plainview intelligence module:
```bash
curl http://localhost:4000/intelligence/insights
```

## Data Flow

### 1. Observation → Advisory
```
Sensor/Camera → Fusion (8002) → Redis Stream → Intelligence (8003) → Advisory
```

### 2. Advisory → Plainview Insight
```
Intelligence Advisory → Plainview Adapter (8005) → Enrichment → Plainview API (4000)
```

### 3. Insight → Dashboard
```
Plainview API → SSE /events → Dashboard → AIAssistant UI
```

## WebSocket Integration

### Connect from Plainview Dashboard:
```typescript
const ws = new WebSocket('ws://localhost:8005/ws/plainview');

ws.onopen = () => {
  console.log('Connected to Summit.OS Intelligence');
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === 'insight') {
    // Display insight in UI
    console.log(msg.data);
  }
};
```

### Request advisories:
```typescript
ws.send(JSON.stringify({
  type: 'request_advisory'
}));
```

## API Reference

### Intelligence Service Plainview Endpoints

#### Analyze Flow Metrics
```bash
POST /plainview/flow/analyze
Content-Type: application/json

{
  "flow_rate_lpm": 150.0,
  "pressure_pa": 2500000.0,
  "temperature_c": 45.0
}

Response:
{
  "is_anomaly": false,
  "confidence": 0.15,
  "anomaly_type": "normal",
  "severity": "low",
  "details": {}
}
```

#### Predict Valve Health
```bash
POST /plainview/valve/health
Content-Type: application/json

{
  "cycles_count": 3000,
  "last_torque_nm": 52.0,
  "age_days": 500,
  "avg_temp_c": 45.0
}

Response:
{
  "health_score": 75.0,
  "maintenance_days": 90,
  "risk_level": "medium",
  "recommendations": [
    "Plan preventive maintenance",
    "Monitor torque readings weekly"
  ]
}
```

#### Assess Pipeline Integrity
```bash
POST /plainview/pipeline/assess
Content-Type: application/json

{
  "section_id": "pipeline-a",
  "pressure_variance": 0.15,
  "flow_consistency": 0.85,
  "age_years": 15.0,
  "inspection_score": 85.0
}

Response:
{
  "integrity_score": 72.5,
  "leak_probability": 0.22,
  "risk_level": "medium",
  "action_required": "Schedule inspection within 7 days"
}
```

### Plainview Intelligence Endpoints

#### Receive Insight
```bash
POST /intelligence/insights
Content-Type: application/json

{
  "type": "intelligence.insight",
  "insight_type": "flow_anomaly",
  "severity": "high",
  "title": "HIGH Risk: Flow Anomaly",
  "description": "Flow rate deviation detected with 85% confidence",
  "confidence": 0.85,
  "asset_id": "flow-system",
  "recommendations": ["Monitor flow metrics for next 30 minutes"],
  "timestamp": "2024-01-01T00:00:00Z",
  "source": "summit-os-intelligence"
}

Response:
{
  "status": "ok",
  "insight_id": "insight-1"
}
```

#### List Insights
```bash
GET /intelligence/insights?limit=10&severity=high

Response:
[
  {
    "id": "insight-1",
    "type": "flow_anomaly",
    "severity": "high",
    "title": "HIGH Risk: Flow Anomaly",
    "description": "...",
    "confidence": 0.85,
    "asset_id": "flow-system",
    "recommendations": [...],
    "timestamp": "2024-01-01T00:00:00Z",
    "source": "summit-os-intelligence",
    "received_at": "2024-01-01T00:00:05Z"
  }
]
```

## Environment Variables

### Summit.OS Intelligence
```bash
POSTGRES_URL=postgresql://summit:summit_password@localhost:5432/summit_os
REDIS_URL=redis://localhost:6379
MODEL_REGISTRY=/models
INTELLIGENCE_ENABLE_XGB=false  # Enable XGBoost models
```

### Plainview Adapter
```bash
INTELLIGENCE_URL=http://intelligence:8003
PLAINVIEW_API_URL=http://host.docker.internal:4000
REDIS_URL=redis://redis:6379
MQTT_BROKER=mqtt://mqtt:1883
```

### Plainview API
```bash
# No special config needed - intelligence module auto-registers
```

## Troubleshooting

### Adapter can't reach Plainview
```bash
# Check Plainview API is running
curl http://localhost:4000/health

# Check adapter logs
docker logs summit-plainview-adapter

# Verify host.docker.internal resolves (MacOS/Windows)
# On Linux, use --add-host=host.docker.internal:host-gateway
```

### No insights appearing
```bash
# 1. Check Intelligence service has advisories
curl http://localhost:8003/advisories

# 2. Check adapter is polling
docker logs summit-plainview-adapter | grep "Processing advisory"

# 3. Check Plainview received insights
curl http://localhost:4000/intelligence/insights

# 4. Check SSE stream
curl http://localhost:4000/events
```

### Models not loading
```bash
# Check models exist
ls -la models/

# Train if missing
python scripts/train_plainview_models.py

# Check Intelligence service logs
docker logs summit-intelligence | grep "model"
```

## Next Steps

### Production Readiness
- [ ] Replace mock ONNX models with real trained models
- [ ] Add authentication between services (JWT/mTLS)
- [ ] Implement advisory deduplication
- [ ] Add retry logic and DLQ for failed forwards
- [ ] Set up monitoring/alerting for adapter
- [ ] Database persistence for Plainview insights

### Enhanced Features
- [ ] Bidirectional event forwarding (Plainview → Summit.OS)
- [ ] Historical trend analysis
- [ ] Multi-site insight aggregation
- [ ] Custom alert rules in Plainview
- [ ] Integration with Sentinel for spread predictions

## Support

For issues or questions:
1. Check logs: `docker logs <container-name>`
2. Verify health endpoints
3. Review environment variables
4. Test API endpoints manually with `curl`

---

**Built with ❤️ for integrated industrial autonomy**
