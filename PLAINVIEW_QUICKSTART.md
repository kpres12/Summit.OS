# Plainview AI Integration - Quick Start

## What Was Built

Complete AI integration between Summit.OS Intelligence and Plainview, including:

✅ **Plainview Intelligence Adapter** (`apps/plainview-adapter/`)
- Bridges Summit.OS advisories to Plainview insights
- WebSocket support for real-time updates
- HTTP endpoints for health checks

✅ **Plainview-specific ML Models** (`apps/intelligence/plainview_models.py`)
- Flow anomaly detection
- Valve health prediction  
- Pipeline integrity assessment
- Rule-based fallback + ONNX support

✅ **Plainview Intelligence Module** (`Plainview/services/api/app/modules/intelligence.py`)
- Receives insights from adapter
- Stores and serves insights via API
- Emits to SSE event stream

✅ **Docker Integration** (updated `infra/docker/docker-compose.yml`)
- Added plainview-adapter service
- Configured networking between systems

✅ **ML Model Training** (`scripts/train_plainview_models.py`)
- Generates ONNX models for Plainview domains
- Synthetic training data (replace with real data)

## 5-Minute Setup

### 1. Train Models (one-time)
```bash
cd /Users/kpres12/Downloads/Summit.OS

# Install dependencies
pip install onnx scikit-learn skl2onnx numpy

# Train models
python scripts/train_plainview_models.py
# Creates: models/flow_anomaly.onnx, valve_health.onnx, pipeline_integrity.onnx
```

### 2. Start Summit.OS
```bash
cd /Users/kpres12/Downloads/Summit.OS

# Start everything
make dev

# OR start just what you need:
make dev-services    # Infrastructure only
make dev-backend     # Backend services
```

**Running on:**
- Intelligence: http://localhost:8003
- Adapter: http://localhost:8005
- Redis: localhost:6379
- Postgres: localhost:5433

### 3. Start Plainview
```bash
cd /Users/kpres12/Downloads/Plainview

# Terminal 1: API
npm run dev -w @plainview/api

# Terminal 2: Dashboard
npm run dev -w @plainview/dashboard
```

**Running on:**
- API: http://localhost:4000
- Dashboard: http://localhost:5173

### 4. Test Integration
```bash
# Check adapter
curl http://localhost:8005/health

# Test flow analysis
curl -X POST http://localhost:8003/plainview/flow/analyze \
  -H "Content-Type: application/json" \
  -d '{"flow_rate_lpm": 180, "pressure_pa": 2600000, "temperature_c": 50}'

# Check insights
curl http://localhost:4000/intelligence/insights
```

## How It Works

```
Summit.OS Intelligence → Adapter → Plainview Intelligence Module → Dashboard
     (advisories)      (enriches)      (stores/emits)          (displays)
```

### Data Flow
1. Intelligence service generates risk advisories from observations
2. Adapter polls Intelligence every 5 seconds for new advisories
3. Adapter enriches with Plainview domain context (FlowIQ, ValveOps, etc.)
4. Adapter forwards to Plainview `/intelligence/insights` endpoint
5. Plainview emits insight to SSE stream (`/events`)
6. Dashboard AIAssistant can display insights

### WebSocket Support
```typescript
// Connect from Plainview dashboard
const ws = new WebSocket('ws://localhost:8005/ws/plainview');

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === 'insight') {
    // Real-time AI insight from Summit.OS
    console.log(msg.data);
  }
};
```

## Key Endpoints

### Summit.OS Intelligence
```bash
GET  /advisories                    # List all advisories
POST /plainview/flow/analyze        # Analyze flow metrics
POST /plainview/valve/health        # Predict valve health
POST /plainview/pipeline/assess     # Assess pipeline integrity
```

### Plainview Adapter
```bash
GET /health                         # Health check
GET /readyz                         # Readiness probe
WS  /ws/plainview                   # WebSocket for insights
```

### Plainview API
```bash
POST /intelligence/insights         # Receive insight (from adapter)
GET  /intelligence/insights         # List insights
GET  /intelligence/insights/{id}    # Get specific insight
```

## Architecture

```
┌─────────────────────────────────────────────┐
│            Summit.OS                        │
│                                              │
│  Fusion → Intelligence → Adapter            │
│   (8002)     (8003)        (8005)           │
│                              │               │
└──────────────────────────────┼───────────────┘
                               │ HTTP/WebSocket
                               ▼
┌──────────────────────────────────────────────┐
│            Plainview                         │
│                                              │
│  Intelligence Module → Event Bus → Dashboard│
│   /intelligence/*        /events   (5173)   │
│                                              │
└──────────────────────────────────────────────┘
```

## Files Created/Modified

### Summit.OS
```
apps/plainview-adapter/
  ├── main.py                        # Adapter service
  ├── requirements.txt
  └── Dockerfile

apps/intelligence/
  └── plainview_models.py            # ML models for Plainview

apps/intelligence/main.py            # Added Plainview endpoints

scripts/
  └── train_plainview_models.py      # Model training

infra/docker/
  └── docker-compose.yml             # Added adapter service

PLAINVIEW_INTEGRATION.md             # Full documentation
PLAINVIEW_QUICKSTART.md              # This file
```

### Plainview
```
services/api/app/modules/
  └── intelligence.py                # New intelligence module

services/api/app/
  └── main.py                        # Register intelligence module
```

## Next Steps

### Connect Dashboard
The AIAssistant component in Plainview can now be connected to:
1. Poll `/intelligence/insights` for new insights
2. Connect via WebSocket to `ws://localhost:8005/ws/plainview`
3. Display real-time AI recommendations

### Enhance Models
Replace synthetic models with real training data:
```bash
# Your real data
python scripts/train_plainview_models.py --data-path /path/to/real/data
```

### Production Deploy
1. Add authentication (JWT/mTLS)
2. Configure environment variables
3. Set up monitoring/alerting
4. Enable ONNX model loading in Intelligence

## Troubleshooting

**Adapter can't reach Plainview?**
- Verify Plainview API is on http://localhost:4000
- Check adapter logs: `docker logs summit-plainview-adapter`
- On Linux: add `--add-host=host.docker.internal:host-gateway` to docker run

**No insights appearing?**
```bash
# Verify chain
curl http://localhost:8003/advisories     # Step 1: Advisories exist?
docker logs summit-plainview-adapter      # Step 2: Adapter processing?
curl http://localhost:4000/intelligence/insights  # Step 3: Plainview received?
```

**Models not loading?**
```bash
ls -la models/  # Check files exist
python scripts/train_plainview_models.py  # Retrain if missing
```

## Support

See full documentation: [PLAINVIEW_INTEGRATION.md](./PLAINVIEW_INTEGRATION.md)

---

**🎉 You now have full AI integration between Summit.OS and Plainview!**
