# Summit.OS Deployment Guide

## Overview

This guide covers deploying Summit.OS across different environments, from local development to production edge and cloud deployments.

## 🏗️ Deployment Architecture

### Development Environment
```
┌─────────────────────────────────────────────────────────────┐
│                    Developer Machine                       │
│ ┌───────────────┬─────────────────────────────────────────┐ │
│ │ FireLine      │ Next.js development server              │ │
│ │ Console       │ http://localhost:3000                   │ │
│ └───────────────┴─────────────────────────────────────────┘ │
│ ┌───────────────┬─────────────────────────────────────────┐ │
│ │ Summit.OS     │ Docker Compose services                 │ │
│ │ Services      │ - API Gateway (8000)                    │ │
│ │               │ - Data Fabric (8001)                   │ │
│ │               │ - Sensor Fusion (8002)                 │ │
│ │               │ - Intelligence (8003)                  │ │
│ │               │ - Mission Tasking (8004)               │ │
│ └───────────────┴─────────────────────────────────────────┘ │
│ ┌───────────────┬─────────────────────────────────────────┐ │
│ │ Infrastructure│ Redis, PostgreSQL, MQTT, Prometheus    │ │
│ └───────────────┴─────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Edge Deployment
```
┌─────────────────────────────────────────────────────────────┐
│                    Edge Environment                        │
│ ┌───────────────┬─────────────────────────────────────────┐ │
│ │ Edge Devices  │ Drones, UGVs, Sensors, Cameras         │ │
│ │               │ - Summit Edge Agent                    │ │
│ │               │ - ONNX AI Models                       │ │
│ │               │ - Local Data Buffer                    │ │
│ └───────────────┴─────────────────────────────────────────┘ │
│ ┌───────────────┬─────────────────────────────────────────┐ │
│ │ Edge Summit.OS│ k3s cluster with Summit.OS services    │ │
│ │               │ - Data Fabric (local)                  │ │
│ │               │ - Sensor Fusion (edge)                │ │
│ │               │ - Mission Tasking (edge)               │ │
│ └───────────────┴─────────────────────────────────────────┘ │
│ ┌───────────────┬─────────────────────────────────────────┐ │
│ │ Edge Storage  │ Local PostgreSQL + Redis               │ │
│ │               │ - Offline operation                    │ │
│ │               │ - Store-and-forward                    │ │
│ └───────────────┴─────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Cloud Deployment
```
┌─────────────────────────────────────────────────────────────┐
│                    Cloud Environment                       │
│ ┌───────────────┬─────────────────────────────────────────┐ │
│ │ Load Balancer │ AWS ALB / GCP LB / Azure LB            │ │
│ └───────────────┴─────────────────────────────────────────┘ │
│ ┌───────────────┬─────────────────────────────────────────┐ │
│ │ Summit.OS     │ EKS / GKE / AKS cluster                 │ │
│ │ Services      │ - Auto-scaling services                 │ │
│ │               │ - Multi-region deployment              │ │
│ │               │ - High availability                   │ │
│ └───────────────┴─────────────────────────────────────────┘ │
│ ┌───────────────┬─────────────────────────────────────────┐ │
│ │ Cloud Storage │ RDS / Cloud SQL / Azure Database       │ │
│ │               │ - Managed databases                    │ │
│ │               │ - Automated backups                    │ │
│ │               │ - Multi-AZ deployment                  │ │
│ └───────────────┴─────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Deployment Methods

### 1. Local Development Deployment

#### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for FireLine Console)
- Python 3.11+ (for Summit.OS services)
- Git

#### Quick Start
```bash
# Clone repository
git clone https://github.com/bigmt/summit-os.git
cd summit-os

# Start development environment
make dev

# Access services
# - FireLine Console: http://localhost:3000
# - API Gateway: http://localhost:8000
# - Grafana: http://localhost:3001
```

#### Manual Setup
```bash
# Start infrastructure services
docker-compose -f infra/docker/docker-compose.yml up -d redis postgres mqtt

# Start Summit.OS services
docker-compose -f infra/docker/docker-compose.yml up -d fabric fusion intelligence tasking api-gateway

# Start FireLine Console
cd apps/console
npm install
npm run dev
```

### 2. Edge Deployment (k3s)

#### Prerequisites
- k3s cluster
- kubectl configured
- Helm 3.x

#### Deploy to Edge
```bash
# Create namespace
kubectl create namespace summit-os

# Deploy Summit.OS to k3s
helm install summit-os ./infra/k8s/summit-os \
  --namespace summit-os \
  --set edge.enabled=true \
  --set storage.local=true \
  --set ai.models.edge=true

# Verify deployment
kubectl get pods -n summit-os
```

#### Edge Configuration
```yaml
# infra/k8s/summit-os/values-edge.yaml
edge:
  enabled: true
  offline: true
  storeAndForward: true

storage:
  local: true
  persistence:
    enabled: true
    size: 100Gi

ai:
  models:
    edge: true
    onnx: true
    localInference: true

networking:
  mqtt:
    enabled: true
    port: 1883
  grpc:
    enabled: true
    port: 9090
```

### 3. Cloud Deployment (EKS/GKE/AKS)

#### AWS EKS Deployment
```bash
# Create EKS cluster
eksctl create cluster --name summit-os --region us-west-2 --nodes 3

# Deploy Summit.OS to EKS
helm install summit-os ./infra/k8s/summit-os \
  --namespace summit-os \
  --set cloud.provider=aws \
  --set storage.rds.enabled=true \
  --set monitoring.prometheus.enabled=true \
  --set monitoring.grafana.enabled=true

# Configure ingress
kubectl apply -f infra/k8s/ingress/aws-alb.yaml
```

#### Google GKE Deployment
```bash
# Create GKE cluster
gcloud container clusters create summit-os \
  --zone us-central1-a \
  --num-nodes 3

# Deploy Summit.OS to GKE
helm install summit-os ./infra/k8s/summit-os \
  --namespace summit-os \
  --set cloud.provider=gcp \
  --set storage.cloudsql.enabled=true \
  --set monitoring.stackdriver.enabled=true
```

#### Azure AKS Deployment
```bash
# Create AKS cluster
az aks create --resource-group summit-os-rg \
  --name summit-os-cluster \
  --node-count 3

# Deploy Summit.OS to AKS
helm install summit-os ./infra/k8s/summit-os \
  --namespace summit-os \
  --set cloud.provider=azure \
  --set storage.cosmosdb.enabled=true \
  --set monitoring.azure.enabled=true
```

### 4. Hybrid Deployment (Edge + Cloud)

#### Edge-Cloud Synchronization
```yaml
# infra/k8s/summit-os/values-hybrid.yaml
edge:
  enabled: true
  cloudSync: true
  syncInterval: 300s

cloud:
  enabled: true
  edgeSync: true
  federation: true

storage:
  hybrid: true
  edge:
    local: true
    sync: true
  cloud:
    managed: true
    backup: true

networking:
  edgeToCloud: true
  cloudToEdge: true
  mqtt:
    bridge: true
    topics:
      - "summit-os/edge/+/telemetry"
      - "summit-os/edge/+/alerts"
      - "summit-os/cloud/+/commands"
```

## 🔧 Configuration Management

### Environment Variables
```bash
# Summit.OS Configuration
export SUMMIT_API_KEY="your-api-key"
export SUMMIT_BASE_URL="https://api.summit-os.bigmt.ai"
export SUMMIT_ENVIRONMENT="production"

# Database Configuration
export POSTGRES_URL="postgresql://user:pass@host:5432/summit_os"
export REDIS_URL="redis://host:6379"

# MQTT Configuration
export MQTT_BROKER="mqtt.summit-os.bigmt.ai"
export MQTT_PORT="1883"
export MQTT_USERNAME="summit"
export MQTT_PASSWORD="your-password"

# AI Configuration
export AI_MODELS_PATH="/models"
export AI_EDGE_INFERENCE="true"
export AI_CLOUD_TRAINING="true"

# Monitoring Configuration
export PROMETHEUS_ENDPOINT="http://prometheus:9090"
export GRAFANA_ENDPOINT="http://grafana:3000"
```

### Configuration Files
```yaml
# config/summit-os.yaml
summit:
  api:
    base_url: "https://api.summit-os.bigmt.ai"
    timeout: 30
    retry_attempts: 3
  
  database:
    postgres:
      url: "postgresql://user:pass@host:5432/summit_os"
      pool_size: 20
      max_overflow: 30
    
    redis:
      url: "redis://host:6379"
      db: 0
      max_connections: 100
  
  mqtt:
    broker: "mqtt.summit-os.bigmt.ai"
    port: 1883
    username: "summit"
    password: "your-password"
    keepalive: 60
  
  ai:
    models:
      path: "/models"
      edge_inference: true
      cloud_training: true
      onnx_optimization: true
  
  monitoring:
    prometheus:
      enabled: true
      endpoint: "http://prometheus:9090"
    
    grafana:
      enabled: true
      endpoint: "http://grafana:3000"
    
    jaeger:
      enabled: true
      endpoint: "http://jaeger:14268"
```

## 📊 Monitoring and Observability

### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'summit-os-services'
    static_configs:
      - targets: ['api-gateway:8000', 'fabric:8001', 'fusion:8002', 'intelligence:8003', 'tasking:8004']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'summit-os-edge'
    static_configs:
      - targets: ['edge-agent:8005']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'infrastructure'
    static_configs:
      - targets: ['redis:6379', 'postgres:5432', 'mqtt:1883']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Grafana Dashboards
```json
{
  "dashboard": {
    "title": "Summit.OS Overview",
    "panels": [
      {
        "title": "Service Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"summit-os-services\"}",
            "legendFormat": "{{instance}}"
          }
        ]
      },
      {
        "title": "Telemetry Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(summit_telemetry_total[5m])",
            "legendFormat": "Telemetry/sec"
          }
        ]
      },
      {
        "title": "Alert Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(summit_alerts_total[5m])",
            "legendFormat": "Alerts/sec"
          }
        ]
      }
    ]
  }
}
```

## 🔒 Security Configuration

### TLS/SSL Setup
```yaml
# security/tls.yaml
tls:
  enabled: true
  certificates:
    - name: "summit-os-tls"
      secret: "summit-os-tls-secret"
      hosts:
        - "api.summit-os.bigmt.ai"
        - "ws.summit-os.bigmt.ai"
        - "mqtt.summit-os.bigmt.ai"
  
  mqtt:
    enabled: true
    port: 8883
    certificate: "/certs/mqtt.crt"
    key: "/certs/mqtt.key"
  
  grpc:
    enabled: true
    port: 9443
    certificate: "/certs/grpc.crt"
    key: "/certs/grpc.key"
```

### Authentication Setup
```yaml
# security/auth.yaml
authentication:
  keycloak:
    enabled: true
    url: "https://auth.summit-os.bigmt.ai"
    realm: "summit-os"
    client_id: "summit-os-client"
  
  jwt:
    secret: "your-jwt-secret"
    expiration: 3600
  
  api_keys:
    enabled: true
    rotation: 30d
  
  mqtt:
    username: "summit"
    password: "your-mqtt-password"
```

## 🚀 CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy Summit.OS

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          make test
          make lint
          make security-scan

  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker images
        run: |
          docker build -t summit-os/api-gateway ./apps/api-gateway
          docker build -t summit-os/fabric ./apps/fabric
          docker build -t summit-os/fusion ./apps/fusion
          docker build -t summit-os/intelligence ./apps/intelligence
          docker build -t summit-os/tasking ./apps/tasking

  deploy-dev:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/develop'
    steps:
      - name: Deploy to development
        run: |
          helm upgrade --install summit-os-dev ./infra/k8s/summit-os \
            --namespace summit-os-dev \
            --set environment=development

  deploy-prod:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          helm upgrade --install summit-os ./infra/k8s/summit-os \
            --namespace summit-os \
            --set environment=production
```

### GitLab CI
```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

test:
  stage: test
  script:
    - make test
    - make lint
    - make security-scan

build:
  stage: build
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

deploy:
  stage: deploy
  script:
    - helm upgrade --install summit-os ./infra/k8s/summit-os \
        --namespace summit-os \
        --set image.tag=$CI_COMMIT_SHA
  only:
    - main
```

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] **Environment Setup** - Configure target environment
- [ ] **Dependencies** - Install required tools and dependencies
- [ ] **Secrets Management** - Configure API keys and passwords
- [ ] **Network Configuration** - Configure networking and firewall rules
- [ ] **Storage Setup** - Configure databases and storage
- [ ] **Monitoring Setup** - Configure monitoring and logging

### Deployment
- [ ] **Infrastructure** - Deploy infrastructure components
- [ ] **Summit.OS Services** - Deploy Summit.OS microservices
- [ ] **FireLine Console** - Deploy FireLine Console frontend
- [ ] **AI Models** - Deploy AI models to edge/cloud
- [ ] **Configuration** - Apply configuration settings
- [ ] **Security** - Configure security settings

### Post-Deployment
- [ ] **Health Checks** - Verify all services are healthy
- [ ] **Integration Tests** - Run integration tests
- [ ] **Performance Tests** - Run performance tests
- [ ] **Monitoring** - Verify monitoring is working
- [ ] **Documentation** - Update deployment documentation
- [ ] **Training** - Train operators on new deployment

## 🔧 Troubleshooting

### Common Issues

#### Service Not Starting
```bash
# Check service logs
kubectl logs -n summit-os deployment/api-gateway
kubectl logs -n summit-os deployment/fabric

# Check service status
kubectl get pods -n summit-os
kubectl describe pod <pod-name> -n summit-os
```

#### Database Connection Issues
```bash
# Check database connectivity
kubectl exec -it <pod-name> -n summit-os -- psql $POSTGRES_URL

# Check Redis connectivity
kubectl exec -it <pod-name> -n summit-os -- redis-cli -u $REDIS_URL ping
```

#### MQTT Connection Issues
```bash
# Check MQTT broker
kubectl exec -it <pod-name> -n summit-os -- mosquitto_pub -h $MQTT_BROKER -t test -m "test"
```

#### AI Model Issues
```bash
# Check AI model deployment
kubectl exec -it <pod-name> -n summit-os -- ls -la /models

# Test AI inference
kubectl exec -it <pod-name> -n summit-os -- python -c "import onnxruntime; print('ONNX Runtime OK')"
```

### Performance Optimization

#### Database Optimization
```sql
-- Create indexes for better performance
CREATE INDEX idx_telemetry_device_timestamp ON telemetry(device_id, timestamp);
CREATE INDEX idx_alerts_severity_timestamp ON alerts(severity, timestamp);
CREATE INDEX idx_missions_status ON missions(status);
```

#### Redis Optimization
```bash
# Configure Redis for high performance
redis-cli CONFIG SET maxmemory 2gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli CONFIG SET save "900 1 300 10 60 10000"
```

#### MQTT Optimization
```conf
# mosquitto.conf
max_connections 10000
max_inflight_messages 100
max_queued_messages 1000
message_size_limit 268435456
```

## 📞 Support

- **Documentation**: https://docs.summit-os.bigmt.ai
- **Deployment Guide**: https://deploy.summit-os.bigmt.ai
- **Support**: support@bigmt.ai
- **Community**: https://community.summit-os.bigmt.ai
- **Status Page**: https://status.summit-os.bigmt.ai
