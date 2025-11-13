# Production Readiness Checklist

## Critical (Must-Fix Before Launch) ✅

- [x] **Secrets externalized** - No hardcoded passwords in docker-compose or code
- [x] **CORS locked down** - Configurable via `CORS_ORIGINS` env var
- [x] **--reload disabled** - Set `UVICORN_RELOAD=true` only in dev
- [x] **CI gates enforced** - Removed `|| true` safety valves; builds fail on errors
- [x] **Frontend tests added** - Console has test script stub
- [x] **Console Dockerfile** - Multi-stage Docker build for Next.js
- [x] **DB migrations enabled** - `FABRIC_SKIP_MIGRATIONS=false` by default
- [x] **Service healthchecks** - All compose services have healthcheck blocks
- [x] **Metrics endpoint** - API Gateway exposes `/metrics`
- [x] **Secrets management docs** - See `docs/SECRETS_MANAGEMENT.md`

## High (Fix Within First Week)

- [ ] **Dependency pinning with hashes** - Run `pip-compile --generate-hashes` for all services
- [ ] **Structured error taxonomy** - Create error codes enum; add `Retry-After` headers
- [ ] **Rate limiting** - Add slowapi or custom limiter to public endpoints
- [ ] **Timeout tuning** - Set httpx timeouts: connect=5s, read=30s, write=10s
- [ ] **Circuit breakers** - Add retry logic with exponential backoff
- [ ] **Metrics on all services** - Add `/metrics` to fabric, fusion, intelligence, tasking
- [ ] **Structured logging** - Ensure all services use structlog with JSON output
- [ ] **Error alerting** - Wire Prometheus alerts for 5xx rate > 1%

## Medium (Fix Within First Month)

- [ ] **SLO tracking** - Define availability (99.9%) and latency (p95 < 500ms) SLOs
- [ ] **Synthetic checks** - Add blackbox_exporter probes for critical paths
- [ ] **Load testing** - Run k6 or Locust to 10x expected traffic
- [ ] **Chaos engineering** - Monthly game days: kill DB, slow network, CPU spike
- [ ] **Backup & restore** - Automate Postgres backups; test restore procedure
- [ ] **Schema versioning** - All API/event schemas have version field
- [ ] **Contract tests** - CI runs Pact-style consumer tests
- [ ] **Cost guardrails** - Tag all cloud resources; alert on budget overruns
- [ ] **Security scan** - Run Trivy/Grype on container images; fail on high CVEs
- [ ] **SBOM generation** - Add CycloneDX/SPDX to CI

## Commands to Run Now

### 1. Dependency Pinning

```bash
# For each Python service
cd apps/api-gateway
pip-compile requirements.txt --generate-hashes --output-file requirements.lock
# Repeat for fabric, fusion, intelligence, tasking
```

### 2. Update Dockerfiles to Use Hashed Requirements

```dockerfile
COPY requirements.lock .
RUN pip install --require-hashes -r requirements.lock
```

### 3. Add Error Taxonomy

Create `apps/shared_errors.py`:

```python
from enum import Enum
from fastapi import HTTPException

class ErrorCode(str, Enum):
    UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"  # Retry after 5s
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"  # Retry after 60s
    VALIDATION_FAILED = "VALIDATION_FAILED"  # Do not retry
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"  # Do not retry
    INTERNAL_ERROR = "INTERNAL_ERROR"  # Retry after 30s

def retriable(code: ErrorCode) -> bool:
    return code in {ErrorCode.UPSTREAM_TIMEOUT, ErrorCode.INTERNAL_ERROR, ErrorCode.RATE_LIMIT_EXCEEDED}

def retry_after_seconds(code: ErrorCode) -> int:
    mapping = {
        ErrorCode.UPSTREAM_TIMEOUT: 5,
        ErrorCode.RATE_LIMIT_EXCEEDED: 60,
        ErrorCode.INTERNAL_ERROR: 30,
    }
    return mapping.get(code, 0)
```

### 4. Add Rate Limiting

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/v1/alerts")
@limiter.limit("20/minute")
async def list_alerts():
    ...
```

### 5. Tune Timeouts

```python
# In all httpx clients
client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
)
```

### 6. Add Circuit Breaker (tenacity)

```bash
pip install tenacity
```

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(httpx.TimeoutException)
)
async def call_upstream():
    async with httpx.AsyncClient() as client:
        return await client.get(url)
```

## Monitoring Setup

### Prometheus Alerts

Add to `infra/docker/prometheus-alerts.yml`:

```yaml
groups:
  - name: summit-os
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(http_errors_total[5m]) > 0.01
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate on {{ $labels.service }}"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "p95 latency > 500ms on {{ $labels.endpoint }}"
```

### Grafana Dashboards

Import standard FastAPI dashboard: https://grafana.com/grafana/dashboards/14859

## Pre-Launch Validation

Run these commands before deploying:

```bash
# 1. Lint passes
make lint

# 2. Tests pass
make test

# 3. Build succeeds
docker-compose build

# 4. Stack starts healthy
docker-compose up -d
sleep 30
make health

# 5. Metrics exposed
curl http://localhost:8000/metrics | grep http_requests_total

# 6. No secrets in logs
docker-compose logs | grep -i "password\|secret\|token" || echo "Clean"
```

## See Also

- [Secrets Management](./SECRETS_MANAGEMENT.md)
- [WARP.md](../WARP.md) - Dev environment guide
