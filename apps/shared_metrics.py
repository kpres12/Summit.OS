"""
Shared Prometheus metrics for Summit.OS services.

Usage in any FastAPI service:
    from shared_metrics import instrument_app, metrics_endpoint
    
    app = FastAPI()
    instrument_app(app, service_name="my-service")
    app.add_route("/metrics", metrics_endpoint)
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Response
from fastapi.responses import Response as FastAPIResponse
import time

# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'HTTP requests in progress',
    ['method', 'endpoint']
)

# Error metrics
http_errors_total = Counter(
    'http_errors_total',
    'Total HTTP errors',
    ['method', 'endpoint', 'status', 'exception']
)

# Application metrics
app_info = Gauge(
    'app_info',
    'Application info',
    ['service', 'version']
)


def instrument_app(app: FastAPI, service_name: str, version: str = "1.0.0"):
    """Add Prometheus instrumentation middleware to FastAPI app."""
    
    app_info.labels(service=service_name, version=version).set(1)
    
    @app.middleware("http")
    async def prometheus_middleware(request, call_next):
        method = request.method
        path = request.url.path
        
        # Skip metrics endpoint
        if path == "/metrics":
            return await call_next(request)
        
        # Track in-progress requests
        http_requests_in_progress.labels(method=method, endpoint=path).inc()
        
        start = time.time()
        status = 500
        exception_type = None
        
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        except Exception as e:
            exception_type = type(e).__name__
            http_errors_total.labels(
                method=method,
                endpoint=path,
                status=status,
                exception=exception_type
            ).inc()
            raise
        finally:
            duration = time.time() - start
            
            # Record metrics
            http_requests_total.labels(
                method=method,
                endpoint=path,
                status=status
            ).inc()
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=path
            ).observe(duration)
            
            http_requests_in_progress.labels(method=method, endpoint=path).dec()


def metrics_endpoint():
    """Return Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
