"""
Distributed Tracing for Heli.OS

OpenTelemetry-compatible tracing that works with or without
the opentelemetry SDK. Provides:
- Span creation with context propagation
- Automatic FastAPI middleware
- Trace export (OTLP, Jaeger, console)
- Correlation ID injection for log → trace mapping

Falls back to a lightweight in-process tracer when OTel is not installed.
"""

from __future__ import annotations

import time
import uuid
import logging
import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional
from contextvars import ContextVar

logger = logging.getLogger("observability.tracing")

# Context propagation
_current_span: ContextVar[Optional["Span"]] = ContextVar("current_span", default=None)
_current_trace: ContextVar[str] = ContextVar("current_trace", default="")


@dataclass
class Span:
    """A trace span."""

    trace_id: str
    span_id: str
    parent_id: str = ""
    operation: str = ""
    service: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = "ok"  # "ok", "error"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None:
        self.events.append(
            {
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {},
            }
        )

    def set_error(self, error: Exception) -> None:
        self.status = "error"
        self.attributes["error.type"] = type(error).__name__
        self.attributes["error.message"] = str(error)

    def to_dict(self) -> Dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "operation": self.operation,
            "service": self.service,
            "start": self.start_time,
            "end": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


class SpanExporter:
    """Base class for span exporters."""

    def export(self, spans: List[Span]) -> None:
        pass

    def shutdown(self) -> None:
        pass


class ConsoleExporter(SpanExporter):
    """Export spans to console/logs."""

    def export(self, spans: List[Span]) -> None:
        for span in spans:
            logger.info(
                f"[TRACE] {span.service}/{span.operation} "
                f"trace={span.trace_id[:8]} span={span.span_id[:8]} "
                f"duration={span.duration_ms:.1f}ms status={span.status}"
            )


class InMemoryExporter(SpanExporter):
    """Keep spans in memory for testing/debugging."""

    def __init__(self, max_spans: int = 10000):
        self.spans: List[Span] = []
        self.max_spans = max_spans

    def export(self, spans: List[Span]) -> None:
        self.spans.extend(spans)
        if len(self.spans) > self.max_spans:
            self.spans = self.spans[-self.max_spans :]

    def get_traces(self) -> Dict[str, List[Span]]:
        traces: Dict[str, List[Span]] = {}
        for span in self.spans:
            if span.trace_id not in traces:
                traces[span.trace_id] = []
            traces[span.trace_id].append(span)
        return traces

    def clear(self) -> None:
        self.spans.clear()


class Tracer:
    """
    Lightweight tracer compatible with OpenTelemetry patterns.

    Uses OTel SDK when available, otherwise falls back to
    in-process tracing.
    """

    def __init__(
        self, service_name: str = "heli-os", exporter: Optional[SpanExporter] = None
    ):
        self.service_name = service_name
        self._exporter = exporter or ConsoleExporter()
        self._otel_available = self._check_otel()
        self._otel_tracer = None

        if self._otel_available:
            self._init_otel()

    @staticmethod
    def _check_otel() -> bool:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider

            return True
        except ImportError:
            return False

    def _init_otel(self):
        import os

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                SimpleSpanProcessor,
                ConsoleSpanExporter,
            )

            resource = Resource.create({"service.name": self.service_name})
            provider = TracerProvider(resource=resource)

            otlp_endpoint = os.getenv("OTLP_ENDPOINT", "")
            if otlp_endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                        OTLPSpanExporter,
                    )

                    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                    logger.info(f"OTLP exporter configured: {otlp_endpoint}")
                except ImportError:
                    logger.warning(
                        "opentelemetry-exporter-otlp-proto-grpc not installed — using console exporter"
                    )
                    provider.add_span_processor(
                        SimpleSpanProcessor(ConsoleSpanExporter())
                    )
            else:
                provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

            trace.set_tracer_provider(provider)
            self._otel_tracer = trace.get_tracer(self.service_name)
            logger.info(f"OpenTelemetry tracer initialized for {self.service_name}")
        except Exception as e:
            logger.warning(f"OTel init failed, using fallback: {e}")

    @contextlib.contextmanager
    def start_span(
        self, operation: str, attributes: Optional[Dict] = None
    ) -> Generator[Span, None, None]:
        """Start a new span (context manager)."""
        parent = _current_span.get()
        trace_id = _current_trace.get() or uuid.uuid4().hex[:32]

        span = Span(
            trace_id=trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent.span_id if parent else "",
            operation=operation,
            service=self.service_name,
            start_time=time.time(),
            attributes=attributes or {},
        )

        token_span = _current_span.set(span)
        token_trace = _current_trace.set(trace_id)

        try:
            yield span
            if span.status != "error":
                span.status = "ok"
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            span.end_time = time.time()
            self._exporter.export([span])
            _current_span.reset(token_span)
            _current_trace.reset(token_trace)

    def get_current_span(self) -> Optional[Span]:
        return _current_span.get()

    def get_current_trace_id(self) -> str:
        return _current_trace.get() or ""


# ── FastAPI Middleware ──────────────────────────────────────


def create_tracing_middleware(tracer: Tracer):
    """Create FastAPI middleware for automatic request tracing."""

    async def middleware(request, call_next):
        # Extract trace context from incoming headers
        trace_id = request.headers.get("X-Trace-ID", "")
        if trace_id:
            _current_trace.set(trace_id)

        operation = f"{request.method} {request.url.path}"
        with tracer.start_span(
            operation,
            attributes={
                "http.method": request.method,
                "http.url": str(request.url),
                "http.route": request.url.path,
            },
        ) as span:
            # Bind trace_id to structlog context so all log lines in this request carry it
            try:
                import structlog

                structlog.contextvars.clear_contextvars()
                structlog.contextvars.bind_contextvars(
                    trace_id=span.trace_id,
                    service=tracer.service_name,
                )
            except ImportError:
                pass

            response = await call_next(request)
            span.set_attribute("http.status_code", response.status_code)
            if response.status_code >= 400:
                span.status = "error"
            # Propagate trace ID to caller
            response.headers["X-Trace-ID"] = span.trace_id
            return response

    return middleware


# ── Global tracer ──────────────────────────────────────────

_tracer_registry: Dict[str, Tracer] = {}


def get_tracer(service_name: str = "heli-os") -> Tracer:
    """Get or create a tracer for the given service name."""
    if service_name not in _tracer_registry:
        _tracer_registry[service_name] = Tracer(service_name)
    return _tracer_registry[service_name]
