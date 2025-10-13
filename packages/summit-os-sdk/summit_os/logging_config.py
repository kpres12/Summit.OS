"""
Summit.OS Structured Logging Configuration

Provides consistent structured logging across all services with:
- JSON formatting for log aggregation
- Request ID tracking
- Correlation IDs for distributed tracing
- Service context
"""

import logging
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Context vars for request tracking
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
correlation_id_ctx: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def configure_logging(service_name: str, log_level: str = "INFO") -> None:
    """
    Configure structured logging for a service.
    
    Args:
        service_name: Name of the service (e.g., "fusion", "intelligence")
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            add_service_context(service_name),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def add_service_context(service_name: str):
    """Add service context to all log messages."""
    def processor(logger, method_name, event_dict):
        event_dict["service"] = service_name
        
        # Add request/correlation IDs if available
        request_id = request_id_ctx.get()
        if request_id:
            event_dict["request_id"] = request_id
        
        correlation_id = correlation_id_ctx.get()
        if correlation_id:
            event_dict["correlation_id"] = correlation_id
        
        return event_dict
    return processor


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for request logging with correlation tracking.
    
    Adds request_id and correlation_id to all logs within a request context.
    """
    
    async def dispatch(self, request: Request, call_next):
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request_id_ctx.set(request_id)
        
        # Extract or generate correlation ID (for distributed tracing)
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        correlation_id_ctx.set(correlation_id)
        
        # Get logger
        logger = structlog.get_logger()
        
        # Log request
        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            client_host=request.client.host if request.client else None,
        )
        
        # Process request
        try:
            response: Response = await call_next(request)
            
            # Log response
            logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
            )
            
            # Add tracking headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
        
        except Exception as e:
            # Log error
            logger.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                exc_info=True,
            )
            raise
        
        finally:
            # Clean up context
            request_id_ctx.set(None)
            correlation_id_ctx.set(None)


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """
    Bind additional context to all logs in the current context.
    
    Example:
        bind_context(user_id="user_123", org_id="org_456")
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """
    Unbind context keys from the current context.
    
    Example:
        unbind_context("user_id", "org_id")
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all context variables."""
    structlog.contextvars.clear_contextvars()
