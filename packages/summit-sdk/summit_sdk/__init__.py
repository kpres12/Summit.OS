"""
Summit.OS SDK

Programmatic Python SDK for interacting with Summit.OS services.
Python SDK for Summit.OS services (REST + WebSocket + gRPC).
"""

from .client import SummitClient
from .integration import IntegrationClient
from .errors import (
    SummitError,
    ErrorCode,
    ConnectionError,
    TimeoutError,
    CircuitOpenError,
    AuthError,
    NotFoundError,
    ValidationError,
    RateLimitError,
    ServerError,
    NotConnectedError,
)
from .retry import RetryPolicy, CircuitBreaker

__version__ = "0.1.0a1"

__all__ = [
    "SummitClient",
    "IntegrationClient",
    # Errors
    "SummitError",
    "ErrorCode",
    "ConnectionError",
    "TimeoutError",
    "CircuitOpenError",
    "AuthError",
    "NotFoundError",
    "ValidationError",
    "RateLimitError",
    "ServerError",
    "NotConnectedError",
    # Resilience
    "RetryPolicy",
    "CircuitBreaker",
]
