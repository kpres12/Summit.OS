"""
Summit.OS SDK — Structured Errors

Provides a typed error hierarchy with machine-readable error codes
so callers can programmatically handle failures.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    """Machine-readable error codes."""
    # Transport
    CONNECTION_FAILED = "SUMMIT-E1001"
    TIMEOUT = "SUMMIT-E1002"
    CIRCUIT_OPEN = "SUMMIT-E1003"

    # Auth
    UNAUTHORIZED = "SUMMIT-E2001"
    FORBIDDEN = "SUMMIT-E2002"
    TOKEN_EXPIRED = "SUMMIT-E2003"

    # Client
    NOT_FOUND = "SUMMIT-E3001"
    VALIDATION_ERROR = "SUMMIT-E3002"
    CONFLICT = "SUMMIT-E3003"
    RATE_LIMITED = "SUMMIT-E3004"

    # Server
    SERVER_ERROR = "SUMMIT-E4001"
    SERVICE_UNAVAILABLE = "SUMMIT-E4002"

    # SDK
    NOT_CONNECTED = "SUMMIT-E5001"
    INVALID_CONFIG = "SUMMIT-E5002"


class SummitError(Exception):
    """Base exception for all Summit SDK errors."""

    def __init__(self, message: str, code: ErrorCode,
                 status: int = 0, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.code = code
        self.status = status
        self.details = details or {}
        super().__init__(f"[{code.value}] {message}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.code.value,
            "message": self.message,
            "status": self.status,
            "details": self.details,
        }


class ConnectionError(SummitError):
    """Failed to connect to Summit.OS API."""
    def __init__(self, message: str = "Connection failed",
                 details: Optional[Dict] = None):
        super().__init__(message, ErrorCode.CONNECTION_FAILED, details=details)


class TimeoutError(SummitError):
    """Request timed out."""
    def __init__(self, message: str = "Request timed out", timeout: float = 0):
        super().__init__(message, ErrorCode.TIMEOUT,
                         details={"timeout_s": timeout})


class CircuitOpenError(SummitError):
    """Circuit breaker is open — calls are being rejected."""
    def __init__(self, message: str = "Circuit breaker open, service unavailable"):
        super().__init__(message, ErrorCode.CIRCUIT_OPEN)


class AuthError(SummitError):
    """Authentication or authorization failure."""
    def __init__(self, message: str = "Unauthorized", status: int = 401):
        code = ErrorCode.UNAUTHORIZED if status == 401 else ErrorCode.FORBIDDEN
        super().__init__(message, code, status=status)


class NotFoundError(SummitError):
    """Requested resource not found."""
    def __init__(self, resource: str = "", resource_id: str = ""):
        msg = f"{resource} '{resource_id}' not found" if resource else "Not found"
        super().__init__(msg, ErrorCode.NOT_FOUND, status=404,
                         details={"resource": resource, "id": resource_id})


class ValidationError(SummitError):
    """Request validation failed."""
    def __init__(self, message: str = "Validation error",
                 fields: Optional[Dict] = None):
        super().__init__(message, ErrorCode.VALIDATION_ERROR, status=422,
                         details={"fields": fields or {}})


class RateLimitError(SummitError):
    """Rate limit exceeded."""
    def __init__(self, retry_after: float = 0):
        super().__init__("Rate limit exceeded", ErrorCode.RATE_LIMITED,
                         status=429, details={"retry_after_s": retry_after})


class ServerError(SummitError):
    """Summit.OS server returned a 5xx error."""
    def __init__(self, message: str = "Internal server error", status: int = 500):
        code = (ErrorCode.SERVICE_UNAVAILABLE if status == 503
                else ErrorCode.SERVER_ERROR)
        super().__init__(message, code, status=status)


class NotConnectedError(SummitError):
    """SDK client is not connected."""
    def __init__(self):
        super().__init__("Client not connected — call await client.connect() first",
                         ErrorCode.NOT_CONNECTED)


# ── HTTP status → error mapping ─────────────────────────────

def raise_for_status(status: int, body: Dict) -> None:
    """Raise the appropriate SummitError for an HTTP error status."""
    if 200 <= status < 300:
        return
    msg = body.get("detail") or body.get("message") or body.get("error", "")
    if status == 401:
        raise AuthError(msg or "Unauthorized", status=401)
    if status == 403:
        raise AuthError(msg or "Forbidden", status=403)
    if status == 404:
        raise NotFoundError(msg)
    if status == 422:
        raise ValidationError(msg, fields=body.get("details"))
    if status == 429:
        raise RateLimitError(retry_after=float(body.get("retry_after", 0)))
    if status >= 500:
        raise ServerError(msg or "Server error", status=status)
    # Generic fallback
    raise SummitError(msg or f"HTTP {status}", ErrorCode.SERVER_ERROR, status=status)
