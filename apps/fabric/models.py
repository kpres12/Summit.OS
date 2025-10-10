"""Data models for Summit.OS Data Fabric Service."""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class SeverityLevel(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DeviceStatus(str, Enum):
    """Device status values."""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class MissionStatus(str, Enum):
    """Mission status values."""
    PLANNED = "planned"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class Location(BaseModel):
    """Geographic location."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    altitude: Optional[float] = Field(None, ge=0)
    accuracy: Optional[float] = Field(None, ge=0)


class TelemetryMessage(BaseModel):
    """Telemetry data from edge devices."""
    device_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    location: Location
    sensors: Dict[str, Any] = Field(default_factory=dict)
    status: DeviceStatus = DeviceStatus.ONLINE
    battery_level: Optional[float] = Field(None, ge=0, le=100)
    signal_strength: Optional[float] = Field(None, ge=-100, le=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AlertMessage(BaseModel):
    """Alert message for incidents and anomalies."""
    alert_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    severity: SeverityLevel
    location: Location
    description: str
    source: str
    category: str = "general"
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


class MissionUpdate(BaseModel):
    """Mission status update."""
    mission_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: MissionStatus
    assets: List[str] = Field(default_factory=list)
    objectives: List[str] = Field(default_factory=list)
    progress: float = Field(0.0, ge=0.0, le=1.0)
    estimated_completion: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SystemMetrics(BaseModel):
    """System performance metrics."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    service: str
    cpu_usage: float = Field(ge=0.0, le=100.0)
    memory_usage: float = Field(ge=0.0, le=100.0)
    active_connections: int = Field(ge=0)
    messages_per_second: float = Field(ge=0.0)
    latency_ms: float = Field(ge=0.0)
    error_rate: float = Field(ge=0.0, le=1.0)


class DeviceInfo(BaseModel):
    """Device information and capabilities."""
    device_id: str
    device_type: str
    capabilities: List[str] = Field(default_factory=list)
    location: Location
    last_seen: datetime
    status: DeviceStatus
    firmware_version: Optional[str] = None
    hardware_info: Dict[str, Any] = Field(default_factory=dict)
