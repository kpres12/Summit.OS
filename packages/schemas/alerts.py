"""Alert data schemas for Summit.OS."""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum

from .telemetry import LocationSchema


class SeverityLevel(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertCategory(str, Enum):
    """Alert categories."""
    FIRE = "fire"
    SMOKE = "smoke"
    WEATHER = "weather"
    EQUIPMENT = "equipment"
    SECURITY = "security"
    COMMUNICATION = "communication"
    NAVIGATION = "navigation"
    POWER = "power"
    GENERAL = "general"


class AlertStatus(str, Enum):
    """Alert status values."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    CANCELLED = "cancelled"


class AlertSchema(BaseModel):
    """Alert message schema."""
    alert_id: str = Field(..., description="Unique alert identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    severity: SeverityLevel
    category: AlertCategory = AlertCategory.GENERAL
    status: AlertStatus = AlertStatus.ACTIVE
    location: LocationSchema
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Detailed alert description")
    source: str = Field(..., description="Alert source (device, system, etc.)")
    tags: List[str] = Field(default_factory=list, description="Alert tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Acknowledgment fields
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    acknowledgment_notes: Optional[str] = None
    
    # Resolution fields
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    # Escalation fields
    escalated: bool = False
    escalated_at: Optional[datetime] = None
    escalated_to: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AlertUpdate(BaseModel):
    """Alert update schema."""
    alert_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    update_type: str  # status_change, acknowledgment, resolution, escalation
    updated_by: str
    changes: Dict[str, Any]
    notes: Optional[str] = None


class AlertFilter(BaseModel):
    """Alert filtering criteria."""
    severity: Optional[List[SeverityLevel]] = None
    category: Optional[List[AlertCategory]] = None
    status: Optional[List[AlertStatus]] = None
    source: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    location_bounds: Optional[Dict[str, float]] = None  # min_lat, max_lat, min_lon, max_lon
    time_range: Optional[Dict[str, datetime]] = None  # start_time, end_time
    limit: Optional[int] = Field(100, ge=1, le=1000)


class AlertSummary(BaseModel):
    """Alert summary statistics."""
    total_alerts: int
    active_alerts: int
    acknowledged_alerts: int
    resolved_alerts: int
    escalated_alerts: int
    by_severity: Dict[SeverityLevel, int]
    by_category: Dict[AlertCategory, int]
    by_status: Dict[AlertStatus, int]
    recent_alerts: List[AlertSchema]
