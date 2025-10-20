"""Device schemas for Summit.OS."""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class DeviceStatus(str, Enum):
    """Device status values."""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class DeviceSchema(BaseModel):
    """Device definition schema."""
    device_id: str = Field(..., description="Unique device identifier")
    device_type: str = Field(..., description="Type of device")
    name: Optional[str] = None
    status: DeviceStatus = DeviceStatus.UNKNOWN
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    location: Optional[Dict[str, float]] = None
    battery_level: Optional[float] = Field(None, ge=0, le=100)
    signal_strength: Optional[float] = None
    last_seen: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)