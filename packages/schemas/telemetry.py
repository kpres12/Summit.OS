"""Telemetry data schemas for Summit.OS."""

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


class LocationSchema(BaseModel):
    """Geographic location schema."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    altitude: Optional[float] = Field(None, ge=0, description="Altitude in meters")
    accuracy: Optional[float] = Field(None, ge=0, description="Location accuracy in meters")
    heading: Optional[float] = Field(None, ge=0, le=360, description="Heading in degrees")
    speed: Optional[float] = Field(None, ge=0, description="Speed in m/s")


class SensorData(BaseModel):
    """Individual sensor reading."""
    sensor_id: str
    sensor_type: str
    value: float
    unit: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    quality: float = Field(ge=0.0, le=1.0, description="Data quality score")


class TelemetrySchema(BaseModel):
    """Telemetry data from edge devices."""
    device_id: str = Field(..., description="Unique device identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    location: LocationSchema
    status: DeviceStatus = DeviceStatus.ONLINE
    battery_level: Optional[float] = Field(None, ge=0, le=100, description="Battery level percentage")
    signal_strength: Optional[float] = Field(None, ge=-100, le=0, description="Signal strength in dBm")
    sensors: List[SensorData] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TelemetryBatch(BaseModel):
    """Batch of telemetry data."""
    device_id: str
    batch_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    telemetry_data: List[TelemetrySchema]
    compression: Optional[str] = None
    checksum: Optional[str] = None


class DeviceCapabilities(BaseModel):
    """Device capabilities and specifications."""
    device_id: str
    device_type: str
    sensors: List[str] = Field(default_factory=list)
    actuators: List[str] = Field(default_factory=list)
    communication: List[str] = Field(default_factory=list)
    processing: Dict[str, Any] = Field(default_factory=dict)
    power: Dict[str, Any] = Field(default_factory=dict)
    environmental: Dict[str, Any] = Field(default_factory=dict)
