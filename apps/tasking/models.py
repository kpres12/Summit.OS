"""Pydantic models for the tasking service."""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class TaskDispatchRequest(BaseModel):
    task_id: str
    asset_id: str
    action: str
    waypoints: list = []


class Task(BaseModel):
    task_id: str
    asset_id: str
    action: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class AssetIn(BaseModel):
    asset_id: str
    type: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None
    battery: Optional[float] = Field(default=None, ge=0, le=100)
    link: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None


class AssetOut(AssetIn):
    updated_at: Optional[datetime] = None


class MissionCreateRequest(BaseModel):
    name: Optional[str] = None
    objectives: List[str] = Field(default_factory=list)
    area: Optional[Dict[str, Any]] = (
        None  # e.g., {center: {lat, lon}, radius_m: 500} or {polygon: [[lat,lon], ...]}
    )
    num_drones: Optional[int] = Field(default=None, ge=1)
    policy_context: Optional[Dict[str, Any]] = (
        None  # airspace/geofence/weather/NOTAMs/operator
    )
    # planning_params:
    #   pattern: "loiter" | "grid"
    #   altitude: meters
    #   speed: m/s
    #   grid_spacing_m: for pattern=grid
    #   heading_deg: lane orientation for grid
    planning_params: Optional[Dict[str, Any]] = (
        None  # pattern: loiter|grid, altitude, speed, grid_spacing_m, heading_deg
    )


class MissionAssignment(BaseModel):
    asset_id: str
    plan: Dict[str, Any]
    status: str


class MissionResponse(BaseModel):
    mission_id: str
    name: Optional[str]
    objectives: List[str]
    status: str
    policy_ok: bool
    assignments: List[MissionAssignment]
    created_at: datetime
    started_at: Optional[datetime] = None


class ValveCommand(BaseModel):
    command: str
    params: Dict[str, Any] | None = None
    safety: Dict[str, Any] | None = None
    mission_id: str | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: Any | None = None
