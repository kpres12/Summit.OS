"""Mission schemas for Summit.OS."""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class MissionStatus(str, Enum):
    """Mission status values."""
    PLANNING = "planning"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class MissionSchema(BaseModel):
    """Mission definition schema."""
    mission_id: str = Field(..., description="Unique mission identifier")
    name: Optional[str] = None
    objectives: List[str] = Field(default_factory=list)
    area: Optional[Dict[str, Any]] = None
    status: MissionStatus = MissionStatus.PLANNING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)