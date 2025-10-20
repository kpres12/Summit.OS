"""Intelligence and risk assessment schemas for Summit.OS."""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level classifications."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAssessment(BaseModel):
    """Risk assessment schema."""
    risk_id: str = Field(..., description="Unique risk assessment ID")
    risk_level: RiskLevel
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    factors: Dict[str, float] = Field(default_factory=dict)
    location: Optional[Dict[str, float]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IntelligenceSchema(BaseModel):
    """Intelligence data schema."""
    intelligence_id: str = Field(..., description="Unique intelligence ID")
    type: str = Field(..., description="Type of intelligence")
    summary: str = Field(..., description="Intelligence summary")
    risk_assessment: Optional[RiskAssessment] = None
    recommendations: List[str] = Field(default_factory=list)
    source_data: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)