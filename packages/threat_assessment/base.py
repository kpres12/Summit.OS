"""Generic threat assessment framework for Summit.OS."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class ThreatLevel(str, Enum):
    """Generic threat level classifications."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class ThreatAssessmentResult(BaseModel):
    """Result of threat assessment analysis."""
    threat_level: ThreatLevel
    severity_score: float = Field(ge=0.0, le=1.0, description="Normalized severity score")
    confidence: float = Field(ge=0.0, le=1.0, description="Assessment confidence")
    escalation_required: bool = False
    escalation_reason: Optional[str] = None
    environmental_factors: Dict[str, float] = Field(default_factory=dict)
    risk_factors: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    time_critical: bool = False
    estimated_containment_window: Optional[float] = Field(None, description="Minutes before escalation needed")


class BaseThreatAssessor(ABC):
    """Abstract base class for domain-specific threat assessment."""
    
    def __init__(self, domain: str, config: Dict[str, Any] = None):
        self.domain = domain
        self.config = config or {}
    
    @abstractmethod
    async def assess_threat(
        self, 
        location: Dict[str, float],
        sensor_data: Dict[str, Any],
        environmental_data: Optional[Dict[str, Any]] = None,
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> ThreatAssessmentResult:
        """Assess threat level based on available data."""
        pass
    
    @abstractmethod
    def get_escalation_thresholds(self) -> Dict[str, float]:
        """Return domain-specific escalation thresholds."""
        pass
    
    @abstractmethod
    def supports_data_type(self, data_type: str) -> bool:
        """Check if assessor can handle this data type."""
        pass


class GenericThreatAssessor(BaseThreatAssessor):
    """Default generic threat assessor for unknown domains."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("generic", config)
    
    async def assess_threat(
        self,
        location: Dict[str, float],
        sensor_data: Dict[str, Any],
        environmental_data: Optional[Dict[str, Any]] = None,
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> ThreatAssessmentResult:
        """Basic threat assessment using generic metrics."""
        
        # Extract common metrics
        confidence = sensor_data.get("confidence", 0.5)
        intensity = sensor_data.get("intensity", 0.0)
        size_metric = sensor_data.get("size", 0.0)
        spread_rate = sensor_data.get("spread_rate", 0.0)
        
        # Simple scoring
        severity_score = min(1.0, (intensity * 0.4 + size_metric * 0.3 + spread_rate * 0.3))
        
        # Determine threat level
        if severity_score < 0.2:
            threat_level = ThreatLevel.LOW
        elif severity_score < 0.4:
            threat_level = ThreatLevel.MODERATE
        elif severity_score < 0.7:
            threat_level = ThreatLevel.HIGH
        else:
            threat_level = ThreatLevel.CRITICAL
        
        # Escalation logic
        escalation_required = severity_score > 0.3
        escalation_reason = f"Severity score {severity_score:.2f} exceeds threshold" if escalation_required else None
        
        return ThreatAssessmentResult(
            threat_level=threat_level,
            severity_score=severity_score,
            confidence=confidence,
            escalation_required=escalation_required,
            escalation_reason=escalation_reason,
            environmental_factors=environmental_data or {},
            time_critical=severity_score > 0.6,
            estimated_containment_window=max(5.0, (1.0 - severity_score) * 60.0)
        )
    
    def get_escalation_thresholds(self) -> Dict[str, float]:
        """Return generic escalation thresholds."""
        return {
            "tier_1_to_2": 0.3,
            "tier_2_to_3": 0.6,
            "tier_3_to_4": 0.8
        }
    
    def supports_data_type(self, data_type: str) -> bool:
        """Generic assessor supports any data type."""
        return True


class ThreatAssessmentRegistry:
    """Registry for domain-specific threat assessors."""
    
    def __init__(self):
        self._assessors: Dict[str, BaseThreatAssessor] = {}
        self._default_assessor = GenericThreatAssessor()
        
    def register_assessor(self, domain: str, assessor: BaseThreatAssessor):
        """Register a domain-specific threat assessor."""
        self._assessors[domain] = assessor
    
    def get_assessor(self, domain: str) -> BaseThreatAssessor:
        """Get assessor for domain, fallback to generic."""
        return self._assessors.get(domain, self._default_assessor)
    
    def list_domains(self) -> List[str]:
        """List all registered domains."""
        return list(self._assessors.keys())


# Global registry instance
threat_registry = ThreatAssessmentRegistry()