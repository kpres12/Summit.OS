"""Drone-specific schemas for tiered response system."""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
from enum import Enum


class DroneType(str, Enum):
    """Drone types for tiered response."""
    SCOUT = "scout"  # Reconnaissance/verification drone
    INTERCEPTOR = "interceptor"  # Intervention/action drone
    RELAY = "relay"  # Communication relay drone
    CARRIER = "carrier"  # Heavy payload drone
    GENERIC = "generic"  # Standard drone


class DroneRole(str, Enum):
    """Operational roles for drones."""
    RECONNAISSANCE = "reconnaissance"
    VERIFICATION = "verification"
    INTERVENTION = "intervention"
    DEPLOYMENT = "deployment"
    RELAY = "relay"
    PATROL = "patrol"
    MONITORING = "monitoring"
    ESCORT = "escort"


class PayloadType(str, Enum):
    """Generic payload types for drones."""
    LIQUID_CAPSULE = "liquid_capsule"
    SOLID_CONTAINER = "solid_container"
    GAS_CANISTER = "gas_canister"
    SIGNAL_BEACON = "signal_beacon"
    THERMAL_BEACON = "thermal_beacon"
    SENSOR_PACKAGE = "sensor_package"
    TOOL_KIT = "tool_kit"
    SAMPLE_CONTAINER = "sample_container"


class MissionTier(str, Enum):
    """Mission tier levels for escalation."""
    TIER_1_VERIFY = "tier_1_verify"  # Initial verification
    TIER_2_SUPPRESS = "tier_2_suppress"  # Direct intervention
    TIER_3_CONTAIN = "tier_3_contain"  # Multi-asset containment
    TIER_4_ESCALATE = "tier_4_escalate"  # Human response


class DroneCapabilities(BaseModel):
    """Extended capabilities for tiered response drones."""
    drone_type: DroneType
    roles: List[DroneRole] = Field(default_factory=list)
    max_speed: float = Field(..., description="Maximum speed in km/h")
    climb_rate: float = Field(..., description="Climb rate in m/s")
    endurance: float = Field(..., description="Flight endurance in minutes")
    payload_capacity: float = Field(..., description="Payload capacity in kg")
    payload_types: List[PayloadType] = Field(default_factory=list)
    sensors: List[str] = Field(default_factory=list)
    environmental_resistance: Dict[str, bool] = Field(default_factory=dict)  # heat, cold, water, etc.
    auto_dock: bool = True
    mavlink_conn: Optional[str] = None
    mesh_radio: bool = False
    
    # Performance characteristics
    dash_speed: Optional[float] = Field(None, description="Dash speed for emergency response")
    response_time: Optional[float] = Field(None, description="Time to reach coordinates in seconds")
    operating_altitude: Dict[str, float] = Field(
        default_factory=lambda: {"min": 10, "max": 120, "optimal": 60}
    )


class PayloadConfig(BaseModel):
    """Configuration for intervention payloads."""
    type: PayloadType
    capacity: float = Field(..., description="Capacity in liters or units")
    deployment_pattern: str = Field(default="single", description="Deployment pattern")
    effective_radius: float = Field(..., description="Effective radius in meters")
    preparation_time: float = Field(default=0, description="Preparation time in seconds")


class InterventionPlan(BaseModel):
    """Plan for drone intervention missions."""
    target_location: Dict[str, float]  # lat, lon
    payload_config: PayloadConfig
    approach_vector: Optional[Dict[str, float]] = None  # bearing, distance
    drop_altitude: float = Field(default=30, description="Drop altitude in meters")
    escape_route: Optional[List[Dict[str, float]]] = None  # waypoints for post-drop
    wind_compensation: bool = True
    terrain_following: bool = False


class SwarmCoordination(BaseModel):
    """Coordination data for multi-drone operations."""
    formation: str = Field(default="containment_ring", description="Formation pattern")
    lead_drone: str = Field(..., description="Lead drone asset_id")
    separation_distance: float = Field(default=50, description="Minimum separation in meters")
    coordination_frequency: Optional[str] = Field(None, description="Radio frequency for coordination")
    anti_collision: bool = True
    shared_telemetry: bool = True


class ThreatThreshold(BaseModel):
    """Generic threat assessment thresholds for escalation decisions."""
    severity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall severity threshold")
    size_metric: Optional[float] = Field(None, description="Size/scale threshold (units depend on domain)")
    spread_rate: Optional[float] = Field(None, description="Spread/growth rate threshold")
    intensity_metric: Optional[float] = Field(None, description="Intensity measurement threshold")
    environmental_factor: float = Field(default=1.0, description="Environmental conditions multiplier")
    terrain_factor: float = Field(default=1.0, description="Terrain/geography multiplier")
    urgency_multiplier: float = Field(default=1.0, description="Time-sensitive urgency factor")


class TieredMissionRequest(BaseModel):
    """Request for tiered mission dispatch."""
    alert_id: str = Field(..., description="Associated alert/detection ID")
    initial_location: Dict[str, float]  # lat, lon of target
    verification_required: bool = True
    intervention_threshold: ThreatThreshold
    max_tier: MissionTier = MissionTier.TIER_3_CONTAIN
    preferred_assets: Optional[List[str]] = Field(None, description="Preferred asset IDs")
    time_limit: Optional[float] = Field(None, description="Mission time limit in minutes")
    environmental_data: Optional[Dict[str, Any]] = None
    terrain_data: Optional[Dict[str, Any]] = None
    domain_context: Optional[Dict[str, Any]] = Field(None, description="Domain-specific context data")


class TieredMissionStatus(BaseModel):
    """Status of tiered mission execution."""
    mission_id: str
    current_tier: MissionTier
    tier_1_status: Optional[str] = None  # PENDING, ACTIVE, COMPLETED, FAILED
    tier_2_status: Optional[str] = None
    tier_3_status: Optional[str] = None
    verification_result: Optional[Dict[str, Any]] = None
    intervention_result: Optional[Dict[str, Any]] = None
    escalation_reason: Optional[str] = None
    next_tier_eta: Optional[float] = None
    assets_deployed: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DroneBoxConfig(BaseModel):
    """Configuration for deployment box containing FireFly + EmberWing."""
    box_id: str
    location: Dict[str, float]  # lat, lon of deployment box
    firefly_id: str
    emberwing_id: str
    shared_charger: bool = True
    launch_sequence_delay: float = Field(default=2.0, description="Delay between launches in seconds")
    recovery_timeout: float = Field(default=1800, description="Auto-recovery timeout in seconds")
    weather_limits: Dict[str, float] = Field(
        default_factory=lambda: {
            "max_wind_speed": 15.0,  # m/s
            "min_visibility": 1000,  # meters
            "max_precipitation": 5.0  # mm/h
        }
    )


# Extended asset model for tiered response
class TieredAsset(BaseModel):
    """Asset model extended for tiered response capabilities."""
    asset_id: str
    type: str = "drone"
    drone_capabilities: Optional[DroneCapabilities] = None
    current_payload: Optional[PayloadConfig] = None
    box_assignment: Optional[str] = Field(None, description="Deployment box ID")
    tier_assignments: List[MissionTier] = Field(default_factory=list)
    battery: Optional[float] = Field(None, ge=0, le=100)
    link: Optional[str] = None
    last_maintenance: Optional[datetime] = None
    flight_hours: float = Field(default=0.0)
    interventions_completed: int = Field(default=0)