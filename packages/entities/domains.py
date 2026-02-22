"""
Multi-Domain Entity Extensions for Summit.OS

Extends the core Entity model with domain-specific attributes for:
- Aerial: fixed-wing, rotary-wing, UAS
- Ground: wheeled, tracked, dismounted
- Maritime: surface, subsurface
- Fixed: installations, towers, sensors
- Sensor: radar, EO/IR, acoustic, SIGINT
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


# ── Aerial Domain ──────────────────────────────────────────

class AircraftType(str, Enum):
    FIXED_WING = "fixed_wing"
    ROTARY_WING = "rotary_wing"
    MULTIROTOR = "multirotor"
    VTOL = "vtol"
    BALLOON = "balloon"
    UNKNOWN = "unknown"


class FlightMode(str, Enum):
    GROUND = "ground"
    TAKEOFF = "takeoff"
    CRUISE = "cruise"
    LOITER = "loiter"
    LANDING = "landing"
    EMERGENCY = "emergency"
    RTL = "rtl"


@dataclass
class AerialAttributes:
    """Domain-specific attributes for aerial entities."""
    aircraft_type: AircraftType = AircraftType.UNKNOWN
    flight_mode: FlightMode = FlightMode.GROUND
    altitude_agl_m: float = 0.0
    altitude_msl_m: float = 0.0
    airspeed_mps: float = 0.0
    groundspeed_mps: float = 0.0
    climb_rate_mps: float = 0.0
    heading_deg: float = 0.0
    bank_angle_deg: float = 0.0
    # Performance limits
    max_altitude_m: float = 5000.0
    max_speed_mps: float = 50.0
    endurance_min: float = 60.0
    # Payload
    payload_kg: float = 0.0
    max_payload_kg: float = 5.0
    # Transponder
    squawk: str = ""
    callsign: str = ""
    icao_hex: str = ""
    # Battery/fuel
    fuel_remaining_pct: float = 100.0

    def to_dict(self) -> Dict:
        return {
            "aircraft_type": self.aircraft_type.value,
            "flight_mode": self.flight_mode.value,
            "altitude_agl_m": self.altitude_agl_m,
            "altitude_msl_m": self.altitude_msl_m,
            "airspeed_mps": self.airspeed_mps,
            "groundspeed_mps": self.groundspeed_mps,
            "climb_rate_mps": self.climb_rate_mps,
            "heading_deg": self.heading_deg,
            "fuel_remaining_pct": self.fuel_remaining_pct,
            "callsign": self.callsign,
        }


# ── Ground Domain ──────────────────────────────────────────

class GroundVehicleType(str, Enum):
    WHEELED = "wheeled"
    TRACKED = "tracked"
    DISMOUNTED = "dismounted"
    UGAV = "ugav"  # Unmanned Ground Autonomous Vehicle
    UNKNOWN = "unknown"


class MobilityState(str, Enum):
    STATIONARY = "stationary"
    MOVING = "moving"
    HALTED = "halted"
    STUCK = "stuck"


@dataclass
class GroundAttributes:
    """Domain-specific attributes for ground entities."""
    vehicle_type: GroundVehicleType = GroundVehicleType.UNKNOWN
    mobility_state: MobilityState = MobilityState.STATIONARY
    speed_mps: float = 0.0
    heading_deg: float = 0.0
    terrain_type: str = ""
    crew_count: int = 0
    # Performance
    max_speed_mps: float = 25.0
    max_grade_pct: float = 60.0
    turn_radius_m: float = 5.0

    def to_dict(self) -> Dict:
        return {
            "vehicle_type": self.vehicle_type.value,
            "mobility_state": self.mobility_state.value,
            "speed_mps": self.speed_mps,
            "heading_deg": self.heading_deg,
            "max_speed_mps": self.max_speed_mps,
        }


# ── Maritime Domain ────────────────────────────────────────

class VesselType(str, Enum):
    SURFACE = "surface"
    SUBSURFACE = "subsurface"
    USV = "usv"  # Unmanned Surface Vehicle
    UUV = "uuv"  # Unmanned Underwater Vehicle
    UNKNOWN = "unknown"


@dataclass
class MaritimeAttributes:
    """Domain-specific attributes for maritime entities."""
    vessel_type: VesselType = VesselType.UNKNOWN
    speed_knots: float = 0.0
    course_deg: float = 0.0
    draft_m: float = 0.0
    depth_m: float = 0.0  # For subsurface
    mmsi: str = ""
    imo: str = ""
    flag_state: str = ""
    destination: str = ""
    cargo_type: str = ""

    def to_dict(self) -> Dict:
        return {
            "vessel_type": self.vessel_type.value,
            "speed_knots": self.speed_knots,
            "course_deg": self.course_deg,
            "mmsi": self.mmsi,
            "depth_m": self.depth_m,
        }


# ── Fixed Site Domain ─────────────────────────────────────

class SiteType(str, Enum):
    COMMAND_POST = "command_post"
    OBSERVATION_POST = "observation_post"
    COMM_TOWER = "comm_tower"
    RADAR_SITE = "radar_site"
    LANDING_ZONE = "landing_zone"
    SUPPLY_POINT = "supply_point"
    UNKNOWN = "unknown"


@dataclass
class FixedSiteAttributes:
    """Domain-specific attributes for fixed installations."""
    site_type: SiteType = SiteType.UNKNOWN
    operational: bool = True
    coverage_radius_m: float = 0.0
    elevation_m: float = 0.0
    personnel_count: int = 0
    power_status: str = "nominal"
    comms_status: str = "nominal"

    def to_dict(self) -> Dict:
        return {
            "site_type": self.site_type.value,
            "operational": self.operational,
            "coverage_radius_m": self.coverage_radius_m,
        }


# ── Sensor Platform Domain ────────────────────────────────

class SensorType(str, Enum):
    RADAR = "radar"
    EO_IR = "eo_ir"
    ACOUSTIC = "acoustic"
    SIGINT = "sigint"
    LIDAR = "lidar"
    CBRN = "cbrn"
    WEATHER = "weather"
    UNKNOWN = "unknown"


@dataclass
class SensorPlatformAttributes:
    """Domain-specific attributes for sensor platforms."""
    sensor_type: SensorType = SensorType.UNKNOWN
    operational: bool = True
    azimuth_coverage_deg: float = 360.0
    elevation_coverage_deg: float = 90.0
    max_range_m: float = 10000.0
    min_range_m: float = 0.0
    update_rate_hz: float = 1.0
    detection_probability: float = 0.9
    current_mode: str = "scan"
    tracks_held: int = 0

    def to_dict(self) -> Dict:
        return {
            "sensor_type": self.sensor_type.value,
            "operational": self.operational,
            "max_range_m": self.max_range_m,
            "update_rate_hz": self.update_rate_hz,
            "current_mode": self.current_mode,
            "tracks_held": self.tracks_held,
        }


# ── Domain Registry ───────────────────────────────────────

DOMAIN_ATTRIBUTE_MAP = {
    "aerial": AerialAttributes,
    "ground": GroundAttributes,
    "maritime": MaritimeAttributes,
    "fixed": FixedSiteAttributes,
    "sensor": SensorPlatformAttributes,
}


def create_domain_attributes(domain: str, **kwargs) -> Any:
    """Factory to create domain-specific attributes."""
    cls = DOMAIN_ATTRIBUTE_MAP.get(domain)
    if cls is None:
        raise ValueError(f"Unknown domain: {domain}. Valid: {list(DOMAIN_ATTRIBUTE_MAP.keys())}")
    return cls(**kwargs)
