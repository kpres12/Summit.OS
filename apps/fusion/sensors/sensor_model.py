"""
Sensor Models for Summit.OS Fusion Engine

Each sensor model describes:
- Measurement type (position, bearing-range, range-only, etc.)
- Noise characteristics (R matrix)
- Field of view / coverage
- Detection probability

Used by the Track Manager to properly weight observations
from heterogeneous sensor types.
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod


@dataclass
class SensorSpec:
    """Static sensor specification."""
    sensor_id: str
    sensor_type: str  # radar, eo_ir, adsb, gps, acoustic, lidar
    # Position of sensor (for bearing-range sensors)
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    # Field of view
    azimuth_fov_deg: float = 360.0
    elevation_fov_deg: float = 90.0
    boresight_az_deg: float = 0.0
    boresight_el_deg: float = 0.0
    # Range limits
    min_range_m: float = 0.0
    max_range_m: float = float("inf")
    # Performance
    detection_probability: float = 0.9
    false_alarm_rate: float = 1e-6
    # Update rate
    update_rate_hz: float = 1.0


class SensorModel(ABC):
    """Abstract sensor model."""

    def __init__(self, spec: SensorSpec):
        self.spec = spec

    @abstractmethod
    def measurement_noise_matrix(self, range_m: float = 0.0) -> np.ndarray:
        """Return the measurement noise covariance matrix R."""
        ...

    @abstractmethod
    def sigma_position_m(self, range_m: float = 0.0) -> float:
        """Return effective position uncertainty in meters."""
        ...

    def detection_probability(self, range_m: float = 0.0) -> float:
        """Probability of detection at given range."""
        if range_m > self.spec.max_range_m:
            return 0.0
        if range_m < self.spec.min_range_m:
            return 0.0
        return self.spec.detection_probability

    def in_field_of_view(self, az_deg: float, el_deg: float) -> bool:
        """Check if a target bearing is within the sensor's FOV."""
        # Normalize azimuth difference to [-180, 180]
        daz = ((az_deg - self.spec.boresight_az_deg + 180) % 360) - 180
        de = el_deg - self.spec.boresight_el_deg
        return (abs(daz) <= self.spec.azimuth_fov_deg / 2 and
                abs(de) <= self.spec.elevation_fov_deg / 2)


class RadarModel(SensorModel):
    """
    Surveillance radar sensor model.

    Measures range + azimuth + elevation with range-dependent noise.
    Position uncertainty grows with range due to angular resolution.
    """

    def __init__(
        self,
        spec: SensorSpec,
        range_sigma_m: float = 15.0,
        azimuth_sigma_deg: float = 1.5,
        elevation_sigma_deg: float = 2.0,
    ):
        super().__init__(spec)
        self.range_sigma_m = range_sigma_m
        self.az_sigma_rad = math.radians(azimuth_sigma_deg)
        self.el_sigma_rad = math.radians(elevation_sigma_deg)

    def measurement_noise_matrix(self, range_m: float = 1000.0) -> np.ndarray:
        """R matrix in (range, azimuth, elevation) space."""
        return np.diag([
            self.range_sigma_m ** 2,
            self.az_sigma_rad ** 2,
            self.el_sigma_rad ** 2,
        ])

    def sigma_position_m(self, range_m: float = 1000.0) -> float:
        """Effective position uncertainty at given range."""
        cross_range = range_m * self.az_sigma_rad
        return math.sqrt(self.range_sigma_m ** 2 + cross_range ** 2)

    def detection_probability(self, range_m: float = 0.0) -> float:
        """Pd drops with range^4 (radar equation)."""
        if range_m > self.spec.max_range_m or range_m < self.spec.min_range_m:
            return 0.0
        # Simple model: Pd degrades as (max_range/range)^2 scaled
        ratio = range_m / self.spec.max_range_m
        return self.spec.detection_probability * max(0.0, 1.0 - ratio ** 2)


class EOIRModel(SensorModel):
    """
    Electro-Optical / Infrared camera model.

    Measures bearing (azimuth + elevation) only — no direct range.
    Position uncertainty depends on range (which must come from fusion).
    """

    def __init__(
        self,
        spec: SensorSpec,
        pixel_sigma_deg: float = 0.05,
    ):
        super().__init__(spec)
        self.pixel_sigma_rad = math.radians(pixel_sigma_deg)

    def measurement_noise_matrix(self, range_m: float = 1000.0) -> np.ndarray:
        """R matrix in (azimuth, elevation) bearing space."""
        return np.diag([
            self.pixel_sigma_rad ** 2,
            self.pixel_sigma_rad ** 2,
        ])

    def sigma_position_m(self, range_m: float = 1000.0) -> float:
        """Bearing-only: position uncertainty = range * angular uncertainty."""
        return range_m * self.pixel_sigma_rad


class ADSBModel(SensorModel):
    """
    ADS-B transponder model.

    Cooperative target self-reports GPS position.
    High accuracy but only for cooperative targets.
    """

    def __init__(
        self,
        spec: SensorSpec,
        position_sigma_m: float = 7.5,
        altitude_sigma_m: float = 15.0,
    ):
        super().__init__(spec)
        self.position_sigma_m = position_sigma_m
        self.altitude_sigma_m = altitude_sigma_m

    def measurement_noise_matrix(self, range_m: float = 0.0) -> np.ndarray:
        """R matrix in (lat_m, lon_m, alt_m) position space."""
        return np.diag([
            self.position_sigma_m ** 2,
            self.position_sigma_m ** 2,
            self.altitude_sigma_m ** 2,
        ])

    def sigma_position_m(self, range_m: float = 0.0) -> float:
        """ADS-B has constant accuracy regardless of range."""
        return self.position_sigma_m


class GPSModel(SensorModel):
    """
    GPS/GNSS position model for own-platform navigation.
    """

    def __init__(
        self,
        spec: SensorSpec,
        horizontal_sigma_m: float = 2.5,
        vertical_sigma_m: float = 5.0,
    ):
        super().__init__(spec)
        self.horizontal_sigma_m = horizontal_sigma_m
        self.vertical_sigma_m = vertical_sigma_m

    def measurement_noise_matrix(self, range_m: float = 0.0) -> np.ndarray:
        return np.diag([
            self.horizontal_sigma_m ** 2,
            self.horizontal_sigma_m ** 2,
            self.vertical_sigma_m ** 2,
        ])

    def sigma_position_m(self, range_m: float = 0.0) -> float:
        return self.horizontal_sigma_m


class AcousticModel(SensorModel):
    """
    Acoustic sensor model (bearing-only, limited range).
    """

    def __init__(
        self,
        spec: SensorSpec,
        bearing_sigma_deg: float = 5.0,
    ):
        super().__init__(spec)
        self.bearing_sigma_rad = math.radians(bearing_sigma_deg)

    def measurement_noise_matrix(self, range_m: float = 0.0) -> np.ndarray:
        return np.diag([self.bearing_sigma_rad ** 2])

    def sigma_position_m(self, range_m: float = 500.0) -> float:
        return range_m * self.bearing_sigma_rad


# ── Sensor Registry ──────────────────────────────────────────

_SENSOR_TYPE_MAP = {
    "radar": RadarModel,
    "eo_ir": EOIRModel,
    "adsb": ADSBModel,
    "gps": GPSModel,
    "acoustic": AcousticModel,
}


class SensorRegistry:
    """Registry of known sensor models."""

    def __init__(self):
        self.sensors: Dict[str, SensorModel] = {}

    def register(self, model: SensorModel):
        self.sensors[model.spec.sensor_id] = model

    def get(self, sensor_id: str) -> Optional[SensorModel]:
        return self.sensors.get(sensor_id)

    def get_sigma(self, sensor_id: str, range_m: float = 0.0) -> float:
        """Get position sigma for a sensor, with fallback default."""
        model = self.sensors.get(sensor_id)
        if model:
            return model.sigma_position_m(range_m)
        return 10.0  # conservative default

    @classmethod
    def from_specs(cls, specs: list[SensorSpec]) -> "SensorRegistry":
        """Build registry from a list of sensor specs."""
        registry = cls()
        for spec in specs:
            model_cls = _SENSOR_TYPE_MAP.get(spec.sensor_type)
            if model_cls:
                registry.register(model_cls(spec))
        return registry
