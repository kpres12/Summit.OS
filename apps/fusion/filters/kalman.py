"""
Extended Kalman Filter (EKF) for Summit.OS Track Fusion

State vector x = [lat, lon, alt, vn, ve, vd]  (6-state)
  - lat, lon in degrees
  - alt in meters MSL
  - vn, ve, vd in m/s (North, East, Down)

The filter operates in a local tangent plane (ENU) for prediction,
then converts back to geodetic for the state output.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

# WGS-84 constants
_A = 6378137.0  # semi-major axis
_E2 = 6.6943799901377997e-3  # eccentricity squared


def _meters_per_deg_lat(lat_deg: float) -> float:
    """Meters per degree of latitude at given latitude."""
    lat = math.radians(lat_deg)
    return math.pi / 180.0 * _A * (1 - _E2) / (1 - _E2 * math.sin(lat) ** 2) ** 1.5


def _meters_per_deg_lon(lat_deg: float) -> float:
    """Meters per degree of longitude at given latitude."""
    lat = math.radians(lat_deg)
    return (
        math.pi / 180.0 * _A * math.cos(lat) / math.sqrt(1 - _E2 * math.sin(lat) ** 2)
    )


@dataclass
class EKFState:
    """Extended Kalman Filter state."""

    x: np.ndarray = field(
        default_factory=lambda: np.zeros(6)
    )  # [lat, lon, alt, vn, ve, vd]
    P: np.ndarray = field(default_factory=lambda: np.eye(6) * 100.0)  # Covariance
    last_update_time: float = 0.0
    initialized: bool = False


class ExtendedKalmanFilter:
    """
    6-state Extended Kalman Filter for geospatial tracking.

    Constant-velocity process model with configurable process noise.
    Supports position-only and position+velocity measurements.
    """

    def __init__(
        self,
        process_noise_pos: float = 0.5,  # m²/s³ position process noise spectral density
        process_noise_vel: float = 2.0,  # m²/s³ velocity process noise spectral density
        initial_pos_var: float = 100.0,  # m² initial position variance
        initial_vel_var: float = 25.0,  # m² initial velocity variance
    ):
        self.q_pos = process_noise_pos
        self.q_vel = process_noise_vel
        self.initial_pos_var = initial_pos_var
        self.initial_vel_var = initial_vel_var

    def initialize(
        self,
        lat: float,
        lon: float,
        alt: float,
        t: float,
        vn: float = 0.0,
        ve: float = 0.0,
        vd: float = 0.0,
    ) -> EKFState:
        """Initialize the filter with a first measurement."""
        state = EKFState()
        state.x = np.array([lat, lon, alt, vn, ve, vd], dtype=np.float64)
        state.P = np.diag(
            [
                self.initial_pos_var,
                self.initial_pos_var,
                self.initial_pos_var,
                self.initial_vel_var,
                self.initial_vel_var,
                self.initial_vel_var,
            ]
        )
        state.last_update_time = t
        state.initialized = True
        return state

    def predict(self, state: EKFState, t: float) -> EKFState:
        """
        Predict state forward to time t using constant-velocity model.

        Converts velocity from m/s to degrees/s for lat/lon propagation.
        """
        if not state.initialized:
            raise ValueError("Filter not initialized")

        dt = t - state.last_update_time
        if dt <= 0:
            return state

        lat, lon, alt, vn, ve, vd = state.x

        # Convert velocity to degrees
        m_per_deg_lat = _meters_per_deg_lat(lat)
        m_per_deg_lon = _meters_per_deg_lon(lat)

        dlat = (vn / m_per_deg_lat) * dt  # degrees
        dlon = (ve / m_per_deg_lon) * dt  # degrees
        dalt = -vd * dt  # positive vd = descending

        # State transition (constant velocity)
        x_pred = np.array(
            [
                lat + dlat,
                lon + dlon,
                alt + dalt,
                vn,
                ve,
                vd,
            ]
        )

        # State transition Jacobian F
        F = np.eye(6)
        F[0, 3] = dt / m_per_deg_lat
        F[1, 4] = dt / m_per_deg_lon
        F[2, 5] = -dt

        # Process noise Q (piecewise white noise jerk model)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        Q = np.zeros((6, 6))
        # Position blocks
        Q[0, 0] = self.q_pos * dt3 / 3 / (m_per_deg_lat**2)
        Q[1, 1] = self.q_pos * dt3 / 3 / (m_per_deg_lon**2)
        Q[2, 2] = self.q_pos * dt3 / 3
        # Velocity blocks
        Q[3, 3] = self.q_vel * dt
        Q[4, 4] = self.q_vel * dt
        Q[5, 5] = self.q_vel * dt
        # Cross terms (pos-vel)
        Q[0, 3] = Q[3, 0] = self.q_pos * dt2 / 2 / m_per_deg_lat
        Q[1, 4] = Q[4, 1] = self.q_pos * dt2 / 2 / m_per_deg_lon
        Q[2, 5] = Q[5, 2] = -self.q_pos * dt2 / 2

        # Covariance propagation
        P_pred = F @ state.P @ F.T + Q

        new_state = EKFState(x=x_pred, P=P_pred, last_update_time=t, initialized=True)
        return new_state

    def update_position(
        self,
        state: EKFState,
        lat_meas: float,
        lon_meas: float,
        alt_meas: float,
        t: float,
        sigma_pos_m: float = 5.0,
    ) -> EKFState:
        """
        Update with a position-only measurement.

        Args:
            sigma_pos_m: 1-sigma position measurement noise in meters
        """
        # First predict to measurement time
        state = self.predict(state, t)

        lat = state.x[0]
        m_per_deg_lat = _meters_per_deg_lat(lat)
        m_per_deg_lon = _meters_per_deg_lon(lat)

        # Measurement matrix H (observe lat, lon, alt)
        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        # Measurement noise R (convert m to degrees)
        R = np.diag(
            [
                (sigma_pos_m / m_per_deg_lat) ** 2,
                (sigma_pos_m / m_per_deg_lon) ** 2,
                sigma_pos_m**2,
            ]
        )

        z = np.array([lat_meas, lon_meas, alt_meas])
        y = z - H @ state.x  # Innovation

        S = H @ state.P @ H.T + R  # Innovation covariance
        K = state.P @ H.T @ np.linalg.inv(S)  # Kalman gain

        x_upd = state.x + K @ y
        P_upd = (np.eye(6) - K @ H) @ state.P

        # Joseph form for numerical stability
        I_KH = np.eye(6) - K @ H
        P_upd = I_KH @ state.P @ I_KH.T + K @ R @ K.T

        return EKFState(x=x_upd, P=P_upd, last_update_time=t, initialized=True)

    def update_position_velocity(
        self,
        state: EKFState,
        lat_meas: float,
        lon_meas: float,
        alt_meas: float,
        vn_meas: float,
        ve_meas: float,
        vd_meas: float,
        t: float,
        sigma_pos_m: float = 5.0,
        sigma_vel_mps: float = 1.0,
    ) -> EKFState:
        """Update with position + velocity measurement (e.g., GPS with velocity)."""
        state = self.predict(state, t)

        lat = state.x[0]
        m_per_deg_lat = _meters_per_deg_lat(lat)
        m_per_deg_lon = _meters_per_deg_lon(lat)

        H = np.eye(6)
        R = np.diag(
            [
                (sigma_pos_m / m_per_deg_lat) ** 2,
                (sigma_pos_m / m_per_deg_lon) ** 2,
                sigma_pos_m**2,
                sigma_vel_mps**2,
                sigma_vel_mps**2,
                sigma_vel_mps**2,
            ]
        )

        z = np.array([lat_meas, lon_meas, alt_meas, vn_meas, ve_meas, vd_meas])
        y = z - H @ state.x

        S = H @ state.P @ H.T + R
        K = state.P @ H.T @ np.linalg.inv(S)

        x_upd = state.x + K @ y
        I_KH = np.eye(6) - K @ H
        P_upd = I_KH @ state.P @ I_KH.T + K @ R @ K.T

        return EKFState(x=x_upd, P=P_upd, last_update_time=t, initialized=True)

    def update_bearing_range(
        self,
        state: EKFState,
        observer_lat: float,
        observer_lon: float,
        observer_alt: float,
        bearing_deg: float,
        range_m: float,
        t: float,
        sigma_bearing_deg: float = 2.0,
        sigma_range_m: float = 10.0,
    ) -> EKFState:
        """
        Update with bearing + range measurement from a sensor.
        Nonlinear measurement — uses EKF linearization.
        """
        state = self.predict(state, t)

        lat, lon, alt = state.x[0], state.x[1], state.x[2]
        m_per_deg_lat = _meters_per_deg_lat(lat)
        m_per_deg_lon = _meters_per_deg_lon(lat)

        # Predicted relative position in meters (ENU)
        dn = (lat - observer_lat) * m_per_deg_lat
        de = (lon - observer_lon) * m_per_deg_lon
        du = alt - observer_alt

        # Predicted bearing and range
        pred_range = math.sqrt(dn**2 + de**2 + du**2) + 1e-10
        pred_bearing = math.degrees(math.atan2(de, dn)) % 360

        # Innovation
        y_bearing = bearing_deg - pred_bearing
        # Wrap to [-180, 180]
        if y_bearing > 180:
            y_bearing -= 360
        elif y_bearing < -180:
            y_bearing += 360
        y_range = range_m - pred_range

        z = np.array([y_bearing, y_range])

        # Jacobian of [bearing, range] w.r.t. [lat, lon, alt, vn, ve, vd]
        H = np.zeros((2, 6))
        horiz_dist_sq = dn**2 + de**2 + 1e-10
        horiz_dist = math.sqrt(horiz_dist_sq)

        # d(bearing)/d(lat) = d(atan2(de,dn))/d(dn) * d(dn)/d(lat)
        H[0, 0] = -de / horiz_dist_sq * m_per_deg_lat * (180 / math.pi)
        H[0, 1] = dn / horiz_dist_sq * m_per_deg_lon * (180 / math.pi)
        # d(range)/d(lat, lon, alt)
        H[1, 0] = dn / pred_range * m_per_deg_lat
        H[1, 1] = de / pred_range * m_per_deg_lon
        H[1, 2] = du / pred_range

        R = np.diag([sigma_bearing_deg**2, sigma_range_m**2])

        S = H @ state.P @ H.T + R
        K = state.P @ H.T @ np.linalg.inv(S)

        x_upd = state.x + K @ z
        I_KH = np.eye(6) - K @ H
        P_upd = I_KH @ state.P @ I_KH.T + K @ R @ K.T

        return EKFState(x=x_upd, P=P_upd, last_update_time=t, initialized=True)

    @staticmethod
    def get_position(state: EKFState) -> Tuple[float, float, float]:
        """Extract lat, lon, alt from state."""
        return float(state.x[0]), float(state.x[1]), float(state.x[2])

    @staticmethod
    def get_velocity(state: EKFState) -> Tuple[float, float, float]:
        """Extract vn, ve, vd from state."""
        return float(state.x[3]), float(state.x[4]), float(state.x[5])

    @staticmethod
    def get_position_uncertainty(state: EKFState) -> float:
        """Get 1-sigma position uncertainty in meters (horizontal CEP)."""
        # Convert lat/lon variance to meters
        lat = state.x[0]
        m_lat = _meters_per_deg_lat(lat)
        m_lon = _meters_per_deg_lon(lat)
        var_n = state.P[0, 0] * m_lat**2
        var_e = state.P[1, 1] * m_lon**2
        return math.sqrt(var_n + var_e)

    @staticmethod
    def mahalanobis_distance(state_a: EKFState, state_b: EKFState) -> float:
        """Compute Mahalanobis distance between two track states (position only)."""
        dx = state_a.x[:3] - state_b.x[:3]
        S = state_a.P[:3, :3] + state_b.P[:3, :3]
        try:
            S_inv = np.linalg.inv(S)
            return float(np.sqrt(dx @ S_inv @ dx))
        except np.linalg.LinAlgError:
            return float("inf")
