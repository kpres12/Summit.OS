"""
Multi-Sensor Track Correlator

Correlates observations from multiple sensors to existing tracks using:
1. Mahalanobis distance gating — reject unlikely associations
2. Hungarian algorithm — optimal global assignment
3. Covariance intersection — fuse track-to-track without cross-correlation
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from apps.fusion.filters.kalman import EKFState, ExtendedKalmanFilter


@dataclass
class Association:
    """Result of observation-to-track association."""

    observation_idx: int
    track_id: str
    distance: float  # Mahalanobis distance
    gate_passed: bool


def mahalanobis_gate(
    state: EKFState,
    z_lat: float,
    z_lon: float,
    z_alt: float,
    sigma_m: float = 10.0,
    gate_threshold: float = 9.21,  # chi-squared 3 DoF, 99% confidence
) -> Tuple[float, bool]:
    """
    Compute Mahalanobis distance between a track state and an observation.

    Args:
        state: Current track EKF state
        z_lat, z_lon, z_alt: Observation position
        sigma_m: Measurement noise (meters)
        gate_threshold: Chi-squared threshold for gating

    Returns:
        (distance, passed_gate)
    """
    from apps.fusion.filters.kalman import _meters_per_deg_lat, _meters_per_deg_lon

    lat = state.x[0]
    m_lat = _meters_per_deg_lat(lat)
    m_lon = _meters_per_deg_lon(lat)

    # Innovation in degrees
    dy = np.array([z_lat - state.x[0], z_lon - state.x[1], z_alt - state.x[2]])

    # Innovation covariance S = H P H^T + R
    H = np.zeros((3, 6))
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0

    R = np.diag(
        [
            (sigma_m / m_lat) ** 2,
            (sigma_m / m_lon) ** 2,
            sigma_m**2,
        ]
    )

    S = H @ state.P @ H.T + R

    try:
        S_inv = np.linalg.inv(S)
        d2 = float(dy @ S_inv @ dy)
        return math.sqrt(max(0.0, d2)), d2 < gate_threshold
    except np.linalg.LinAlgError:
        return float("inf"), False


def hungarian_assignment(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """
    Solve the linear assignment problem using the Hungarian algorithm.

    Args:
        cost_matrix: (N_observations x M_tracks) cost matrix.
                     Use large value (1e9) for gated-out pairs.

    Returns:
        List of (observation_idx, track_idx) assignments
    """
    n_obs, n_trk = cost_matrix.shape
    if n_obs == 0 or n_trk == 0:
        return []

    # Pad to square matrix if needed
    size = max(n_obs, n_trk)
    padded = np.full((size, size), 1e9)
    padded[:n_obs, :n_trk] = cost_matrix

    # Hungarian algorithm (Kuhn-Munkres)
    # Using a straightforward implementation
    u = np.zeros(size + 1)
    v = np.zeros(size + 1)
    p = np.zeros(size + 1, dtype=int)
    way = np.zeros(size + 1, dtype=int)

    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0

            for j in range(1, size + 1):
                if not used[j]:
                    cur = padded[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]

    # Extract assignments
    assignments = []
    for j in range(1, size + 1):
        if p[j] != 0 and p[j] - 1 < n_obs and j - 1 < n_trk:
            obs_idx = p[j] - 1
            trk_idx = j - 1
            if cost_matrix[obs_idx, trk_idx] < 1e8:  # Not a dummy assignment
                assignments.append((obs_idx, trk_idx))

    return assignments


def covariance_intersection(
    x_a: np.ndarray,
    P_a: np.ndarray,
    x_b: np.ndarray,
    P_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Covariance Intersection (CI) fusion of two estimates.

    Fuses two track estimates without requiring knowledge of their
    cross-correlation. Conservative but guaranteed consistent.

    Returns:
        (x_fused, P_fused)
    """
    n = len(x_a)

    # Find optimal omega that minimizes trace of fused covariance
    best_omega = 0.5
    best_trace = float("inf")

    for omega_i in range(11):  # Search omega in [0, 1]
        omega = omega_i / 10.0
        if omega < 0.01:
            omega = 0.01
        if omega > 0.99:
            omega = 0.99

        try:
            P_inv_a = np.linalg.inv(P_a)
            P_inv_b = np.linalg.inv(P_b)
            P_fused_inv = omega * P_inv_a + (1 - omega) * P_inv_b
            P_fused = np.linalg.inv(P_fused_inv)
            tr = np.trace(P_fused)
            if tr < best_trace:
                best_trace = tr
                best_omega = omega
        except np.linalg.LinAlgError:
            continue

    # Compute with best omega
    omega = best_omega
    try:
        P_inv_a = np.linalg.inv(P_a)
        P_inv_b = np.linalg.inv(P_b)
        P_fused_inv = omega * P_inv_a + (1 - omega) * P_inv_b
        P_fused = np.linalg.inv(P_fused_inv)
        x_fused = P_fused @ (omega * P_inv_a @ x_a + (1 - omega) * P_inv_b @ x_b)
        return x_fused, P_fused
    except np.linalg.LinAlgError:
        # Fallback: simple weighted average
        w_a = 1.0 / (np.trace(P_a) + 1e-10)
        w_b = 1.0 / (np.trace(P_b) + 1e-10)
        w_sum = w_a + w_b
        x_fused = (w_a * x_a + w_b * x_b) / w_sum
        P_fused = (P_a + P_b) / 2.0
        return x_fused, P_fused


class TrackCorrelator:
    """
    Correlates observations from multiple sensors to existing tracks.

    Pipeline:
    1. Predict all tracks to observation time
    2. Compute Mahalanobis distance matrix (observations × tracks)
    3. Gate: set distance to INF for unlikely pairs
    4. Hungarian assignment for optimal global association
    5. Return matched, unmatched observations, unmatched tracks
    """

    def __init__(
        self,
        gate_threshold: float = 9.21,  # chi-squared 3DoF, 99%
        max_distance: float = 50.0,  # Max Mahalanobis distance to consider
    ):
        self.gate_threshold = gate_threshold
        self.max_distance = max_distance

    def correlate(
        self,
        observations: List[
            Tuple[float, float, float, float, str]
        ],  # (lat, lon, alt, sigma_m, sensor_id)
        tracks: Dict[str, EKFState],
        t: float,
        ekf: ExtendedKalmanFilter,
    ) -> Tuple[
        List[Tuple[int, str, float]],  # matched: (obs_idx, track_id, distance)
        List[int],  # unmatched_observations
        List[str],  # unmatched_tracks
    ]:
        """
        Correlate observations to tracks.

        Returns:
            (matched, unmatched_obs, unmatched_tracks)
        """
        track_ids = list(tracks.keys())
        n_obs = len(observations)
        n_trk = len(track_ids)

        if n_obs == 0:
            return [], [], track_ids

        if n_trk == 0:
            return [], list(range(n_obs)), []

        # Predict all tracks to observation time
        predicted: Dict[str, EKFState] = {}
        for tid in track_ids:
            predicted[tid] = ekf.predict(tracks[tid], t)

        # Build cost matrix
        cost = np.full((n_obs, n_trk), 1e9)

        for i, (lat, lon, alt, sigma, sid) in enumerate(observations):
            for j, tid in enumerate(track_ids):
                dist, passed = mahalanobis_gate(
                    predicted[tid],
                    lat,
                    lon,
                    alt,
                    sigma_m=sigma,
                    gate_threshold=self.gate_threshold,
                )
                if passed and dist < self.max_distance:
                    cost[i, j] = dist

        # Hungarian assignment
        assignments = hungarian_assignment(cost)

        matched = []
        matched_obs = set()
        matched_trk = set()

        for obs_idx, trk_idx in assignments:
            tid = track_ids[trk_idx]
            dist = cost[obs_idx, trk_idx]
            matched.append((obs_idx, tid, dist))
            matched_obs.add(obs_idx)
            matched_trk.add(trk_idx)

        unmatched_obs = [i for i in range(n_obs) if i not in matched_obs]
        unmatched_trks = [track_ids[j] for j in range(n_trk) if j not in matched_trk]

        return matched, unmatched_obs, unmatched_trks
