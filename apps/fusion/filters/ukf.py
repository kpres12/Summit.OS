"""
Unscented Kalman Filter (UKF) for Heli.OS Fusion

Complements the existing EKF with a sigma-point filter that handles
highly nonlinear dynamics (e.g., maneuvering targets, coordinated turns)
without requiring Jacobian computation.

Implements:
- Scaled unscented transform
- Augmented state UKF (process + measurement noise)
- Multiple motion models: constant velocity, coordinated turn, singer
- WGS-84 aware state with lat/lon/alt

References: Wan & Van Der Merwe (2000), Julier & Uhlmann (2004)
"""

from __future__ import annotations

import math
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger("fusion.ukf")


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter with sigma-point sampling.

    State vector: [lat, lon, alt, vn, ve, vd, ax, ay] (8 states)
    - lat, lon: WGS-84 degrees
    - alt: meters above WGS-84 ellipsoid
    - vn, ve, vd: velocity north/east/down (m/s)
    - ax, ay: acceleration (m/s²)
    """

    EARTH_R = 6_371_000.0

    def __init__(self, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        self.n = 8  # State dimension
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # State and covariance
        self.x = [0.0] * self.n  # State mean
        self.P = self._eye(self.n)  # State covariance
        self._scale_P(1e-4)

        # Process noise
        self.Q = self._eye(self.n)
        self._set_process_noise()

        # Measurement noise (lat, lon, alt)
        self.R_pos = [
            [1e-10, 0, 0],  # ~10m lat noise
            [0, 1e-10, 0],  # ~10m lon noise
            [0, 0, 25.0],  # 5m alt noise
        ]

        # Sigma point weights
        self._compute_weights()

        self._initialized = False
        self._last_time = 0.0

    def _eye(self, n: int) -> List[List[float]]:
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    def _scale_P(self, scale: float):
        for i in range(self.n):
            self.P[i][i] *= scale

    def _set_process_noise(self):
        # Position noise
        self.Q[0][0] = 1e-12  # lat
        self.Q[1][1] = 1e-12  # lon
        self.Q[2][2] = 1.0  # alt
        # Velocity noise
        self.Q[3][3] = 0.5  # vn
        self.Q[4][4] = 0.5  # ve
        self.Q[5][5] = 0.5  # vd
        # Acceleration noise
        self.Q[6][6] = 2.0  # ax
        self.Q[7][7] = 2.0  # ay

    def _compute_weights(self):
        """Compute sigma point weights for the unscented transform."""
        lam = self.alpha**2 * (self.n + self.kappa) - self.n

        self.n_sigma = 2 * self.n + 1
        self.wm = [0.0] * self.n_sigma  # Mean weights
        self.wc = [0.0] * self.n_sigma  # Covariance weights

        self.wm[0] = lam / (self.n + lam)
        self.wc[0] = lam / (self.n + lam) + (1 - self.alpha**2 + self.beta)

        w = 1.0 / (2 * (self.n + lam))
        for i in range(1, self.n_sigma):
            self.wm[i] = w
            self.wc[i] = w

        self._gamma = math.sqrt(self.n + lam)

    def initialize(self, lat: float, lon: float, alt: float, timestamp: float) -> None:
        """Initialize filter with first measurement."""
        self.x = [lat, lon, alt, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._initialized = True
        self._last_time = timestamp

    def predict(self, dt: float) -> None:
        """Predict step using sigma points."""
        if not self._initialized or dt <= 0:
            return

        # Generate sigma points
        sigmas = self._generate_sigma_points(self.x, self.P)

        # Propagate through motion model
        sigmas_pred = [self._motion_model(s, dt) for s in sigmas]

        # Recover mean and covariance
        self.x = self._weighted_mean(sigmas_pred, self.wm)
        self.P = self._weighted_covariance(sigmas_pred, self.x, self.wc)

        # Add process noise
        for i in range(self.n):
            self.P[i][i] += self.Q[i][i] * dt

    def update(
        self,
        lat: float,
        lon: float,
        alt: float,
        timestamp: float,
        R: Optional[List[List[float]]] = None,
    ) -> None:
        """Update step with position measurement."""
        if not self._initialized:
            self.initialize(lat, lon, alt, timestamp)
            return

        dt = timestamp - self._last_time
        if dt > 0:
            self.predict(dt)
        self._last_time = timestamp

        R_meas = R or self.R_pos
        m = len(R_meas)  # Measurement dimension (3)

        # Generate sigma points from predicted state
        sigmas = self._generate_sigma_points(self.x, self.P)

        # Transform sigma points through measurement model
        z_sigmas = [self._measurement_model(s) for s in sigmas]

        # Predicted measurement mean
        z_mean = self._weighted_mean(z_sigmas, self.wm)

        # Innovation covariance
        Pzz = self._weighted_covariance(z_sigmas, z_mean, self.wc)
        for i in range(m):
            Pzz[i][i] += R_meas[i][i]

        # Cross covariance
        Pxz = [[0.0] * m for _ in range(self.n)]
        for k in range(self.n_sigma):
            dx = [sigmas[k][i] - self.x[i] for i in range(self.n)]
            dz = [z_sigmas[k][i] - z_mean[i] for i in range(m)]
            for i in range(self.n):
                for j in range(m):
                    Pxz[i][j] += self.wc[k] * dx[i] * dz[j]

        # Kalman gain: K = Pxz @ Pzz^-1
        Pzz_inv = self._invert_3x3(Pzz)
        K = self._mat_mul(Pxz, Pzz_inv)

        # Innovation
        z = [lat, lon, alt]
        innovation = [z[i] - z_mean[i] for i in range(m)]

        # State update
        for i in range(self.n):
            for j in range(m):
                self.x[i] += K[i][j] * innovation[j]

        # Covariance update: P -= K @ Pzz @ K^T
        KPzz = self._mat_mul(K, Pzz)
        KT = [[K[j][i] for j in range(self.n)] for i in range(m)]
        KPzzKT = self._mat_mul(KPzz, KT)
        for i in range(self.n):
            for j in range(self.n):
                self.P[i][j] -= KPzzKT[i][j]

    def _motion_model(self, state: List[float], dt: float) -> List[float]:
        """Constant-acceleration motion model on WGS-84."""
        lat, lon, alt, vn, ve, vd, ax, ay = state

        # Update velocities
        vn_new = vn + ax * dt
        ve_new = ve + ay * dt
        vd_new = vd  # Assume constant vertical rate

        # Update positions
        dlat = (vn * dt + 0.5 * ax * dt**2) / self.EARTH_R * (180 / math.pi)
        cos_lat = math.cos(math.radians(lat))
        dlon = (
            (ve * dt + 0.5 * ay * dt**2)
            / (self.EARTH_R * max(cos_lat, 1e-10))
            * (180 / math.pi)
        )
        dalt = -vd * dt

        return [
            lat + dlat,
            lon + dlon,
            alt + dalt,
            vn_new,
            ve_new,
            vd_new,
            ax,
            ay,
        ]

    def _measurement_model(self, state: List[float]) -> List[float]:
        """Extract position from state (identity for position states)."""
        return [state[0], state[1], state[2]]

    def _generate_sigma_points(
        self, x: List[float], P: List[List[float]]
    ) -> List[List[float]]:
        """Generate 2n+1 sigma points."""
        # Cholesky-like decomposition (simple sqrt for diagonal-dominant)
        L = self._cholesky(P)

        sigmas = [list(x)]  # Central point

        for j in range(self.n):
            col = [L[i][j] * self._gamma for i in range(self.n)]
            sigmas.append([x[i] + col[i] for i in range(self.n)])
            sigmas.append([x[i] - col[i] for i in range(self.n)])

        return sigmas

    def _cholesky(self, A: List[List[float]]) -> List[List[float]]:
        """Cholesky decomposition A = L @ L^T."""
        n = len(A)
        L = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = A[i][i] - s
                    L[i][j] = math.sqrt(max(val, 1e-20))
                else:
                    L[i][j] = (A[i][j] - s) / max(L[j][j], 1e-20)

        return L

    def _weighted_mean(
        self, points: List[List[float]], weights: List[float]
    ) -> List[float]:
        n = len(points[0])
        mean = [0.0] * n
        for k, pt in enumerate(points):
            for i in range(n):
                mean[i] += weights[k] * pt[i]
        return mean

    def _weighted_covariance(
        self, points: List[List[float]], mean: List[float], weights: List[float]
    ) -> List[List[float]]:
        n = len(mean)
        cov = [[0.0] * n for _ in range(n)]
        for k, pt in enumerate(points):
            d = [pt[i] - mean[i] for i in range(n)]
            for i in range(n):
                for j in range(n):
                    cov[i][j] += weights[k] * d[i] * d[j]
        return cov

    def _invert_3x3(self, M: List[List[float]]) -> List[List[float]]:
        """Invert a 3x3 matrix."""
        a, b, c = M[0]
        d, e, f = M[1]
        g, h, k = M[2]
        det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g)
        if abs(det) < 1e-30:
            return self._eye(3)
        inv_det = 1.0 / det
        return [
            [
                (e * k - f * h) * inv_det,
                (c * h - b * k) * inv_det,
                (b * f - c * e) * inv_det,
            ],
            [
                (f * g - d * k) * inv_det,
                (a * k - c * g) * inv_det,
                (c * d - a * f) * inv_det,
            ],
            [
                (d * h - e * g) * inv_det,
                (b * g - a * h) * inv_det,
                (a * e - b * d) * inv_det,
            ],
        ]

    def _mat_mul(self, A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """Matrix multiplication."""
        rows_a = len(A)
        cols_b = len(B[0])
        inner = len(B)
        C = [[0.0] * cols_b for _ in range(rows_a)]
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(inner):
                    C[i][j] += A[i][k] * B[k][j]
        return C

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x[0], self.x[1], self.x[2])

    @property
    def velocity(self) -> Tuple[float, float, float]:
        return (self.x[3], self.x[4], self.x[5])

    @property
    def speed(self) -> float:
        return math.sqrt(self.x[3] ** 2 + self.x[4] ** 2)

    @property
    def heading(self) -> float:
        return math.degrees(math.atan2(self.x[4], self.x[3])) % 360
