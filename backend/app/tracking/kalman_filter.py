"""
Constant-Velocity Kalman Filter for 3D Object Tracking.
Problem Statement: SIH 26053

State vector:
    x = [x, y, z, vx, vy, vz]^T

Measurement vector:
    z = [x, y, z]^T  (or [x, y, z, vx, vy, vz]^T when Doppler/velocity is measured)

Features:
- Numerically stable Joseph form covariance update
- Symmetry enforcement on error covariance P
- Dynamic process noise scaling based on frame delta time dt
"""

import math
from typing import Optional, Tuple
import numpy as np


class KalmanFilter3D:
    """
    Constant-Velocity 3D Kalman Filter.
    Tracks 3D position [x, y, z] and velocity [vx, vy, vz].
    """

    def __init__(
        self,
        initial_pos: np.ndarray,
        initial_vel: Optional[np.ndarray] = None,
        accel_variance: float = 2.0,  # m/s^2 acceleration noise
        measurement_pos_std: float = 0.05,  # 5cm LiDAR position uncertainty
        measurement_vel_std: float = 0.20,  # 20cm/s velocity uncertainty
    ):
        # State dimension: 6 [x, y, z, vx, vy, vz]
        self.dim_x = 6
        self.dim_z = 3

        # State vector
        self.x = np.zeros((self.dim_x, 1), dtype=np.float64)
        self.x[0:3, 0] = initial_pos

        if initial_vel is not None:
            self.x[3:6, 0] = initial_vel

        # Error covariance matrix P
        self.P = np.diag([
            measurement_pos_std**2,
            measurement_pos_std**2,
            measurement_pos_std**2,
            1.0,  # Initial velocity variance
            1.0,
            1.0,
        ]).astype(np.float64)

        self.accel_var = accel_variance
        self.pos_std = measurement_pos_std
        self.vel_std = measurement_vel_std

        # Measurement noise covariance R (3x3 for position only)
        self.R_pos = np.eye(3, dtype=np.float64) * (self.pos_std**2)
        # 6x6 for position + velocity
        self.R_full = np.diag([
            self.pos_std**2,
            self.pos_std**2,
            self.pos_std**2,
            self.vel_std**2,
            self.vel_std**2,
            self.vel_std**2,
        ]).astype(np.float64)

        # Measurement matrices
        self.H_pos = np.zeros((3, 6), dtype=np.float64)
        self.H_pos[0, 0] = 1.0
        self.H_pos[1, 1] = 1.0
        self.H_pos[2, 2] = 1.0

        self.H_full = np.eye(6, dtype=np.float64)

    def _build_f_and_q(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Construct state transition matrix F and process noise Q for timestep dt."""
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Continuous white-noise acceleration model
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        q_block = np.array([
            [dt4 / 4.0, dt3 / 2.0],
            [dt3 / 2.0, dt2],
        ], dtype=np.float64) * self.accel_var

        Q = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            Q[i, i] = q_block[0, 0]
            Q[i, i + 3] = q_block[0, 1]
            Q[i + 3, i] = q_block[1, 0]
            Q[i + 3, i + 3] = q_block[1, 1]

        return F, Q

    def predict(self, dt: float = 0.0667) -> np.ndarray:
        """
        Kalman prediction step:
            x' = F * x
            P' = F * P * F^T + Q
        """
        F, Q = self._build_f_and_q(dt)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # Enforce covariance symmetry
        self.P = 0.5 * (self.P + self.P.T)
        return self.x

    def update(self, measurement: np.ndarray) -> None:
        """
        Kalman update step:
            K = P' * H^T * (H * P' * H^T + R)^-1
            x = x' + K * (z - H * x')
            P = (I - K * H) * P' * (I - K * H)^T + K * R * K^T (Joseph form)

        Args:
            measurement: (3,) [x, y, z] or (6,) [x, y, z, vx, vy, vz].
        """
        meas = np.asarray(measurement, dtype=np.float64).reshape(-1, 1)

        if meas.shape[0] == 6:
            H = self.H_full
            R = self.R_full
        else:
            H = self.H_pos
            R = self.R_pos

        # Innovation: y = z - H * x'
        y = meas - (H @ self.x)

        # Innovation covariance: S = H * P' * H^T + R
        S = H @ self.P @ H.T + R

        # Kalman gain: K = P' * H^T * S^-1
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)

        # Updated state: x = x' + K * y
        self.x = self.x + K @ y

        # Joseph form update for numerical stability and positive semi-definiteness
        I_KH = np.eye(self.dim_x, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # Symmetry enforcement
        self.P = 0.5 * (self.P + self.P.T)
        return self.position

    @property
    def position(self) -> np.ndarray:
        """Current estimated 3D position [x, y, z]."""
        return self.x[0:3, 0].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Current estimated 3D velocity [vx, vy, vz]."""
        return self.x[3:6, 0].copy()

    @property
    def speed_2d(self) -> float:
        """Magnitude of planar speed in m/s: sqrt(vx^2 + vy^2)."""
        vx = float(self.x[3, 0])
        vy = float(self.x[4, 0])
        return math.hypot(vx, vy)

    @property
    def heading_deg(self) -> float:
        """Estimated planar movement heading in degrees (-180 to 180)."""
        vx = float(self.x[3, 0])
        vy = float(self.x[4, 0])
        deg = math.degrees(math.atan2(vy, vx))
        return (deg + 360.0) % 360.0
