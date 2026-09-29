"""
Point Cloud Preprocessing Pipeline for MI Sense.
Problem Statement: SIH 26053
"""

from typing import Tuple, List
import numpy as np

from backend.app.config.settings import settings


class Preprocessor:
    """
    Performs Region-Of-Interest (ROI) spatial cropping, ground classification,
    and downsampling for real-time WebGL rendering and processing.
    """

    def __init__(
        self,
        x_min: float = settings.ROI_X_MIN,
        x_max: float = settings.ROI_X_MAX,
        y_min: float = settings.ROI_Y_MIN,
        y_max: float = settings.ROI_Y_MAX,
        z_min: float = settings.ROI_Z_MIN,
        z_max: float = settings.ROI_Z_MAX,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

    def crop_roi(self, points: np.ndarray) -> np.ndarray:
        """Filter points strictly within vehicle perception ROI."""
        if len(points) == 0:
            return points

        mask = (
            (points[:, 0] >= self.x_min) & (points[:, 0] <= self.x_max) &
            (points[:, 1] >= self.y_min) & (points[:, 1] <= self.y_max) &
            (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)
        )
        return points[mask]

    def classify_ground(
        self,
        points: np.ndarray,
        ground_height_threshold: float = 0.28,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify point cloud into ground plane returns vs above-ground obstacles.
        """
        if len(points) == 0:
            return points, points

        ground_mask = points[:, 2] <= ground_height_threshold
        ground_points = points[ground_mask]
        obstacle_points = points[~ground_mask]
        return ground_points, obstacle_points

    def downsample(self, points: np.ndarray, max_points: int = 15000) -> np.ndarray:
        """Subsample point cloud uniformly if count exceeds max_points."""
        n = len(points)
        if n <= max_points:
            return points
        stride = int(np.ceil(n / float(max_points)))
        return points[::stride]

    def prepare_webgl_payload(self, points: np.ndarray, max_render_pts: int = 6000) -> List[List[float]]:
        """
        Format point cloud for low-latency JSON WebSocket broadcast and WebGL rendering.
        """
        pts = self.downsample(points, max_render_pts)
        if len(pts) == 0:
            return []

        # Return list of [x, y, z, intensity] rounded to 2 decimal places
        return [
            [round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2), round(float(p[3]), 2)]
            for p in pts
        ]
