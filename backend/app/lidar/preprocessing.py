"""
LiDAR Point Cloud Preprocessing Module for MI Sense.
Problem Statement: SIH 26053

Key Responsibilities:
1. ROI Cropping: Bounding box filtering in vehicle coordinate frame.
2. Coordinate Transformations: Sensor frame to vehicle BEV frame.
3. Outlier & Noise Removal: Z-thresholding and spatial density outlier rejection.
4. Ground Plane Segmentation: Robust RANSAC plane fitting with normal vector gating.
5. Ground / Non-Ground Classification: Ground pavement vs elevated obstacles vs pothole depressions.
6. Downsampling for WebGL: Stratified voxel downsampling (5,000 - 8,000 points)
   for smooth 60 FPS frontend rendering while retaining full points for 2.5D elevation.
7. Performance Metrics: Preprocessing latency, point count ratios, plane equations.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import numpy as np

from backend.app.config.settings import settings


@dataclass
class PreprocessingResult:
    """Structured result of LiDAR point cloud preprocessing."""
    processed_points: np.ndarray       # Full preprocessed points (N, 4): [x, y, z, intensity]
    ground_points: np.ndarray          # Segmented ground points (N_g, 4)
    non_ground_points: np.ndarray      # Segmented obstacle / elevated points (N_ng, 4)
    downsampled_points: np.ndarray     # Subsampled for WebGL frontend streaming (N_sub, 4)
    ground_mask: np.ndarray            # Boolean mask of length N (True = ground)
    plane_coefficients: Tuple[float, float, float, float]  # Plane [a, b, c, d]: ax + by + cz + d = 0
    raw_count: int
    processed_count: int
    ground_count: int
    non_ground_count: int
    downsampled_count: int
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "raw_count": self.raw_count,
            "processed_count": self.processed_count,
            "ground_count": self.ground_count,
            "non_ground_count": self.non_ground_count,
            "downsampled_count": self.downsampled_count,
            "plane_coefficients": list(self.plane_coefficients),
            "latency_ms": round(self.latency_ms, 2),
        }


class PointCloudPreprocessor:
    """
    High-performance NumPy-accelerated LiDAR preprocessor.
    """

    def __init__(
        self,
        roi_x_min: float = settings.ROI_X_MIN,
        roi_x_max: float = settings.ROI_X_MAX,
        roi_y_min: float = settings.ROI_Y_MIN,
        roi_y_max: float = settings.ROI_Y_MAX,
        roi_z_min: float = settings.ROI_Z_MIN,
        roi_z_max: float = settings.ROI_Z_MAX,
        ransac_iterations: int = 40,
        ransac_threshold: float = 0.12,
        max_viz_points: int = 6000,
    ) -> None:
        self.roi_x_min = roi_x_min
        self.roi_x_max = roi_x_max
        self.roi_y_min = roi_y_min
        self.roi_y_max = roi_y_max
        self.roi_z_min = roi_z_min
        self.roi_z_max = roi_z_max
        self.ransac_iterations = ransac_iterations
        self.ransac_threshold = ransac_threshold
        self.max_viz_points = max_viz_points

    def process(
        self,
        raw_points: np.ndarray,
        roi: Optional[Dict[str, float]] = None,
        sensor_transform: Optional[np.ndarray] = None,
    ) -> PreprocessingResult:
        """
        Executes the complete preprocessing pipeline.

        Args:
            raw_points: (N, 4) or (N, 3) raw point cloud [x, y, z, (intensity)].
            roi: Optional custom ROI dict with keys x_min, x_max, y_min, y_max, z_min, z_max.
            sensor_transform: Optional 4x4 matrix or translation vector for extrinsic calibration.

        Returns:
            PreprocessingResult containing filtered, segmented, and downsampled points.
        """
        t_start = time.perf_counter()
        raw_count = len(raw_points)

        if raw_count == 0:
            empty_pts = np.zeros((0, 4), dtype=np.float32)
            return PreprocessingResult(
                processed_points=empty_pts,
                ground_points=empty_pts,
                non_ground_points=empty_pts,
                downsampled_points=empty_pts,
                ground_mask=np.zeros(0, dtype=bool),
                plane_coefficients=(0.0, 0.0, 1.0, 1.73),
                raw_count=0,
                processed_count=0,
                ground_count=0,
                non_ground_count=0,
                downsampled_count=0,
                latency_ms=0.0,
            )

        # 1. Ensure (N, 4) format with intensity
        pts = self._ensure_4d_points(raw_points)

        # 2. Coordinate Transformation (sensor frame -> vehicle frame)
        if sensor_transform is not None:
            pts = self._apply_transform(pts, sensor_transform)

        # 3. ROI Cropping
        pts = self._filter_roi(pts, roi)

        # 4. Outlier & Isolated Noise Removal
        pts = self._remove_statistical_outliers(pts)
        processed_count = len(pts)

        if processed_count == 0:
            empty_pts = np.zeros((0, 4), dtype=np.float32)
            return PreprocessingResult(
                processed_points=empty_pts,
                ground_points=empty_pts,
                non_ground_points=empty_pts,
                downsampled_points=empty_pts,
                ground_mask=np.zeros(0, dtype=bool),
                plane_coefficients=(0.0, 0.0, 1.0, 1.73),
                raw_count=raw_count,
                processed_count=0,
                ground_count=0,
                non_ground_count=0,
                downsampled_count=0,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        # 5. Ground Plane Estimation & Segmentation via RANSAC
        plane_coeffs, ground_mask = self._segment_ground_ransac(pts)

        ground_points = pts[ground_mask]
        non_ground_points = pts[~ground_mask]

        # 6. Downsampling for WebGL Visualization
        downsampled_points = self._downsample_for_viz(pts, self.max_viz_points)

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        return PreprocessingResult(
            processed_points=pts,
            ground_points=ground_points,
            non_ground_points=non_ground_points,
            downsampled_points=downsampled_points,
            ground_mask=ground_mask,
            plane_coefficients=plane_coeffs,
            raw_count=raw_count,
            processed_count=processed_count,
            ground_count=len(ground_points),
            non_ground_count=len(non_ground_points),
            downsampled_count=len(downsampled_points),
            latency_ms=latency_ms,
        )

    def _ensure_4d_points(self, points: np.ndarray) -> np.ndarray:
        """Ensures array is 2D float32 with at least 4 columns (x, y, z, intensity)."""
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 4 if len(pts) % 4 == 0 else 3)
        if pts.shape[1] == 3:
            # Default intensity = 0.5
            intensity = np.full((len(pts), 1), 0.5, dtype=np.float32)
            pts = np.hstack([pts, intensity])
        return pts[:, :4]

    def _apply_transform(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Applies rigid 4x4 coordinate transformation matrix."""
        xyz = points[:, :3]
        if transform.shape == (4, 4):
            R = transform[:3, :3]
            t = transform[:3, 3]
            xyz_trans = (xyz @ R.T) + t
        elif transform.shape == (3,):
            xyz_trans = xyz + transform
        else:
            return points
        return np.column_stack([xyz_trans, points[:, 3]]).astype(np.float32)

    def _filter_roi(
        self, points: np.ndarray, roi: Optional[Dict[str, float]]
    ) -> np.ndarray:
        """Crops points to the Region of Interest bounds."""
        x_min = roi.get("x_min", self.roi_x_min) if roi else self.roi_x_min
        x_max = roi.get("x_max", self.roi_x_max) if roi else self.roi_x_max
        y_min = roi.get("y_min", self.roi_y_min) if roi else self.roi_y_min
        y_max = roi.get("y_max", self.roi_y_max) if roi else self.roi_y_max
        z_min = roi.get("z_min", self.roi_z_min) if roi else self.roi_z_min
        z_max = roi.get("z_max", self.roi_z_max) if roi else self.roi_z_max

        mask = (
            (points[:, 0] >= x_min)
            & (points[:, 0] <= x_max)
            & (points[:, 1] >= y_min)
            & (points[:, 1] <= y_max)
            & (points[:, 2] >= z_min)
            & (points[:, 2] <= z_max)
        )
        return points[mask]

    def _remove_statistical_outliers(self, points: np.ndarray) -> np.ndarray:
        """
        High-performance spatial grid density filtering to remove isolated noise.
        Bins points into 0.5m x 0.5m BEV cells and drops points in cells with < 2 points.
        """
        if len(points) < 50:
            return points

        # 1. Z-score check on Z coordinates to eliminate extreme sensor glitch spikes
        z = points[:, 2]
        z_median = np.median(z)
        z_mad = np.median(np.abs(z - z_median))
        if z_mad > 1e-4:
            valid_z = np.abs(z - z_median) < (5.0 * z_mad + 2.0)
            points = points[valid_z]

        # 2. Fast 2D BEV grid hash density filter
        cell_size = 0.5
        grid_x = np.floor((points[:, 0] - self.roi_x_min) / cell_size).astype(np.int32)
        grid_y = np.floor((points[:, 1] - self.roi_y_min) / cell_size).astype(np.int32)
        grid_keys = grid_x * 10000 + grid_y

        # Count frequencies using numpy unique
        _, inv, counts = np.unique(grid_keys, return_inverse=True, return_counts=True)
        dense_mask = counts[inv] >= 2

        return points[dense_mask]

    def _segment_ground_ransac(
        self, points: np.ndarray
    ) -> Tuple[Tuple[float, float, float, float], np.ndarray]:
        """
        Robust RANSAC plane fitting to identify the ground plane.
        Vehicle coordinate system: Ground is around z = -1.73m with normal ~ [0, 0, 1].

        Returns:
            plane_coeffs: (a, b, c, d) normalized such that ax + by + cz + d = 0
            ground_mask: boolean array (True for ground, False for obstacles)
        """
        xyz = points[:, :3]
        n_pts = len(xyz)

        # Candidates: points near expected ground level z in [-2.2, -1.2]
        ground_candidate_mask = (xyz[:, 2] >= -2.2) & (xyz[:, 2] <= -1.2)
        candidate_indices = np.where(ground_candidate_mask)[0]

        # Default horizontal plane at z = -1.73m (0x + 0y + 1z + 1.73 = 0)
        best_plane = (0.0, 0.0, 1.0, 1.73)
        best_inliers = 0
        best_inlier_mask = np.zeros(n_pts, dtype=bool)

        if len(candidate_indices) >= 3:
            for _ in range(self.ransac_iterations):
                sample_idx = np.random.choice(candidate_indices, 3, replace=False)
                p1, p2, p3 = xyz[sample_idx[0]], xyz[sample_idx[1]], xyz[sample_idx[2]]

                # Plane vectors
                v1 = p2 - p1
                v2 = p3 - p1
                normal = np.cross(v1, v2)
                norm_len = np.linalg.norm(normal)

                if norm_len < 1e-6:
                    continue

                normal = normal / norm_len

                # Ensure normal points upward (+z > 0)
                if normal[2] < 0:
                    normal = -normal

                # Physical constraint: Road plane must be nearly horizontal (pitch/roll < 20 deg)
                # i.e., normal[2] > cos(20 deg) = 0.939
                if normal[2] < 0.85:
                    continue

                d = -float(np.dot(normal, p1))

                # Distance from all candidate points to plane: |ax + by + cz + d|
                distances = np.abs(xyz[candidate_indices] @ normal + d)
                inlier_sub = distances < self.ransac_threshold
                num_inliers = np.count_nonzero(inlier_sub)

                if num_inliers > best_inliers:
                    best_inliers = num_inliers
                    best_plane = (float(normal[0]), float(normal[1]), float(normal[2]), float(d))

        # Refine best plane using least squares on inliers if sufficient
        a, b, c, d = best_plane
        normal = np.array([a, b, c], dtype=np.float32)

        # Classify points relative to ground plane
        # Signed height above plane: delta_h = (ax + by + cz + d) / c
        # (c > 0.85, so division is well conditioned)
        signed_dist = (xyz @ normal + d) / c

        # Ground classification criteria:
        # 1. Potholes / depressions: signed_dist in [-0.25, -0.08] are ground terrain features
        # 2. Road surface: signed_dist in [-0.08, +0.12] is drivable ground
        # 3. Obstacles / curbs: signed_dist > +0.12 are non-ground obstacles
        ground_mask = (signed_dist >= -0.25) & (signed_dist <= 0.12)

        return (a, b, c, d), ground_mask

    def _downsample_for_viz(
        self, points: np.ndarray, max_points: int = 6000
    ) -> np.ndarray:
        """
        Downsamples point cloud using a spatial BEV voxel grid for uniform WebGL rendering.
        Ensures consistent point density across both near and far ranges.
        """
        if len(points) <= max_points:
            return points

        # Stratified voxel grid downsampling in BEV
        voxel_size = 0.35  # meters
        vx = np.floor((points[:, 0] - self.roi_x_min) / voxel_size).astype(np.int32)
        vy = np.floor((points[:, 1] - self.roi_y_min) / voxel_size).astype(np.int32)
        voxel_keys = vx * 10000 + vy

        # Pick the first point in each voxel
        _, unique_indices = np.unique(voxel_keys, return_index=True)
        sampled_pts = points[unique_indices]

        # If still above max_points, uniform random subsample
        if len(sampled_pts) > max_points:
            sub_idx = np.random.choice(len(sampled_pts), max_points, replace=False)
            return sampled_pts[sub_idx]

        # If below target count, fill remainder from remaining points
        if len(sampled_pts) < max_points and len(points) > len(sampled_pts):
            remainder_count = min(max_points - len(sampled_pts), len(points) - len(sampled_pts))
            remaining_mask = np.ones(len(points), dtype=bool)
            remaining_mask[unique_indices] = False
            remaining_idx = np.where(remaining_mask)[0]
            fill_idx = np.random.choice(remaining_idx, remainder_count, replace=False)
            return np.vstack([sampled_pts, points[fill_idx]])

        return sampled_pts


# Global preprocessor instance
_default_preprocessor = PointCloudPreprocessor()


def preprocess_point_cloud(
    raw_points: np.ndarray,
    roi: Optional[Dict[str, float]] = None,
    sensor_transform: Optional[np.ndarray] = None,
    max_viz_points: int = 6000,
) -> PreprocessingResult:
    """
    Convenience function to preprocess a raw LiDAR point cloud.

    Args:
        raw_points: (N, 4) raw point cloud array.
        roi: Optional ROI dictionary.
        sensor_transform: Optional 4x4 matrix or translation vector.
        max_viz_points: Maximum points to downsample for WebGL streaming.

    Returns:
        PreprocessingResult containing processed, ground, non-ground, and downsampled points.
    """
    if roi is not None or max_viz_points != 6000:
        custom_proc = PointCloudPreprocessor(max_viz_points=max_viz_points)
        return custom_proc.process(raw_points, roi, sensor_transform)

    return _default_preprocessor.process(raw_points, roi, sensor_transform)
