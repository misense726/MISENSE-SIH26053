"""
2.5D Elevation Mapping Engine for MI Sense Adaptive LiDAR Mapping.
Problem Statement: SIH 26053

Calculates elevation statistics per spatial region while strictly filtering
out dynamic points belonging to moving objects tracked by the tracker.
"""

from dataclasses import dataclass, field
import math
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np

from backend.app.config.settings import settings
from backend.app.models.schemas import TrackedObject, Detection3D


@dataclass
class ElevationCellStats:
    """Elevation and roughness metrics for a 2.5D grid cell."""
    point_count: int = 0
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0
    variance: float = 0.0
    slope_deg: float = 0.0
    occupancy: float = 0.0
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float32))

    @property
    def mean_z(self) -> float:
        return self.mean

    @property
    def min_z(self) -> float:
        return self.min

    @property
    def max_z(self) -> float:
        return self.max

    @property
    def variance_z(self) -> float:
        return self.variance

    @property
    def elevation_mean(self) -> float:
        return self.mean

    @property
    def elevation_min(self) -> float:
        return self.min

    @property
    def elevation_max(self) -> float:
        return self.max

    @property
    def elevation_variance(self) -> float:
        return self.variance

    @property
    def has_points(self) -> bool:
        return self.point_count > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_count": int(self.point_count),
            "mean": float(self.mean),
            "min": float(self.min),
            "max": float(self.max),
            "variance": float(self.variance),
            "slope_deg": float(self.slope_deg),
            "occupancy": float(self.occupancy),
            "has_points": bool(self.has_points),
        }


class ElevationMap25D:
    """
    2.5D Elevation Map Generator with dynamic obstacle exclusion and
    hierarchical spatial querying.
    """

    def __init__(
        self,
        ground_z_ref: float = 0.0,
        roi_bounds: Optional[Tuple[float, float, float, float]] = None,
        default_res: float = settings.MIN_CELL_SIZE,
        **kwargs,
    ):
        if roi_bounds is not None:
            self.roi_x_min, self.roi_x_max, self.roi_y_min, self.roi_y_max = roi_bounds
        else:
            self.roi_x_min = kwargs.get("roi_x_min", settings.ROI_X_MIN)
            self.roi_x_max = kwargs.get("roi_x_max", settings.ROI_X_MAX)
            self.roi_y_min = kwargs.get("roi_y_min", settings.ROI_Y_MIN)
            self.roi_y_max = kwargs.get("roi_y_max", settings.ROI_Y_MAX)

        self.roi_z_min = kwargs.get("roi_z_min", settings.ROI_Z_MIN)
        self.roi_z_max = kwargs.get("roi_z_max", settings.ROI_Z_MAX)
        self.ground_z_ref = float(ground_z_ref)
        self.fine_res = float(default_res)

        # 2D bin configuration
        self.num_bins_x = max(1, int(round((self.roi_x_max - self.roi_x_min) / self.fine_res)))
        self.num_bins_y = max(1, int(round((self.roi_y_max - self.roi_y_min) / self.fine_res)))
        self.total_bins = self.num_bins_x * self.num_bins_y

        # Point buffers
        self.raw_points: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.static_points: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.dynamic_points: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.dynamic_mask: np.ndarray = np.empty((0,), dtype=bool)

        # Spatial index
        self.bin_ptrs: np.ndarray = np.zeros(self.total_bins + 1, dtype=np.int32)
        self.sorted_static_indices: np.ndarray = np.empty((0,), dtype=np.int32)
        self.has_index: bool = False

        self._cached_road_ref: Optional[float] = None

    def update(
        self,
        points: Union[np.ndarray, List[List[float]]],
        dynamic_objects: Optional[List[Any]] = None,
        dynamic_tracks: Optional[List[TrackedObject]] = None,
        filter_dynamic: bool = True,
        margin: float = 0.25,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Updates the elevation map with incoming point cloud data and filters out dynamic objects.
        Returns (static_count, dynamic_count).
        """
        self._cached_road_ref = None

        if isinstance(points, list):
            pts_arr = np.asarray(points, dtype=np.float32) if len(points) > 0 else np.empty((0, 3), dtype=np.float32)
        else:
            pts_arr = np.asarray(points, dtype=np.float32)

        if pts_arr.ndim == 1 and pts_arr.size == 0:
            pts_arr = np.empty((0, 3), dtype=np.float32)
        elif pts_arr.ndim == 2 and pts_arr.shape[1] > 3:
            pts_arr = pts_arr[:, :3]

        self.raw_points = pts_arr

        if len(self.raw_points) == 0:
            self.static_points = np.empty((0, 3), dtype=np.float32)
            self.dynamic_points = np.empty((0, 3), dtype=np.float32)
            self.dynamic_mask = np.empty((0,), dtype=bool)
            self.has_index = False
            return 0, 0

        # Merge dynamic object inputs
        dyn_actors = []
        if dynamic_objects:
            dyn_actors.extend(dynamic_objects)
        if dynamic_tracks:
            dyn_actors.extend(dynamic_tracks)

        if filter_dynamic and len(dyn_actors) > 0:
            self.static_points = self.filter_dynamic_points(self.raw_points, dyn_actors, margin=margin)
            # Dynamic count
            d_count = len(self.raw_points) - len(self.static_points)
        else:
            self.static_points = self.raw_points
            self.dynamic_points = np.empty((0, 3), dtype=np.float32)
            d_count = 0

        self._build_spatial_index()
        return len(self.static_points), d_count

    def filter_dynamic_points(
        self,
        points: np.ndarray,
        dynamic_tracks: List[Any],
        margin: float = 0.35,
    ) -> np.ndarray:
        """
        Exclude points falling inside 3D bounding boxes of moving/dynamic tracked objects.
        Ensures dynamic obstacles do not contaminate the static elevation map.
        """
        if len(dynamic_tracks) == 0 or len(points) == 0:
            return points

        keep_mask = np.ones(points.shape[0], dtype=bool)

        for track in dynamic_tracks:
            # Check dynamic state or speed threshold
            is_dynamic = False
            if isinstance(track, TrackedObject):
                is_dynamic = (track.dynamic_state == "DYNAMIC") or (track.speed >= settings.DYNAMIC_SPEED_THRESHOLD)
            elif isinstance(track, Detection3D):
                if track.velocity is not None:
                    is_dynamic = np.linalg.norm(track.velocity[:2]) >= settings.DYNAMIC_SPEED_THRESHOLD
                else:
                    is_dynamic = track.class_name in ("vehicle", "pedestrian", "cyclist")
            elif isinstance(track, dict):
                state = track.get("dynamic_state")
                speed = track.get("speed", 0.0)
                is_dynamic = (state == "DYNAMIC") or (speed >= settings.DYNAMIC_SPEED_THRESHOLD)
            else:
                is_dynamic = getattr(track, "dynamic_state", "STATIC") == "DYNAMIC"

            if not is_dynamic:
                continue

            if isinstance(track, (TrackedObject, Detection3D)):
                tx, ty, tz = track.position[0], track.position[1], track.position[2]
                dx, dy, dz = track.dimensions[0], track.dimensions[1], track.dimensions[2]
                yaw = float(track.yaw)
            else:
                pos = track.get("position", [0.0, 0.0, 0.0])
                dims = track.get("dimensions", [4.0, 2.0, 1.5])
                tx, ty, tz = float(pos[0]), float(pos[1]), float(pos[2])
                dx, dy, dz = float(dims[0]), float(dims[1]), float(dims[2])
                yaw = float(track.get("yaw", 0.0))

            half_l = (dx * 0.5) + margin
            half_w = (dy * 0.5) + margin
            half_h = (dz * 0.5) + margin

            coarse_r = max(half_l, half_w)
            dist_sq = (points[:, 0] - tx) ** 2 + (points[:, 1] - ty) ** 2
            candidate_indices = np.where((dist_sq < (coarse_r * 1.5) ** 2) & keep_mask)[0]

            if len(candidate_indices) == 0:
                continue

            cand_pts = points[candidate_indices]
            rel_x = cand_pts[:, 0] - tx
            rel_y = cand_pts[:, 1] - ty
            rel_z = cand_pts[:, 2] - tz

            cos_y = math.cos(-yaw)
            sin_y = math.sin(-yaw)
            rot_x = rel_x * cos_y - rel_y * sin_y
            rot_y = rel_x * sin_y + rel_y * cos_y

            in_box = (
                (np.abs(rot_x) <= half_l) &
                (np.abs(rot_y) <= half_w) &
                (np.abs(rel_z) <= half_h)
            )

            keep_mask[candidate_indices[in_box]] = False

        return points[keep_mask]

    def _build_spatial_index(self):
        """Builds 2D grid spatial index over static points."""
        n_pts = len(self.static_points)
        if n_pts == 0:
            self.bin_ptrs = np.zeros(self.total_bins + 1, dtype=np.int32)
            self.sorted_static_indices = np.empty((0,), dtype=np.int32)
            self.has_index = False
            return

        x = self.static_points[:, 0]
        y = self.static_points[:, 1]

        ix = np.floor((x - self.roi_x_min) / self.fine_res).astype(np.int32)
        iy = np.floor((y - self.roi_y_min) / self.fine_res).astype(np.int32)

        valid = (ix >= 0) & (ix < self.num_bins_x) & (iy >= 0) & (iy < self.num_bins_y)
        valid_indices = np.flatnonzero(valid)

        if len(valid_indices) == 0:
            self.bin_ptrs = np.zeros(self.total_bins + 1, dtype=np.int32)
            self.sorted_static_indices = np.empty((0,), dtype=np.int32)
            self.has_index = False
            return

        bin_keys = ix[valid] * self.num_bins_y + iy[valid]
        sort_order = np.argsort(bin_keys, kind="stable")
        sorted_keys = bin_keys[sort_order]
        self.sorted_static_indices = valid_indices[sort_order]

        counts = np.bincount(sorted_keys, minlength=self.total_bins)
        self.bin_ptrs = np.zeros(self.total_bins + 1, dtype=np.int32)
        self.bin_ptrs[1:] = np.cumsum(counts)
        self.has_index = True

    def get_points_in_cell(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> np.ndarray:
        """Fast retrieval of static points inside bounding box [x_min, x_max, y_min, y_max]."""
        if not self.has_index or len(self.static_points) == 0:
            return np.empty((0, 3), dtype=np.float32)

        b_x0 = max(0, int(np.floor((x_min - self.roi_x_min) / self.fine_res)))
        b_x1 = min(self.num_bins_x - 1, int(np.floor((x_max - 1e-4 - self.roi_x_min) / self.fine_res)))
        b_y0 = max(0, int(np.floor((y_min - self.roi_y_min) / self.fine_res)))
        b_y1 = min(self.num_bins_y - 1, int(np.floor((y_max - 1e-4 - self.roi_y_min) / self.fine_res)))

        if b_x0 > b_x1 or b_y0 > b_y1:
            return np.empty((0, 3), dtype=np.float32)

        slices = []
        for bx in range(b_x0, b_x1 + 1):
            start_bin = bx * self.num_bins_y + b_y0
            end_bin = bx * self.num_bins_y + b_y1
            p_start = self.bin_ptrs[start_bin]
            p_end = self.bin_ptrs[end_bin + 1]
            if p_end > p_start:
                slices.append(self.sorted_static_indices[p_start:p_end])

        if not slices:
            return np.empty((0, 3), dtype=np.float32)

        all_idx = np.concatenate(slices)
        pts = self.static_points[all_idx]

        mask = (
            (pts[:, 0] >= x_min) & (pts[:, 0] < x_max) &
            (pts[:, 1] >= y_min) & (pts[:, 1] < y_max)
        )
        return pts[mask]

    def compute_cell_stats(
        self,
        *args,
        **kwargs,
    ) -> ElevationCellStats:
        """
        Calculates 2.5D elevation representation for a cell.
        Supports both signatures:
        1. compute_cell_stats(cell_points: np.ndarray)
        2. compute_cell_stats(x_min: float, x_max: float, y_min: float, y_max: float, points=None)
        """
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            cell_points = args[0]
            area = 1.0
        elif "cell_points" in kwargs:
            cell_points = kwargs["cell_points"]
            area = 1.0
        elif len(args) >= 4:
            x_min, x_max, y_min, y_max = float(args[0]), float(args[1]), float(args[2]), float(args[3])
            pts_kw = kwargs.get("points", args[4] if len(args) > 4 else None)
            if pts_kw is not None:
                cell_points = pts_kw
            else:
                cell_points = self.get_points_in_cell(x_min, x_max, y_min, y_max)
            area = max(0.0625, (x_max - x_min) * (y_max - y_min))
        else:
            return ElevationCellStats()

        if len(cell_points) == 0:
            return ElevationCellStats()

        z_vals = cell_points[:, 2]
        count = int(len(z_vals))
        mean_z = float(np.mean(z_vals))
        min_z = float(np.min(z_vals))
        max_z = float(np.max(z_vals))
        var_z = float(np.var(z_vals)) if count > 1 else 0.0

        # Occupancy
        occupancy = float(np.clip(count / max(1.0, area * 16.0), 0.05, 1.0))

        # Slope estimation via planar fit
        slope_deg = 0.0
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        if count >= 3:
            x_vals = cell_points[:, 0] - np.mean(cell_points[:, 0])
            y_vals = cell_points[:, 1] - np.mean(cell_points[:, 1])
            z_centered = z_vals - mean_z

            s_xx = float(np.sum(x_vals * x_vals))
            s_yy = float(np.sum(y_vals * y_vals))
            s_xy = float(np.sum(x_vals * y_vals))
            s_xz = float(np.sum(x_vals * z_centered))
            s_yz = float(np.sum(y_vals * z_centered))

            det = s_xx * s_yy - s_xy * s_xy
            if det > 1e-6:
                a = (s_yy * s_xz - s_xy * s_yz) / det
                b = (s_xx * s_yz - s_xy * s_xz) / det
                norm = math.sqrt(a * a + b * b + 1.0)
                normal = np.array([-a / norm, -b / norm, 1.0 / norm], dtype=np.float32)
                cos_theta = min(1.0, max(0.0, 1.0 / norm))
                slope_deg = math.degrees(math.acos(cos_theta))
            else:
                # SVD fallback
                coords = np.column_stack([x_vals, y_vals, z_centered])
                try:
                    _, _, vh = np.linalg.svd(coords, full_matrices=False)
                    n = vh[2].astype(np.float32)
                    if n[2] < 0:
                        n = -n
                    n_len = np.linalg.norm(n)
                    if n_len > 1e-6:
                        normal = n / n_len
                        cos_theta = float(np.clip(normal[2], 0.0, 1.0))
                        slope_deg = math.degrees(math.acos(cos_theta))
                except Exception:
                    slope_deg = 0.0

        return ElevationCellStats(
            point_count=count,
            mean=round(mean_z, 3),
            min=round(min_z, 3),
            max=round(max_z, 3),
            variance=round(var_z, 5),
            slope_deg=round(slope_deg, 2),
            occupancy=round(occupancy, 3),
            normal=normal,
        )

    def get_stats_at(self, x: float, y: float, cell_size: float = 1.0) -> ElevationCellStats:
        """Retrieves 2.5D cell stats centered at (x, y) with specified cell size."""
        half = cell_size / 2.0
        return self.compute_cell_stats(x - half, x + half, y - half, y + half)

    def get_road_reference_elevation(
        self,
        center_x: float = 0.0,
        center_y: float = 8.0,
        half_width: float = 3.0,
        half_length: float = 12.0,
    ) -> float:
        """Estimates the reference ground/road plane height in the immediate ego corridor."""
        if self._cached_road_ref is not None:
            return self._cached_road_ref

        pts = self.get_points_in_cell(
            x_min=center_x - half_width,
            x_max=center_x + half_width,
            y_min=center_y - half_length,
            y_max=center_y + half_length,
        )

        if len(pts) > 10:
            ref_z = float(np.percentile(pts[:, 2], 20))
        elif len(self.static_points) > 0:
            ref_z = float(np.percentile(self.static_points[:, 2], 15))
        else:
            ref_z = self.ground_z_ref

        self._cached_road_ref = ref_z
        return ref_z
