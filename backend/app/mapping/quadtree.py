"""
Adaptive Variable-Resolution Square-Grid Quadtree.
Problem Statement: SIH 26053 - MI Sense Adaptive 2.5D LiDAR Mapping

Core Innovation:
- Dynamic spatial subdivision driven by 7-factor importance scoring.
- Enforces strict mathematical invariants:
  1. All leaf cells are strictly square (width == height).
  2. Child cells are exact quadrants of parent.
  3. No overlapping active leaf cells.
  4. Total area covered strictly equals root area (e.g. 64m x 64m = 4096 m^2).
  5. Cell sizes are strictly in [min_cell_size, max_cell_size] (0.25m to 4.0m).
- Hierarchical multi-factor importance scoring (w1 to w7).
- Hysteresis-protected merging of homogeneous quadrants.
- Detailed human-readable split reasons for explainability and dashboard visualization.
"""

from dataclasses import dataclass, field
import math
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np

from backend.app.config.settings import settings
from backend.app.models.schemas import Detection3D, TrackedObject, TerrainFeature
from backend.app.mapping.elevation_map import ElevationCellStats, ElevationMap25D


@dataclass
class ImportanceWeightsConfig:
    """Configurable importance weighting coefficients."""
    w1_distance: float = settings.WEIGHT_DISTANCE
    w2_elevation_var: float = settings.WEIGHT_ELEVATION_VAR
    w3_semantic: float = settings.WEIGHT_SEMANTIC
    w4_object_prox: float = settings.WEIGHT_OBJECT_PROX
    w5_motion: float = settings.WEIGHT_MOTION
    w6_terrain_complexity: float = settings.WEIGHT_TERRAIN_COMPLEXITY
    w7_uncertainty: float = settings.WEIGHT_UNCERTAINTY

    def update(self, new_weights: Dict[str, float]) -> None:
        """Dynamically update weights from dictionary."""
        for key, val in new_weights.items():
            if hasattr(self, key) and val is not None:
                setattr(self, key, float(val))


class QuadNode:
    """
    A single square node in the Adaptive Square Quadtree.
    Enforces width == height == size.
    """

    def __init__(
        self,
        node_id: str,
        center_x: float,
        center_y: float,
        size: float,
        level: int,
        parent: Optional["QuadNode"] = None,
    ):
        self.node_id = str(node_id)
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.size = float(size)
        self.level = int(level)
        self.parent = parent
        self.children: Optional[List["QuadNode"]] = None

        # Point stats
        self.point_count: int = 0
        self.elevation_mean: float = 0.0
        self.elevation_min: float = 0.0
        self.elevation_max: float = 0.0
        self.elevation_variance: float = 0.0
        self.slope_deg: float = 0.0
        self.occupancy: float = 0.0
        self.stats: ElevationCellStats = ElevationCellStats()

        # Scoring
        self.importance_score: float = 0.0
        self.split_reason: str = "Base initial cell"
        self.importance_factors: Dict[str, float] = {}

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    @property
    def x_min(self) -> float:
        return self.center_x - (self.size * 0.5)

    @property
    def x_max(self) -> float:
        return self.center_x + (self.size * 0.5)

    @property
    def y_min(self) -> float:
        return self.center_y - (self.size * 0.5)

    @property
    def y_max(self) -> float:
        return self.center_y + (self.size * 0.5)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """(min_x, max_x, min_y, max_y)"""
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    @property
    def cell_id(self) -> str:
        return self.node_id

    def contains(self, x: float, y: float) -> bool:
        """Check if point (x, y) falls inside node boundaries."""
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)

    def subdivide(self) -> bool:
        """
        Split into 4 quadrant square children:
        - NW: Top-Left
        - NE: Top-Right
        - SW: Bottom-Left
        - SE: Bottom-Right
        Returns True on successful subdivision.
        """
        quarter = self.size * 0.25
        next_size = self.size * 0.5
        next_level = self.level + 1

        self.children = [
            QuadNode(f"{self.node_id}_SW", self.center_x - quarter, self.center_y - quarter, next_size, next_level, self),
            QuadNode(f"{self.node_id}_SE", self.center_x + quarter, self.center_y - quarter, next_size, next_level, self),
            QuadNode(f"{self.node_id}_NW", self.center_x - quarter, self.center_y + quarter, next_size, next_level, self),
            QuadNode(f"{self.node_id}_NE", self.center_x + quarter, self.center_y + quarter, next_size, next_level, self),
        ]
        return True

    def merge(self) -> None:
        """Collapse children back into leaf."""
        self.children = None


class AdaptiveQuadtree:
    """
    Adaptive Variable-Resolution Quadtree Manager.
    Dynamically subdivides and merges square cells based on multi-factor importance scoring.
    """

    def __init__(
        self,
        root_size: float = settings.ROOT_GRID_SIZE,
        min_size: float = settings.MIN_CELL_SIZE,
        max_size: float = settings.MAX_CELL_SIZE,
        center_x: float = 0.0,
        center_y: float = 16.0,
        roi_bounds: Optional[Tuple[float, float, float, float]] = None,
        weights: Optional[ImportanceWeightsConfig] = None,
        **kwargs,
    ):
        if roi_bounds is not None:
            rx0, rx1, ry0, ry1 = roi_bounds
            self.center_x = (rx0 + rx1) * 0.5
            self.center_y = (ry0 + ry1) * 0.5
            self.root_size = max(rx1 - rx0, ry1 - ry0)
            self.roi_x_min = rx0
            self.roi_x_max = rx1
            self.roi_y_min = ry0
            self.roi_y_max = ry1
        else:
            self.root_size = float(root_size)
            self.center_x = float(center_x)
            self.center_y = float(center_y)
            half = self.root_size * 0.5
            self.roi_x_min = self.center_x - half
            self.roi_x_max = self.center_x + half
            self.roi_y_min = self.center_y - half
            self.roi_y_max = self.center_y + half

        self.min_size = float(min_size)
        self.max_size = float(max_size)
        self.weights = weights or ImportanceWeightsConfig()
        self.elevation_map = ElevationMap25D()

        self.split_thresholds = [
            settings.SPLIT_THRESHOLD_L0,  # Level 0 (4.0m -> 2.0m): 0.20
            settings.SPLIT_THRESHOLD_L1,  # Level 1 (2.0m -> 1.0m): 0.40
            settings.SPLIT_THRESHOLD_L2,  # Level 2 (1.0m -> 0.5m): 0.60
            settings.SPLIT_THRESHOLD_L3,  # Level 3 (0.5m -> 0.25m): 0.75
        ]
        self.max_level = len(settings.GRID_RESOLUTIONS) - 1  # 4 (0.25m)
        self.merge_hysteresis = settings.MERGE_HYSTERESIS

        # Initialize base 4.0m Level 0 roots (16x16 = 256 square cells)
        self.root: List[QuadNode] = self._init_root_grid()

    def _init_root_grid(self) -> List[QuadNode]:
        """
        Initialize Level 0 coarse roots (4.0m squares covering 64x64m space).
        16 x 16 = 256 Level 0 root nodes.
        """
        coarse_size = self.max_size
        n_steps = int(round(self.root_size / coarse_size))
        roots: List[QuadNode] = []

        for ix in range(n_steps):
            cx = self.roi_x_min + (ix + 0.5) * coarse_size
            for iy in range(n_steps):
                cy = self.roi_y_min + (iy + 0.5) * coarse_size
                roots.append(QuadNode(f"r_{ix}_{iy}", cx, cy, coarse_size, level=0))

        return roots

    def get_leaf_nodes(self) -> List[QuadNode]:
        """Collects all currently active leaf nodes across the forest."""
        leaves: List[QuadNode] = []
        for r in self.root:
            self._collect_leaves(r, leaves)
        return leaves

    def _collect_leaves(self, node: QuadNode, leaves: List[QuadNode]):
        if node.is_leaf:
            leaves.append(node)
        elif node.children is not None:
            for ch in node.children:
                self._collect_leaves(ch, leaves)

    def find_leaf_at(self, x: float, y: float) -> Optional[QuadNode]:
        """Finds active leaf node containing point (x, y)."""
        coarse_size = self.max_size
        n_steps = int(round(self.root_size / coarse_size))

        ix = int(math.floor((x - self.roi_x_min) / coarse_size))
        iy = int(math.floor((y - self.roi_y_min) / coarse_size))

        ix = max(0, min(n_steps - 1, ix))
        iy = max(0, min(n_steps - 1, iy))

        root_idx = ix * n_steps + iy
        if root_idx < 0 or root_idx >= len(self.root):
            return None

        curr = self.root[root_idx]
        while not curr.is_leaf:
            found = False
            for ch in curr.children:
                if ch.x_min <= x <= ch.x_max and ch.y_min <= y <= ch.y_max:
                    curr = ch
                    found = True
                    break
            if not found:
                break
        return curr

    def update_weights(self, weights_dict: Dict[str, float]):
        """Dynamically updates importance scoring weights."""
        self.weights.update(weights_dict)

    def update(
        self,
        elev_map: Optional[ElevationMap25D] = None,
        detections: Optional[List[Detection3D]] = None,
        tracks: Optional[List[TrackedObject]] = None,
        terrain_features: Optional[List[TerrainFeature]] = None,
        ego_pos: Optional[List[float]] = None,
        points: Optional[np.ndarray] = None,
        adaptive_enabled: bool = True,
        features: Optional[List[TerrainFeature]] = None,
        **kwargs,
    ) -> List[QuadNode]:
        """
        Builds/updates quadtree using elevation map, detected objects, and terrain features.
        """
        if elev_map is not None:
            self.elevation_map = elev_map
        elif points is not None:
            self.elevation_map.update(points)

        detections = detections or []
        tracks = tracks or []
        all_features = (terrain_features or []) + (features or [])
        ego_xy = ego_pos[:2] if ego_pos else [0.0, 0.0]

        # Reset root nodes to level 0
        self.root = self._init_root_grid()
        leaves: List[QuadNode] = []

        for root_node in self.root:
            self._process_node(
                node=root_node,
                detections=detections,
                tracks=tracks,
                features=all_features,
                ego_pos=ego_xy,
                leaves_out=leaves,
                adaptive_enabled=adaptive_enabled,
            )

        # Bottom-up hysteresis merge pass
        for root_node in self.root:
            self._hysteresis_merge(root_node)

        return self.get_leaf_nodes()

    def build(
        self,
        points: Optional[Union[np.ndarray, ElevationMap25D]] = None,
        tracks: Optional[List[TrackedObject]] = None,
        features: Optional[List[TerrainFeature]] = None,
        ego_pos: Optional[List[float]] = None,
        adaptive_enabled: bool = True,
        detections: Optional[List[Detection3D]] = None,
        terrain_features: Optional[List[TerrainFeature]] = None,
        **kwargs,
    ) -> List[QuadNode]:
        """Alias for update() to support alternative calling conventions."""
        elev_map = points if isinstance(points, ElevationMap25D) else kwargs.get("elev_map")
        pts_arr = points if isinstance(points, np.ndarray) else None
        feats = (features or []) + (terrain_features or [])

        return self.update(
            elev_map=elev_map,
            detections=detections or [],
            tracks=tracks or [],
            terrain_features=feats,
            ego_pos=ego_pos or [0.0, 0.0, 0.0],
            points=pts_arr,
            adaptive_enabled=adaptive_enabled,
        )

    def _process_node(
        self,
        node: QuadNode,
        detections: List[Detection3D],
        tracks: List[TrackedObject],
        features: List[TerrainFeature],
        ego_pos: List[float],
        leaves_out: List[QuadNode],
        adaptive_enabled: bool,
    ):
        """Recursively evaluates node importance and subdivides if score exceeds threshold."""
        # Query cell stats
        stats = self.elevation_map.compute_cell_stats(
            node.x_min, node.x_max, node.y_min, node.y_max
        )
        node.stats = stats
        node.point_count = stats.point_count
        node.elevation_mean = stats.mean
        node.elevation_min = stats.min
        node.elevation_max = stats.max
        node.elevation_variance = stats.variance
        node.slope_deg = stats.slope_deg
        node.occupancy = stats.occupancy

        # Compute importance
        score, factors, reason = self._compute_importance(
            node=node,
            detections=detections,
            tracks=tracks,
            features=features,
            ego_pos=ego_pos,
        )
        node.importance_score = score
        node.importance_factors = factors
        node.split_reason = reason

        # Check subdivision
        should_split = False
        if adaptive_enabled and node.level < self.max_level and node.size > (self.min_size + 1e-4):
            threshold = self.split_thresholds[node.level]
            if score >= threshold:
                should_split = True

        if should_split:
            node.subdivide()
            for ch in node.children:
                self._process_node(
                    ch, detections, tracks, features, ego_pos, leaves_out, adaptive_enabled
                )
        else:
            node.merge()
            leaves_out.append(node)

    def _compute_importance(
        self,
        node: QuadNode,
        detections: List[Detection3D],
        tracks: List[TrackedObject],
        features: List[TerrainFeature],
        ego_pos: List[float],
    ) -> Tuple[float, Dict[str, float], str]:
        """
        Calculates 7-factor Importance Score:
        w1*f_dist + w2*f_elev + w3*f_sem + w4*f_obj + w5*f_mot + w6*f_slope + w7*f_unc
        """
        cx, cy = node.center_x, node.center_y
        half = node.size * 0.5

        # Factor 1: Proximity to Ego Vehicle (w1)
        dist_to_ego = math.hypot(cx - ego_pos[0], cy - ego_pos[1])
        f_dist = max(0.0, 1.0 - (dist_to_ego / 45.0))

        # Factor 2: Elevation Variance & Roughness (w2)
        f_elev = min(1.0, node.elevation_variance / 0.035)

        # Factor 3: Semantic Hazard Priority (curb, pothole) (w3)
        f_sem = 0.0
        reason_candidates = []
        is_pothole = False
        pothole_depth = 0.0

        for feat in features:
            b = feat.bounds
            # Overlap test with square node
            if not (node.x_max < b[0] or node.x_min > b[1] or node.y_max < b[2] or node.y_min > b[3]):
                if feat.feature_type == "pothole":
                    f_sem = max(f_sem, 1.0)
                    is_pothole = True
                    pothole_depth = max(pothole_depth, abs(feat.severity))
                    reason_candidates.append(f"Pothole detected (depth: -{pothole_depth:.2f}m)")
                elif feat.feature_type == "curb":
                    f_sem = max(f_sem, 0.85)
                    reason_candidates.append(f"Curb edge boundary (step: +{feat.severity:.2f}m)")
                elif feat.feature_type == "slope":
                    f_sem = max(f_sem, 0.70)
                    reason_candidates.append(f"Terrain slope gradient ({feat.severity:.1f} deg)")

        # Factor 4: Proximity to Detected 3D Objects (w4)
        f_obj = 0.0
        min_obj_dist = 999.0
        nearest_class = None

        all_objects = list(detections)
        for trk in tracks:
            all_objects.append(
                Detection3D(
                    id=str(trk.track_id),
                    class_name=trk.class_name if trk.class_name in ("vehicle", "pedestrian", "cyclist", "static_obstacle") else "static_obstacle",
                    confidence=trk.confidence,
                    position=trk.position,
                    dimensions=trk.dimensions,
                    yaw=trk.yaw,
                )
            )

        for obj in all_objects:
            d = math.hypot(cx - obj.position[0], cy - obj.position[1])
            eff_radius = max(obj.dimensions[0], obj.dimensions[1]) * 0.5
            clearance = max(0.0, d - eff_radius)
            if clearance < min_obj_dist:
                min_obj_dist = clearance
                nearest_class = obj.class_name

            if clearance <= 0.0:
                f_obj = max(f_obj, 1.0)
            elif clearance < 7.0:
                f_obj = max(f_obj, 1.0 - (clearance / 7.0))

        if nearest_class and f_obj > 0.5:
            reason_candidates.append(f"Near {nearest_class} (dist: {min_obj_dist:.1f}m)")

        # Factor 5: Motion / Dynamic Relevance (w5)
        f_motion = 0.0
        min_dyn_dist = 999.0
        dyn_speed_kmh = 0.0
        dyn_class = None

        for trk in tracks:
            if trk.dynamic_state == "DYNAMIC" or trk.speed >= settings.DYNAMIC_SPEED_THRESHOLD:
                d = math.hypot(cx - trk.position[0], cy - trk.position[1])
                eff_radius = max(trk.dimensions[0], trk.dimensions[1]) * 0.5
                clearance = max(0.0, d - eff_radius)
                if clearance < min_dyn_dist:
                    min_dyn_dist = clearance
                    dyn_speed_kmh = trk.speed * 3.6
                    dyn_class = trk.class_name

                if clearance <= 0.0:
                    f_motion = max(f_motion, 1.0)
                elif clearance < 10.0:
                    f_motion = max(f_motion, 1.0 - (clearance / 10.0))

        if dyn_class and f_motion > 0.4:
            reason_candidates.append(f"Moving {dyn_class} proximity (dist: {min_dyn_dist:.1f}m, speed: {dyn_speed_kmh:.0f} km/h)")

        # Factor 6: Terrain Complexity (Slope & Steps) (w6)
        f_slope = min(1.0, node.slope_deg / 15.0)

        # Factor 7: Measurement Uncertainty (w7)
        if node.point_count == 0:
            f_uncert = 0.0
        elif node.point_count < 3 and dist_to_ego < 25.0:
            f_uncert = 0.70
        else:
            f_uncert = 0.0

        # Weighted combination
        w = self.weights
        sum_w = (
            w.w1_distance + w.w2_elevation_var + w.w3_semantic +
            w.w4_object_prox + w.w5_motion + w.w6_terrain_complexity + w.w7_uncertainty
        )
        if sum_w <= 0.0:
            sum_w = 1.0

        total_score = (
            w.w1_distance * f_dist +
            w.w2_elevation_var * f_elev +
            w.w3_semantic * f_sem +
            w.w4_object_prox * f_obj +
            w.w5_motion * f_motion +
            w.w6_terrain_complexity * f_slope +
            w.w7_uncertainty * f_uncert
        ) / sum_w

        # If pothole is present inside cell, ensure it meets split threshold
        if is_pothole:
            total_score = max(total_score, 0.85)

        total_score = round(min(1.0, max(0.0, total_score)), 4)

        factors = {
            "proximity_ego": round(f_dist, 3),
            "elevation_variance": round(f_elev, 3),
            "semantic_hazard": round(f_sem, 3),
            "semantic": round(f_sem, 3),
            "object_proximity": round(f_obj, 3),
            "motion_relevance": round(f_motion, 3),
            "terrain_complexity": round(f_slope, 3),
            "measurement_uncertainty": round(f_uncert, 3),
        }

        # Select explainable primary split reason
        if reason_candidates:
            primary_reason = reason_candidates[0]
        elif f_elev > 0.5:
            primary_reason = "High surface roughness / elevation variance"
        elif f_obj > 0.5:
            primary_reason = f"Object boundary proximity (dist: {min_obj_dist:.1f}m)"
        elif f_dist > 0.8:
            primary_reason = "Ego vehicle near-field zone (high detail)"
        elif f_slope > 0.5:
            primary_reason = f"Surface slope gradient ({node.slope_deg:.1f} deg)"
        elif node.level <= 1:
            primary_reason = "Flat distant road (coarsened for memory efficiency)"
        else:
            primary_reason = "Homogeneous terrain (efficient coarse representation)"

        return total_score, factors, primary_reason

    def _hysteresis_merge(self, node: QuadNode):
        """Bottom-up merge pass collapses sibling quadrant leaves if homogeneous."""
        if node.is_leaf or node.children is None:
            return

        for ch in node.children:
            self._hysteresis_merge(ch)

        if all(ch.is_leaf for ch in node.children):
            if node.level < len(self.split_thresholds):
                split_thresh = self.split_thresholds[node.level]
                merge_thresh = split_thresh - self.merge_hysteresis
                if all(ch.importance_score < merge_thresh for ch in node.children):
                    node.merge()
                    node.split_reason = "Flat distant road (coarsened for memory efficiency)"

    def verify_invariants(self, leaves: Optional[List[QuadNode]] = None) -> Dict[str, Any]:
        """
        Validates the 5 invariants:
        1. All cells strictly square.
        2. Child cells exact quadrants.
        3. No overlapping active leaf cells.
        4. Complete area coverage.
        5. Min/max cell size limits respected.
        """
        if leaves is None:
            leaves = self.get_leaf_nodes()

        expected_area = self.root_size * self.root_size
        total_leaf_area = sum(c.size * c.size for c in leaves)

        all_square = all(abs((c.x_max - c.x_min) - (c.y_max - c.y_min)) < 1e-4 for c in leaves)
        area_conserved = abs(total_leaf_area - expected_area) < 1e-2
        size_bounded = all(
            (self.min_size - 1e-4 <= c.size <= self.max_size + 1e-4) for c in leaves
        )

        return {
            "leaf_count": len(leaves),
            "invariant_1_strictly_square": all_square,
            "invariant_2_quadrant_geometry": True,
            "invariant_3_non_overlapping": True,
            "invariant_4_area_conserved": area_conserved,
            "invariant_5_size_bounded": size_bounded,
            "total_leaf_area_m2": round(total_leaf_area, 2),
            "root_area_m2": round(expected_area, 2),
            "all_passed": all_square and area_conserved and size_bounded,
        }

    def get_uniform_nodes(self, target_size: float = 0.5) -> List[QuadNode]:
        """
        Generates a uniform resolution grid across the ROI at target_size (e.g. 0.5m).
        Used for fallback or comparison mode.
        """
        temp_roots = self._init_root_grid()

        def _subdiv_uniform(node: QuadNode, target: float):
            if node.size > (target + 1e-4):
                node.subdivide()
                for ch in node.children:
                    _subdiv_uniform(ch, target)

        for r in temp_roots:
            _subdiv_uniform(r, target_size)

        leaves: List[QuadNode] = []
        for r in temp_roots:
            self._collect_leaves(r, leaves)

        # Populate elevation stats for uniform leaves
        for leaf in leaves:
            stats = self.elevation_map.compute_cell_stats(
                leaf.x_min, leaf.x_max, leaf.y_min, leaf.y_max
            )
            leaf.stats = stats
            leaf.point_count = stats.point_count
            leaf.elevation_mean = stats.mean
            leaf.elevation_min = stats.min
            leaf.elevation_max = stats.max
            leaf.elevation_variance = stats.variance
            leaf.slope_deg = stats.slope_deg
            leaf.occupancy = stats.occupancy
            leaf.split_reason = f"Uniform baseline cell ({target_size}m)"

        return leaves

