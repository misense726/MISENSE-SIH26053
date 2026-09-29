"""
Semantic Fusion Engine for MI Sense Adaptive 2.5D Mapping.
Problem Statement: SIH 26053

Combines quadtree spatial cells, geometric elevation statistics, terrain
segmentation labels, and multi-object track dynamics into unified AdaptiveCell schemas.
"""

import math
from typing import List, Dict, Any, Optional
import numpy as np

from backend.app.models.schemas import AdaptiveCell, TrackedObject, TerrainFeature
from backend.app.mapping.quadtree import QuadNode


class SemanticFusionEngine:
    """
    Fuses multiple perception modalities into individual adaptive grid cells:
    - 2.5D geometric elevation stats (mean, min, max, variance, slope)
    - Terrain semantic class and drivability boolean
    - 3D object association (object class, track ID, dynamic state)
    - Multi-factor importance scoring and explainability tags
    """

    def __init__(self, terrain_segmenter: Optional[Any] = None):
        self.terrain_segmenter = terrain_segmenter

    def fuse(
        self,
        leaf_nodes: List[QuadNode],
        points: Optional[np.ndarray] = None,
        tracks: Optional[List[TrackedObject]] = None,
        features: Optional[List[TerrainFeature]] = None,
        elev_map: Optional[Any] = None,
        **kwargs,
    ) -> List[AdaptiveCell]:
        """
        Produce fully enriched list of AdaptiveCell models.
        """
        tracks = tracks or []
        features = features or []

        if points is None:
            if elev_map is not None:
                points = getattr(elev_map, "static_points", np.empty((0, 3), dtype=np.float32))
            else:
                points = np.empty((0, 3), dtype=np.float32)

        # Run terrain segmentation if segmenter is provided
        segmentation_results: Dict[str, Dict[str, Any]] = {}
        if self.terrain_segmenter is not None and hasattr(self.terrain_segmenter, "segment"):
            try:
                segmentation_results = self.terrain_segmenter.segment(points, leaf_nodes, features)
            except Exception:
                segmentation_results = {}

        adaptive_cells: List[AdaptiveCell] = []

        # Index potholes, curbs, slopes for fast overlap testing
        potholes = [f for f in features if f.feature_type == "pothole"]
        curbs = [f for f in features if f.feature_type == "curb"]
        slopes = [f for f in features if f.feature_type == "slope"]

        for node in leaf_nodes:
            cell_id = node.node_id
            cx, cy = node.center_x, node.center_y
            size = node.size

            # Check overlap with tracked objects
            obj_class = None
            obj_track_id = None
            cell_dynamic_state = "EMPTY"

            for trk in tracks:
                tx, ty, _ = trk.position
                dx, dy, _ = trk.dimensions
                # Check bounding box overlap
                if abs(cx - tx) <= (size * 0.5 + dx * 0.5) and abs(cy - ty) <= (size * 0.5 + dy * 0.5):
                    obj_class = trk.class_name
                    obj_track_id = trk.track_id
                    cell_dynamic_state = trk.dynamic_state
                    break

            # Fallback local semantic classification if segmenter did not provide it
            if cell_id in segmentation_results:
                seg = segmentation_results[cell_id]
                sem_class = seg.get("semantic_class", "road")
                drivable = seg.get("drivable", True)
                conf = seg.get("confidence", 0.85)
            else:
                # Local heuristic classification
                is_pothole = any(
                    not (node.x_max < f.bounds[0] or node.x_min > f.bounds[1] or node.y_max < f.bounds[2] or node.y_min > f.bounds[3])
                    for f in potholes
                )
                is_curb = any(
                    not (node.x_max < f.bounds[0] or node.x_min > f.bounds[1] or node.y_max < f.bounds[2] or node.y_min > f.bounds[3])
                    for f in curbs
                )
                is_slope = node.slope_deg >= 10.0 or any(
                    not (node.x_max < f.bounds[0] or node.x_min > f.bounds[1] or node.y_max < f.bounds[2] or node.y_min > f.bounds[3])
                    for f in slopes
                )

                if obj_class is not None:
                    sem_class = "obstacle"
                    drivable = False
                    conf = 0.90
                elif is_pothole:
                    sem_class = "pothole"
                    drivable = False
                    conf = 0.92
                elif is_curb:
                    sem_class = "curb"
                    drivable = False
                    conf = 0.88
                elif is_slope:
                    sem_class = "slope"
                    drivable = (node.slope_deg <= 12.0)
                    conf = 0.85
                elif abs(cx) <= 3.75:
                    sem_class = "road"
                    drivable = True
                    conf = 0.95
                else:
                    sem_class = "terrain"
                    drivable = False
                    conf = 0.80

            # Override drivability if cell contains dynamic or static obstacle
            if obj_class is not None or cell_dynamic_state == "DYNAMIC":
                drivable = False
                if obj_class is not None:
                    sem_class = "obstacle"

            # Calculate occupancy: ratio of point count relative to expected density
            expected_pts = max(1, int(size * size * 12))  # ~12 pts/m^2 nominal
            occupancy = round(min(1.0, node.point_count / float(expected_pts)), 2)

            if cell_dynamic_state == "EMPTY" and node.point_count > 0:
                cell_dynamic_state = "STATIC"

            adaptive_cells.append(
                AdaptiveCell(
                    cell_id=cell_id,
                    x=round(cx, 3),
                    y=round(cy, 3),
                    size=round(size, 2),
                    level=node.level,
                    elevation_mean=round(node.elevation_mean, 3),
                    elevation_min=round(node.elevation_min, 3),
                    elevation_max=round(node.elevation_max, 3),
                    elevation_variance=round(node.elevation_variance, 5),
                    slope_deg=round(node.slope_deg, 1),
                    occupancy=occupancy,
                    point_count=node.point_count,
                    semantic_class=sem_class,  # type: ignore
                    drivable=drivable,
                    object_class=obj_class,
                    object_track_id=obj_track_id,
                    dynamic_state=cell_dynamic_state,  # type: ignore
                    importance_score=round(node.importance_score, 3),
                    confidence=round(conf, 2),
                    split_reason=node.split_reason,
                    importance_factors=node.importance_factors,
                )
            )

        return adaptive_cells


def fuse_semantic_cells(
    leaf_nodes: List[QuadNode],
    elev_map: Optional[Any] = None,
    tracks: Optional[List[TrackedObject]] = None,
    terrain_features: Optional[List[TerrainFeature]] = None,
    segmenter: Optional[Any] = None,
    points: Optional[np.ndarray] = None,
    **kwargs,
) -> List[AdaptiveCell]:
    """Convenience function for semantic fusion matching simulation engine signature."""
    engine = SemanticFusionEngine(terrain_segmenter=segmenter)
    return engine.fuse(
        leaf_nodes=leaf_nodes,
        points=points,
        tracks=tracks or [],
        features=terrain_features or [],
        elev_map=elev_map,
        **kwargs,
    )
