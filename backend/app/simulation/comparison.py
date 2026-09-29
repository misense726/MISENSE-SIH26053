"""Detached pipeline inputs and on-demand uniform 2.5D comparisons."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple
from uuid import uuid4

import numpy as np

from backend.app.mapping.elevation_map import ElevationMap25D
from backend.app.mapping.quadtree import AdaptiveQuadtree
from backend.app.mapping.semantic_fusion import SemanticFusionEngine
from backend.app.models.schemas import (
    AdaptiveCell, ComparisonRow, ComparisonSnapshot, FrameMetrics, LidarFrame,
    TerrainFeature, TrackedObject,
)
from backend.app.perception.terrain_segmenter import SimulationTerrainSegmenter


def compact_cell(cell: AdaptiveCell) -> ComparisonRow:
    """x/y are centers; z is measured mean elevation, not a synthesized height."""
    return (
        cell.x, cell.y, cell.size, cell.elevation_mean, cell.point_count,
        cell.semantic_class, cell.drivable,
    )


@dataclass(frozen=True)
class PipelineSnapshot:
    """Private owned copies. No references to live points, nodes, or model lists."""

    snapshot_id: str
    scene_id: str
    frame_id: int
    revision: int
    timestamp: float
    static_points: np.ndarray
    ego_position: Tuple[float, ...]
    tracks: Tuple[TrackedObject, ...]
    features: Tuple[TerrainFeature, ...]
    adaptive: Tuple[ComparisonRow, ...]
    metrics: FrameMetrics
    bounds: Tuple[float, float, float, float]
    min_cell_size: float
    max_cell_size: float
    bytes_per_uniform_cell: int
    bytes_per_adaptive_cell: int
    terrain_segmenter: SimulationTerrainSegmenter

    @classmethod
    def capture(cls, frame, static_points, quadtree, evaluator, terrain_segmenter):
        points = np.array(static_points, copy=True)
        points.setflags(write=False)
        return cls(
            snapshot_id=uuid4().hex,
            scene_id=frame.scene_id,
            frame_id=frame.frame_id,
            revision=frame.revision,
            timestamp=frame.timestamp,
            static_points=points,
            ego_position=(0.0, 0.0, 0.0),
            tracks=tuple(track.model_copy(deep=True) for track in frame.tracks),
            features=tuple(feature.model_copy(deep=True) for feature in frame.terrain_features),
            adaptive=tuple(compact_cell(cell) for cell in frame.adaptive_cells),
            metrics=frame.metrics.model_copy(deep=True),
            bounds=(quadtree.roi_x_min, quadtree.roi_x_max,
                    quadtree.roi_y_min, quadtree.roi_y_max),
            min_cell_size=quadtree.min_size,
            max_cell_size=quadtree.max_size,
            bytes_per_uniform_cell=evaluator.bytes_per_uniform,
            bytes_per_adaptive_cell=evaluator.bytes_per_adaptive,
            terrain_segmenter=deepcopy(terrain_segmenter),
        )


def build_comparison(snapshot: PipelineSnapshot) -> ComparisonSnapshot:
    """CPU worker. Only reads a captured input; never accesses the live engine."""
    tree = AdaptiveQuadtree(
        roi_bounds=snapshot.bounds,
        min_size=snapshot.min_cell_size,
        max_size=snapshot.max_cell_size,
    )
    tree.elevation_map = ElevationMap25D(
        roi_bounds=snapshot.bounds, default_res=snapshot.min_cell_size,
    )
    tree.elevation_map.update(snapshot.static_points, filter_dynamic=False)
    leaves = tree.get_uniform_nodes(target_size=snapshot.min_cell_size)
    fusion = SemanticFusionEngine(terrain_segmenter=deepcopy(snapshot.terrain_segmenter))
    cells = fusion.fuse(
        leaf_nodes=leaves,
        points=snapshot.static_points,
        tracks=list(snapshot.tracks),
        features=list(snapshot.features),
    )
    return ComparisonSnapshot(
        snapshot_id=snapshot.snapshot_id,
        scene_id=snapshot.scene_id,
        frame_id=snapshot.frame_id,
        revision=snapshot.revision,
        timestamp=snapshot.timestamp,
        bounds=snapshot.bounds,
        min_cell_size=snapshot.min_cell_size,
        uniform=[compact_cell(cell) for cell in cells],
        adaptive=list(snapshot.adaptive),
        # These are the captured frame's metrics, not baseline-build timings.
        # Preserve the evaluator's rounded storage formula and KiB values.
        metrics=snapshot.metrics.model_copy(deep=True),
        bytes_per_uniform_cell=snapshot.bytes_per_uniform_cell,
        bytes_per_adaptive_cell=snapshot.bytes_per_adaptive_cell,
    )
