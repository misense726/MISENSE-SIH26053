"""Tests for Metrics and Performance Evaluator Module."""

import pytest
from backend.app.config.settings import settings
from backend.app.models.schemas import AdaptiveCell, TrackedObject, FrameMetrics
from backend.app.metrics.evaluator import PerformanceEvaluator, MetricsEvaluator


def test_evaluator_uniform_baseline_and_reduction():
    evaluator = PerformanceEvaluator()

    # 64m x 64m at 0.25m = 256 * 256 = 65,536 cells
    assert evaluator.uniform_cells_count == 65536
    # 65536 * 48 / 1024 = 3072.0 KB
    assert evaluator.uniform_memory_kb == pytest.approx(3072.0, 0.1)

    # Simulate 5000 adaptive cells
    adaptive_cells = []
    # 1000 fine cells (0.25m)
    for i in range(1000):
        adaptive_cells.append(
            AdaptiveCell(
                cell_id=f"fine_{i}",
                x=0.0, y=float(i), size=0.25, level=4,
                elevation_mean=0.0, elevation_min=0.0, elevation_max=0.0,
                elevation_variance=0.0, slope_deg=0.0, occupancy=0.5, point_count=5,
                semantic_class="road", drivable=True, dynamic_state="STATIC",
                importance_score=0.8, confidence=0.9, split_reason="Fine detail",
            )
        )
    # 2000 medium cells (0.5m)
    for i in range(2000):
        adaptive_cells.append(
            AdaptiveCell(
                cell_id=f"med_{i}",
                x=1.0, y=float(i), size=0.5, level=3,
                elevation_mean=0.0, elevation_min=0.0, elevation_max=0.0,
                elevation_variance=0.0, slope_deg=0.0, occupancy=0.5, point_count=5,
                semantic_class="road", drivable=True, dynamic_state="STATIC",
                importance_score=0.5, confidence=0.9, split_reason="Medium detail",
            )
        )
    # 2000 coarse cells (2.0m)
    for i in range(2000):
        adaptive_cells.append(
            AdaptiveCell(
                cell_id=f"coarse_{i}",
                x=2.0, y=float(i), size=2.0, level=1,
                elevation_mean=0.0, elevation_min=0.0, elevation_max=0.0,
                elevation_variance=0.0, slope_deg=0.0, occupancy=0.5, point_count=5,
                semantic_class="terrain", drivable=False, dynamic_state="STATIC",
                importance_score=0.2, confidence=0.8, split_reason="Coarse background",
            )
        )

    # 1 dynamic track, 1 static track
    tracks = [
        TrackedObject(
            track_id=1, class_name="vehicle", position=[0.0, 10.0, 0.0],
            dimensions=[4.0, 2.0, 1.5], yaw=0.0, velocity=[5.0, 0.0], speed=5.0,
            heading=0.0, dynamic_state="DYNAMIC", age=3, hits=3, confidence=0.9,
        ),
        TrackedObject(
            track_id=2, class_name="static_obstacle", position=[5.0, 10.0, 0.0],
            dimensions=[1.0, 1.0, 1.0], yaw=0.0, velocity=[0.0, 0.0], speed=0.0,
            heading=0.0, dynamic_state="STATIC", age=5, hits=5, confidence=0.85,
        ),
    ]

    timings = {
        "preprocessing": 2.5,
        "detection": 5.0,
        "tracking": 1.2,
        "quadtree": 3.8,
        "total": 12.5,
    }

    metrics = evaluator.evaluate(
        frame_id=1,
        raw_points_count=20000,
        processed_points_count=18000,
        adaptive_cells=adaptive_cells,
        tracks=tracks,
        timings=timings,
        current_fps=20.0,
    )

    assert isinstance(metrics, FrameMetrics)
    assert metrics.frame_id == 1
    assert metrics.uniform_cells_count == 65536
    assert metrics.adaptive_cells_count == 5000

    # Cell reduction %: (1 - 5000 / 65536) * 100 ~ 92.37%
    assert 91.0 < metrics.cell_reduction_percent < 93.0

    # Memory saved %: uniform_mem = 3072KB, adaptive_mem = 5000 * 56 / 1024 = 273.44KB -> ~91.1%
    assert 90.0 < metrics.memory_saved_percent < 92.0

    # Counts
    assert metrics.fine_cells_count == 1000
    assert metrics.medium_cells_count == 2000
    assert metrics.coarse_cells_count == 2000
    assert metrics.active_tracks_count == 2
    assert metrics.dynamic_objects_count == 1
    assert metrics.fps == 20.0
