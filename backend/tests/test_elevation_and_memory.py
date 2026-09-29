"""
Test suite for 2.5D Elevation Map, Geometric Features, and Live Metrics.
Problem Statement: SIH 26053 - MI Sense
"""

import pytest
import numpy as np
from backend.app.mapping.elevation_map import ElevationMap25D
from backend.app.mapping.geometric_features import GeometricFeatureExtractor
from backend.app.metrics.evaluator import MetricsEvaluator
from backend.app.models.schemas import AdaptiveCell, TerrainFeature


def test_elevation_map_calculation():
    """Verify that 2.5D elevation map calculates accurate mean, min, max, and variance."""
    elev_map = ElevationMap25D(roi_bounds=(-10.0, 10.0, -10.0, 10.0), default_res=1.0)

    # Synthetic points around (0, 0)
    # Heights: 0.1, 0.2, 0.3
    points = np.array([
        [0.1, 0.1, 0.1, 0.5],
        [0.2, 0.2, 0.2, 0.5],
        [0.3, 0.3, 0.3, 0.5],
    ], dtype=np.float32)

    elev_map.update(points)
    stats = elev_map.get_stats_at(0.2, 0.2)

    assert stats is not None
    assert abs(stats.elevation_mean - 0.2) < 1e-3
    assert abs(stats.elevation_min - 0.1) < 1e-3
    assert abs(stats.elevation_max - 0.3) < 1e-3
    assert stats.point_count == 3


def test_geometric_pothole_detection():
    """Verify that negative elevation deviations relative to road reference are detected as potholes."""
    elev_map = ElevationMap25D(roi_bounds=(-10.0, 10.0, 0.0, 20.0), default_res=0.5)

    # Road points at z = 0.0 with realistic scan density
    road_pts = []
    for x in np.linspace(-3, 3, 40):
        for y in np.linspace(0, 20, 80):
            # Introduce a pothole depression at (0, 10) with depth -0.15m
            dist = np.hypot(x - 0.0, y - 10.0)
            if dist < 0.6:
                z = -0.15
            else:
                z = 0.0
            road_pts.append([x, y, z, 0.4])

    elev_map.update(np.array(road_pts, dtype=np.float32))

    extractor = GeometricFeatureExtractor(pothole_depth_thresh=-0.08)
    features = extractor.extract(elev_map, road_bounds=(-3, 3, 0, 20))

    potholes = [f for f in features if f.feature_type == "pothole"]
    assert len(potholes) > 0, "Failed to detect geometric pothole!"
    assert abs(potholes[0].position[1] - 10.0) < 1.0, f"Pothole Y coordinate mismatch: {potholes[0].position}"
    assert potholes[0].severity >= 0.08


def test_live_memory_and_cell_reduction_calculation():
    """Verify that MetricsEvaluator calculates true reduction percentages from current scene."""
    evaluator = MetricsEvaluator(roi_size=64.0, uniform_res=0.25)

    # Dummy adaptive cells (e.g. 5,000 cells)
    dummy_cells = [
        AdaptiveCell(
            cell_id=f"c_{i}",
            x=0.0,
            y=0.0,
            size=0.25 if i < 1000 else (1.0 if i < 3000 else 4.0),
            level=4 if i < 1000 else (2 if i < 3000 else 0),
            elevation_mean=0.0,
            elevation_min=0.0,
            elevation_max=0.0,
            elevation_variance=0.001,
            slope_deg=1.0,
            occupancy=0.5,
            point_count=10,
            semantic_class="road",
            drivable=True,
            importance_score=0.3,
            confidence=0.9,
            split_reason="Road",
            importance_factors={},
        )
        for i in range(5000)
    ]

    metrics = evaluator.evaluate(
        frame_id=1,
        raw_points_count=20000,
        processed_points_count=18000,
        adaptive_cells=dummy_cells,
        tracks=[],
        timings={"preprocessing": 1.2, "detection": 3.0, "tracking": 0.5, "quadtree": 1.5, "total": 6.2},
    )

    # 64 / 0.25 = 256 -> 256^2 = 65,536 uniform cells
    assert metrics.uniform_cells_count == 65536
    assert metrics.adaptive_cells_count == 5000

    # Reduction: (1 - 5000 / 65536) * 100 = ~92.37%
    expected_red = (1.0 - 5000.0 / 65536.0) * 100.0
    assert abs(metrics.cell_reduction_percent - expected_red) < 0.1
    assert metrics.cell_reduction_percent > 90.0
    assert metrics.fine_cells_count == 1000
    assert metrics.coarse_cells_count == 2000
