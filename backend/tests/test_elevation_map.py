"""Tests for 2.5D Elevation Map and Dynamic Object Exclusion."""

import numpy as np
import pytest
from backend.app.mapping.elevation_map import ElevationMap25D, ElevationCellStats
from backend.app.models.schemas import TrackedObject, Detection3D


def test_empty_elevation_map():
    emap = ElevationMap25D()
    stats = emap.compute_cell_stats(0.0, 2.0, 0.0, 2.0)
    assert isinstance(stats, ElevationCellStats)
    assert stats.point_count == 0
    assert stats.has_points is False
    assert stats.occupancy == 0.0
    assert stats.slope_deg == 0.0


def test_elevation_map_basic_statistics():
    emap = ElevationMap25D()
    # Create points inside [0, 2] x [0, 2] with z values
    pts = np.array([
        [0.5, 0.5, 1.0],
        [1.0, 1.0, 2.0],
        [1.5, 1.5, 3.0],
        [0.5, 1.5, 2.0],
    ], dtype=np.float32)

    s_count, d_count = emap.update(pts)
    assert s_count == 4
    assert d_count == 0

    stats = emap.compute_cell_stats(0.0, 2.0, 0.0, 2.0)
    assert stats.point_count == 4
    assert stats.has_points is True
    assert stats.min_z == pytest.approx(1.0, 1e-4)
    assert stats.max_z == pytest.approx(3.0, 1e-4)
    assert stats.mean_z == pytest.approx(2.0, 1e-4)
    assert stats.variance_z == pytest.approx(0.5, 1e-4)
    assert stats.occupancy > 0.0


def test_slope_estimation_tilted_plane():
    emap = ElevationMap25D()
    # Points on a plane inclined at 45 degrees along X: z = x
    x = np.linspace(0.1, 1.9, 10)
    y = np.linspace(0.1, 1.9, 10)
    xx, yy = np.meshgrid(x, y)
    zz = xx  # slope dz/dx = 1 => 45 degrees
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)

    emap.update(pts)
    stats = emap.compute_cell_stats(0.0, 2.0, 0.0, 2.0)
    assert stats.point_count == 100
    # Slope should be approximately 45 degrees
    assert abs(stats.slope_deg - 45.0) < 1.0


def test_dynamic_object_exclusion():
    emap = ElevationMap25D()

    # Create static ground points
    ground_pts = np.array([
        [0.0, 5.0, 0.0],
        [1.0, 5.0, 0.0],
        [0.0, 6.0, 0.0],
        [1.0, 6.0, 0.0],
    ], dtype=np.float32)

    # Moving vehicle at position [10.0, 20.0, 1.0], length 4m, width 2m, height 1.5m
    dynamic_car = TrackedObject(
        track_id=1,
        class_name="vehicle",
        position=[10.0, 20.0, 1.0],
        dimensions=[4.0, 2.0, 1.5],
        yaw=0.0,
        velocity=[8.0, 0.0],
        speed=8.0,
        heading=0.0,
        dynamic_state="DYNAMIC",
        age=5,
        hits=5,
        confidence=0.95,
    )

    # Points on the moving car
    car_pts = np.array([
        [10.0, 20.0, 1.0],
        [10.5, 20.2, 1.2],
        [9.5, 19.8, 0.8],
    ], dtype=np.float32)

    all_pts = np.vstack([ground_pts, car_pts])

    s_count, d_count = emap.update(all_pts, dynamic_objects=[dynamic_car], filter_dynamic=True)
    assert s_count == 4
    assert d_count == 3

    # Car cell should now have 0 static points
    car_cell_stats = emap.compute_cell_stats(9.0, 11.0, 19.0, 21.0)
    assert car_cell_stats.point_count == 0
    assert car_cell_stats.has_points is False


def test_road_reference_elevation():
    emap = ElevationMap25D()
    # Road surface at z = -1.6m with small noise
    rng = np.random.default_rng(42)
    x = rng.uniform(-2.5, 2.5, 200)
    y = rng.uniform(0.0, 20.0, 200)
    z = -1.6 + rng.normal(0.0, 0.02, 200)
    pts = np.column_stack([x, y, z]).astype(np.float32)

    emap.update(pts)
    ref_z = emap.get_road_reference_elevation()
    assert abs(ref_z - (-1.6)) < 0.05
