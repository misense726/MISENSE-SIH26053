"""Tests for Geometric Features Detection (Potholes, Curbs, Slopes)."""

import numpy as np
import pytest
from backend.app.mapping.elevation_map import ElevationMap25D
from backend.app.mapping.geometric_features import GeometricFeatureDetector


def test_pothole_detection():
    detector = GeometricFeatureDetector(pothole_depth_thresh=-0.08)
    emap = ElevationMap25D()

    # Create road ground plane at z = -1.6m
    rng = np.random.default_rng(123)
    road_x = rng.uniform(-3.0, 3.0, 500)
    road_y = rng.uniform(0.0, 20.0, 500)
    road_z = np.full(500, -1.6, dtype=np.float32)

    # Add pothole points at (0.0, 5.0) with depth -0.15m -> z = -1.75m
    ph_x = rng.uniform(-0.3, 0.3, 30)
    ph_y = rng.uniform(4.7, 5.3, 30)
    ph_z = np.full(30, -1.75, dtype=np.float32)

    all_pts = np.vstack([
        np.column_stack([road_x, road_y, road_z]),
        np.column_stack([ph_x, ph_y, ph_z]),
    ])
    emap.update(all_pts)

    features = detector.detect_potholes(emap, road_reference_z=-1.6)
    potholes = [f for f in features if f.feature_type == "pothole"]
    assert len(potholes) >= 1
    ph = potholes[0]
    assert ph.severity >= 0.08
    assert "Pothole detected" in ph.description
    assert ph.confidence >= 0.5


def test_curb_detection_from_cells():
    detector = GeometricFeatureDetector(curb_height_thresh=0.12)
    emap = ElevationMap25D()

    # Provide candidate cells with curb step height
    cells = [
        # Flat road cell
        {
            "x": 0.0, "y": 10.0, "size": 0.5,
            "elevation_mean": -1.6, "elevation_min": -1.62, "elevation_max": -1.58,
            "slope_deg": 1.0, "point_count": 20,
        },
        # Curb boundary cell with 0.15m vertical step
        {
            "x": 3.5, "y": 10.0, "size": 0.5,
            "elevation_mean": -1.50, "elevation_min": -1.60, "elevation_max": -1.45,
            "slope_deg": 8.0, "point_count": 25,
        },
    ]

    curbs = detector.detect_curbs(emap, cells=cells, road_reference_z=-1.6)
    assert len(curbs) == 1
    assert curbs[0].feature_type == "curb"
    assert curbs[0].severity == pytest.approx(0.15, 1e-2)
    assert "Curb edge detected" in curbs[0].description


def test_slope_classification():
    detector = GeometricFeatureDetector()

    assert detector.classify_slope(1.5) == "Flat"
    assert detector.classify_slope(5.0) == "Mild slope"
    assert detector.classify_slope(14.0) == "Steep slope"

    cells = [
        {"x": 1.0, "y": 5.0, "size": 1.0, "elevation_mean": 0.0, "slope_deg": 2.0, "point_count": 10},
        {"x": 2.0, "y": 5.0, "size": 1.0, "elevation_mean": 0.5, "slope_deg": 7.5, "point_count": 10},
        {"x": 3.0, "y": 5.0, "size": 1.0, "elevation_mean": 1.5, "slope_deg": 16.0, "point_count": 10},
    ]

    slopes = detector.estimate_slopes(cells)
    assert len(slopes) == 2  # Only slopes >= 3 deg
    types = [s.description for s in slopes]
    assert any("Mild slope" in d for d in types)
    assert any("Steep slope" in d for d in types)
