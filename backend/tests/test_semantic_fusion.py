"""Tests for Semantic Fusion Engine."""

import pytest
from backend.app.models.schemas import AdaptiveCell, TrackedObject, TerrainFeature
from backend.app.mapping.quadtree import QuadNode
from backend.app.mapping.elevation_map import ElevationMap25D
from backend.app.mapping.semantic_fusion import SemanticFusionEngine, fuse_semantic_cells


def test_semantic_fusion_obstacle_and_pothole():
    engine = SemanticFusionEngine()

    # Create two quad nodes
    node_obstacle = QuadNode("node_obs", center_x=0.0, center_y=10.0, size=1.0, level=2)
    node_obstacle.point_count = 15
    node_obstacle.elevation_mean = -1.5
    node_obstacle.slope_deg = 1.0

    node_pothole = QuadNode("node_ph", center_x=1.5, center_y=15.0, size=0.5, level=3)
    node_pothole.point_count = 8
    node_pothole.elevation_mean = -1.75
    node_pothole.slope_deg = 3.0

    # Moving vehicle at (0.0, 10.0)
    tracks = [
        TrackedObject(
            track_id=101, class_name="vehicle", position=[0.0, 10.0, -1.0],
            dimensions=[4.0, 2.0, 1.5], yaw=0.0, velocity=[6.0, 0.0], speed=6.0,
            heading=0.0, dynamic_state="DYNAMIC", age=5, hits=5, confidence=0.95,
        )
    ]

    # Pothole feature at (1.5, 15.0)
    features = [
        TerrainFeature(
            feature_type="pothole", position=[1.5, 15.0, -1.75],
            bounds=[1.25, 1.75, 14.75, 15.25], severity=0.15,
            description="Pothole detected", confidence=0.92,
        )
    ]

    cells = engine.fuse(
        leaf_nodes=[node_obstacle, node_pothole],
        tracks=tracks,
        features=features,
    )

    assert len(cells) == 2

    # Check obstacle cell
    c_obs = cells[0]
    assert c_obs.cell_id == "node_obs"
    assert c_obs.semantic_class == "obstacle"
    assert c_obs.dynamic_state == "DYNAMIC"
    assert c_obs.object_class == "vehicle"
    assert c_obs.object_track_id == 101
    assert c_obs.drivable is False

    # Check pothole cell
    c_ph = cells[1]
    assert c_ph.cell_id == "node_ph"
    assert c_ph.semantic_class == "pothole"
    assert c_ph.drivable is False


def test_fuse_semantic_cells_function():
    node_road = QuadNode("node_road", center_x=0.0, center_y=5.0, size=2.0, level=1)
    node_road.point_count = 20
    node_road.elevation_mean = -1.6
    node_road.slope_deg = 1.0

    cells = fuse_semantic_cells(
        leaf_nodes=[node_road],
        tracks=[],
        terrain_features=[],
    )

    assert len(cells) == 1
    assert cells[0].semantic_class == "road"
    assert cells[0].drivable is True
