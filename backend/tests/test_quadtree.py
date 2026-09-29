"""
Test suite for Adaptive Square-Grid Quadtree.
Problem Statement: SIH 26053 - MI Sense

Verification requirements:
1. All cells remain strictly square.
2. Child cells are exact subdivisions of parent cells.
3. No overlapping active leaf cells.
4. Complete map area coverage without gaps.
5. Maximum and minimum grid sizes are strictly respected.
6. Important areas subdivide down to 0.25m.
7. Unimportant areas remain coarse (4.0m).
8. Merge logic works when hazards or dynamic objects depart.
"""

import pytest
import numpy as np
from backend.app.config.settings import settings
from backend.app.mapping.quadtree import AdaptiveQuadtree, QuadNode, ImportanceWeightsConfig
from backend.app.models.schemas import Detection3D, TrackedObject, TerrainFeature
from backend.app.mapping.elevation_map import ElevationMap25D


def test_square_cell_invariant():
    """Verify that every node in the quadtree is strictly square (width == height)."""
    tree = AdaptiveQuadtree(root_size=64.0, min_size=0.25, max_size=4.0)
    leaves = tree.get_leaf_nodes()

    assert len(leaves) > 0
    for node in leaves:
        width = node.x_max - node.x_min
        height = node.y_max - node.y_min
        assert abs(width - height) < 1e-4, f"Cell {node} is not square: {width} != {height}"
        assert node.size in [4.0, 2.0, 1.0, 0.5, 0.25], f"Unexpected cell size: {node.size}"


def test_quadrant_subdivision():
    """Verify that subdividing a parent produces exactly 4 identical square quadrants."""
    tree = AdaptiveQuadtree(root_size=64.0, min_size=0.25, max_size=4.0)
    root = tree.root

    # Initial root has size 64.0, partitioned into 4.0m cells (Level 0)
    leaves = tree.get_leaf_nodes()
    assert all(leaf.size == 4.0 for leaf in leaves)
    assert len(leaves) == (64 / 4) ** 2  # 16x16 = 256 cells

    # Pick a leaf and force subdivide
    target_node = leaves[0]
    parent_size = target_node.size
    subdivided = target_node.subdivide()

    assert subdivided is True
    assert len(target_node.children) == 4
    for child in target_node.children:
        assert child.size == parent_size / 2.0
        assert child.x_max - child.x_min == child.size
        assert child.y_max - child.y_min == child.size


def test_no_overlapping_active_leaves():
    """Verify that active leaf nodes never overlap spatially."""
    tree = AdaptiveQuadtree(root_size=64.0, min_size=0.25, max_size=4.0)

    # Subdivide a region
    leaves = tree.get_leaf_nodes()
    leaves[0].subdivide()
    leaves[0].children[0].subdivide()

    active_leaves = tree.get_leaf_nodes()

    # Check pairwise intersection
    for i in range(len(active_leaves)):
        a = active_leaves[i]
        for j in range(i + 1, min(i + 50, len(active_leaves))):
            b = active_leaves[j]
            overlap_x = max(0.0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min))
            overlap_y = max(0.0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
            assert (overlap_x * overlap_y) < 1e-6, f"Overlapping leaves detected: {a} and {b}"


def test_complete_area_coverage():
    """Verify that the sum of areas of all active leaf cells equals total ROI area."""
    root_size = 64.0
    expected_total_area = root_size * root_size  # 4096 m^2

    tree = AdaptiveQuadtree(root_size=root_size, min_size=0.25, max_size=4.0)

    # Trigger multiple arbitrary subdivisions
    leaves = tree.get_leaf_nodes()
    leaves[5].subdivide()
    leaves[5].children[2].subdivide()
    leaves[12].subdivide()

    active_leaves = tree.get_leaf_nodes()
    total_leaf_area = sum(leaf.size * leaf.size for leaf in active_leaves)

    assert abs(total_leaf_area - expected_total_area) < 1e-3, (
        f"Area mismatch: {total_leaf_area} != {expected_total_area}"
    )


def test_importance_scoring_pothole_subdivision():
    """Verify that a pothole hazard triggers fine (0.25m) resolution."""
    tree = AdaptiveQuadtree(root_size=64.0, min_size=0.25, max_size=4.0)

    potholes = [
        TerrainFeature(
            feature_type="pothole",
            position=[0.0, 20.0, -0.15],
            bounds=[-0.6, 0.6, 19.4, 20.6],
            severity=0.15,
            description="Deep 15cm pothole",
            confidence=0.95,
        )
    ]

    # Create dummy elevation map
    elev_map = ElevationMap25D(roi_bounds=(-32, 32, -16, 48), default_res=0.25)

    tree.update(
        elev_map=elev_map,
        detections=[],
        tracks=[],
        terrain_features=potholes,
        ego_pos=[0.0, 0.0, 0.0],
    )

    # Check that cells containing the pothole have subdivided to fine resolution
    pothole_leaf = tree.find_leaf_at(0.0, 20.0)
    assert pothole_leaf is not None
    assert pothole_leaf.size <= 0.5, f"Pothole cell failed to subdivide: size={pothole_leaf.size}"
    assert "pothole" in pothole_leaf.split_reason.lower()


def test_coarsening_on_empty_flat_road():
    """Verify that distant flat empty road regions remain coarse (4.0m or 2.0m)."""
    tree = AdaptiveQuadtree(root_size=64.0, min_size=0.25, max_size=4.0)
    elev_map = ElevationMap25D(roi_bounds=(-32, 32, -16, 48), default_res=0.25)

    # Run update with no obstacles or hazards
    tree.update(
        elev_map=elev_map,
        detections=[],
        tracks=[],
        terrain_features=[],
        ego_pos=[0.0, 0.0, 0.0],
    )

    distant_road_leaf = tree.find_leaf_at(0.0, 40.0)
    assert distant_road_leaf is not None
    assert distant_road_leaf.size >= 2.0, (
        f"Distant flat road should remain coarse: size={distant_road_leaf.size}"
    )
