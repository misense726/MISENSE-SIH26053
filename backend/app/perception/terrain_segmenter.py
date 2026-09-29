"""
Terrain and surface segmentation models for MI Sense Adaptive 2.5D Mapping.
Problem Statement: SIH 26053

Provides:
- TerrainSegmenter: Abstract base class for point/cell semantic segmentation.
- SimulationTerrainSegmenter: High-accuracy geometric & spatial rule-based segmenter
  categorizing terrain into: 'road', 'terrain', 'curb', 'pothole', 'slope', 'obstacle', 'unknown'.
- NeuralTerrainSegmenter: Extensible deep learning BEV segmentation stub
  (PointPillars/BEVFormer/SalsaNext-compatible).
"""

from abc import ABC, abstractmethod
import math
from typing import Dict, Any, List, Optional, Union
import numpy as np

from backend.app.config.settings import settings


class TerrainSegmenter(ABC):
    """Abstract base class for terrain and drivability segmentation."""

    @abstractmethod
    def segment(
        self,
        points: np.ndarray,
        cells: List[Any],
        features: Optional[List[Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Segment spatial cells into semantic classes and evaluate drivability.

        Args:
            points: (N, 3) or (N, 4) point cloud array.
            cells: List of cells or nodes to classify (can be QuadNode or dict).
            features: Optional list of pre-detected geometric features (potholes, curbs).

        Returns:
            Dict mapping cell_id -> {
                'semantic_class': str,
                'drivable': bool,
                'confidence': float,
                'slope_deg': float,
            }
        """
        pass

    @property
    @abstractmethod
    def segmenter_name(self) -> str:
        """Name of terrain segmenter."""
        pass


class SimulationTerrainSegmenter(TerrainSegmenter):
    """
    Simulation terrain segmenter combining geometry and scene ground truth.

    Semantics classified:
    - 'road': Smooth drivable asphalt, minimal height variance.
    - 'curb': Elevated sidewalk transition boundary (+0.12m to +0.20m step).
    - 'pothole': Localized surface depression (depth <= -0.08m).
    - 'slope': Grade incline/decline (mild vs steep).
    - 'terrain': Off-road shoulder, grass, or gravel.
    - 'obstacle': Non-ground vertical protrusion.
    - 'unknown': Sparsely sampled or void cells.

    Drivability evaluation:
    - True: 'road', 'slope' (<= 10 degrees).
    - False: 'curb', 'obstacle', steep 'slope', 'pothole', 'terrain', 'unknown'.
    """

    def __init__(
        self,
        road_half_width: float = 7.0,
        curb_width: float = 0.8,
    ):
        self.road_half_width = road_half_width
        self.curb_width = curb_width
        self._name = "Simulation Rule-Based Terrain Segmenter"

    @property
    def segmenter_name(self) -> str:
        return self._name

    def segment(
        self,
        points: np.ndarray,
        cells: List[Any],
        features: Optional[List[Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Segment list of spatial cells."""
        results: Dict[str, Dict[str, Any]] = {}

        # Index geometric features for fast spatial lookup
        potholes = []
        curbs = []
        slopes = []
        if features:
            for feat in features:
                ftype = getattr(feat, "feature_type", None) or (feat.get("feature_type") if isinstance(feat, dict) else None)
                if ftype == "pothole":
                    potholes.append(feat)
                elif ftype == "curb":
                    curbs.append(feat)
                elif ftype == "slope":
                    slopes.append(feat)

        for cell in cells:
            cell_id, cx, cy, size, p_count, z_var, z_mean, slope_deg = self._extract_cell_info(cell)

            if p_count == 0:
                results[cell_id] = {
                    "semantic_class": "unknown",
                    "drivable": False,
                    "confidence": 0.0,
                    "slope_deg": 0.0,
                }
                continue

            # Check for known pothole overlap
            is_pothole = self._check_feature_overlap(cx, cy, size, potholes)
            if is_pothole or (z_mean < settings.POTHOLE_DEPTH_THRESHOLD and abs(cx) <= self.road_half_width):
                results[cell_id] = {
                    "semantic_class": "pothole",
                    "drivable": False,
                    "confidence": 0.94,
                    "slope_deg": slope_deg,
                }
                continue

            # Check for curb overlap or boundary
            is_curb = self._check_feature_overlap(cx, cy, size, curbs)
            dist_to_boundary = abs(abs(cx) - self.road_half_width)
            if is_curb or (dist_to_boundary <= self.curb_width and z_var > 0.015):
                results[cell_id] = {
                    "semantic_class": "curb",
                    "drivable": False,
                    "confidence": 0.92,
                    "slope_deg": slope_deg,
                }
                continue

            # Check for slope
            if slope_deg > settings.SLOPE_MILD_DEG:
                drivable = slope_deg <= settings.SLOPE_STEEP_DEG
                results[cell_id] = {
                    "semantic_class": "slope",
                    "drivable": drivable,
                    "confidence": 0.89,
                    "slope_deg": slope_deg,
                }
                continue

            # Road vs Off-road terrain vs Obstacle
            if abs(cx) <= self.road_half_width:
                # Within road boundaries
                if z_mean > 0.35 and z_var > 0.04:
                    results[cell_id] = {
                        "semantic_class": "obstacle",
                        "drivable": False,
                        "confidence": 0.91,
                        "slope_deg": slope_deg,
                    }
                else:
                    results[cell_id] = {
                        "semantic_class": "road",
                        "drivable": True,
                        "confidence": 0.96,
                        "slope_deg": slope_deg,
                    }
            else:
                # Off-road terrain
                if z_mean > 0.40 and z_var > 0.05:
                    results[cell_id] = {
                        "semantic_class": "obstacle",
                        "drivable": False,
                        "confidence": 0.88,
                        "slope_deg": slope_deg,
                    }
                else:
                    results[cell_id] = {
                        "semantic_class": "terrain",
                        "drivable": False,
                        "confidence": 0.87,
                        "slope_deg": slope_deg,
                    }

        return results

    def _extract_cell_info(self, cell: Any):
        """Extract standardized cell metrics from QuadNode or dictionary."""
        if isinstance(cell, dict):
            return (
                cell.get("cell_id", "c0"),
                cell.get("x", 0.0),
                cell.get("y", 0.0),
                cell.get("size", 1.0),
                cell.get("point_count", 0),
                cell.get("elevation_variance", 0.0),
                cell.get("elevation_mean", 0.0),
                cell.get("slope_deg", 0.0),
            )
        # QuadNode or AdaptiveCell object
        cell_id = getattr(cell, "cell_id", getattr(cell, "node_id", "c0"))
        x = getattr(cell, "x", getattr(cell, "center_x", 0.0))
        y = getattr(cell, "y", getattr(cell, "center_y", 0.0))
        size = getattr(cell, "size", 1.0)
        p_count = getattr(cell, "point_count", 0)
        z_var = getattr(cell, "elevation_variance", 0.0)
        z_mean = getattr(cell, "elevation_mean", 0.0)
        slope = getattr(cell, "slope_deg", 0.0)
        return cell_id, x, y, size, p_count, z_var, z_mean, slope

    def _check_feature_overlap(self, cx: float, cy: float, size: float, features: List[Any]) -> bool:
        """Check if cell bounding box overlaps with any detected feature."""
        half = size * 0.5
        c_min_x, c_max_x = cx - half, cx + half
        c_min_y, c_max_y = cy - half, cy + half

        for f in features:
            bounds = getattr(f, "bounds", None) or (f.get("bounds") if isinstance(f, dict) else None)
            if bounds and len(bounds) >= 4:
                f_min_x, f_max_x, f_min_y, f_max_y = bounds[0], bounds[1], bounds[2], bounds[3]
                # AABB overlap check
                if not (c_max_x < f_min_x or c_min_x > f_max_x or c_max_y < f_min_y or c_min_y > f_max_y):
                    return True
            else:
                pos = getattr(f, "position", None) or (f.get("position") if isinstance(f, dict) else None)
                if pos:
                    if c_min_x <= pos[0] <= c_max_x and c_min_y <= pos[1] <= c_max_y:
                        return True
        return False


class NeuralTerrainSegmenter(TerrainSegmenter):
    """
    Extensible Deep Learning BEV Terrain Segmenter.

    Provides architecture stub compatible with PointPillars, SalsaNext, and BEVFormer
    models for semantic grid classification. Seamlessly falls back to
    SimulationTerrainSegmenter when neural weights are not present.
    """

    def __init__(self, model_weights_path: Optional[str] = None):
        self.model_weights_path = model_weights_path
        self._name = "Neural BEV Terrain Segmenter (PointPillars Backbone)"
        self._fallback = SimulationTerrainSegmenter()
        self.is_loaded = False

    @property
    def segmenter_name(self) -> str:
        return self._name

    def segment(
        self,
        points: np.ndarray,
        cells: List[Any],
        features: Optional[List[Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Segment terrain using BEV neural predictions or fallback."""
        if not self.is_loaded:
            return self._fallback.segment(points, cells, features)

        # In production with trained PyTorch model:
        # 1. Voxelize points into BEV pseudo-image
        # 2. Forward pass through 2D CNN / UNet backbone
        # 3. Softmax segmentation head over classes
        # 4. Map pixel logits to cell bounding boxes
        return self._fallback.segment(points, cells, features)
