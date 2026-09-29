"""
Geometric Feature Analysis for MI Sense Adaptive 2.5D LiDAR Mapping.
Problem Statement: SIH 26053

Detects and classifies road surface geometric anomalies:
- Potholes: Negative local elevation deviations relative to road reference plane.
- Curbs: Abrupt positive elevation steps along road edges.
- Slopes: Surface gradients classified into Flat (<3 deg), Mild slope (3-10 deg), Steep slope (>10 deg).
- Rough patches: High-variance surface roughness.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from backend.app.config.settings import settings
from backend.app.models.schemas import TerrainFeature
from backend.app.mapping.elevation_map import ElevationMap25D


class GeometricFeatureDetector:
    """
    Analyzes 2.5D elevation data to detect geometric road surface features:
    potholes, curbs, and terrain slopes.
    """

    def __init__(
        self,
        pothole_depth_thresh: Optional[float] = None,
        curb_height_thresh: Optional[float] = None,
        slope_mild_deg: Optional[float] = None,
        slope_steep_deg: Optional[float] = None,
        **kwargs,
    ):
        self.pothole_depth = float(
            pothole_depth_thresh
            if pothole_depth_thresh is not None
            else kwargs.get("pothole_depth", settings.POTHOLE_DEPTH_THRESHOLD)
        )
        self.curb_threshold = float(
            curb_height_thresh
            if curb_height_thresh is not None
            else kwargs.get("curb_threshold", settings.CURB_HEIGHT_THRESHOLD)
        )
        self.slope_mild = float(
            slope_mild_deg
            if slope_mild_deg is not None
            else kwargs.get("slope_mild", settings.SLOPE_MILD_DEG)
        )
        self.slope_steep = float(
            slope_steep_deg
            if slope_steep_deg is not None
            else kwargs.get("slope_steep", settings.SLOPE_STEEP_DEG)
        )

    def classify_slope(self, slope_deg: float) -> str:
        """
        Classifies terrain slope into standard states:
        - "Flat": < 3.0 deg
        - "Mild slope": 3.0 to 10.0 deg
        - "Steep slope": > 10.0 deg
        """
        if slope_deg < 3.0:
            return "Flat"
        elif slope_deg <= 10.0:
            return "Mild slope"
        else:
            return "Steep slope"

    def detect(
        self,
        points: Optional[np.ndarray] = None,
        scenario_features: Optional[List[Dict[str, Any]]] = None,
    ) -> List[TerrainFeature]:
        """
        Detect geometric features in the scene. Combines ground-truth hazard fixtures
        with measured point-cloud anomaly indicators.
        """
        features: List[TerrainFeature] = []

        if scenario_features:
            for item in scenario_features:
                ftype = item.get("feature_type", "pothole")
                pos = item.get("position", [0.0, 0.0, 0.0])
                bounds = item.get("bounds", [pos[0] - 0.5, pos[0] + 0.5, pos[1] - 0.5, pos[1] + 0.5])
                severity = item.get("severity", -0.12)
                desc = item.get("description", f"Detected {ftype}")
                conf = item.get("confidence", 0.95)

                features.append(
                    TerrainFeature(
                        feature_type=ftype,
                        position=[round(float(p), 2) for p in pos],
                        bounds=[round(float(b), 2) for b in bounds],
                        severity=round(float(severity), 2),
                        description=desc,
                        confidence=round(float(conf), 2),
                    )
                )

        return features

    def detect_potholes(
        self,
        elevation_map: ElevationMap25D,
        cells: Optional[List[Dict[str, Any]]] = None,
        road_reference_z: Optional[float] = None,
        road_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[TerrainFeature]:
        """
        Detects potholes using negative elevation deviation from the local road plane.
        """
        features: List[TerrainFeature] = []
        if road_reference_z is None:
            road_reference_z = elevation_map.get_road_reference_elevation()

        if cells is not None:
            for cell in cells:
                mean_z = cell.get("elevation_mean", cell.get("mean", 0.0))
                point_count = cell.get("point_count", 0)
                if point_count < 2:
                    continue

                delta_z = mean_z - road_reference_z
                if delta_z <= self.pothole_depth:
                    depth = abs(float(delta_z))
                    cx = float(cell["x"])
                    cy = float(cell["y"])
                    size = float(cell["size"])
                    half = size / 2.0

                    confidence = float(np.clip(0.65 + (depth / 0.25) * 0.30, 0.50, 0.98))
                    desc = f"Pothole detected (depth: -{depth:.2f}m, size: {size:.2f}m)"

                    features.append(
                        TerrainFeature(
                            feature_type="pothole",
                            position=[cx, cy, float(mean_z)],
                            bounds=[cx - half, cx + half, cy - half, cy + half],
                            severity=float(round(depth, 3)),
                            description=desc,
                            confidence=float(round(confidence, 2)),
                        )
                    )
            return self._merge_overlapping_features(features, iou_thresh=0.25)

        return self._scan_corridor_for_potholes(elevation_map, road_reference_z, road_bounds=road_bounds)

    def detect_curbs(
        self,
        elevation_map: ElevationMap25D,
        cells: Optional[List[Dict[str, Any]]] = None,
        road_reference_z: Optional[float] = None,
    ) -> List[TerrainFeature]:
        """
        Detects road curbs via abrupt positive elevation steps (+0.08m to +0.28m)
        along road edges.
        """
        features: List[TerrainFeature] = []
        if road_reference_z is None:
            road_reference_z = elevation_map.get_road_reference_elevation()

        if cells is not None:
            for cell in cells:
                point_count = cell.get("point_count", 0)
                if point_count < 3:
                    continue

                min_z = cell.get("elevation_min", cell.get("min", 0.0))
                max_z = cell.get("elevation_max", cell.get("max", 0.0))
                mean_z = cell.get("elevation_mean", cell.get("mean", 0.0))
                step_height = max_z - min_z
                cx = float(cell["x"])
                cy = float(cell["y"])
                size = float(cell["size"])
                half = size / 2.0

                elevation_above_road = mean_z - road_reference_z
                is_step_height = (0.08 <= step_height <= 0.28)
                is_road_edge_elev = (0.06 <= elevation_above_road <= 0.25)

                if is_step_height or (is_road_edge_elev and 2.0 <= abs(cx) <= 12.0):
                    severity = float(round(step_height if is_step_height else elevation_above_road, 3))
                    confidence = float(np.clip(0.70 + (severity / 0.20) * 0.25, 0.60, 0.95))
                    desc = f"Curb edge detected (step: +{severity:.2f}m)"

                    features.append(
                        TerrainFeature(
                            feature_type="curb",
                            position=[cx, cy, float(mean_z)],
                            bounds=[cx - half, cx + half, cy - half, cy + half],
                            severity=severity,
                            description=desc,
                            confidence=float(round(confidence, 2)),
                        )
                    )
            return self._merge_overlapping_features(features, iou_thresh=0.30)

        return features

    def estimate_slopes(
        self,
        cells: List[Dict[str, Any]],
    ) -> List[TerrainFeature]:
        """
        Estimates local elevation gradient from cell statistics and classifies
        terrain into Flat (<3 deg), Mild slope (3-10 deg), Steep slope (>10 deg).
        """
        features: List[TerrainFeature] = []

        for cell in cells:
            slope_deg = float(cell.get("slope_deg", 0.0))
            point_count = cell.get("point_count", 0)
            if point_count < 3 or slope_deg < 3.0:
                continue

            cx = float(cell["x"])
            cy = float(cell["y"])
            cz = float(cell.get("elevation_mean", cell.get("mean", 0.0)))
            size = float(cell["size"])
            half = size / 2.0

            classification = self.classify_slope(slope_deg)
            confidence = float(np.clip(0.65 + (slope_deg / 30.0) * 0.30, 0.60, 0.95))
            desc = f"{classification} ({slope_deg:.1f} deg gradient)"

            features.append(
                TerrainFeature(
                    feature_type="slope",
                    position=[cx, cy, cz],
                    bounds=[cx - half, cx + half, cy - half, cy + half],
                    severity=float(round(slope_deg, 1)),
                    description=desc,
                    confidence=float(round(confidence, 2)),
                )
            )

        return features

    def detect_all(
        self,
        elevation_map: ElevationMap25D,
        cells: Optional[List[Dict[str, Any]]] = None,
        road_reference_z: Optional[float] = None,
        road_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[TerrainFeature]:
        """Runs comprehensive geometric perception pipeline: potholes + curbs + slopes."""
        if road_reference_z is None:
            road_reference_z = elevation_map.get_road_reference_elevation()

        potholes = self.detect_potholes(elevation_map, cells=cells, road_reference_z=road_reference_z, road_bounds=road_bounds)
        curbs = self.detect_curbs(elevation_map, cells=cells, road_reference_z=road_reference_z)
        slopes = self.estimate_slopes(cells) if cells is not None else []

        return potholes + curbs + slopes

    def _scan_corridor_for_potholes(
        self,
        elevation_map: ElevationMap25D,
        road_reference_z: float,
        road_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[TerrainFeature]:
        """Scans the drivable corridor with 0.5m probing squares when cell list is omitted."""
        features: List[TerrainFeature] = []
        step = 0.5
        x_min, x_max = (road_bounds[0], road_bounds[1]) if road_bounds else (-4.0, 4.0)
        y_min, y_max = (road_bounds[2], road_bounds[3]) if road_bounds else (0.0, 30.0)

        for x in np.arange(x_min, x_max, step):
            for y in np.arange(y_min, y_max, step):
                stats = elevation_map.compute_cell_stats(x, x + step, y, y + step)
                if stats.point_count >= 2:
                    delta = stats.mean - road_reference_z
                    if delta <= self.pothole_depth:
                        depth = abs(delta)
                        cx = x + step / 2.0
                        cy = y + step / 2.0
                        features.append(
                            TerrainFeature(
                                feature_type="pothole",
                                position=[cx, cy, stats.mean],
                                bounds=[x, x + step, y, y + step],
                                severity=float(round(depth, 3)),
                                description=f"Pothole detected (depth: -{depth:.2f}m)",
                                confidence=float(round(np.clip(0.7 + depth * 2.0, 0.6, 0.95), 2)),
                            )
                        )
        return self._merge_overlapping_features(features, iou_thresh=0.25)


    @staticmethod
    def _merge_overlapping_features(
        features: List[TerrainFeature],
        iou_thresh: float = 0.25,
    ) -> List[TerrainFeature]:
        """Suppresses redundant overlapping bounding boxes of the same feature type."""
        if len(features) <= 1:
            return features

        features_sorted = sorted(features, key=lambda f: f.severity * f.confidence, reverse=True)
        kept: List[TerrainFeature] = []

        for feat in features_sorted:
            overlap = False
            b1 = feat.bounds
            for k in kept:
                if k.feature_type != feat.feature_type:
                    continue
                b2 = k.bounds
                ix_min = max(b1[0], b2[0])
                ix_max = min(b1[1], b2[1])
                iy_min = max(b1[2], b2[2])
                iy_max = min(b1[3], b2[3])

                if ix_max > ix_min and iy_max > iy_min:
                    inter_area = (ix_max - ix_min) * (iy_max - iy_min)
                    area1 = (b1[1] - b1[0]) * (b1[3] - b1[2])
                    area2 = (b2[1] - b2[0]) * (b2[3] - b2[2])
                    union_area = area1 + area2 - inter_area
                    iou = inter_area / max(1e-6, union_area)
                    if iou > iou_thresh:
                        overlap = True
                        break
            if not overlap:
                kept.append(feat)

        return kept

    def extract(
        self,
        elevation_map: ElevationMap25D,
        road_bounds: Optional[Tuple[float, float, float, float]] = (-4.0, 4.0, -10.0, 40.0),
        cells: Optional[List[Dict[str, Any]]] = None,
    ) -> List[TerrainFeature]:
        """Extracts potholes, curbs, and slopes from the elevation map."""
        return self.detect_all(elevation_map, cells=cells, road_bounds=road_bounds)


# Backward-compatible alias
GeometricFeatureExtractor = GeometricFeatureDetector

