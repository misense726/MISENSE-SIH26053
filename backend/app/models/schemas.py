"""
Data schemas for MI Sense Adaptive 2.5D LiDAR Mapping.
Problem Statement: SIH 26053
"""

from typing import List, Optional, Dict, Any, Literal, Tuple
from pydantic import BaseModel, Field


class PointCloudData(BaseModel):
    point_count: int
    points: List[List[float]]  # [[x, y, z, intensity], ...]
    timestamp: float


class Detection3D(BaseModel):
    id: str
    class_name: Literal["vehicle", "pedestrian", "cyclist", "static_obstacle"]
    confidence: float
    position: List[float]  # [x, y, z] center
    dimensions: List[float]  # [dx, dy, dz] length, width, height
    yaw: float  # orientation angle in radians
    velocity: Optional[List[float]] = None  # [vx, vy, vz]
    detector_source: str = "Simulated DSVT Adapter"


class TrackedObject(BaseModel):
    track_id: int
    class_name: str
    position: List[float]  # [x, y, z]
    dimensions: List[float]  # [dx, dy, dz]
    yaw: float
    velocity: List[float]  # [vx, vy]
    speed: float  # m/s
    heading: float  # degrees
    dynamic_state: Literal["DYNAMIC", "STATIC"]
    age: int
    hits: int
    confidence: float
    history: List[List[float]] = Field(default_factory=list)  # Trajectory points [[x, y], ...]


class AdaptiveCell(BaseModel):
    cell_id: str
    x: float  # Center x
    y: float  # Center y
    size: float  # Edge size in meters (0.25, 0.5, 1.0, 2.0, 4.0)
    level: int  # 0 (coarsest: 4m) to 4 (finest: 0.25m)
    elevation_mean: float
    elevation_min: float
    elevation_max: float
    elevation_variance: float
    slope_deg: float
    occupancy: float  # 0.0 to 1.0
    point_count: int
    semantic_class: Literal[
        "road", "terrain", "curb", "pothole", "slope", "obstacle", "unknown"
    ]
    drivable: bool
    object_class: Optional[str] = None
    object_track_id: Optional[int] = None
    dynamic_state: Literal["STATIC", "DYNAMIC", "EMPTY"] = "EMPTY"
    importance_score: float
    confidence: float
    split_reason: str
    importance_factors: Dict[str, float] = Field(default_factory=dict)


class TerrainFeature(BaseModel):
    feature_type: Literal["pothole", "curb", "slope", "rough_patch"]
    position: List[float]  # [x, y, z]
    bounds: List[float]  # [min_x, max_x, min_y, max_y]
    severity: float  # depth in meters or slope in degrees
    description: str
    confidence: float


class FrameMetrics(BaseModel):
    frame_id: int
    timestamp: float
    fps: float
    raw_points_count: int
    processed_points_count: int
    uniform_cells_count: int
    adaptive_cells_count: int
    cell_reduction_percent: float
    uniform_memory_kb: float
    adaptive_memory_kb: float
    memory_saved_percent: float
    fine_cells_count: int    # 0.25m
    medium_cells_count: int  # 0.5m - 1.0m
    coarse_cells_count: int  # 2.0m - 4.0m
    active_tracks_count: int
    dynamic_objects_count: int
    processing_time_ms: Dict[str, float]  # preprocessing, detection, tracking, quadtree, total


class EgoVehicleState(BaseModel):
    position: List[float]  # [x, y, z]
    velocity: List[float]  # [vx, vy, vz]
    speed: float  # km/h
    yaw: float  # degrees
    steering_angle: float  # degrees


class SimulationState(BaseModel):
    is_running: bool
    current_scene: str
    sim_time: float
    ego_state: EgoVehicleState
    playback_speed: float = 1.0


class SessionState(BaseModel):
    scene_id: str
    revision: int
    frame_id: int = -1
    is_running: bool
    playback_speed: float
    adaptive_enabled: bool
    weights: Dict[str, float]
    detector_source: str


# JSON serializes these tuples as compact arrays, with cell-center coordinates.
ComparisonRow = Tuple[float, float, float, float, int, str, bool]


class ComparisonSnapshot(BaseModel):
    snapshot_id: str
    scene_id: str
    frame_id: int
    revision: int
    timestamp: float
    bounds: Tuple[float, float, float, float]
    min_cell_size: float
    uniform: List[ComparisonRow]
    adaptive: List[ComparisonRow]
    metrics: FrameMetrics
    bytes_per_uniform_cell: int
    bytes_per_adaptive_cell: int


class LidarFrame(BaseModel):
    scene_id: str = "normal_road"
    revision: int = 0
    frame_id: int
    timestamp: float
    ego_state: EgoVehicleState
    points: List[List[float]]  # Subsampled points for WebGL: [[x, y, z, intensity], ...]
    adaptive_cells: List[AdaptiveCell]
    detections: List[Detection3D]
    tracks: List[TrackedObject]
    terrain_features: List[TerrainFeature]
    metrics: FrameMetrics
    detector_source: str = "Simulated DSVT Adapter (OpenPCDet Schema)"


class ControlMessage(BaseModel):
    request_id: Optional[str] = None
    action: Literal[
        "play",
        "pause",
        "step",
        "reset",
        "set_scene",
        "set_speed",
        "update_weights",
        "toggle_layer",
        "set_adaptive_enabled",
        "jury_demo_step"
    ]
    scene_id: Optional[str] = None
    speed: Optional[float] = None
    weights: Optional[Dict[str, float]] = None
    layer: Optional[str] = None
    enabled: Optional[bool] = None
    step_number: Optional[int] = None


class SceneInfo(BaseModel):
    id: str
    name: str
    description: str
    features: List[str]
    expected_behavior: str


class ImportanceWeights(BaseModel):
    w1_distance: float
    w2_elevation_var: float
    w3_semantic: float
    w4_object_prox: float
    w5_motion: float
    w6_terrain_complexity: float
    w7_uncertainty: float

