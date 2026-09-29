"""
System configuration for MI Sense Adaptive 2.5D LiDAR Mapping.
Problem Statement: SIH 26053
"""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # System & Server
    PROJECT_NAME: str = "MI Sense - Adaptive 2.5D LiDAR Mapping"
    VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    SIMULATION_FPS: int = 15

    # LiDAR Specifications
    LIDAR_BEAMS: int = 64
    LIDAR_RANGE_MIN: float = 0.5
    LIDAR_RANGE_MAX: float = 50.0
    LIDAR_FOV_HORIZONTAL: float = 360.0  # degrees
    LIDAR_FOV_VERTICAL_MIN: float = -25.0  # degrees
    LIDAR_FOV_VERTICAL_MAX: float = 15.0  # degrees
    LIDAR_NOISE_STD: float = 0.02  # meters
    LIDAR_DROPOUT_RATE: float = 0.03

    # Region of Interest (ROI) in vehicle frame (meters)
    ROI_X_MIN: float = -32.0  # Left
    ROI_X_MAX: float = 32.0   # Right
    ROI_Y_MIN: float = -16.0  # Behind
    ROI_Y_MAX: float = 48.0   # Ahead
    ROI_Z_MIN: float = -2.5   # Below ground
    ROI_Z_MAX: float = 4.0    # Overhead

    # Quadtree Grid Resolutions (Square cell sizes in meters)
    # Level 0 (Coarsest) to Level 4 (Finest)
    GRID_RESOLUTIONS: List[float] = [4.0, 2.0, 1.0, 0.5, 0.25]
    MIN_CELL_SIZE: float = 0.25
    MAX_CELL_SIZE: float = 4.0
    ROOT_GRID_SIZE: float = 64.0  # 64m x 64m square bounds

    # Importance Scoring Weights (w1 to w7)
    # Importance Score = sum(wi * factor_i)
    WEIGHT_DISTANCE: float = Field(default=0.15, description="w1: Proximity to ego vehicle")
    WEIGHT_ELEVATION_VAR: float = Field(default=0.20, description="w2: Local height variance & roughness")
    WEIGHT_SEMANTIC: float = Field(default=0.20, description="w3: Semantic class priority (curb, road edge)")
    WEIGHT_OBJECT_PROX: float = Field(default=0.20, description="w4: Proximity to detected objects")
    WEIGHT_MOTION: float = Field(default=0.10, description="w5: Dynamic/moving object relevance")
    WEIGHT_TERRAIN_COMPLEXITY: float = Field(default=0.10, description="w6: Slopes, steps, non-planar terrain")
    WEIGHT_UNCERTAINTY: float = Field(default=0.05, description="w7: Low point count / measurement uncertainty")

    # Quadtree Subdivision Thresholds (Normalized 0.0 - 1.0)
    SPLIT_THRESHOLD_L0: float = 0.20  # 4.0m -> 2.0m
    SPLIT_THRESHOLD_L1: float = 0.40  # 2.0m -> 1.0m
    SPLIT_THRESHOLD_L2: float = 0.60  # 1.0m -> 0.5m
    SPLIT_THRESHOLD_L3: float = 0.75  # 0.5m -> 0.25m
    MERGE_HYSTERESIS: float = 0.10    # Merge threshold = Split threshold - hysteresis

    # Geometric Perception Thresholds
    POTHOLE_DEPTH_THRESHOLD: float = -0.08  # meters relative to local road reference
    POTHOLE_MIN_RADIUS: float = 0.25
    CURB_HEIGHT_THRESHOLD: float = 0.12    # meters abrupt step
    SLOPE_MILD_DEG: float = 5.0            # degrees
    SLOPE_STEEP_DEG: float = 12.0          # degrees

    # Tracking Configuration
    TRACK_MAX_AGE: int = 5                 # Frames to keep track without detection
    TRACK_MIN_HITS: int = 2                # Frames needed to confirm track
    DYNAMIC_SPEED_THRESHOLD: float = 0.5   # m/s to classify as dynamic
    DYNAMIC_DISPLACEMENT_THRESHOLD: float = 1.0 # meters total displacement
    HUNGARIAN_MAX_DISTANCE: float = 3.5    # Association gate in meters

    # Memory Calculation Constants
    BYTES_PER_UNIFORM_CELL: int = 48       # Structured 2.5D cell bytes
    BYTES_PER_ADAPTIVE_CELL: int = 56      # Includes hierarchy pointer / size overhead

    model_config = SettingsConfigDict(env_prefix="MI_SENSE_")


settings = Settings()
