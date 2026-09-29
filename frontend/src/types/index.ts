/**
 * Type definitions for MI Sense Adaptive 2.5D LiDAR Mapping
 * Problem Statement: SIH 26053
 */

export type SemanticClass =
  "road" | "terrain" | "curb" | "pothole" | "slope" | "obstacle" | "unknown";

export type DynamicState = "STATIC" | "DYNAMIC" | "EMPTY";

export type ObjectClass =
  "vehicle" | "pedestrian" | "cyclist" | "static_obstacle";

export interface Detection3D {
  id: string;
  class_name: ObjectClass;
  confidence: number;
  position: [number, number, number]; // [x, y, z]
  dimensions: [number, number, number]; // [length, width, height], forward is +Y
  yaw: number; // radians
  velocity?: [number, number, number];
  detector_source: string;
}

export interface TrackedObject {
  track_id: number;
  class_name: string;
  position: [number, number, number];
  dimensions: [number, number, number]; // [length, width, height], forward is +Y
  yaw: number;
  velocity: [number, number];
  speed: number; // m/s
  heading: number; // deg
  dynamic_state: "DYNAMIC" | "STATIC";
  age: number;
  hits: number;
  confidence: number;
  history: [number, number][]; // [[x, y], ...]
}

export interface AdaptiveCell {
  cell_id: string;
  x: number;
  y: number;
  size: number; // 0.25, 0.5, 1.0, 2.0, 4.0
  level: number; // 0 (4m) to 4 (0.25m)
  elevation_mean: number;
  elevation_min: number;
  elevation_max: number;
  elevation_variance: number;
  slope_deg: number;
  occupancy: number;
  point_count: number;
  semantic_class: SemanticClass;
  drivable: boolean;
  object_class?: string | null;
  object_track_id?: number | null;
  dynamic_state: DynamicState;
  importance_score: number;
  confidence: number;
  split_reason: string;
  importance_factors: Record<string, number>;
}

export interface TerrainFeature {
  feature_type: "pothole" | "curb" | "slope" | "rough_patch";
  position: [number, number, number];
  bounds: [number, number, number, number];
  severity: number;
  description: string;
  confidence: number;
}

export interface FrameMetrics {
  frame_id: number;
  timestamp: number;
  fps: number;
  raw_points_count: number;
  processed_points_count: number;
  uniform_cells_count: number;
  adaptive_cells_count: number;
  cell_reduction_percent: number;
  uniform_memory_kb: number;
  adaptive_memory_kb: number;
  memory_saved_percent: number;
  fine_cells_count: number;
  medium_cells_count: number;
  coarse_cells_count: number;
  active_tracks_count: number;
  dynamic_objects_count: number;
  processing_time_ms: {
    preprocessing: number;
    detection: number;
    tracking: number;
    quadtree: number;
    total: number;
  };
}

export interface EgoVehicleState {
  position: [number, number, number];
  velocity: [number, number, number];
  speed: number; // km/h
  yaw: number;
  steering_angle: number;
}

export interface LidarFrame {
  scene_id: string;
  revision: number;
  frame_id: number;
  timestamp: number;
  ego_state: EgoVehicleState;
  points: [number, number, number, number][]; // [x, y, z, intensity]
  adaptive_cells: AdaptiveCell[];
  detections: Detection3D[];
  tracks: TrackedObject[];
  terrain_features: TerrainFeature[];
  metrics: FrameMetrics;
  detector_source: string;
}

export interface SceneInfo {
  id: string;
  name: string;
  description: string;
  features: string[];
  expected_behavior: string;
}

export interface LayerVisibility {
  surface: boolean;
  points: boolean;
  elevation: boolean;
  adaptiveGrid: boolean;
  semanticMap: boolean;
  detections: boolean;
  tracks: boolean;
  potholes: boolean;
  curbs: boolean;
  slopes: boolean;
}

export interface ImportanceWeights {
  w1_distance: number;
  w2_elevation_var: number;
  w3_semantic: number;
  w4_object_prox: number;
  w5_motion: number;
  w6_terrain_complexity: number;
  w7_uncertainty: number;
}

export type ViewMode = "adaptive" | "comparison" | "architecture" | "dossier";
export type MapPreset = "terrain" | "resolution" | "scan";
export type CameraPreset = "angled" | "top" | "focus";
export type ConnectionState =
  "connecting" | "connected" | "reconnecting" | "disconnected";

export interface SessionState {
  scene_id: string;
  revision: number;
  frame_id: number;
  is_running: boolean;
  playback_speed: number;
  adaptive_enabled: boolean;
  weights: ImportanceWeights;
  detector_source: string;
}

export interface SystemConfig {
  session: SessionState;
  grid_resolutions: number[];
  roi: [number, number, number, number];
  weights: ImportanceWeights;
  detector_source: string;
}

export interface ControlCommand {
  action:
    | "play"
    | "pause"
    | "step"
    | "reset"
    | "set_scene"
    | "set_speed"
    | "update_weights"
    | "set_adaptive_enabled";
  scene_id?: string;
  speed?: number;
  weights?: Partial<ImportanceWeights>;
  enabled?: boolean;
}

export interface CommandResult {
  status: "success" | "error";
  message?: string;
  state: SessionState;
}

export type ComparisonCell = [
  x: number,
  y: number,
  size: number,
  z: number,
  pointCount: number,
  semantic: string,
  drivable: boolean,
];
export interface ComparisonSnapshot {
  snapshot_id: string;
  scene_id: string;
  frame_id: number;
  revision: number;
  timestamp: number;
  bounds: [number, number, number, number];
  min_cell_size: number;
  uniform: ComparisonCell[];
  adaptive: ComparisonCell[];
  metrics: FrameMetrics;
  bytes_per_uniform_cell: number;
  bytes_per_adaptive_cell: number;
}

export interface JuryStep {
  step: number;
  title: string;
  scene_id: string;
  highlight: string;
  message: string;
  technical_note: string;
  recommended_layers: Partial<LayerVisibility>;
  camera_preset: "bev" | "perspective" | "close_up";
}
