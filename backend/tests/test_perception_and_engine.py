"""
Comprehensive unit and integration test suite for Perception Models,
Simulation Engine, and REST/WebSocket API endpoints.
Problem Statement: SIH 26053
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.app.config.settings import settings
from backend.app.models.schemas import Detection3D, TrackedObject, LidarFrame, ControlMessage
from backend.app.perception.detector import SimulatedDSVTAdapter, OpenPCDetDSVTAdapter
from backend.app.perception.terrain_segmenter import (
    SimulationTerrainSegmenter,
    NeuralTerrainSegmenter,
)
from backend.app.simulation.engine import SimulationEngine
from backend.app.simulation.world import SimulationWorld
from backend.app.main import app


# 1. Perception Detector Tests
def test_simulated_dsvt_adapter_output_schema():
    """Verify SimulatedDSVTAdapter produces detections strictly conforming to OpenPCDet schema."""
    adapter = SimulatedDSVTAdapter(noise_std=0.01, miss_rate=0.0)

    # Mock ground truth actors
    actors = [
        {
            "id": "veh_1",
            "class_name": "vehicle",
            "position": [2.0, 15.0, 0.75],
            "dimensions": [4.6, 2.0, 1.6],
            "yaw": 0.05,
            "velocity": [0.0, 10.0, 0.0],
        },
        {
            "id": "ped_1",
            "class_name": "pedestrian",
            "position": [-3.0, 8.0, 0.85],
            "dimensions": [0.6, 0.6, 1.7],
            "yaw": 1.57,
            "velocity": [1.2, 0.0, 0.0],
        },
    ]

    dummy_points = np.zeros((100, 4), dtype=np.float32)
    detections = adapter.detect(dummy_points, ground_truth_actors=actors)

    assert len(detections) == 2
    veh_det = next(d for d in detections if d.class_name == "vehicle")
    assert abs(veh_det.position[0] - 2.0) <= 0.05
    assert abs(veh_det.position[1] - 15.0) <= 0.05
    assert 0.85 <= veh_det.confidence <= 0.99
    assert veh_det.detector_source == "Simulated DSVT Adapter (OpenPCDet Schema)"
    assert veh_det.velocity is not None
    assert len(veh_det.dimensions) == 3


def test_openpcdet_adapter_fallback():
    """Verify OpenPCDetDSVTAdapter detects environment status and gracefully falls back."""
    adapter = OpenPCDetDSVTAdapter()

    # In CPU/test environment, should fall back cleanly
    assert adapter.is_fallback is True
    assert "Simulated Fallback" in adapter.status_message or "OpenPCDet" in adapter.detector_source

    actors = [
        {
            "id": "cyclist_1",
            "class_name": "cyclist",
            "position": [1.0, 12.0, 0.8],
            "dimensions": [1.8, 0.7, 1.4],
            "yaw": 0.0,
            "velocity": [0.0, 4.0, 0.0],
        }
    ]

    dummy_points = np.zeros((50, 4), dtype=np.float32)
    dets = adapter.detect(dummy_points, ground_truth_actors=actors)
    assert len(dets) == 1
    assert dets[0].class_name == "cyclist"


# 2. Terrain Segmenter Tests
def test_terrain_segmenter_classes_and_drivability():
    """Verify SimulationTerrainSegmenter assigns correct classes and drivability booleans."""
    segmenter = SimulationTerrainSegmenter(road_half_width=7.0)

    cells = [
        # Road cell
        {"cell_id": "c_road", "x": 0.0, "y": 10.0, "size": 1.0, "point_count": 20, "elevation_mean": 0.0, "elevation_variance": 0.002, "slope_deg": 1.0},
        # Pothole cell
        {"cell_id": "c_pothole", "x": 1.0, "y": 12.0, "size": 0.5, "point_count": 15, "elevation_mean": -0.14, "elevation_variance": 0.02, "slope_deg": 2.0},
        # Curb cell
        {"cell_id": "c_curb", "x": 7.2, "y": 10.0, "size": 0.5, "point_count": 18, "elevation_mean": 0.15, "elevation_variance": 0.03, "slope_deg": 4.0},
        # Steep slope
        {"cell_id": "c_slope", "x": 0.0, "y": 25.0, "size": 1.0, "point_count": 25, "elevation_mean": 1.5, "elevation_variance": 0.01, "slope_deg": 14.0},
        # Off-road terrain
        {"cell_id": "c_terrain", "x": 12.0, "y": 10.0, "size": 2.0, "point_count": 10, "elevation_mean": 0.1, "elevation_variance": 0.02, "slope_deg": 2.0},
    ]

    points = np.zeros((100, 4), dtype=np.float32)
    results = segmenter.segment(points, cells)

    assert results["c_road"]["semantic_class"] == "road"
    assert results["c_road"]["drivable"] is True

    assert results["c_pothole"]["semantic_class"] == "pothole"
    assert results["c_pothole"]["drivable"] is False

    assert results["c_curb"]["semantic_class"] == "curb"
    assert results["c_curb"]["drivable"] is False

    assert results["c_slope"]["semantic_class"] == "slope"
    assert results["c_slope"]["drivable"] is False  # Steep slope (>12 deg) is non-drivable

    assert results["c_terrain"]["semantic_class"] == "terrain"
    assert results["c_terrain"]["drivable"] is False


def test_neural_terrain_segmenter_fallback():
    """Verify NeuralTerrainSegmenter falls back gracefully."""
    neural_seg = NeuralTerrainSegmenter()
    cells = [
        {"cell_id": "c1", "x": 0.0, "y": 5.0, "size": 1.0, "point_count": 20, "elevation_mean": 0.0, "elevation_variance": 0.001, "slope_deg": 1.0},
    ]
    res = neural_seg.segment(np.zeros((10, 4)), cells)
    assert "c1" in res
    assert res["c1"]["semantic_class"] == "road"


# 3. Master Simulation Engine Tests
def test_simulation_engine_cycle_and_frame_generation():
    """Verify SimulationEngine runs full cycle and generates valid LidarFrame."""
    engine = SimulationEngine(initial_scene_id="complex_environment")

    frame = engine.step_frame()

    assert isinstance(frame, LidarFrame)
    assert frame.frame_id == 0
    assert len(frame.points) > 0
    assert len(frame.adaptive_cells) > 0
    assert frame.metrics.adaptive_cells_count == len(frame.adaptive_cells)
    assert frame.metrics.cell_reduction_percent > 90.0
    assert frame.metrics.memory_saved_percent > 90.0
    assert "preprocessing" in frame.metrics.processing_time_ms
    assert "detection" in frame.metrics.processing_time_ms
    assert "tracking" in frame.metrics.processing_time_ms
    assert "quadtree" in frame.metrics.processing_time_ms


def test_dynamic_point_exclusion_in_engine():
    """Verify dynamic vehicle points are excluded from the static elevation map in the engine."""
    engine = SimulationEngine(initial_scene_id="moving_vehicle")

    # Run 5 frames so Kalman tracker confirms dynamic track
    for _ in range(5):
        frame = engine.step_frame()

    # Verify at least one dynamic track exists
    dyn_tracks = [t for t in frame.tracks if t.dynamic_state == "DYNAMIC"]
    assert len(dyn_tracks) >= 1
    car_track = dyn_tracks[0]

    # Verify the cell directly under the moving vehicle does NOT have its static elevation smeared
    car_x, car_y = car_track.position[0], car_track.position[1]
    matching_cells = [
        c for c in frame.adaptive_cells
        if abs(c.x - car_x) <= (c.size * 0.5) and abs(c.y - car_y) <= (c.size * 0.5)
    ]

    # The cells should represent road surface, not car height (> 0.5m)
    for c in matching_cells:
        assert c.elevation_mean < 0.35, f"Dynamic vehicle points contaminated static elevation map! mean={c.elevation_mean}"


def test_simulation_engine_command_handling():
    """Verify control commands: play, pause, step, reset, set_scene, set_speed, update_weights."""
    engine = SimulationEngine()

    # Pause
    r_pause = engine.handle_command(ControlMessage(action="pause"))
    assert r_pause["status"] == "success"
    assert engine.is_running is False

    # Play
    r_play = engine.handle_command(ControlMessage(action="play"))
    assert r_play["status"] == "success"
    assert engine.is_running is True

    # Set Scene
    r_scene = engine.handle_command(ControlMessage(action="set_scene", scene_id="pothole"))
    assert r_scene["status"] == "success"
    assert engine.current_scene_id == "pothole"

    # Set Speed
    r_speed = engine.handle_command(ControlMessage(action="set_speed", speed=2.0))
    assert r_speed["status"] == "success"
    assert engine.playback_speed == 2.0

    # Update Weights
    r_w = engine.handle_command(ControlMessage(action="update_weights", weights={"w2_elevation_var": 0.35}))
    assert r_w["status"] == "success"
    assert engine.weights_config.w2_elevation_var == 0.35

    # Jury Demo Step
    r_jury = engine.handle_command(ControlMessage(action="jury_demo_step", step_number=2))
    assert r_jury["status"] == "success"
    assert r_jury["scene"] == "pothole"


# 4. REST API Endpoint Tests
def test_api_endpoints():
    """Verify all REST API endpoints via FastAPI TestClient."""
    client = TestClient(app)

    # GET /
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert r_root.json()["problem_statement"] == "SIH 26053"

    # GET /api/health
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    health_data = r_health.json()
    assert health_data["status"] == "healthy"
    assert "uptime" in health_data
    assert "fps" in health_data

    # GET /api/scenes
    r_scenes = client.get("/api/scenes")
    assert r_scenes.status_code == 200
    scenes = r_scenes.json()
    assert len(scenes) == 6
    scene_ids = {s["id"] for s in scenes}
    assert "normal_road" in scene_ids
    assert "pothole" in scene_ids
    assert "curb_boundary" in scene_ids
    assert "moving_vehicle" in scene_ids
    assert "pedestrian" in scene_ids
    assert "complex_environment" in scene_ids

    # GET /api/config
    r_config = client.get("/api/config")
    assert r_config.status_code == 200
    cfg = r_config.json()
    assert "weights" in cfg
    assert "grid_resolutions" in cfg
    assert cfg["grid_resolutions"] == [4.0, 2.0, 1.0, 0.5, 0.25]

    # POST /api/config/weights
    r_post_weights = client.post("/api/config/weights", json={"w1_distance": 0.25, "w3_semantic": 0.30})
    assert r_post_weights.status_code == 200
    assert r_post_weights.json()["updated_weights"]["w1_distance"] == 0.25

    # GET /api/architecture
    r_arch = client.get("/api/architecture")
    assert r_arch.status_code == 200
    arch = r_arch.json()
    assert len(arch["nodes"]) >= 8
    assert len(arch["links"]) >= 8

    # GET /api/explain
    r_exp = client.get("/api/explain")
    assert r_exp.status_code == 200
    exp = r_exp.json()
    assert len(exp["factors"]) == 7
    assert "dynamic_point_exclusion" in exp

    # GET /api/jury-demo/steps
    r_jury = client.get("/api/jury-demo/steps")
    assert r_jury.status_code == 200
    steps = r_jury.json()
    assert len(steps) == 4
    assert steps[0]["step"] == 1
    assert steps[1]["scene_id"] == "pothole"
    assert steps[3]["step"] == 4
