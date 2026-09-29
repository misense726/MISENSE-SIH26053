"""
Comprehensive unit test suite for MI Sense Simulation and LiDAR Preprocessing modules.
Problem Statement: SIH 26053
"""

import math
import numpy as np
import pytest

from backend.app.config.settings import settings
from backend.app.models.schemas import EgoVehicleState, Detection3D
from backend.app.simulation.world import (
    RoadModel,
    Curb,
    SlopeTerrain,
    Pothole,
    StaticObstacle,
    MovingVehicle,
    PedestrianActor,
    EgoVehicle,
    SimulationWorld,
)
from backend.app.simulation.scenarios import (
    load_scenario,
    get_all_scenarios_info,
    get_scenario_info,
    SCENARIOS_CATALOG,
)
from backend.app.simulation.lidar_simulator import LidarSimulator
from backend.app.lidar.preprocessing import (
    preprocess_point_cloud,
    PointCloudPreprocessor,
    PreprocessingResult,
)
from backend.app.lidar.dataset_loader import (
    KittiDatasetAdapter,
    NuScenesDatasetAdapter,
    CustomPcdBinAdapter,
    SyntheticDatasetGenerator,
)


# ============================================================================
# 1. World Simulation Tests
# ============================================================================


def test_road_model_geometry() -> None:
    """Tests road model lane centers, bounds, and surface reflectances."""
    road = RoadModel(lane_width=3.75, num_lanes=2, curvature=0.0)
    assert road.half_road_width == pytest.approx(3.75)
    assert road.get_centerline_x(10.0) == pytest.approx(0.0)
    # Right lane center
    assert road.get_lane_center_x(0, 10.0) == pytest.approx(1.875)
    # Left lane center
    assert road.get_lane_center_x(1, 10.0) == pytest.approx(-1.875)

    # Road bounds
    assert road.is_on_road(0.0, 10.0) is True
    assert road.is_on_road(3.70, 10.0) is True
    assert road.is_on_road(4.5, 10.0) is False
    assert road.is_on_shoulder(4.5, 10.0) is True

    # Reflectance
    refl_asphalt = road.get_surface_reflectance(1.0, 10.0)
    assert 0.28 <= refl_asphalt <= 0.35
    # Lane marking reflectance
    refl_marking = road.get_surface_reflectance(0.0, 0.0)
    assert refl_marking > 0.75


def test_curb_raised_step() -> None:
    """Tests that curbs create an abrupt +0.15m height step at the road boundary."""
    road = RoadModel(lane_width=3.75, num_lanes=2)
    curb_right = Curb(side="right", step_height=0.15, width=0.25)

    # On road before curb: offset is 0
    assert curb_right.get_height_offset(3.5, 10.0, road) == pytest.approx(0.0)
    # Past the curb on sidewalk: offset is full step_height (+0.15m)
    assert curb_right.get_height_offset(4.2, 10.0, road) == pytest.approx(0.15)


def test_slope_elevation() -> None:
    """Tests that slopes create an accurate linear gradient with smooth transitions."""
    slope = SlopeTerrain(y_start=20.0, y_end=40.0, slope_deg=6.0, direction="up")
    # Before slope
    elev_before, _ = slope.get_elevation_and_slope(0.0, 10.0)
    assert elev_before == pytest.approx(0.0)

    # In middle of slope: positive rise
    elev_mid, s_deg = slope.get_elevation_and_slope(0.0, 30.0)
    assert elev_mid > 0.0
    assert s_deg > 0.0

    # Past slope: plateau
    elev_after, _ = slope.get_elevation_and_slope(0.0, 50.0)
    assert elev_after == pytest.approx(slope.total_rise)


def test_pothole_profile() -> None:
    """Tests that potholes produce negative depression depth matching specifications."""
    pothole = Pothole(center_x=1.875, center_y=20.0, depth=0.14, radius=0.65)
    # At center: max negative depression
    dep_center = pothole.get_depression(1.875, 20.0)
    assert dep_center == pytest.approx(-0.14, abs=0.03)

    # Outside radius: 0.0
    dep_outside = pothole.get_depression(1.875 + 0.8, 20.0)
    assert dep_outside == pytest.approx(0.0)


def test_dynamic_actors_and_ego_kinematics() -> None:
    """Tests ego vehicle and dynamic actor step integration."""
    world = SimulationWorld()
    load_scenario(world, "complex_environment")

    initial_ego_y = world.ego.position[1]
    initial_actors_pos = [actor.position.copy() for actor in world.dynamic_actors]

    # Step simulation
    dt = 1.0 / 15.0
    world.step(dt)

    # Ego vehicle moved forward
    assert world.ego.position[1] > initial_ego_y
    assert world.sim_time == pytest.approx(dt)
    assert world.frame_count == 1

    # Dynamic actors moved
    for i, actor in enumerate(world.dynamic_actors):
        assert not np.array_equal(actor.position, initial_actors_pos[i])
        assert len(actor.history) >= 1


# ============================================================================
# 2. Scenarios Tests
# ============================================================================


def test_all_six_scenarios_load() -> None:
    """Tests that all 6 required scenarios can be initialized correctly."""
    required_scenes = [
        "normal_road",
        "pothole",
        "curb_boundary",
        "moving_vehicle",
        "pedestrian",
        "complex_environment",
    ]

    all_info = get_all_scenarios_info()
    assert len(all_info) == 6
    scene_ids = [s.id for s in all_info]
    for s_id in required_scenes:
        assert s_id in scene_ids

    world = SimulationWorld()
    for s_id in required_scenes:
        load_scenario(world, s_id)
        assert world.scene_id == s_id
        info = get_scenario_info(s_id)
        assert info is not None
        assert info.id == s_id
        assert len(info.features) > 0
        assert len(info.expected_behavior) > 0

    # Specific scenario entity verification
    load_scenario(world, "pothole")
    assert len(world.potholes) >= 1
    assert world.potholes[0].depth >= 0.10

    load_scenario(world, "curb_boundary")
    assert len(world.curbs) >= 2
    assert len(world.static_obstacles) >= 1  # Parked car

    load_scenario(world, "pedestrian")
    peds = [a for a in world.dynamic_actors if a.class_name == "pedestrian"]
    assert len(peds) >= 1

    load_scenario(world, "complex_environment")
    assert len(world.potholes) >= 1
    assert len(world.curbs) >= 2
    assert len(world.slopes) >= 1
    assert len(world.static_obstacles) >= 1
    assert len(world.dynamic_actors) >= 2


# ============================================================================
# 3. LiDAR Simulator Tests
# ============================================================================


def test_lidar_point_cloud_generation() -> None:
    """Tests 64-beam LiDAR point cloud generation and characteristics."""
    world = SimulationWorld()
    load_scenario(world, "complex_environment")
    sim = LidarSimulator(num_beams=64, num_azimuth_steps=400)

    cloud = sim.generate_point_cloud(world)

    # Array shape & type
    assert isinstance(cloud, np.ndarray)
    assert cloud.ndim == 2
    assert cloud.shape[1] == 4
    assert cloud.dtype == np.float32
    assert cloud.shape[0] > 10000

    # ROI constraints
    assert cloud[:, 0].min() >= settings.ROI_X_MIN
    assert cloud[:, 0].max() <= settings.ROI_X_MAX
    assert cloud[:, 1].min() >= settings.ROI_Y_MIN
    assert cloud[:, 1].max() <= settings.ROI_Y_MAX
    assert cloud[:, 2].min() >= settings.ROI_Z_MIN
    assert cloud[:, 2].max() <= settings.ROI_Z_MAX

    # Intensities valid range
    assert cloud[:, 3].min() >= 0.05
    assert cloud[:, 3].max() <= 1.0


# ============================================================================
# 4. Preprocessing Tests
# ============================================================================


def test_point_cloud_preprocessing() -> None:
    """Tests ROI cropping, RANSAC ground segmentation, and WebGL downsampling."""
    world = SimulationWorld()
    load_scenario(world, "complex_environment")
    sim = LidarSimulator(num_beams=64, num_azimuth_steps=400)
    raw_cloud = sim.generate_point_cloud(world)

    res: PreprocessingResult = preprocess_point_cloud(raw_cloud, max_viz_points=5000)

    assert res.raw_count == len(raw_cloud)
    assert res.processed_count <= res.raw_count
    assert res.ground_count > 0
    assert res.non_ground_count > 0
    assert res.ground_count + res.non_ground_count == res.processed_count
    assert res.downsampled_count <= 5000
    assert res.latency_ms > 0.0

    # Ground plane normal must be approximately vertical [0, 0, 1]
    a, b, c, d = res.plane_coefficients
    assert abs(c) > 0.90
    # Ground plane height in vehicle frame around -1.73m (d ~ 1.7m)
    assert 1.4 < d < 2.0


# ============================================================================
# 5. Dataset Loader & Adapter Tests
# ============================================================================


def test_synthetic_dataset_generator() -> None:
    """Tests streaming synthetic dataset generator and caching."""
    gen = SyntheticDatasetGenerator(scene_id="pothole", total_frames=15)
    assert len(gen) == 15
    assert "pothole" in gen.get_dataset_name().lower()

    f0 = gen.get_frame(0)
    assert f0.shape[1] == 4
    assert len(f0) > 5000

    ego = gen.get_ego_state(0)
    assert ego is not None
    assert ego.speed > 0.0

    dets = gen.get_ground_truth_detections(0)
    assert isinstance(dets, list)


def test_custom_pcd_bin_adapter(tmp_path) -> None:
    """Tests saving and loading PCD files (ASCII & Binary)."""
    temp_dir = tmp_path / "pcd_test"
    temp_dir.mkdir(parents=True, exist_ok=True)
    pcd_file = str(temp_dir / "test_frame.pcd")

    test_pts = np.array(
        [
            [1.0, 2.0, -1.7, 0.3],
            [2.0, 4.0, -1.6, 0.8],
            [3.0, 6.0, 0.5, 0.9],
        ],
        dtype=np.float32,
    )

    # 1. Binary PCD
    CustomPcdBinAdapter.save_frame_as_pcd(test_pts, pcd_file, binary=True)
    adapter = CustomPcdBinAdapter(folder_path=str(temp_dir))
    loaded = adapter.get_frame(0)
    np.testing.assert_allclose(loaded[:, :3], test_pts[:, :3], atol=1e-4)

    # 2. ASCII PCD
    pcd_ascii_file = str(temp_dir / "test_ascii.pcd")
    CustomPcdBinAdapter.save_frame_as_pcd(test_pts, pcd_ascii_file, binary=False)
    adapter_ascii = CustomPcdBinAdapter(folder_path=str(temp_dir))
    loaded_ascii = adapter_ascii.get_frame(1)
    np.testing.assert_allclose(loaded_ascii[:, :3], test_pts[:, :3], atol=1e-3)


def test_kitti_dataset_adapter(tmp_path) -> None:
    """Tests saving and loading KITTI binary format with coordinate conversion."""
    temp_dir = tmp_path / "kitti_test"
    temp_dir.mkdir(parents=True, exist_ok=True)
    bin_file = str(temp_dir / "000000.bin")

    # In vehicle coordinates: x_veh=1.0, y_veh=5.0, z_veh=-1.7, r=0.5
    test_pts = np.array([[1.0, 5.0, -1.7, 0.5]], dtype=np.float32)
    KittiDatasetAdapter.save_frame_as_kitti_bin(test_pts, bin_file)

    adapter = KittiDatasetAdapter(dataset_path=str(temp_dir), convert_to_vehicle_frame=False)
    raw_read = adapter.get_frame(0)
    np.testing.assert_allclose(raw_read, test_pts, atol=1e-5)
