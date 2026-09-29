"""
Simulation Scenarios for MI Sense Adaptive 2.5D Mapping.
Problem Statement: SIH 26053

Defines 6 diverse evaluation scenarios:
1. normal_road: Flat, smooth asphalt demonstrating massive memory savings.
2. pothole: Sharp negative depressions triggering localized 0.25m leaf cells.
3. curb_boundary: Stepped curbs (+0.15m) delineating drivable road edges.
4. moving_vehicle: Dynamic actor moving at speed; points excluded from static elevation map.
5. pedestrian: Vulnerable road users crossing navigation corridor.
6. complex_environment: Combined benchmark with potholes, curbs, parked & moving actors.
"""

from dataclasses import dataclass, field
import math
from typing import List, Dict, Any, Optional

from backend.app.models.schemas import EgoVehicleState, SceneInfo


class SimulatedActor:
    """Simulated dynamic or static actor in the perception world."""

    def __init__(
        self,
        actor_id: str,
        class_name: str,
        position: List[float],
        dimensions: List[float],
        yaw: float = 0.0,
        velocity: Optional[List[float]] = None,
        trajectory_type: str = "linear",  # linear, patrol, static
        bounds: Optional[List[float]] = None,
    ):
        self.id = actor_id
        self.class_name = class_name
        self.initial_pos = list(position)
        self.position = list(position)
        self.dimensions = list(dimensions)
        self.yaw = yaw
        self.initial_velocity = list(velocity) if velocity else [0.0, 0.0, 0.0]
        self.velocity = list(self.initial_velocity)
        self.trajectory_type = trajectory_type
        self.bounds = bounds  # [min_y, max_y] or [min_x, max_x]
        self.time_alive = 0.0

    def step(self, dt: float) -> None:
        """Advance actor along simulated path."""
        self.time_alive += dt

        if self.trajectory_type == "static":
            return

        if self.trajectory_type == "linear":
            # Linear motion
            self.position[0] += self.velocity[0] * dt
            self.position[1] += self.velocity[1] * dt
            self.position[2] += self.velocity[2] * dt

            # Wrap around boundaries to keep demo continuous
            if self.bounds and len(self.bounds) >= 2:
                if self.velocity[1] > 0 and self.position[1] > self.bounds[1]:
                    self.position[1] = self.bounds[0]
                elif self.velocity[1] < 0 and self.position[1] < self.bounds[0]:
                    self.position[1] = self.bounds[1]
                elif self.velocity[0] > 0 and self.position[0] > self.bounds[1]:
                    self.position[0] = self.bounds[0]
                elif self.velocity[0] < 0 and self.position[0] < self.bounds[0]:
                    self.position[0] = self.bounds[1]

        elif self.trajectory_type == "patrol":
            # Oscillate back and forth
            self.position[0] += self.velocity[0] * dt
            self.position[1] += self.velocity[1] * dt
            if self.bounds and len(self.bounds) >= 2:
                if self.position[0] > self.bounds[1]:
                    self.position[0] = self.bounds[1]
                    self.velocity[0] = -abs(self.velocity[0])
                    self.yaw = math.pi
                elif self.position[0] < self.bounds[0]:
                    self.position[0] = self.bounds[0]
                    self.velocity[0] = abs(self.velocity[0])
                    self.yaw = 0.0

    def reset(self) -> None:
        """Reset actor to starting position and velocity."""
        self.position = list(self.initial_pos)
        self.velocity = list(self.initial_velocity)
        self.time_alive = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "class_name": self.class_name,
            "position": self.position,
            "dimensions": self.dimensions,
            "yaw": self.yaw,
            "velocity": self.velocity,
        }


class Scenario:
    """Container for scenario environment, ego trajectory, and actors."""

    def __init__(
        self,
        scene_id: str,
        name: str,
        description: str,
        features: List[str],
        expected_behavior: str,
        actors: List[SimulatedActor],
        hazards: Optional[List[Dict[str, Any]]] = None,
        ego_speed: float = 30.0,  # km/h
    ):
        self.id = scene_id
        self.name = name
        self.description = description
        self.features = features
        self.expected_behavior = expected_behavior
        self.actors = actors
        self.hazards = hazards or []
        self.ego_speed = ego_speed

    def to_info(self) -> SceneInfo:
        return SceneInfo(
            id=self.id,
            name=self.name,
            description=self.description,
            features=self.features,
            expected_behavior=self.expected_behavior,
        )


def build_all_scenarios() -> Dict[str, Scenario]:
    """Instantiate the 6 standard benchmark scenarios."""
    scenarios = {}

    # 1. Normal Road
    scenarios["normal_road"] = Scenario(
        scene_id="normal_road",
        name="Normal Highway Road",
        description="Uniform asphalt highway with minimal roughness. Demonstrates >95% cell reduction.",
        features=["Smooth asphalt surface", "Uniform elevation", "Sparse shoulder clutter"],
        expected_behavior="Quadtree maintains coarse 4.0m and 2.0m cells everywhere, maximizing memory compression.",
        actors=[
            SimulatedActor(
                actor_id="parked_1",
                class_name="vehicle",
                position=[8.5, 25.0, 0.75],
                dimensions=[4.5, 1.9, 1.5],
                yaw=0.0,
                trajectory_type="static",
            ),
        ],
        hazards=[],
        ego_speed=40.0,
    )

    # 2. Pothole Hazard
    scenarios["pothole"] = Scenario(
        scene_id="pothole",
        name="Road Potholes & Depressions",
        description="Two sharp potholes (-0.12m and -0.15m deep) directly in vehicle lane.",
        features=["Sharp negative depth step", "Local roughness", "High semantic priority"],
        expected_behavior="Quadtree recursively splits down to 0.25m leaf cells over potholes while keeping flat road coarse.",
        actors=[],
        hazards=[
            {
                "feature_type": "pothole",
                "position": [-1.2, 14.0, -0.12],
                "bounds": [-1.8, -0.6, 13.4, 14.6],
                "radius": 0.6,
                "severity": -0.12,
                "description": "Hazard Pothole A (-12cm deep)",
                "confidence": 0.96,
            },
            {
                "feature_type": "pothole",
                "position": [1.5, 26.0, -0.15],
                "bounds": [0.8, 2.2, 25.3, 26.7],
                "radius": 0.7,
                "severity": -0.15,
                "description": "Severe Pothole B (-15cm deep)",
                "confidence": 0.98,
            },
        ],
        ego_speed=25.0,
    )

    # 3. Curb Boundary
    scenarios["curb_boundary"] = Scenario(
        scene_id="curb_boundary",
        name="Curb & Sidewalk Delineation",
        description="Raised concrete curbs (+0.15m) separating road from sidewalks.",
        features=["Abrupt 15cm height step", "Road edge boundary", "Off-road transition"],
        expected_behavior="Fine 0.5m-0.25m cells align along curb lines, preserving sharp boundary for path planners.",
        actors=[
            SimulatedActor(
                actor_id="barrier_1",
                class_name="static_obstacle",
                position=[-7.5, 18.0, 0.5],
                dimensions=[1.2, 0.4, 0.8],
                yaw=0.0,
                trajectory_type="static",
            ),
        ],
        hazards=[
            {
                "feature_type": "curb",
                "position": [-7.0, 16.0, 0.15],
                "bounds": [-7.5, -6.5, -16.0, 48.0],
                "severity": 0.15,
                "description": "Left Curb Line (+15cm)",
                "confidence": 0.95,
            },
            {
                "feature_type": "curb",
                "position": [7.0, 16.0, 0.15],
                "bounds": [6.5, 7.5, -16.0, 48.0],
                "severity": 0.15,
                "description": "Right Curb Line (+15cm)",
                "confidence": 0.95,
            },
        ],
        ego_speed=30.0,
    )

    # 4. Moving Vehicle
    scenarios["moving_vehicle"] = Scenario(
        scene_id="moving_vehicle",
        name="Overtaking Moving Vehicle",
        description="Vehicle traveling at 12 m/s ahead. Demonstrates dynamic tracking and point filtering.",
        features=["Moving dynamic vehicle", "Kalman velocity estimation", "Static map exclusion"],
        expected_behavior="Tracker flags vehicle as DYNAMIC; its points are filtered from the elevation map to prevent ghost trails.",
        actors=[
            SimulatedActor(
                actor_id="car_ahead",
                class_name="vehicle",
                position=[2.0, 10.0, 0.75],
                dimensions=[4.6, 2.0, 1.5],
                yaw=0.0,
                velocity=[0.0, 8.5, 0.0],
                trajectory_type="linear",
                bounds=[-10.0, 45.0],
            ),
        ],
        hazards=[],
        ego_speed=25.0,
    )

    # 5. Crossing Pedestrian
    scenarios["pedestrian"] = Scenario(
        scene_id="pedestrian",
        name="Crossing Pedestrian",
        description="Pedestrians walking across the roadway. Dynamic motion relevance scoring in action.",
        features=["Pedestrian walking path", "Dynamic track trajectory", "High navigation priority"],
        expected_behavior="Pedestrians tracked with Kalman filter; fine grid resolves walking corridors.",
        actors=[
            SimulatedActor(
                actor_id="ped_cross",
                class_name="pedestrian",
                position=[-4.5, 18.0, 0.85],
                dimensions=[0.6, 0.6, 1.7],
                yaw=0.0,
                velocity=[1.2, 0.0, 0.0],
                trajectory_type="patrol",
                bounds=[-5.0, 5.0],
            ),
            SimulatedActor(
                actor_id="ped_sidewalk",
                class_name="pedestrian",
                position=[8.0, 12.0, 0.85],
                dimensions=[0.6, 0.6, 1.7],
                yaw=math.pi * 0.5,
                velocity=[0.0, 1.0, 0.0],
                trajectory_type="linear",
                bounds=[-5.0, 35.0],
            ),
        ],
        hazards=[],
        ego_speed=20.0,
    )

    # 6. Complex Environment
    scenarios["complex_environment"] = Scenario(
        scene_id="complex_environment",
        name="Complex Urban Intersection",
        description="Multi-hazard environment with potholes, curbs, parked cars, moving traffic, and pedestrians.",
        features=["Potholes + Curbs", "Moving & Parked Cars", "Crossing Pedestrian", "Full Adaptivity"],
        expected_behavior="Simultaneously demonstrates high compression on road, fine cells on hazards, and dynamic point filtering.",
        actors=[
            SimulatedActor(
                actor_id="moving_sedan",
                class_name="vehicle",
                position=[2.2, 12.0, 0.75],
                dimensions=[4.5, 1.9, 1.5],
                yaw=0.0,
                velocity=[0.0, 7.0, 0.0],
                trajectory_type="linear",
                bounds=[-5.0, 44.0],
            ),
            SimulatedActor(
                actor_id="parked_van",
                class_name="vehicle",
                position=[-8.2, 15.0, 1.0],
                dimensions=[5.2, 2.1, 2.0],
                yaw=0.0,
                trajectory_type="static",
            ),
            SimulatedActor(
                actor_id="walker_1",
                class_name="pedestrian",
                position=[-3.0, 24.0, 0.85],
                dimensions=[0.6, 0.6, 1.7],
                yaw=0.0,
                velocity=[1.1, 0.0, 0.0],
                trajectory_type="patrol",
                bounds=[-4.0, 4.0],
            ),
        ],
        hazards=[
            {
                "feature_type": "pothole",
                "position": [-1.0, 19.0, -0.14],
                "bounds": [-1.6, -0.4, 18.4, 19.6],
                "radius": 0.6,
                "severity": -0.14,
                "description": "Critical Pothole (-14cm)",
                "confidence": 0.97,
            },
            {
                "feature_type": "curb",
                "position": [-7.0, 16.0, 0.15],
                "bounds": [-7.5, -6.5, -16.0, 48.0],
                "severity": 0.15,
                "description": "Left Curb Line",
                "confidence": 0.94,
            },
            {
                "feature_type": "curb",
                "position": [7.0, 16.0, 0.15],
                "bounds": [6.5, 7.5, -16.0, 48.0],
                "severity": 0.15,
                "description": "Right Curb Line",
                "confidence": 0.94,
            },
        ],
        ego_speed=30.0,
    )

    return scenarios


# Cached global scenario dictionary
SCENARIOS_REGISTRY: Dict[str, Scenario] = build_all_scenarios()
SCENARIOS_CATALOG: Dict[str, Scenario] = SCENARIOS_REGISTRY


def get_all_scenarios_info() -> List[SceneInfo]:
    """Retrieve metadata for all 6 scenarios matching the SceneInfo schema."""
    return [sc.to_info() for sc in SCENARIOS_REGISTRY.values()]


def get_scenario_info(scene_id: str) -> Optional[SceneInfo]:
    """Retrieve metadata for a specific scenario by id."""
    sc = SCENARIOS_REGISTRY.get(scene_id)
    return sc.to_info() if sc else None


def load_scenario(arg1: Any, arg2: Optional[str] = None) -> Any:
    """
    Load scenario by ID or configure world object.
    Supports both load_scenario(scene_id) and load_scenario(world, scene_id).
    """
    if arg2 is not None:
        world, scene_id = arg1, arg2
        if hasattr(world, "set_scenario"):
            world.set_scenario(scene_id)
        elif hasattr(world, "current_scene_id"):
            world.current_scene_id = scene_id
        return world
    else:
        scene_id = str(arg1)
        return SCENARIOS_REGISTRY.get(scene_id, SCENARIOS_REGISTRY["normal_road"])


def switch_scene(arg1: Any, arg2: Optional[str] = None) -> Any:
    """
    Switch active scenario and reset actor positions.
    Supports both switch_scene(scene_id) and switch_scene(world, scene_id).
    """
    if arg2 is not None:
        return load_scenario(arg1, arg2)
    else:
        sc = load_scenario(arg1)
        for a in sc.actors:
            a.reset()
        return sc


