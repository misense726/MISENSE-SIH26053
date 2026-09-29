"""
Simulated driving world model for MI Sense Adaptive 2.5D LiDAR Mapping.
Problem Statement: SIH 26053

Defines:
- Ego vehicle state: position, velocity, yaw, steering angle, kinematic bicycle model.
- Static elements: road (with lanes, curvature), roadside terrain/shoulders,
  curbs (raised step ~15cm), slopes (mild ~5-8 deg, steep ~12-15 deg),
  potholes (depression ~10-15cm deep, radius ~0.4-0.8m), static obstacles (traffic barrels, barriers, parked car).
- Dynamic elements: moving vehicles (speeds 20-50 km/h, lane following or overtaking),
  pedestrians (walking crossing road or on shoulder at 1-1.5 m/s).
- Step function `world.step(dt)` that advances ego vehicle and all dynamic actors smoothly.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
import math
import numpy as np

from backend.app.config.settings import settings
from backend.app.models.schemas import (
    EgoVehicleState,
    Detection3D,
    TerrainFeature,
)


class RoadModel:
    """
    Parametric road model with lanes, shoulders, and curvature.
    World coordinates: Y is longitudinal (forward), X is lateral (right/left).
    Baseline road surface is at z = 0.0m.
    """

    def __init__(
        self,
        lane_width: float = 3.75,
        num_lanes: int = 2,
        curvature: float = 0.0,
        shoulder_width: float = 2.5,
        half_width: Optional[float] = None,
    ) -> None:
        if half_width is not None:
            self.half_road_width = half_width
            self.num_lanes = num_lanes
            self.lane_width = (2.0 * half_width) / max(1, num_lanes)
        else:
            self.lane_width = lane_width
            self.num_lanes = num_lanes
            self.half_road_width = (num_lanes * lane_width) / 2.0  # 3.75m for 2 lanes

        self.curvature = curvature  # Curvature coefficient (0.0 for straight)
        self.shoulder_width = shoulder_width
        self.total_half_width = self.half_road_width + shoulder_width

    def get_centerline_x(self, y: float) -> float:
        """Returns the lateral center X of the road at longitudinal position y."""
        if abs(self.curvature) < 1e-6:
            return 0.0
        return self.curvature * math.sin(0.04 * y) * 20.0

    def get_lane_center_x(self, lane_index: int, y: float) -> float:
        """
        Returns lateral X for lane index at longitudinal position y.
        lane_index 0: Right lane (+X)
        lane_index 1: Left lane (-X)
        """
        road_center = self.get_centerline_x(y)
        if lane_index == 0:
            return road_center + self.lane_width / 2.0
        elif lane_index == 1:
            return road_center - self.lane_width / 2.0
        else:
            offset = (lane_index - 0.5) * self.lane_width
            return road_center + offset

    def is_on_road(self, x: float, y: float) -> bool:
        """Check if (x, y) is within road pavement boundaries."""
        center_x = self.get_centerline_x(y)
        return abs(x - center_x) <= self.half_road_width

    def is_on_shoulder(self, x: float, y: float) -> bool:
        """Check if (x, y) is on the roadside shoulder."""
        center_x = self.get_centerline_x(y)
        dist = abs(x - center_x)
        return self.half_road_width < dist <= self.total_half_width

    def get_surface_reflectance(self, x: float, y: float) -> float:
        """
        Calculates surface reflectance intensity:
        - Lane markings (center dashed, edge solid): ~0.80 - 0.85
        - Clean asphalt: ~0.30
        - Roadside shoulder/gravel: ~0.22
        """
        center_x = self.get_centerline_x(y)
        dx = x - center_x
        abs_dx = abs(dx)

        if abs_dx > self.half_road_width:
            return 0.22  # Shoulder

        # Center dashed line (width 0.15m, 3m dashed with 3m gap)
        if abs_dx < 0.10:
            if (int(y / 3.0) % 2) == 0:
                return 0.82

        # Outer edge lines (solid white lines near curbs, width 0.15m)
        if abs(abs_dx - (self.half_road_width - 0.15)) < 0.10:
            return 0.85

        return 0.30  # Asphalt


class Curb:
    """
    Raised curb along the road boundary with a sharp step height (~0.15m).
    """

    def __init__(
        self,
        side: str = "right",
        step_height: float = 0.15,
        width: float = 0.25,
        y_min: float = -100.0,
        y_max: float = 100.0,
        x_offset: Optional[float] = None,
        height: Optional[float] = None,
    ) -> None:
        self.side = side
        self.step_height = height if height is not None else step_height
        self.width = width
        self.y_min = y_min
        self.y_max = y_max
        self.x_offset = x_offset
        self.reflectance = 0.50

    def get_height_offset(self, x: float, y: float, road: RoadModel) -> float:
        """
        Returns the height contribution of the curb.
        Step transition from 0.0 to +step_height across curb width.
        Beyond the curb (sidewalk), elevation remains at +step_height.
        """
        if y < self.y_min or y > self.y_max:
            return 0.0

        if self.x_offset is not None:
            road_edge = self.x_offset
        else:
            road_center = road.get_centerline_x(y)
            road_edge = (
                road_center - road.half_road_width
                if self.side == "left"
                else road_center + road.half_road_width
            )

        if self.side == "left" or (self.x_offset is not None and self.x_offset < 0):
            dist_past_edge = road_edge - x
            if dist_past_edge < 0:
                return 0.0
            elif dist_past_edge < self.width:
                t = dist_past_edge / self.width
                return self.step_height * (3 * t**2 - 2 * t**3)
            else:
                return self.step_height
        else:
            dist_past_edge = x - road_edge
            if dist_past_edge < 0:
                return 0.0
            elif dist_past_edge < self.width:
                t = dist_past_edge / self.width
                return self.step_height * (3 * t**2 - 2 * t**3)
            else:
                return self.step_height


class SlopeTerrain:
    """
    Parametric ramp / slope terrain segment.
    Mild slope: ~5-8 degrees.
    Steep slope: ~12-15 degrees.
    """

    def __init__(
        self,
        y_start: float = 20.0,
        y_end: float = 40.0,
        slope_deg: float = 6.0,
        direction: str = "up",
        x_min: float = -32.0,
        x_max: float = 32.0,
        start_y: Optional[float] = None,
        end_y: Optional[float] = None,
        angle_deg: Optional[float] = None,
    ) -> None:
        self.y_start = start_y if start_y is not None else y_start
        self.y_end = end_y if end_y is not None else y_end
        self.slope_deg = angle_deg if angle_deg is not None else slope_deg
        self.direction = direction
        self.x_min = x_min
        self.x_max = x_max
        self.slope_rad = math.radians(self.slope_deg)
        self.gradient = math.tan(self.slope_rad)
        if direction == "down":
            self.gradient = -self.gradient
        self.length = max(1.0, self.y_end - self.y_start)
        self.total_rise = self.gradient * self.length

    def get_elevation_and_slope(self, x: float, y: float) -> Tuple[float, float]:
        """
        Returns (elevation_offset, slope_deg) at (x, y).
        Uses smooth cubic Hermite blend at entry and exit.
        """
        if x < self.x_min or x > self.x_max or y < self.y_start:
            return 0.0, 0.0

        if y > self.y_end:
            return self.total_rise, 0.0

        dy = y - self.y_start
        u = dy / self.length
        h_factor = 3 * (u**2) - 2 * (u**3)
        elev = self.total_rise * h_factor
        current_slope_deg = self.slope_deg * (6 * u * (1 - u)) * 1.5
        return elev, min(current_slope_deg, self.slope_deg)


class Pothole:
    """
    Road depression with an inverted bowl profile.
    Depth: ~10-15cm deep (0.10m - 0.15m).
    Radius: ~0.4-0.8m (0.4m - 0.8m).
    """

    def __init__(
        self,
        center_x: float = 0.0,
        center_y: float = 20.0,
        depth: float = 0.14,
        radius: float = 0.65,
        roughness: float = 0.02,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> None:
        self.center_x = x if x is not None else center_x
        self.center_y = y if y is not None else center_y
        self.depth = abs(depth)
        self.radius = radius
        self.roughness = roughness
        self.reflectance = 0.20

    def get_depression(self, px: float, py: float) -> float:
        """
        Returns negative depression depth (in meters) at (px, py).
        Zero if outside pothole radius.
        """
        dx = px - self.center_x
        dy = py - self.center_y
        r2 = dx * dx + dy * dy
        R2 = self.radius * self.radius

        if r2 >= R2:
            return 0.0

        ratio = r2 / R2
        factor = (1.0 - ratio) ** 2
        noise = self.roughness * math.sin(dx * 25.0) * math.cos(dy * 25.0)
        return -(self.depth * factor + noise * factor)


class StaticObstacle:
    """
    Static obstacle in the scene:
    traffic_barrel, jersey_barrier, guardrail, parked_car.
    """

    def __init__(
        self,
        obstacle_id: str,
        obstacle_type: str,
        position: Union[List[float], Tuple[float, float, float]],
        dimensions: Union[List[float], Tuple[float, float, float]],
        yaw: float = 0.0,
        reflectance: float = 0.80,
    ) -> None:
        self.id = obstacle_id
        self.obstacle_type = obstacle_type
        self.class_name = "static_obstacle"
        self.position = np.array(position, dtype=np.float32)
        self.dimensions = np.array(dimensions, dtype=np.float32)
        self.yaw = yaw
        self.reflectance = reflectance
        self.velocity = np.zeros(3, dtype=np.float32)

    def to_detection_3d(self, ego_pos: np.ndarray, ego_yaw: float) -> Detection3D:
        """Transforms obstacle to ego vehicle coordinate frame as Detection3D."""
        dx = self.position[0] - ego_pos[0]
        dy = self.position[1] - ego_pos[1]
        dz = self.position[2] - ego_pos[2]

        rad = math.radians(-ego_yaw)
        c, s = math.cos(rad), math.sin(rad)
        rel_x = c * dx - s * dy
        rel_y = s * dx + c * dy
        rel_z = dz

        rel_yaw = math.radians(self.yaw - ego_yaw)

        return Detection3D(
            id=self.id,
            class_name="static_obstacle",
            confidence=0.98,
            position=[float(rel_x), float(rel_y), float(rel_z)],
            dimensions=[
                float(self.dimensions[0]),
                float(self.dimensions[1]),
                float(self.dimensions[2]),
            ],
            yaw=float(rel_yaw),
            velocity=[0.0, 0.0, 0.0],
            detector_source="Simulated DSVT Adapter",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "class_name": self.class_name,
            "position": self.position.tolist(),
            "dimensions": self.dimensions.tolist(),
            "yaw": self.yaw,
            "velocity": [0.0, 0.0, 0.0],
        }


class DynamicActor:
    """
    Base class for moving actors in the world.
    Tracks trajectory history and velocity.
    """

    def __init__(
        self,
        actor_id: Union[int, str],
        class_name: str,
        position: Union[List[float], Tuple[float, float, float]],
        dimensions: Union[List[float], Tuple[float, float, float]],
        yaw: float = 0.0,
        speed_ms: float = 0.0,
        reflectance: float = 0.80,
        velocity: Optional[List[float]] = None,
        trajectory_type: str = "linear",
        bounds: Optional[List[float]] = None,
    ) -> None:
        self.id = str(actor_id)
        self.class_name = class_name
        self.initial_pos = list(position)
        self.position = np.array(position, dtype=np.float32)
        self.dimensions = np.array(dimensions, dtype=np.float32)
        self.yaw = yaw
        self.speed = speed_ms
        self.initial_velocity = list(velocity) if velocity else [0.0, float(speed_ms), 0.0]
        self.velocity = np.array(self.initial_velocity, dtype=np.float32)
        self.reflectance = reflectance
        self.trajectory_type = trajectory_type
        self.bounds = bounds
        self.history: List[List[float]] = []
        self.max_history = 30
        self.age = 0
        self.hits = 5

    def step(self, dt: float, world: Optional["SimulationWorld"] = None) -> None:
        """Advance actor state by dt seconds."""
        self.age += 1
        self.history.append([float(self.position[0]), float(self.position[1])])
        if len(self.history) > self.max_history:
            self.history.pop(0)

        if self.trajectory_type == "static":
            return

        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt
        self.position[2] += self.velocity[2] * dt

        if self.bounds and len(self.bounds) >= 2:
            if self.velocity[1] > 0 and self.position[1] > self.bounds[1]:
                self.position[1] = self.bounds[0]
            elif self.velocity[1] < 0 and self.position[1] < self.bounds[0]:
                self.position[1] = self.bounds[1]

    def reset(self) -> None:
        """Reset actor to starting configuration."""
        self.position = np.array(self.initial_pos, dtype=np.float32)
        self.velocity = np.array(self.initial_velocity, dtype=np.float32)
        self.history.clear()
        self.age = 0

    def to_detection_3d(self, ego_pos: np.ndarray, ego_yaw: float) -> Detection3D:
        """Converts actor to vehicle coordinate frame Detection3D."""
        dx = self.position[0] - ego_pos[0]
        dy = self.position[1] - ego_pos[1]
        dz = self.position[2] - ego_pos[2]

        rad = math.radians(-ego_yaw)
        c, s = math.cos(rad), math.sin(rad)
        rel_x = c * dx - s * dy
        rel_y = s * dx + c * dy
        rel_z = dz

        vx = self.velocity[0]
        vy = self.velocity[1]
        rel_vx = c * vx - s * vy
        rel_vy = s * vx + c * vy

        rel_yaw = math.radians(self.yaw - ego_yaw)

        valid_class = "vehicle"
        if self.class_name in ["vehicle", "pedestrian", "cyclist", "static_obstacle"]:
            valid_class = self.class_name

        return Detection3D(
            id=f"actor_{self.id}",
            class_name=valid_class,  # type: ignore
            confidence=0.96,
            position=[float(rel_x), float(rel_y), float(rel_z)],
            dimensions=[
                float(self.dimensions[0]),
                float(self.dimensions[1]),
                float(self.dimensions[2]),
            ],
            yaw=float(rel_yaw),
            velocity=[float(rel_vx), float(rel_vy), 0.0],
            detector_source="Simulated DSVT Adapter",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "class_name": self.class_name,
            "position": self.position.tolist(),
            "dimensions": self.dimensions.tolist(),
            "yaw": self.yaw,
            "velocity": self.velocity.tolist(),
        }


class MovingVehicle(DynamicActor):
    """
    Moving vehicle with realistic lane-following, overtaking, or oncoming behavior.
    Speeds: 20-50 km/h (5.5 - 13.9 m/s).
    """

    def __init__(
        self,
        actor_id: Union[int, str],
        position: Union[List[float], Tuple[float, float, float]],
        speed_kmh: float = 35.0,
        behavior: str = "lane_following",
        lane_idx: int = 0,
        dimensions: Union[List[float], Tuple[float, float, float]] = (4.5, 1.9, 1.5),
        bounds: Optional[List[float]] = None,
        velocity: Optional[List[float]] = None,
        yaw: float = 0.0,
        trajectory_type: str = "linear",
    ) -> None:
        super().__init__(
            actor_id=actor_id,
            class_name="vehicle",
            position=position,
            dimensions=dimensions,
            yaw=yaw if yaw != 0.0 else (0.0 if behavior != "oncoming" else 180.0),
            speed_ms=speed_kmh / 3.6,
            reflectance=0.85,
            velocity=velocity,
            trajectory_type=trajectory_type,
            bounds=bounds or [-10.0, 50.0],
        )
        self.speed_kmh = speed_kmh
        self.behavior = behavior
        self.lane_idx = lane_idx
        self.overtake_phase = 0
        self.phase_time = 0.0

    def step(self, dt: float, world: Optional["SimulationWorld"] = None) -> None:
        self.age += 1
        self.phase_time += dt

        self.history.append([float(self.position[0]), float(self.position[1])])
        if len(self.history) > self.max_history:
            self.history.pop(0)

        speed_ms = self.speed_kmh / 3.6

        if world is not None and hasattr(world, "road"):
            road = world.road
            ego_y = world.ego.position[1]
        else:
            road = RoadModel()
            ego_y = 0.0

        if self.behavior == "lane_following":
            target_x = road.get_lane_center_x(self.lane_idx, self.position[1])
            dx = (target_x - self.position[0]) * 2.0
            self.velocity = np.array([dx, speed_ms, 0.0], dtype=np.float32)
            self.position[0] += dx * dt
            self.position[1] += speed_ms * dt
            self.yaw = math.degrees(math.atan2(dx, speed_ms))

            if self.bounds and self.position[1] > self.bounds[1]:
                self.position[1] = self.bounds[0]

        elif self.behavior == "oncoming":
            target_x = road.get_lane_center_x(1, self.position[1])
            dx = (target_x - self.position[0]) * 2.0
            self.velocity = np.array([dx, -speed_ms, 0.0], dtype=np.float32)
            self.position[0] += dx * dt
            self.position[1] -= speed_ms * dt
            self.yaw = 180.0 + math.degrees(math.atan2(dx, speed_ms))

            if self.position[1] < ego_y - 25.0:
                self.position[1] = ego_y + 60.0

        elif self.behavior == "overtaking":
            rel_y = self.position[1] - ego_y
            lane_0_x = road.get_lane_center_x(0, self.position[1])
            lane_1_x = road.get_lane_center_x(1, self.position[1])

            if self.overtake_phase == 0:
                target_x = lane_0_x
                current_speed = speed_ms * 1.05
                if rel_y > -10.0 or self.phase_time > 3.0:
                    self.overtake_phase = 1
                    self.phase_time = 0.0
            elif self.overtake_phase == 1:
                target_x = lane_1_x
                current_speed = speed_ms * 1.25
                if abs(self.position[0] - lane_1_x) < 0.2 and self.phase_time > 2.0:
                    self.overtake_phase = 2
                    self.phase_time = 0.0
            elif self.overtake_phase == 2:
                target_x = lane_1_x
                current_speed = speed_ms * 1.30
                if rel_y > 15.0 and self.phase_time > 3.0:
                    self.overtake_phase = 3
                    self.phase_time = 0.0
            else:
                target_x = lane_0_x
                current_speed = speed_ms * 1.0
                if abs(self.position[0] - lane_0_x) < 0.2 and self.phase_time > 2.5:
                    self.overtake_phase = 0
                    self.phase_time = 0.0

            lateral_err = target_x - self.position[0]
            lat_v = float(np.clip(lateral_err * 2.5, -2.5, 2.5))
            self.velocity = np.array([lat_v, current_speed, 0.0], dtype=np.float32)
            self.position[0] += lat_v * dt
            self.position[1] += current_speed * dt
            self.yaw = math.degrees(math.atan2(lat_v, current_speed))

        else:
            self.position[0] += self.velocity[0] * dt
            self.position[1] += self.velocity[1] * dt


class PedestrianActor(DynamicActor):
    """
    Moving pedestrian walking across road or along shoulder.
    Speeds: 1.0 - 1.5 m/s.
    Dimensions: 0.5m x 0.5m x 1.75m.
    """

    def __init__(
        self,
        actor_id: Union[int, str],
        position: Union[List[float], Tuple[float, float, float]],
        speed_ms: float = 1.3,
        behavior: str = "crossing",
        cross_direction: int = -1,
        y_travel: float = 0.0,
        bounds: Optional[List[float]] = None,
        velocity: Optional[List[float]] = None,
        yaw: float = 0.0,
        trajectory_type: str = "patrol",
    ) -> None:
        super().__init__(
            actor_id=actor_id,
            class_name="pedestrian",
            position=position,
            dimensions=(0.5, 0.5, 1.75),
            yaw=yaw if yaw != 0.0 else (-90.0 if cross_direction < 0 else 90.0),
            speed_ms=speed_ms,
            reflectance=0.42,
            velocity=velocity,
            trajectory_type=trajectory_type,
            bounds=bounds,
        )
        self.behavior = behavior
        self.cross_direction = cross_direction
        self.y_travel = y_travel
        self.wait_timer = 0.0

    def step(self, dt: float, world: Optional["SimulationWorld"] = None) -> None:
        self.age += 1
        self.history.append([float(self.position[0]), float(self.position[1])])
        if len(self.history) > self.max_history:
            self.history.pop(0)

        if self.wait_timer > 0.0:
            self.wait_timer -= dt
            self.velocity = np.zeros(3, dtype=np.float32)
            return

        if self.behavior == "crossing":
            if world is not None and hasattr(world, "road"):
                road_center = world.road.get_centerline_x(self.position[1])
                half_w = world.road.half_road_width
            else:
                road_center = 0.0
                half_w = 3.75

            left_bound = self.bounds[0] if self.bounds else (road_center - half_w - 1.0)
            right_bound = self.bounds[1] if self.bounds else (road_center + half_w + 1.0)

            vx = self.cross_direction * self.speed
            vy = self.y_travel
            self.velocity = np.array([vx, vy, 0.0], dtype=np.float32)

            self.position[0] += vx * dt
            self.position[1] += vy * dt
            self.yaw = 90.0 if self.cross_direction > 0 else -90.0

            if self.cross_direction < 0 and self.position[0] <= left_bound:
                self.cross_direction = 1
                self.wait_timer = 1.0
            elif self.cross_direction > 0 and self.position[0] >= right_bound:
                self.cross_direction = -1
                self.wait_timer = 1.0

        elif self.behavior == "shoulder_walking":
            vy = self.speed
            self.velocity = np.array([0.0, vy, 0.0], dtype=np.float32)
            self.position[1] += vy * dt
            self.yaw = 0.0
            if self.bounds and self.position[1] > self.bounds[1]:
                self.position[1] = self.bounds[0]
        else:
            self.position[0] += self.velocity[0] * dt
            self.position[1] += self.velocity[1] * dt


# Compatibility alias
SimulatedActor = DynamicActor


class EgoVehicle:
    """
    Ego vehicle model with kinematic steering and position integration.
    """

    def __init__(
        self,
        position: Union[List[float], Tuple[float, float, float]] = (0.0, 0.0, 1.73),
        speed_kmh: float = 30.0,
        yaw_deg: float = 0.0,
        steering_angle: float = 0.0,
        wheelbase: float = 2.7,
        state: Optional[EgoVehicleState] = None,
    ) -> None:
        if state is not None:
            self.position = np.array(state.position, dtype=np.float32)
            self.speed_kmh = state.speed
            self.yaw = state.yaw
            self.steering_angle = state.steering_angle
        else:
            self.position = np.array(position, dtype=np.float32)
            self.speed_kmh = speed_kmh
            self.yaw = yaw_deg
            self.steering_angle = steering_angle

        self.wheelbase = wheelbase
        self.target_speed_kmh = self.speed_kmh
        self.sensor_height = 1.73

    @property
    def speed_ms(self) -> float:
        return self.speed_kmh / 3.6

    def step(self, dt: float, road: Optional[RoadModel] = None) -> EgoVehicleState:
        """Advances ego vehicle state using kinematic bicycle model."""
        v = self.speed_ms
        self.speed_kmh += (self.target_speed_kmh - self.speed_kmh) * 0.1 * dt

        delta_rad = math.radians(self.steering_angle)
        yaw_rate_rad = (v / self.wheelbase) * math.tan(delta_rad)
        self.yaw += math.degrees(yaw_rate_rad * dt)

        yaw_rad = math.radians(self.yaw)
        vx = v * math.sin(yaw_rad)
        vy = v * math.cos(yaw_rad)

        self.position[0] += vx * dt
        self.position[1] += vy * dt

        velocity = np.array([vx, vy, 0.0], dtype=np.float32)

        return EgoVehicleState(
            position=[float(self.position[0]), float(self.position[1]), float(self.position[2])],
            velocity=[float(velocity[0]), float(velocity[1]), float(velocity[2])],
            speed=float(self.speed_kmh),
            yaw=float(self.yaw),
            steering_angle=float(self.steering_angle),
        )


class SimulationWorld:
    """
    Rich simulated world containing road network, static terrain features,
    curbs, slopes, potholes, obstacles, dynamic actors, and ego vehicle.
    """

    def __init__(self, initial_scene_id: str = "normal_road") -> None:
        self.road = RoadModel()
        self.curbs: List[Curb] = []
        self.potholes: List[Pothole] = []
        self.slopes: List[SlopeTerrain] = []
        self.static_obstacles: List[StaticObstacle] = []
        self.dynamic_actors: List[DynamicActor] = []
        self.ego = EgoVehicle()
        self.sim_time = 0.0
        self.frame_count = 0
        self.scene_id = initial_scene_id
        self.current_scene_id = initial_scene_id
        self._init_scenario(initial_scene_id)

    def _init_scenario(self, scene_id: str) -> None:
        """Initializes entities based on the scene id."""
        from backend.app.simulation.scenarios import build_all_scenarios
        self.scenarios = build_all_scenarios()
        if scene_id not in self.scenarios:
            scene_id = "normal_road"
        self.current_scene_id = scene_id
        self.scene_id = scene_id
        self.scenario = self.scenarios[scene_id]

        self.clear()

        # Configure ego
        self.ego = EgoVehicle(
            position=(0.0, 0.0, 1.73),
            speed_kmh=self.scenario.ego_speed,
            yaw_deg=0.0,
            steering_angle=0.0,
        )

        # Standard road
        self.road = RoadModel(lane_width=3.75, num_lanes=2, curvature=0.0)

        # Build entities from scenario definition
        for actor in self.scenario.actors:
            if actor.class_name == "static_obstacle" or actor.trajectory_type == "static":
                self.static_obstacles.append(
                    StaticObstacle(
                        obstacle_id=actor.id,
                        obstacle_type=actor.class_name,
                        position=actor.position,
                        dimensions=actor.dimensions,
                        yaw=actor.yaw,
                    )
                )
            elif actor.class_name == "pedestrian":
                self.dynamic_actors.append(
                    PedestrianActor(
                        actor_id=actor.id,
                        position=actor.position,
                        speed_ms=float(np.linalg.norm(actor.velocity[:2])) if any(actor.velocity) else 1.2,
                        behavior="crossing" if actor.trajectory_type == "patrol" else "shoulder_walking",
                        bounds=actor.bounds,
                        velocity=actor.velocity,
                        yaw=actor.yaw,
                    )
                )
            else:
                self.dynamic_actors.append(
                    MovingVehicle(
                        actor_id=actor.id,
                        position=actor.position,
                        speed_kmh=float(actor.velocity[1] * 3.6) if actor.velocity[1] != 0 else 35.0,
                        dimensions=actor.dimensions,
                        bounds=actor.bounds,
                        velocity=actor.velocity,
                        yaw=actor.yaw,
                    )
                )

        # Hazards: potholes and curbs
        for h in self.scenario.hazards:
            ftype = h.get("feature_type")
            pos = h.get("position", [0, 0, 0])
            if ftype == "pothole":
                self.potholes.append(
                    Pothole(
                        center_x=pos[0],
                        center_y=pos[1],
                        depth=abs(h.get("severity", 0.14)),
                        radius=h.get("radius", 0.65),
                    )
                )
            elif ftype == "curb":
                side = "left" if pos[0] < 0 else "right"
                self.curbs.append(
                    Curb(
                        side=side,
                        step_height=h.get("severity", 0.15),
                        width=0.25,
                        x_offset=pos[0],
                    )
                )

        # Scene-specific additions (curbs and slopes)
        if scene_id in ["curb_boundary", "complex_environment", "pedestrian"]:
            if not any(c.side == "left" for c in self.curbs):
                self.curbs.append(Curb(side="left", step_height=0.15, width=0.25, x_offset=-7.0))
            if not any(c.side == "right" for c in self.curbs):
                self.curbs.append(Curb(side="right", step_height=0.15, width=0.25, x_offset=7.0))

        if scene_id == "complex_environment":
            self.slopes.append(
                SlopeTerrain(
                    y_start=28.0,
                    y_end=45.0,
                    slope_deg=7.5,
                    direction="up",
                )
            )

    def clear(self) -> None:
        """Clears all entities."""
        self.curbs.clear()
        self.potholes.clear()
        self.slopes.clear()
        self.static_obstacles.clear()
        self.dynamic_actors.clear()
        self.sim_time = 0.0
        self.frame_count = 0

    def reset(self) -> None:
        """Resets simulation time and actors."""
        self.sim_time = 0.0
        self.frame_count = 0
        self.ego.position = np.array([0.0, 0.0, 1.73], dtype=np.float32)
        self.ego.yaw = 0.0
        self.ego.steering_angle = 0.0
        for actor in self.dynamic_actors:
            actor.reset()

    def set_scenario(self, scene_id: str) -> bool:
        """Switch to a different scenario."""
        self._init_scenario(scene_id)
        return True

    def step(self, dt: float = 1.0 / 15.0) -> EgoVehicleState:
        """
        Advances the simulation by dt seconds.
        Updates ego vehicle, all dynamic actors, and simulation clock.
        """
        self.sim_time += dt
        self.frame_count += 1

        ego_state = self.ego.step(dt, self.road)

        for actor in self.dynamic_actors:
            actor.step(dt, self)

        return ego_state

    @property
    def ego_state(self) -> EgoVehicleState:
        v = self.ego.speed_ms
        yaw_rad = math.radians(self.ego.yaw)
        vx = v * math.sin(yaw_rad)
        vy = v * math.cos(yaw_rad)
        return EgoVehicleState(
            position=[float(self.ego.position[0]), float(self.ego.position[1]), float(self.ego.position[2])],
            velocity=[float(vx), float(vy), 0.0],
            speed=float(self.ego.speed_kmh),
            yaw=float(self.ego.yaw),
            steering_angle=float(self.ego.steering_angle),
        )

    @property
    def actors(self) -> List[Any]:
        return self.dynamic_actors + self.static_obstacles

    @property
    def hazards(self) -> List[Dict[str, Any]]:
        return [f.model_dump() for f in self.get_ground_truth_terrain_features()]

    def get_elevation_at(self, x: float, y: float) -> Tuple[float, str, float]:
        """
        Calculates terrain elevation z, semantic class name, and slope angle (deg)
        at world coordinate (x, y).
        """
        base_z = 0.0
        slope_deg = 0.0
        sem_class = "road" if self.road.is_on_road(x, y) else "terrain"

        for slope in self.slopes:
            s_elev, s_deg = slope.get_elevation_and_slope(x, y)
            if abs(s_elev) > 1e-4:
                base_z += s_elev
                slope_deg = max(slope_deg, s_deg)
                if s_deg >= settings.SLOPE_MILD_DEG:
                    sem_class = "slope"

        curb_offset = 0.0
        for curb in self.curbs:
            c_h = curb.get_height_offset(x, y, self.road)
            if c_h > 0.0:
                curb_offset = max(curb_offset, c_h)
                if 0.02 < c_h < curb.step_height - 0.01:
                    sem_class = "curb"
                elif c_h >= curb.step_height - 0.01 and sem_class != "slope":
                    sem_class = "terrain"

        total_z = base_z + curb_offset

        for pothole in self.potholes:
            dep = pothole.get_depression(x, y)
            if dep < -0.01:
                total_z += dep
                if dep <= settings.POTHOLE_DEPTH_THRESHOLD:
                    sem_class = "pothole"

        return float(total_z), sem_class, float(slope_deg)

    def get_surface_reflectance(self, x: float, y: float) -> float:
        """Returns surface reflectance intensity at (x, y)."""
        for pothole in self.potholes:
            if pothole.get_depression(x, y) < -0.02:
                return pothole.reflectance

        for curb in self.curbs:
            c_h = curb.get_height_offset(x, y, self.road)
            if 0.02 < c_h < curb.step_height - 0.01:
                return curb.reflectance

        return self.road.get_surface_reflectance(x, y)

    def get_ground_truth_detections(self) -> List[Detection3D]:
        """Returns 3D bounding boxes for all active objects in vehicle frame."""
        detections: List[Detection3D] = []
        ego_pos = self.ego.position
        ego_yaw = self.ego.yaw

        for actor in self.dynamic_actors:
            det = actor.to_detection_3d(ego_pos, ego_yaw)
            px, py, pz = det.position
            if (
                settings.ROI_X_MIN <= px <= settings.ROI_X_MAX
                and settings.ROI_Y_MIN <= py <= settings.ROI_Y_MAX
            ):
                detections.append(det)

        for obs in self.static_obstacles:
            det = obs.to_detection_3d(ego_pos, ego_yaw)
            px, py, pz = det.position
            if (
                settings.ROI_X_MIN <= px <= settings.ROI_X_MAX
                and settings.ROI_Y_MIN <= py <= settings.ROI_Y_MAX
            ):
                detections.append(det)

        return detections

    def get_ground_truth_terrain_features(self) -> List[TerrainFeature]:
        """Returns ground truth terrain features (potholes, curbs, slopes)."""
        features: List[TerrainFeature] = []
        ego_pos = self.ego.position
        ego_yaw = self.ego.yaw

        rad = math.radians(-ego_yaw)
        c, s = math.cos(rad), math.sin(rad)

        for p in self.potholes:
            dx = p.center_x - ego_pos[0]
            dy = p.center_y - ego_pos[1]
            rel_x = c * dx - s * dy
            rel_y = s * dx + c * dy
            elev, _, _ = self.get_elevation_at(p.center_x, p.center_y)
            rel_z = elev - ego_pos[2]

            features.append(
                TerrainFeature(
                    feature_type="pothole",
                    position=[float(rel_x), float(rel_y), float(rel_z)],
                    bounds=[
                        float(rel_x - p.radius),
                        float(rel_x + p.radius),
                        float(rel_y - p.radius),
                        float(rel_y + p.radius),
                    ],
                    severity=float(p.depth),
                    description=f"Pothole depression depth={p.depth*100:.1f}cm radius={p.radius:.2f}m",
                    confidence=0.95,
                )
            )

        for curb in self.curbs:
            edge_x = curb.x_offset if curb.x_offset is not None else (
                self.road.get_centerline_x(ego_pos[1]) - self.road.half_road_width
                if curb.side == "left"
                else self.road.get_centerline_x(ego_pos[1]) + self.road.half_road_width
            )
            dx = edge_x - ego_pos[0]
            rel_x = c * dx
            features.append(
                TerrainFeature(
                    feature_type="curb",
                    position=[float(rel_x), 10.0, -self.ego.sensor_height + curb.step_height],
                    bounds=[
                        float(rel_x - 0.2),
                        float(rel_x + 0.2),
                        -16.0,
                        48.0,
                    ],
                    severity=float(curb.step_height),
                    description=f"Raised {curb.side} curb step height={curb.step_height*100:.0f}cm",
                    confidence=0.98,
                )
            )

        for slope in self.slopes:
            dy = slope.y_start - ego_pos[1]
            features.append(
                TerrainFeature(
                    feature_type="slope",
                    position=[0.0, float(dy + slope.length / 2.0), 0.0],
                    bounds=[slope.x_min, slope.x_max, float(dy), float(dy + slope.length)],
                    severity=float(slope.slope_deg),
                    description=f"{slope.direction.capitalize()} ramp slope={slope.slope_deg:.1f} deg",
                    confidence=0.92,
                )
            )

        return features


# Universal alias for World
World = SimulationWorld
