"""
64-Beam LiDAR Sensor Simulator for MI Sense.
Problem Statement: SIH 26053
"""

import math
import random
from typing import List, Dict, Any, Optional
import numpy as np

from backend.app.config.settings import settings


class LidarSimulator:
    """
    Simulates a 64-beam mechanical/solid-state LiDAR scanner.
    Synthesizes realistic ground returns, obstacles, hazard fixtures, and moving actors.
    """

    def __init__(
        self,
        beams: int = settings.LIDAR_BEAMS,
        range_max: float = settings.LIDAR_RANGE_MAX,
        noise_std: float = settings.LIDAR_NOISE_STD,
    ):
        self.beams = beams
        self.range_max = range_max
        self.noise_std = noise_std

        # Vertical beam angles from -25° to +15°
        self.v_angles = np.linspace(
            math.radians(settings.LIDAR_FOV_VERTICAL_MIN),
            math.radians(settings.LIDAR_FOV_VERTICAL_MAX),
            self.beams,
        )

    def generate_point_cloud(
        self,
        ego_pose: List[float],  # [x, y, z, yaw]
        actors: List[Any],
        hazards: Optional[List[Dict[str, Any]]] = None,
        road_half_width: float = 7.0,
    ) -> np.ndarray:
        """
        Synthesize LiDAR point cloud around ego vehicle.

        Returns:
            (N, 4) numpy array of [x, y, z, intensity].
        """
        points_list: List[List[float]] = []

        # 1. Ground Plane Points (Road, Curbs, Sidewalks, Potholes)
        # Sample concentric rings based on beam pitch angles hitting the ground (z=0)
        sensor_height = 1.8  # LiDAR mounted 1.8m above ground on vehicle roof

        for v_ang in self.v_angles:
            if v_ang >= -0.01:
                # Upward beam, doesn't hit ground within close range
                continue

            # Theoretical ground intercept distance: d = sensor_height / -tan(v_ang)
            ground_dist = sensor_height / math.tan(-v_ang)
            if ground_dist > self.range_max or ground_dist < 1.0:
                continue

            # Generate azimuth sweep around ego
            # Higher density in forward sector (-90 to +90 deg), full 360 degree coverage
            azimuth_steps = max(36, int(round(ground_dist * 4)))
            azimuths = np.linspace(-math.pi, math.pi, azimuth_steps, endpoint=False)

            for az in azimuths:
                # Calculate vehicle-frame coordinates
                r = ground_dist + random.gauss(0, self.noise_std)
                # Sensor dropout
                if random.random() < settings.LIDAR_DROPOUT_RATE:
                    continue

                px = r * math.sin(az)
                py = r * math.cos(az)

                # Check ROI bounds
                if not (settings.ROI_X_MIN <= px <= settings.ROI_X_MAX and settings.ROI_Y_MIN <= py <= settings.ROI_Y_MAX):
                    continue

                # Ground elevation logic
                pz = -sensor_height  # Relative to sensor, ground is at z = -sensor_height (0m in world)
                intensity = 0.4  # Default asphalt reflectance

                # Check if on road vs curb vs off-road
                dist_from_centerline = abs(px)
                if dist_from_centerline > road_half_width:
                    # Curb step (+0.15m) or elevated sidewalk/grass
                    if dist_from_centerline <= (road_half_width + 0.8):
                        pz += 0.15 + random.gauss(0, 0.01)  # Raised curb
                        intensity = 0.65  # Concrete curb paint
                    else:
                        pz += random.gauss(0, 0.03) + 0.05  # Grass/dirt shoulder
                        intensity = 0.25

                # Check for pothole depressions
                if hazards:
                    for h in hazards:
                        if h.get("feature_type") == "pothole":
                            hx, hy = h.get("position", [0, 0, 0])[:2]
                            d_pot = math.hypot(px - hx, py - hy)
                            radius = h.get("radius", 0.6)
                            if d_pot < radius:
                                depth = h.get("severity", -0.12)
                                # Bowl shape depression
                                pz += depth * (1.0 - (d_pot / radius)**2)
                                intensity = 0.15  # Darker cavity

                points_list.append([px, py, pz + sensor_height, intensity])

        # 2. Obstacle & Actor Surface Points
        for actor in actors:
            act_data = self._get_actor_dict(actor)
            if not act_data:
                continue

            pos = act_data["position"]
            dims = act_data["dimensions"]
            yaw = act_data["yaw"]
            c_name = act_data.get("class_name", "vehicle")

            # Distance to actor
            dist = math.hypot(pos[0], pos[1])
            if dist > self.range_max:
                continue

            # Number of surface returns scales inversely with distance squared
            base_pts = 120 if c_name == "vehicle" else 40
            n_pts = max(10, int(base_pts * (15.0 / max(dist, 5.0))))

            half_l, half_w, half_h = dims[0] * 0.5, dims[1] * 0.5, dims[2] * 0.5
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)

            for _ in range(n_pts):
                # Sample random surface points on box
                side = random.choice(["top", "side_x", "side_y"])
                if side == "top":
                    lx = random.uniform(-half_l, half_l)
                    ly = random.uniform(-half_w, half_w)
                    lz = half_h
                elif side == "side_x":
                    lx = random.choice([-half_l, half_l])
                    ly = random.uniform(-half_w, half_w)
                    lz = random.uniform(-half_h, half_h)
                else:
                    lx = random.uniform(-half_l, half_l)
                    ly = random.choice([-half_w, half_w])
                    lz = random.uniform(-half_h, half_h)

                # Rotate to actor yaw
                gx = pos[0] + (lx * cos_y - ly * sin_y) + random.gauss(0, self.noise_std)
                gy = pos[1] + (lx * sin_y + ly * cos_y) + random.gauss(0, self.noise_std)
                gz = pos[2] + lz + random.gauss(0, self.noise_std)

                points_list.append([gx, gy, gz, 0.85])

        if not points_list:
            return np.zeros((0, 4), dtype=np.float32)

        return np.array(points_list, dtype=np.float32)

    def _get_actor_dict(self, actor: Any) -> Optional[Dict[str, Any]]:
        """Standardize actor format."""
        if isinstance(actor, dict):
            return actor
        if hasattr(actor, "to_dict"):
            return actor.to_dict()
        data = {}
        for attr in ["id", "class_name", "position", "dimensions", "yaw", "velocity"]:
            if hasattr(actor, attr):
                data[attr] = getattr(actor, attr)
            else:
                return None
        return data
