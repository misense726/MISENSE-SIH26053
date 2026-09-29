"""
Perception detection models and adapters for MI Sense.
Problem Statement: SIH 26053

Provides:
- PerceptionModel: Abstract base class for 3D LiDAR detectors.
- SimulatedDSVTAdapter: Realistic 3D bounding box detection simulation matching
  OpenPCDet DSVT-Pillar output schema.
- OpenPCDetDSVTAdapter: Dependency probe with simulated detections. Model
  loading and trained-model inference are not implemented yet.
"""

from abc import ABC, abstractmethod
import logging
import math
import random
from typing import List, Dict, Any, Optional, Union
import numpy as np

from backend.app.config.settings import settings
from backend.app.models.schemas import Detection3D

logger = logging.getLogger(__name__)


class PerceptionModel(ABC):
    """Abstract base class for 3D LiDAR object perception detectors."""

    @abstractmethod
    def detect(
        self,
        points: np.ndarray,
        ground_truth_actors: Optional[List[Any]] = None,
    ) -> List[Detection3D]:
        """
        Run 3D object detection on LiDAR point cloud.

        Args:
            points: (N, 3) or (N, 4) numpy array of LiDAR points [x, y, z, (intensity)].
            ground_truth_actors: Optional list of simulated world actors (for adapters).

        Returns:
            List of 3D detection bounding boxes matching Detection3D schema.
        """
        pass

    @property
    @abstractmethod
    def detector_source(self) -> str:
        """Name of detection source for metadata & telemetry."""
        pass

    @property
    @abstractmethod
    def is_fallback(self) -> bool:
        """True if running in simulated fallback mode."""
        pass


class SimulatedDSVTAdapter(PerceptionModel):
    """
    Simulated DSVT-Pillar 3D Object Detector.

    Faithfully reproduces the output schema, coordinate frame, and uncertainty
    profile of OpenPCDet's Dynamic Sparse Voxel Transformer (DSVT-Pillar) model:
    - Classes: 'vehicle', 'pedestrian', 'cyclist', 'static_obstacle'
    - High-fidelity confidence scoring (0.85 - 0.98) with range attenuation
    - Gaussian position jitter (clipped to ±0.03m)
    - Realistic sensor occlusion / dropouts (< 2% miss rate)
    """

    def __init__(
        self,
        noise_std: float = 0.015,
        max_jitter: float = 0.03,
        miss_rate: float = 0.018,
    ):
        self.noise_std = noise_std
        self.max_jitter = max_jitter
        self.miss_rate = miss_rate
        self._source_name = "Simulated DSVT Adapter (OpenPCDet Schema)"

    @property
    def detector_source(self) -> str:
        return self._source_name

    @property
    def is_fallback(self) -> bool:
        return True

    def detect(
        self,
        points: np.ndarray,
        ground_truth_actors: Optional[List[Any]] = None,
    ) -> List[Detection3D]:
        """
        Produce realistic DSVT 3D detections from world ground truth actors.
        """
        if not ground_truth_actors:
            return []

        detections: List[Detection3D] = []

        for actor in ground_truth_actors:
            # Extract actor attributes (supports both dict and class object)
            actor_data = self._extract_actor_data(actor)
            if actor_data is None:
                continue

            pos = actor_data["position"]
            dims = actor_data["dimensions"]
            yaw = actor_data["yaw"]
            vel = actor_data.get("velocity", [0.0, 0.0, 0.0])
            class_name = actor_data.get("class_name", "vehicle")
            actor_id = str(actor_data.get("id", "det_0"))

            # Range & FOV Check
            dist = math.hypot(pos[0], pos[1])
            if dist > settings.LIDAR_RANGE_MAX or dist < settings.LIDAR_RANGE_MIN:
                continue

            # Check vehicle ROI bounds
            if not (
                settings.ROI_X_MIN <= pos[0] <= settings.ROI_X_MAX
                and settings.ROI_Y_MIN <= pos[1] <= settings.ROI_Y_MAX
            ):
                continue

            # Simulate occasional sensor occlusion / miss (< 2%)
            if random.random() < self.miss_rate:
                continue

            # Simulate realistic detection jitter
            jitter_x = float(np.clip(random.gauss(0, self.noise_std), -self.max_jitter, self.max_jitter))
            jitter_y = float(np.clip(random.gauss(0, self.noise_std), -self.max_jitter, self.max_jitter))
            jitter_z = float(np.clip(random.gauss(0, self.noise_std * 0.5), -self.max_jitter * 0.5, self.max_jitter * 0.5))

            detected_pos = [
                round(pos[0] + jitter_x, 3),
                round(pos[1] + jitter_y, 3),
                round(pos[2] + jitter_z, 3),
            ]

            # Slight dimension uncertainty (±0.02m)
            dim_noise = [random.uniform(-0.02, 0.02) for _ in range(3)]
            detected_dims = [
                max(0.2, round(dims[0] + dim_noise[0], 2)),
                max(0.2, round(dims[1] + dim_noise[1], 2)),
                max(0.2, round(dims[2] + dim_noise[2], 2)),
            ]

            # Orientation jitter (±0.015 rad)
            yaw_noise = random.gauss(0, 0.015)
            detected_yaw = round((yaw + yaw_noise + math.pi) % (2 * math.pi) - math.pi, 3)

            # Realistic confidence score (0.85 - 0.98), slightly lower at longer range
            range_penalty = min(0.08, (dist / settings.LIDAR_RANGE_MAX) * 0.08)
            base_conf = random.uniform(0.92, 0.98)
            confidence = round(max(0.85, min(0.99, base_conf - range_penalty)), 3)

            # Velocity vector with small measurement noise
            vel_noise_x = random.gauss(0, 0.02)
            vel_noise_y = random.gauss(0, 0.02)
            detected_vel = [
                round(vel[0] + vel_noise_x, 3),
                round(vel[1] + vel_noise_y, 3),
                round(vel[2], 3),
            ]

            valid_classes = {"vehicle", "pedestrian", "cyclist", "static_obstacle"}
            normalized_class = class_name.lower()
            if normalized_class not in valid_classes:
                normalized_class = "static_obstacle"

            detections.append(
                Detection3D(
                    id=f"det_{actor_id}",
                    class_name=normalized_class,  # type: ignore
                    confidence=confidence,
                    position=detected_pos,
                    dimensions=detected_dims,
                    yaw=detected_yaw,
                    velocity=detected_vel,
                    detector_source=self._source_name,
                )
            )

        return detections

    def _extract_actor_data(self, actor: Any) -> Optional[Dict[str, Any]]:
        """Extract standardized dictionary from actor object or dict."""
        if isinstance(actor, dict):
            return actor
        if hasattr(actor, "to_dict"):
            return actor.to_dict()

        # Handle object with attributes
        data = {}
        for attr in ["id", "class_name", "position", "dimensions", "yaw", "velocity"]:
            if hasattr(actor, attr):
                data[attr] = getattr(actor, attr)
            else:
                return None
        return data


class OpenPCDetDSVTAdapter(PerceptionModel):
    """Probe OpenPCDet dependencies without claiming they provide working inference.

    This adapter has no model loader or inference implementation. Even when CUDA,
    OpenPCDet, and a checkpoint path are present, detections remain simulated.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self._is_fallback = True
        self._source_name = "Simulated DSVT Adapter (OpenPCDet Schema)"
        self._fallback_adapter = SimulatedDSVTAdapter()

        # Check PyTorch and CUDA availability
        self.torch_available = False
        self.cuda_available = False
        self.pcdet_available = False
        self.model = None

        self._check_dependencies()

        if self._is_fallback:
            self._fallback_adapter = SimulatedDSVTAdapter()
        else:
            self._fallback_adapter = None

    def _check_dependencies(self) -> None:
        """Inspect PyTorch and OpenPCDet installation status."""
        try:
            import torch
            self.torch_available = True
            self.cuda_available = torch.cuda.is_available()
        except ImportError:
            self.torch_available = False
            self.cuda_available = False

        try:
            import pcdet
            self.pcdet_available = True
        except ImportError:
            self.pcdet_available = False

        if not self.torch_available or not self.cuda_available or not self.pcdet_available:
            reasons = []
            if not self.torch_available:
                reasons.append("PyTorch missing")
            elif not self.cuda_available:
                reasons.append("CUDA GPU unavailable")
            if not self.pcdet_available:
                reasons.append("OpenPCDet package missing")

            self._is_fallback = True
            self._status_message = f"Simulated Fallback ({', '.join(reasons)})"
            self._source_name = "Simulated DSVT Adapter (OpenPCDet Schema)"
            logger.info(
                "OpenPCDet DSVT-Pillar not active: %s. Using high-fidelity Simulated DSVT Adapter.",
                ", ".join(reasons),
            )
        else:
            self._status_message = "Dependencies available; inference is not implemented. Using simulated detections."
            self._source_name = "Simulated DSVT Adapter (OpenPCDet Schema)"

    @property
    def detector_source(self) -> str:
        return self._source_name

    @property
    def is_fallback(self) -> bool:
        return self._is_fallback

    @property
    def status_message(self) -> str:
        return self._status_message

    def detect(
        self,
        points: np.ndarray,
        ground_truth_actors: Optional[List[Any]] = None,
    ) -> List[Detection3D]:
        """
        Run DSVT inference or dispatch to fallback simulator.
        """
        if self._is_fallback or self._fallback_adapter is not None:
            return self._fallback_adapter.detect(points, ground_truth_actors)

        # In production CUDA environment with loaded model:
        # Construct OpenPCDet input dictionary, run model forward pass,
        # and decode predicted 3D bounding boxes.
        try:
            # Placeholder for PyTorch inference pass
            return []
        except Exception as e:
            logger.warning("Error during OpenPCDet inference: %s. Falling back to simulation.", e)
            if self._fallback_adapter is None:
                self._fallback_adapter = SimulatedDSVTAdapter()
            return self._fallback_adapter.detect(points, ground_truth_actors)
