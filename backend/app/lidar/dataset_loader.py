"""
Dataset Loader & Sensor Adapters for MI Sense.
Problem Statement: SIH 26053

Provides:
- Abstract base class `DatasetLoader`.
- `KittiDatasetAdapter`: Reads raw KITTI Velodyne .bin files and calibration.
- `NuScenesDatasetAdapter`: Reads NuScenes 5-channel LIDAR_TOP .bin files.
- `CustomPcdBinAdapter`: Parses standard .pcd (ASCII/Binary) and custom .bin files.
- `SyntheticDatasetGenerator`: Streams scenario-driven simulation frames and exports
  sample datasets to disk for offline evaluation and testing.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Iterator, Dict, Any
from pathlib import Path
import os
import struct
import numpy as np

from backend.app.config.settings import settings
from backend.app.models.schemas import EgoVehicleState, Detection3D
from backend.app.simulation.world import SimulationWorld
from backend.app.simulation.scenarios import load_scenario
from backend.app.simulation.lidar_simulator import LidarSimulator


class DatasetLoader(ABC):
    """
    Abstract base class defining the standard interface for point cloud data loaders.
    """

    @abstractmethod
    def get_frame(self, frame_id: int) -> np.ndarray:
        """
        Retrieves a single LiDAR frame.

        Args:
            frame_id: 0-indexed integer frame identifier.

        Returns:
            np.ndarray of shape (N, 4): [x, y, z, intensity] as float32.
        """
        pass

    @abstractmethod
    def get_num_frames(self) -> int:
        """Returns the total number of frames in the dataset."""
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Returns human-readable dataset name."""
        pass

    def get_ego_state(self, frame_id: int) -> Optional[EgoVehicleState]:
        """Optionally returns ego vehicle state for the given frame."""
        return None

    def get_ground_truth_detections(self, frame_id: int) -> List[Detection3D]:
        """Optionally returns ground truth 3D detections for the frame."""
        return []

    def reset(self) -> None:
        """Resets stream reader index to the beginning."""
        pass

    def __len__(self) -> int:
        return self.get_num_frames()

    def __iter__(self) -> Iterator[np.ndarray]:
        for idx in range(self.get_num_frames()):
            yield self.get_frame(idx)


class KittiDatasetAdapter(DatasetLoader):
    """
    Adapter for KITTI Raw / Odometry / Tracking LiDAR datasets.
    KITTI point format: binary float32 [x, y, z, reflectance].
    Velodyne coordinates: +X forward, +Y left, +Z up.
    Converts to MI Sense vehicle coordinates: +Y forward, +X right, +Z up.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        convert_to_vehicle_frame: bool = True,
    ) -> None:
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.convert_to_vehicle_frame = convert_to_vehicle_frame
        self.bin_files: List[Path] = []
        self._fallback_sim: Optional[SyntheticDatasetGenerator] = None

        if self.dataset_path and self.dataset_path.exists():
            # Look for standard velodyne subfolder or root folder
            velo_dir = self.dataset_path / "velodyne"
            if not velo_dir.exists():
                velo_dir = self.dataset_path / "velodyne_points" / "data"
            if not velo_dir.exists():
                velo_dir = self.dataset_path

            self.bin_files = sorted(list(velo_dir.glob("*.bin")))

        if not self.bin_files:
            # Graceful synthetic fallback when real dataset folder is not mounted
            self._fallback_sim = SyntheticDatasetGenerator(scene_id="normal_road", total_frames=100)

    def get_frame(self, frame_id: int) -> np.ndarray:
        if self._fallback_sim is not None:
            return self._fallback_sim.get_frame(frame_id)

        idx = frame_id % len(self.bin_files)
        file_path = self.bin_files[idx]

        # Read binary float32
        raw_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

        if self.convert_to_vehicle_frame:
            # KITTI: X_kitti is forward, Y_kitti is left, Z_kitti is up
            # MI Sense: Y_veh is forward, X_veh is right, Z_veh is up
            x_kitti = raw_data[:, 0]
            y_kitti = raw_data[:, 1]
            z_kitti = raw_data[:, 2]
            intensity = raw_data[:, 3]

            x_veh = -y_kitti  # right
            y_veh = x_kitti   # forward
            z_veh = z_kitti   # up

            return np.column_stack([x_veh, y_veh, z_veh, intensity]).astype(np.float32)

        return raw_data.astype(np.float32)

    def get_num_frames(self) -> int:
        if self._fallback_sim is not None:
            return self._fallback_sim.get_num_frames()
        return len(self.bin_files)

    def get_dataset_name(self) -> str:
        source = "Mounted" if self._fallback_sim is None else "Synthetic Fallback"
        return f"KITTI Velodyne 64-Beam Dataset ({source})"

    @staticmethod
    def save_frame_as_kitti_bin(points: np.ndarray, output_path: str) -> None:
        """Exports point cloud array [x, y, z, intensity] to KITTI binary format."""
        pts = np.asarray(points, dtype=np.float32)
        if pts.shape[1] == 3:
            pts = np.column_stack([pts, np.full((len(pts), 1), 0.5, dtype=np.float32)])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pts.tofile(output_path)


class NuScenesDatasetAdapter(DatasetLoader):
    """
    Adapter for nuScenes LiDAR point clouds (LIDAR_TOP).
    nuScenes point format: binary float32 [x, y, z, intensity, ring_index].
    nuScenes coordinates: +X right, +Y forward, +Z up (matches MI Sense vehicle frame).
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
    ) -> None:
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.bin_files: List[Path] = []
        self._fallback_sim: Optional[SyntheticDatasetGenerator] = None

        if self.dataset_path and self.dataset_path.exists():
            lidar_dir = self.dataset_path / "samples" / "LIDAR_TOP"
            if not lidar_dir.exists():
                lidar_dir = self.dataset_path
            self.bin_files = sorted(list(lidar_dir.glob("*.bin")))

        if not self.bin_files:
            self._fallback_sim = SyntheticDatasetGenerator(scene_id="moving_vehicle", total_frames=100)

    def get_frame(self, frame_id: int) -> np.ndarray:
        if self._fallback_sim is not None:
            return self._fallback_sim.get_frame(frame_id)

        idx = frame_id % len(self.bin_files)
        file_path = self.bin_files[idx]

        # 5 floats per point: [x, y, z, intensity, ring]
        raw_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)
        # Extract first 4 columns [x, y, z, intensity]
        # Normalize nuScenes intensity (usually 0-255) to 0.0-1.0 if needed
        xyz = raw_data[:, :3]
        intensity = raw_data[:, 3]
        if np.max(intensity) > 1.5:
            intensity = intensity / 255.0

        return np.column_stack([xyz, intensity]).astype(np.float32)

    def get_num_frames(self) -> int:
        if self._fallback_sim is not None:
            return self._fallback_sim.get_num_frames()
        return len(self.bin_files)

    def get_dataset_name(self) -> str:
        source = "Mounted" if self._fallback_sim is None else "Synthetic Fallback"
        return f"nuScenes 32-Beam LiDAR Dataset ({source})"


class CustomPcdBinAdapter(DatasetLoader):
    """
    Universal reader and writer for custom .pcd (ASCII and Binary) and .bin files.
    """

    def __init__(
        self,
        folder_path: Optional[str] = None,
    ) -> None:
        self.folder_path = Path(folder_path) if folder_path else None
        self.files: List[Path] = []
        self._fallback_sim: Optional[SyntheticDatasetGenerator] = None

        if self.folder_path and self.folder_path.exists():
            # Match both .pcd and .bin files
            pcd_files = list(self.folder_path.glob("*.pcd"))
            bin_files = list(self.folder_path.glob("*.bin"))
            self.files = sorted(pcd_files + bin_files)

        if not self.files:
            self._fallback_sim = SyntheticDatasetGenerator(scene_id="complex_environment", total_frames=100)

    def get_frame(self, frame_id: int) -> np.ndarray:
        if self._fallback_sim is not None:
            return self._fallback_sim.get_frame(frame_id)

        idx = frame_id % len(self.files)
        file_path = self.files[idx]

        if file_path.suffix.lower() == ".pcd":
            return self._read_pcd_file(file_path)
        else:
            raw = np.fromfile(file_path, dtype=np.float32)
            if len(raw) % 4 == 0:
                return raw.reshape(-1, 4)
            elif len(raw) % 3 == 0:
                pts = raw.reshape(-1, 3)
                intensity = np.full((len(pts), 1), 0.5, dtype=np.float32)
                return np.hstack([pts, intensity])
            else:
                return raw[: (len(raw) // 4) * 4].reshape(-1, 4)

    def get_num_frames(self) -> int:
        if self._fallback_sim is not None:
            return self._fallback_sim.get_num_frames()
        return len(self.files)

    def get_dataset_name(self) -> str:
        source = "Custom Directory" if self._fallback_sim is None else "Synthetic Fallback"
        return f"Custom PCD/BIN Dataset ({source})"

    def _read_pcd_file(self, file_path: Path) -> np.ndarray:
        """Parses standard PCD file (supports ASCII and Binary data)."""
        with open(file_path, "rb") as f:
            header_lines = []
            while True:
                line = f.readline().decode("ascii", errors="ignore").strip()
                header_lines.append(line)
                if line.startswith("DATA"):
                    break

            data_mode = header_lines[-1].split()[-1].lower()

            points_count = 0
            fields = ["x", "y", "z"]
            for h in header_lines:
                if h.startswith("POINTS"):
                    points_count = int(h.split()[-1])
                elif h.startswith("FIELDS"):
                    fields = h.split()[1:]

            if data_mode == "ascii":
                pts = []
                for line in f:
                    parts = line.decode("ascii", errors="ignore").strip().split()
                    if parts:
                        pts.append([float(p) for p in parts[:4]])
                arr = np.array(pts, dtype=np.float32)
                if arr.shape[1] == 3:
                    arr = np.column_stack([arr, np.full((len(arr), 1), 0.5, dtype=np.float32)])
                return arr

            elif data_mode == "binary":
                num_fields = len(fields)
                raw_bytes = f.read()
                arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(-1, num_fields)
                if arr.shape[1] >= 4:
                    return arr[:, :4]
                else:
                    return np.column_stack([arr[:, :3], np.full((len(arr), 1), 0.5, dtype=np.float32)])

        return np.zeros((0, 4), dtype=np.float32)

    @staticmethod
    def save_frame_as_pcd(points: np.ndarray, file_path: str, binary: bool = True) -> None:
        """
        Saves a point cloud array to a standard .pcd file.

        Args:
            points: (N, 4) point cloud [x, y, z, intensity].
            file_path: Destination .pcd file path.
            binary: If True, writes compact binary format; else ASCII.
        """
        pts = np.asarray(points, dtype=np.float32)
        if pts.shape[1] == 3:
            pts = np.column_stack([pts, np.full((len(pts), 1), 0.5, dtype=np.float32)])

        n_pts = len(pts)
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        header = (
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z intensity\n"
            "SIZE 4 4 4 4\n"
            "TYPE F F F F\n"
            "COUNT 1 1 1 1\n"
            f"WIDTH {n_pts}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {n_pts}\n"
            f"DATA {'binary' if binary else 'ascii'}\n"
        )

        with open(file_path, "wb") as f:
            f.write(header.encode("ascii"))
            if binary:
                f.write(pts.tobytes())
            else:
                for pt in pts:
                    f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {pt[3]:.4f}\n".encode("ascii"))


class SyntheticDatasetGenerator(DatasetLoader):
    """
    Streams realistic 64-beam synthetic LiDAR frames generated on the fly.
    Also provides batch pre-generation and file export capabilities.
    """

    def __init__(
        self,
        scene_id: str = "complex_environment",
        total_frames: int = 150,
        fps: int = 15,
    ) -> None:
        self.scene_id = scene_id
        self.total_frames = total_frames
        self.dt = 1.0 / float(fps)
        self.current_frame = 0

        self.world = SimulationWorld()
        load_scenario(self.world, self.scene_id)
        self.simulator = LidarSimulator()

        # Cache of generated frames for instant repeatability
        self._frame_cache: Dict[int, Tuple[np.ndarray, EgoVehicleState, List[Detection3D]]] = {}

    def get_frame(self, frame_id: int) -> np.ndarray:
        idx = frame_id % self.total_frames
        if idx not in self._frame_cache:
            self._advance_to_frame(idx)
        return self._frame_cache[idx][0]

    def get_ego_state(self, frame_id: int) -> Optional[EgoVehicleState]:
        idx = frame_id % self.total_frames
        if idx not in self._frame_cache:
            self._advance_to_frame(idx)
        return self._frame_cache[idx][1]

    def get_ground_truth_detections(self, frame_id: int) -> List[Detection3D]:
        idx = frame_id % self.total_frames
        if idx not in self._frame_cache:
            self._advance_to_frame(idx)
        return self._frame_cache[idx][2]

    def get_num_frames(self) -> int:
        return self.total_frames

    def get_dataset_name(self) -> str:
        return f"MI Sense Synthetic 64-Beam Simulator ({self.scene_id})"

    def reset(self) -> None:
        self.current_frame = 0
        load_scenario(self.world, self.scene_id)
        self._frame_cache.clear()

    def set_scene(self, scene_id: str) -> None:
        """Switches simulation scenario and clears frame cache."""
        self.scene_id = scene_id
        self.reset()

    def _advance_to_frame(self, target_frame: int) -> None:
        """Generates simulation state up to target_frame and caches result."""
        # If target frame is behind current simulation state, reset
        if target_frame < self.current_frame:
            load_scenario(self.world, self.scene_id)
            self.current_frame = 0

        while self.current_frame <= target_frame:
            ego_state = self.world.step(self.dt)
            cloud = self.simulator.generate_point_cloud(self.world)
            dets = self.world.get_ground_truth_detections()
            self._frame_cache[self.current_frame] = (cloud, ego_state, dets)
            self.current_frame += 1

    def export_sequence_to_disk(
        self,
        output_dir: str,
        num_frames: int = 30,
        file_format: str = "bin",  # "bin" or "pcd"
    ) -> List[str]:
        """
        Generates and saves a sequence of frames to disk.

        Args:
            output_dir: Target directory path.
            num_frames: Number of sequential frames to generate and export.
            file_format: "bin" (KITTI binary) or "pcd" (standard Point Cloud Data).

        Returns:
            List of generated file paths.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved_files = []

        self.reset()
        for f in range(num_frames):
            cloud = self.get_frame(f)
            if file_format == "pcd":
                dest = str(out_path / f"frame_{f:06d}.pcd")
                CustomPcdBinAdapter.save_frame_as_pcd(cloud, dest, binary=True)
            else:
                dest = str(out_path / f"frame_{f:06d}.bin")
                KittiDatasetAdapter.save_frame_as_kitti_bin(cloud, dest)
            saved_files.append(dest)

        return saved_files
