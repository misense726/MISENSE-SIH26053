"""
Performance, memory efficiency, and compression evaluator for MI Sense.
Problem Statement: SIH 26053 - MI Sense Adaptive 2.5D LiDAR Mapping

Calculates live computational and memory efficiency metrics:
- Uniform vs. Adaptive cell counts (e.g., 65,536 uniform vs. ~4,000-8,000 adaptive).
- Cell reduction percentage: (1 - adaptive / uniform) * 100.
- Memory consumption in KB (Uniform: 48 bytes/cell, Adaptive: 56 bytes/cell).
- Memory savings percentage: (1 - adaptive_mem / uniform_mem) * 100.
- Granular resolution breakdown: fine (0.25m), medium (0.5m-1.0m), coarse (2.0m-4.0m).
- Stage latency tracking (preprocessing, detection, tracking, quadtree, total) and live FPS.
"""

import time
from typing import List, Dict, Any, Optional, Union
import numpy as np

from backend.app.config.settings import settings
from backend.app.models.schemas import FrameMetrics, AdaptiveCell, TrackedObject
from backend.app.mapping.quadtree import QuadNode


class PerformanceEvaluator:
    """
    Evaluates runtime performance and memory savings of adaptive quadtree representation
    versus a baseline uniform high-resolution 0.25m grid.
    """

    def __init__(
        self,
        roi_size: Optional[float] = None,
        uniform_res: Optional[float] = None,
        bytes_uniform_cell: Optional[int] = None,
        bytes_adaptive_cell: Optional[int] = None,
        **kwargs,
    ):
        self.roi_size = float(roi_size if roi_size is not None else settings.ROOT_GRID_SIZE)
        self.uniform_res = float(uniform_res if uniform_res is not None else settings.MIN_CELL_SIZE)
        self.bytes_per_uniform = int(bytes_uniform_cell if bytes_uniform_cell is not None else settings.BYTES_PER_UNIFORM_CELL)
        self.bytes_per_adaptive = int(bytes_adaptive_cell if bytes_adaptive_cell is not None else settings.BYTES_PER_ADAPTIVE_CELL)

        # Baseline uniform grid dimensions: 64m x 64m at 0.25m resolution = 256 x 256 cells = 65,536 cells
        cells_per_side = int(round(self.roi_size / self.uniform_res))
        self.uniform_cells_count = cells_per_side * cells_per_side
        self.uniform_memory_kb = round(
            (self.uniform_cells_count * self.bytes_per_uniform) / 1024.0, 2
        )

        self.last_timestamp = time.time()
        self.fps = float(settings.SIMULATION_FPS)

    def evaluate(
        self,
        frame_id: int,
        raw_points_count: int = 0,
        processed_points_count: int = 0,
        adaptive_cells: Optional[Union[List[AdaptiveCell], List[QuadNode]]] = None,
        tracks: Optional[List[TrackedObject]] = None,
        timings: Optional[Dict[str, float]] = None,
        processing_times_ms: Optional[Dict[str, float]] = None,
        current_fps: Optional[float] = None,
        timestamp: Optional[float] = None,
        **kwargs,
    ) -> FrameMetrics:
        """
        Compute frame metrics and memory savings.
        Supports all simulation engine and standalone benchmark calling conventions.
        """
        now = timestamp if timestamp is not None else time.time()

        if current_fps is not None and current_fps > 0:
            self.fps = round(float(current_fps), 1)
        else:
            dt = now - self.last_timestamp
            if dt > 0.001:
                instant_fps = 1.0 / dt
                self.fps = round(0.9 * self.fps + 0.1 * instant_fps, 1)
        self.last_timestamp = now

        adaptive_cells = adaptive_cells or []
        tracks = tracks or []
        adaptive_cells_count = len(adaptive_cells)

        # Count by resolution bins
        fine_count = 0
        medium_count = 0
        coarse_count = 0

        for c in adaptive_cells:
            size = getattr(c, "size", 1.0)
            level = getattr(c, "level", None)

            if level == 4 or size <= 0.25 + 1e-4:
                fine_count += 1
            elif level in (2, 3) or (0.25 < size <= 1.0 + 1e-4):
                medium_count += 1
            else:
                coarse_count += 1

        # Calculate memory
        adaptive_memory_kb = round((adaptive_cells_count * self.bytes_per_adaptive) / 1024.0, 2)

        # Percentages
        if self.uniform_cells_count > 0:
            cell_reduction_percent = 100.0 * (1.0 - (adaptive_cells_count / float(self.uniform_cells_count)))
            cell_reduction_percent = round(np.clip(cell_reduction_percent, 0.0, 100.0), 2)
        else:
            cell_reduction_percent = 0.0

        if self.uniform_memory_kb > 0:
            memory_saved_percent = 100.0 * (1.0 - (adaptive_memory_kb / float(self.uniform_memory_kb)))
            memory_saved_percent = round(np.clip(memory_saved_percent, 0.0, 100.0), 2)
        else:
            memory_saved_percent = 0.0

        # Track counts
        active_tracks_count = len(tracks)
        dynamic_objects_count = sum(
            1 for t in tracks
            if (getattr(t, "dynamic_state", "STATIC") == "DYNAMIC" or getattr(t, "speed", 0.0) >= settings.DYNAMIC_SPEED_THRESHOLD)
        )

        # Combine timings
        final_timings = {
            "preprocessing": 0.0,
            "detection": 0.0,
            "tracking": 0.0,
            "quadtree": 0.0,
            "total": 0.0,
        }
        if timings:
            final_timings.update(timings)
        if processing_times_ms:
            final_timings.update(processing_times_ms)

        if final_timings["total"] <= 0.0:
            final_timings["total"] = round(
                final_timings["preprocessing"] +
                final_timings["detection"] +
                final_timings["tracking"] +
                final_timings["quadtree"],
                2,
            )

        return FrameMetrics(
            frame_id=int(frame_id),
            timestamp=float(round(now, 4)),
            fps=float(self.fps),
            raw_points_count=int(raw_points_count),
            processed_points_count=int(processed_points_count),
            uniform_cells_count=int(self.uniform_cells_count),
            adaptive_cells_count=int(adaptive_cells_count),
            cell_reduction_percent=float(cell_reduction_percent),
            uniform_memory_kb=float(self.uniform_memory_kb),
            adaptive_memory_kb=float(adaptive_memory_kb),
            memory_saved_percent=float(memory_saved_percent),
            fine_cells_count=int(fine_count),
            medium_cells_count=int(medium_count),
            coarse_cells_count=int(coarse_count),
            active_tracks_count=int(active_tracks_count),
            dynamic_objects_count=int(dynamic_objects_count),
            processing_time_ms={
                k: float(round(v, 2)) for k, v in final_timings.items()
            },
        )

    def evaluate_frame(self, *args, **kwargs) -> FrameMetrics:
        """Alias for evaluate() to support alternative calling convention."""
        return self.evaluate(*args, **kwargs)


# Backward-compatible alias
MetricsEvaluator = PerformanceEvaluator
