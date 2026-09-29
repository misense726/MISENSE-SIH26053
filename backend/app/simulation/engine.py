"""
Master Simulation & Perception Pipeline Engine for MI Sense.
Problem Statement: SIH 26053

Coordinates the end-to-end perception loop at 15 Hz:
1. Steps the simulation world (ego vehicle + dynamic/static actors).
2. Generates 64-beam raw LiDAR point clouds.
3. Crops to ROI, downsamples, and classifies ground plane.
4. Performs 3D DSVT bounding box detection.
5. Updates Kalman Filter + Hungarian tracker, separating dynamic & static tracks.
6. Filters dynamic points to build pristine static 2.5D elevation models.
7. Detects geometric features (potholes, curbs, slopes).
8. Updates adaptive variable-resolution square quadtree (4m down to 0.25m).
9. Fuses semantic terrain classes, object metadata, and explainability factors.
10. Calculates real-time FPS, memory reduction, and compression metrics.
11. Streams LidarFrame payloads to all connected WebSocket clients.
"""

import asyncio
import logging
import math
import time
from typing import Dict, Any, Optional, Set, List
import numpy as np
from fastapi import WebSocket

from backend.app.config.settings import settings
from backend.app.models.schemas import (
    LidarFrame,
    ControlMessage,
    EgoVehicleState,
    FrameMetrics,
    ImportanceWeights,
    AdaptiveCell,
    TrackedObject,
    Detection3D,
    TerrainFeature,
    SessionState,
    ComparisonSnapshot,
)
from backend.app.simulation.world import SimulationWorld
from backend.app.simulation.scenarios import SCENARIOS_CATALOG
from backend.app.simulation.comparison import PipelineSnapshot, build_comparison
from backend.app.simulation.lidar_simulator import LidarSimulator
from backend.app.lidar.preprocessor import Preprocessor
from backend.app.perception.detector import SimulatedDSVTAdapter, OpenPCDetDSVTAdapter
from backend.app.perception.terrain_segmenter import SimulationTerrainSegmenter
from backend.app.tracking.tracker import MultiObjectTracker
from backend.app.mapping.elevation_map import ElevationMap25D
from backend.app.mapping.geometric_features import GeometricFeatureDetector
from backend.app.mapping.quadtree import AdaptiveQuadtree, ImportanceWeightsConfig
from backend.app.mapping.semantic_fusion import SemanticFusionEngine
from backend.app.metrics.evaluator import PerformanceEvaluator

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    Master perception engine executing the real-time processing pipeline.
    """

    def __init__(self, initial_scene_id: str = "normal_road"):
        self.fps = settings.SIMULATION_FPS
        self.playback_speed = 1.0
        self.is_running = True
        self.adaptive_enabled = True
        self.current_scene_id = initial_scene_id
        # Revisions identify scene resets and mapping configuration changes.
        # Play/pause/speed do not invalidate an already captured frame.
        self.revision = 0
        self._latest_snapshot: Optional[PipelineSnapshot] = None
        self._latest_comparison: Optional[ComparisonSnapshot] = None
        self._comparison_lock = asyncio.Lock()
        self._comparison_task: Optional[asyncio.Task] = None

        # Active layer visibility
        self.layers = {
            "points": True,
            "elevation": True,
            "adaptiveGrid": True,
            "semanticMap": True,
            "detections": True,
            "tracks": True,
            "potholes": True,
            "curbs": True,
            "slopes": True,
        }

        # WebSocket subscribers
        self.active_connections: Set[WebSocket] = set()

        # Frame counter & timing
        self.frame_id = 0
        self.last_frame: Optional[LidarFrame] = None
        self._loop_task: Optional[asyncio.Task] = None

        # Core perception modules
        self.world = SimulationWorld(initial_scene_id=self.current_scene_id)
        self.lidar_sim = LidarSimulator(
            num_beams=settings.LIDAR_BEAMS,
            range_max=settings.LIDAR_RANGE_MAX,
            noise_std=settings.LIDAR_NOISE_STD,
            dropout_rate=settings.LIDAR_DROPOUT_RATE,
        )
        self.preprocessor = Preprocessor()
        self.detector = OpenPCDetDSVTAdapter()
        self.terrain_segmenter = SimulationTerrainSegmenter()
        self.tracker = MultiObjectTracker()
        self.elevation_map = ElevationMap25D()
        self.geometric_detector = GeometricFeatureDetector()
        self.weights_config = ImportanceWeightsConfig()
        self.quadtree = AdaptiveQuadtree(weights=self.weights_config)
        self.semantic_fusion = SemanticFusionEngine(terrain_segmenter=self.terrain_segmenter)
        self.metrics_evaluator = PerformanceEvaluator()

    async def start(self) -> None:
        """Start the background 15 Hz simulation loop."""
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(self._run_loop())
            logger.info("SimulationEngine started background pipeline at %d Hz.", self.fps)

    async def stop(self) -> None:
        """Stop the background simulation loop."""
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
            logger.info("SimulationEngine background loop stopped.")

    async def _run_loop(self) -> None:
        """Continuous execution loop maintaining target frame rate."""
        frame_interval = 1.0 / float(self.fps)

        while True:
            cycle_start = time.perf_counter()

            if self.is_running:
                try:
                    frame = self.step_frame()
                    await self.broadcast_frame(frame)
                except Exception as e:
                    logger.exception("Error executing simulation frame %d: %s", self.frame_id, e)

            # Throttle loop to target FPS scaled by playback speed
            elapsed = time.perf_counter() - cycle_start
            sleep_time = max(0.001, frame_interval - elapsed)
            await asyncio.sleep(sleep_time)

    def step_frame(self) -> LidarFrame:
        """
        Execute one complete perception pipeline cycle.
        """
        t_total_start = time.perf_counter()
        dt = (1.0 / float(self.fps)) * self.playback_speed

        # 1. Step world simulation
        self.world.step(dt)
        ego_state = self.world.ego_state

        # 2. Generate raw 64-beam LiDAR point cloud
        raw_points = self.lidar_sim.generate_point_cloud(self.world)
        # The simulator returns sensor-relative z. Mapping uses road datum z=0.
        # Translate once so elevations, object boxes and feature markers agree.
        raw_points[:, 2] += self.lidar_sim.sensor_height

        # 3. Preprocess points (ROI crop, ground classification)
        t_pre_start = time.perf_counter()
        cropped_points = self.preprocessor.crop_roi(raw_points)
        t_pre = (time.perf_counter() - t_pre_start) * 1000.0

        # 4. 3D Object Detection (OpenPCDet / Simulated DSVT)
        t_det_start = time.perf_counter()
        actor_inputs = []
        for detection in self.world.get_ground_truth_detections():
            actor = detection.model_dump()
            actor["position"][2] += self.lidar_sim.sensor_height
            actor_inputs.append(actor)
        detections = self.detector.detect(cropped_points, ground_truth_actors=actor_inputs)
        t_det = (time.perf_counter() - t_det_start) * 1000.0

        # 5. Multi-Object Tracking (Kalman + Hungarian)
        t_trk_start = time.perf_counter()
        dynamic_tracks, static_tracks = self.tracker.update(detections, dt=dt)
        all_tracks = dynamic_tracks + static_tracks
        t_trk = (time.perf_counter() - t_trk_start) * 1000.0

        # 6. Exclude dynamic points from static 2.5D elevation mapping
        static_points = self.elevation_map.filter_dynamic_points(cropped_points, dynamic_tracks)

        # 7. Detect geometric features (potholes, curbs, slopes)
        terrain_features = self.geometric_detector.detect(
            static_points,
            scenario_features=self.world.hazards,
        )
        for feature in terrain_features:
            if feature.feature_type != "slope":
                feature.position[2] += self.lidar_sim.sensor_height
        self.terrain_segmenter.road_half_width = self.world.road.half_road_width

        # 8. Adaptive Square Quadtree Construction & Importance Scoring
        t_quad_start = time.perf_counter()
        leaf_nodes = self.quadtree.build(
            points=static_points,
            tracks=all_tracks,
            features=terrain_features,
            ego_pos=[0.0, 0.0, 0.0],
            adaptive_enabled=self.adaptive_enabled,
        )

        # 9. Semantic Fusion into Adaptive Cells
        adaptive_cells = self.semantic_fusion.fuse(
            leaf_nodes=leaf_nodes,
            points=static_points,
            tracks=all_tracks,
            features=terrain_features,
        )
        t_quad = (time.perf_counter() - t_quad_start) * 1000.0

        t_total = (time.perf_counter() - t_total_start) * 1000.0

        # 10. Performance & Memory Evaluation
        processing_times = {
            "preprocessing": round(t_pre, 2),
            "detection": round(t_det, 2),
            "tracking": round(t_trk, 2),
            "quadtree": round(t_quad, 2),
            "total": round(t_total, 2),
        }

        metrics = self.metrics_evaluator.evaluate(
            frame_id=self.frame_id,
            timestamp=time.time(),
            raw_points_count=len(raw_points),
            processed_points_count=len(static_points),
            adaptive_cells=adaptive_cells,
            tracks=all_tracks,
            processing_times_ms=processing_times,
        )

        # 11. Prepare lightweight points for WebGL rendering
        webgl_points = self.preprocessor.prepare_webgl_payload(raw_points, max_render_pts=4500)

        frame = LidarFrame(
            scene_id=self.current_scene_id,
            revision=self.revision,
            frame_id=self.frame_id,
            timestamp=time.time(),
            ego_state=ego_state,
            points=webgl_points,
            adaptive_cells=adaptive_cells,
            detections=detections,
            tracks=all_tracks,
            terrain_features=terrain_features,
            metrics=metrics,
            detector_source=self.detector.detector_source,
        )

        self._publish_snapshot(frame, static_points)
        return frame

    def _publish_snapshot(self, frame: LidarFrame, static_points: np.ndarray) -> None:
        # This synchronous assignment completes before a comparison can capture it.
        self._latest_snapshot = PipelineSnapshot.capture(
            frame, static_points, self.quadtree, self.metrics_evaluator,
            self.terrain_segmenter,
        )
        self.last_frame = frame
        self.frame_id += 1

    def _refresh_mapping(self) -> None:
        """Rebuild the same sensor input after tuning, even while paused."""
        snapshot = self._latest_snapshot
        if snapshot is None or self.last_frame is None:
            frame = self.step_frame()
        else:
            started = time.perf_counter()
            leaves = self.quadtree.build(
                points=snapshot.static_points,
                tracks=list(snapshot.tracks),
                features=list(snapshot.features),
                ego_pos=list(snapshot.ego_position),
                adaptive_enabled=self.adaptive_enabled,
            )
            cells = self.semantic_fusion.fuse(
                leaf_nodes=leaves, points=snapshot.static_points,
                tracks=list(snapshot.tracks), features=list(snapshot.features),
            )
            timings = dict(snapshot.metrics.processing_time_ms)
            timings["quadtree"] = (time.perf_counter() - started) * 1000.0
            timings["total"] = sum(timings[key] for key in (
                "preprocessing", "detection", "tracking", "quadtree",
            ))
            metrics = self.metrics_evaluator.evaluate(
                frame_id=self.frame_id, timestamp=snapshot.timestamp,
                raw_points_count=snapshot.metrics.raw_points_count,
                processed_points_count=len(snapshot.static_points),
                adaptive_cells=cells, tracks=list(snapshot.tracks),
                processing_times_ms=timings, current_fps=snapshot.metrics.fps,
            )
            frame = self.last_frame.model_copy(deep=True, update={
                "frame_id": self.frame_id, "revision": self.revision,
                "adaptive_cells": cells, "metrics": metrics,
            })
            self._publish_snapshot(frame, snapshot.static_points)
        self._safe_broadcast(frame)

    async def capture_comparison(self) -> ComparisonSnapshot:
        """Serialize builds, reuse duplicate snapshots, and retain only one result."""
        async with self._comparison_lock:
            task = self._comparison_task
            if task is None or task.done():
                snapshot = self._latest_snapshot
                if snapshot is None:
                    raise ValueError("No pipeline frame is available yet")
                cached = self._latest_comparison
                if cached is not None and cached.snapshot_id == snapshot.snapshot_id:
                    return cached.model_copy(deep=True)
                task = asyncio.create_task(self._build_comparison(snapshot))
                self._comparison_task = task
        # Concurrent requests join one worker instead of queuing snapshots.
        # Disconnecting a caller does not cancel the shared CPU work.
        result = await asyncio.shield(task)
        return result.model_copy(deep=True)

    async def _build_comparison(self, snapshot: PipelineSnapshot) -> ComparisonSnapshot:
        try:
            result = await asyncio.to_thread(build_comparison, snapshot)
            if result.revision == self.revision:
                self._latest_comparison = result
            return result
        finally:
            self._comparison_task = None

    def get_session_state(self) -> Dict[str, Any]:
        return SessionState(
            scene_id=self.current_scene_id, revision=self.revision,
            frame_id=self.last_frame.frame_id if self.last_frame else -1,
            is_running=self.is_running, playback_speed=self.playback_speed,
            adaptive_enabled=self.adaptive_enabled, weights=self.get_weights(),
            detector_source=self.detector.detector_source,
        ).model_dump()

    def get_mapping_metadata(self) -> Dict[str, Any]:
        tree = self.quadtree
        return {
            "roi": [tree.roi_x_min, tree.roi_x_max, tree.roi_y_min, tree.roi_y_max],
            "min_cell_size": tree.min_size,
            "max_cell_size": tree.max_size,
            "grid_resolutions": list(settings.GRID_RESOLUTIONS),
            "lidar_range_max": self.lidar_sim.range_max,
            "bytes_per_uniform_cell": self.metrics_evaluator.bytes_per_uniform,
            "bytes_per_adaptive_cell": self.metrics_evaluator.bytes_per_adaptive,
            "memory_unit": "KiB",
            "memory_measurement": "estimated_grid_storage",
            "memory_baseline": "uniform_2.5d",
        }

    async def broadcast_frame(self, frame: LidarFrame) -> None:
        """Stream frame to all active WebSocket clients."""
        if not self.active_connections:
            return

        payload = frame.model_dump_json()
        dead_connections = set()

        for ws in tuple(self.active_connections):
            try:
                await ws.send_text(payload)
            except Exception:
                dead_connections.add(ws)

        for dead in dead_connections:
            self.active_connections.discard(dead)

    def _safe_broadcast(self, frame: LidarFrame) -> None:
        """Schedules broadcast if an event loop is currently running."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.broadcast_frame(frame))
        except RuntimeError:
            pass  # No running event loop (e.g. running in synchronous tests)

    def handle_command(self, cmd: ControlMessage) -> Dict[str, Any]:
        """Every acknowledgement, including failures, carries authoritative state."""
        try:
            result = self._apply_command(cmd)
        except Exception as exc:
            logger.exception("Command %s failed", cmd.action)
            result = {"status": "error", "message": str(exc)}
        result.setdefault("action", cmd.action)
        result["state"] = self.get_session_state()
        return result

    def _invalidate_frame(self) -> None:
        self.revision += 1
        self.frame_id = 0
        self.last_frame = None
        self._latest_snapshot = None
        self._latest_comparison = None

    def _apply_command(self, cmd: ControlMessage) -> Dict[str, Any]:
        action = cmd.action

        if action == "play":
            self.is_running = True
            return {"status": "success", "action": "play", "is_running": True}

        elif action == "pause":
            self.is_running = False
            return {"status": "success", "action": "pause", "is_running": False}

        elif action == "step":
            self.is_running = False
            frame = self.step_frame()
            self._safe_broadcast(frame)
            return {"status": "success", "action": "step", "frame_id": frame.frame_id}

        elif action == "reset":
            self.world.reset()
            self.tracker.reset()
            self._invalidate_frame()
            frame = self.step_frame()
            self._safe_broadcast(frame)
            return {"status": "success", "action": "reset"}

        elif action == "set_scene":
            scene_id = cmd.scene_id
            if not scene_id:
                return {"status": "error", "message": "Missing scene_id"}
            if scene_id not in SCENARIOS_CATALOG:
                return {"status": "error", "message": f"Scene '{scene_id}' not found"}
            success = self.world.set_scenario(scene_id)
            if success:
                self.current_scene_id = scene_id
                self.tracker.reset()
                self._invalidate_frame()
                frame = self.step_frame()
                self._safe_broadcast(frame)
                return {"status": "success", "scene": scene_id}
            return {"status": "error", "message": f"Scene '{scene_id}' not found"}

        elif action == "set_speed":
            if cmd.speed is not None and math.isfinite(cmd.speed) and cmd.speed > 0:
                self.playback_speed = float(cmd.speed)
                return {"status": "success", "speed": self.playback_speed}
            return {"status": "error", "message": "Invalid speed value"}

        elif action == "update_weights":
            if not cmd.weights:
                return {"status": "error", "message": "No weights provided"}
            current = self.get_weights()
            if any(key not in current or not math.isfinite(value) or value < 0
                   for key, value in cmd.weights.items()):
                return {"status": "error", "message": "Weights must be known, finite, non-negative values"}
            if any(current[key] != value for key, value in cmd.weights.items()):
                self.weights_config.update(cmd.weights)
                self.revision += 1
                self._latest_comparison = None
                self._refresh_mapping()
            return {"status": "success", "weights": self.get_weights()}

        elif action == "toggle_layer":
            if cmd.layer and cmd.layer in self.layers:
                val = cmd.enabled if cmd.enabled is not None else not self.layers[cmd.layer]
                self.layers[cmd.layer] = val
                return {"status": "success", "layer": cmd.layer, "enabled": val}
            return {"status": "error", "message": "Unknown layer"}

        elif action == "set_adaptive_enabled":
            if cmd.enabled is not None:
                if self.adaptive_enabled != cmd.enabled:
                    self.adaptive_enabled = cmd.enabled
                    self.revision += 1
                    self._latest_comparison = None
                    self._refresh_mapping()
                return {"status": "success", "adaptive_enabled": self.adaptive_enabled}
            return {"status": "error", "message": "Missing enabled flag"}

        elif action == "jury_demo_step":
            step = cmd.step_number or 1
            return self.apply_jury_demo_step(step)

        return {"status": "error", "message": f"Unknown action: {action}"}

    def apply_jury_demo_step(self, step: int) -> Dict[str, Any]:
        """Configure scenario and view settings for specified Jury Demo step."""
        demo_map = {
            1: ("normal_road", "Step 1: Baseline Highway (Uniform vs Adaptive)"),
            2: ("pothole", "Step 2: Road Surface Hazard (Pothole 0.25m Cells)"),
            3: ("curb_boundary", "Step 3: Road Delineation (Curb Elevation Steps)"),
            4: ("moving_vehicle", "Step 4: Dynamic Object Tracking (Kalman + Hungarian)"),
            5: ("moving_vehicle", "Step 5: Dynamic Point Filtering (Ghost Elimination)"),
            6: ("pedestrian", "Step 6: Vulnerable Road Users (Pedestrian Corridor)"),
            7: ("complex_environment", "Step 7: Complex Multi-Hazard Benchmark"),
            8: ("complex_environment", "Step 8: Dynamic Weight Sensitivity Tuning"),
        }

        if step not in demo_map:
            step = 1

        scene_id, title = demo_map[step]
        self.world.set_scenario(scene_id)
        self.current_scene_id = scene_id
        self.tracker.reset()
        self._invalidate_frame()

        if step == 8:
            # Emphasize elevation variance & hazards in step 8
            self.weights_config.w2_elevation_var = 0.35
            self.weights_config.w3_semantic = 0.35
            self.weights_config.w1_distance = 0.10

        frame = self.step_frame()
        self._safe_broadcast(frame)

        return {
            "status": "success",
            "step": step,
            "title": title,
            "scene": scene_id,
        }

    def register_client(self, websocket: WebSocket) -> None:
        """Register a new WebSocket client for live telemetry streaming."""
        self.active_connections.add(websocket)
        logger.info("Client connected. Total clients: %d", len(self.active_connections))

    def unregister_client(self, websocket: WebSocket) -> None:
        """Unregister a disconnected WebSocket client."""
        self.active_connections.discard(websocket)
        logger.info("Client disconnected. Remaining clients: %d", len(self.active_connections))

    def get_weights(self) -> Dict[str, float]:
        """Return active importance weights dictionary."""
        w = self.weights_config
        return {
            "w1_distance": w.w1_distance,
            "w2_elevation_var": w.w2_elevation_var,
            "w3_semantic": w.w3_semantic,
            "w4_object_prox": w.w4_object_prox,
            "w5_motion": w.w5_motion,
            "w6_terrain_complexity": w.w6_terrain_complexity,
            "w7_uncertainty": w.w7_uncertainty,
        }

    def step(self, dt: Optional[float] = None) -> LidarFrame:
        """Alias for step_frame with optional dt."""
        return self.step_frame()

    def set_scene(self, scene_id: str) -> bool:
        """Alias for setting scenario scene."""
        return self.handle_command(ControlMessage(
            action="set_scene", scene_id=scene_id,
        ))["status"] == "success"

    def set_speed(self, speed: float) -> None:
        """Set simulation playback speed."""
        self.handle_command(ControlMessage(action="set_speed", speed=speed))

    def set_adaptive_enabled(self, enabled: bool) -> None:
        """Enable or disable adaptive variable-resolution quadtree."""
        self.handle_command(ControlMessage(action="set_adaptive_enabled", enabled=enabled))

    def update_weights(self, weights: Dict[str, float]) -> None:
        """Update quadtree importance weights."""
        self.handle_command(ControlMessage(action="update_weights", weights=weights))

    def toggle_layer(self, layer: str, enabled: Optional[bool] = None) -> None:
        """Toggle layer visibility."""
        if layer in self.layers:
            self.layers[layer] = enabled if enabled is not None else not self.layers[layer]

    def reset(self) -> None:
        """Reset world, tracker, and frame count."""
        self.handle_command(ControlMessage(action="reset"))


# Singleton engine instance and compatibility alias
simulation_engine = SimulationEngine()
engine = simulation_engine

