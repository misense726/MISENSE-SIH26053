"""
FastAPI REST and WebSocket Endpoints for MI Sense.
Problem Statement: SIH 26053

Endpoints:
- GET /api/health: Healthcheck, uptime, active scene.
- GET /api/scenes: List of all 6 scenarios with metadata and descriptions.
- GET /api/config: Current configuration and importance weights.
- POST /api/config/weights: Update importance weights dynamically.
- GET /api/architecture: Return detailed architecture graph data and node explanations.
- GET /api/explain: Return explanations for why resolution adapts.
- GET /api/jury-demo/steps: Return four-step guided demo metadata.
- WebSocket /ws/stream: Real-time streaming connection for frames and bi-directional controls.
"""

import time
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

from backend.app.config.settings import settings
from backend.app.models.schemas import SceneInfo, ControlMessage, ImportanceWeights
from backend.app.simulation.engine import simulation_engine
from backend.app.simulation.scenarios import get_all_scenarios_info

router = APIRouter()
start_time = time.time()


class WeightUpdateRequest(BaseModel):
    w1_distance: Optional[float] = None
    w2_elevation_var: Optional[float] = None
    w3_semantic: Optional[float] = None
    w4_object_prox: Optional[float] = None
    w5_motion: Optional[float] = None
    w6_terrain_complexity: Optional[float] = None
    w7_uncertainty: Optional[float] = None


@router.get("/health")
async def get_health() -> Dict[str, Any]:
    """Healthcheck and operational telemetry."""
    return {
        "status": "healthy",
        "session": simulation_engine.get_session_state(),
        "mapping": simulation_engine.get_mapping_metadata(),
        "model": {"inference_implemented": False, "model_loaded": False, "simulated": True},
        "uptime": round(time.time() - start_time, 1),
        "scene": simulation_engine.current_scene_id,
        "is_running": simulation_engine.is_running,
        "fps": simulation_engine.fps,
        "frame_id": simulation_engine.frame_id,
        "detector_source": simulation_engine.detector.detector_source,
        "active_clients": len(simulation_engine.active_connections),
    }


@router.get("/scenes", response_model=List[SceneInfo])
async def get_scenes() -> List[SceneInfo]:
    """List all 6 evaluation scenarios with detailed metadata."""
    return get_all_scenarios_info()


@router.get("/config")
async def get_config() -> Dict[str, Any]:
    """Return active system configuration and importance weights."""
    return {
        "session": simulation_engine.get_session_state(),
        **simulation_engine.get_mapping_metadata(),
        "weights": simulation_engine.get_weights(),
        "grid_resolutions": settings.GRID_RESOLUTIONS,
        "split_thresholds": {
            "L0_to_L1_4m_to_2m": settings.SPLIT_THRESHOLD_L0,
            "L1_to_L2_2m_to_1m": settings.SPLIT_THRESHOLD_L1,
            "L2_to_L3_1m_to_05m": settings.SPLIT_THRESHOLD_L2,
            "L3_to_L4_05m_to_025m": settings.SPLIT_THRESHOLD_L3,
        },
        "merge_hysteresis": settings.MERGE_HYSTERESIS,
        "fps": simulation_engine.fps,
        "playback_speed": simulation_engine.playback_speed,
        "adaptive_enabled": simulation_engine.adaptive_enabled,
        "detector_source": simulation_engine.detector.detector_source,
    }


@router.post("/config/weights")
async def update_weights(req: WeightUpdateRequest) -> Dict[str, Any]:
    """Dynamically update importance scoring weights."""
    new_weights = {k: v for k, v in req.model_dump().items() if v is not None}
    if not new_weights:
        raise HTTPException(status_code=400, detail="No weights provided")

    result = simulation_engine.handle_command(ControlMessage(action="update_weights", weights=new_weights))
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    result["updated_weights"] = simulation_engine.get_weights()
    return result


@router.post("/comparison")
async def comparison():
    try:
        return await simulation_engine.capture_comparison()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/architecture")
async def get_architecture() -> Dict[str, Any]:
    """Detailed pipeline architecture graph nodes and dataflow specifications."""
    return {
        "title": "MI Sense Adaptive 2.5D LiDAR Perception Architecture",
        "problem_statement": "SIH 26053",
        "nodes": [
            {
                "id": "sensor",
                "name": "64-Beam LiDAR Scanner",
                "stage": "Acquisition",
                "type": "Hardware / Sensor Simulation",
                "latency_budget": "N/A",
                "description": "Simulates 64 vertical beams (-25° to +15°) with 360° horizontal sweep. Generates ~30,000 raw points per frame.",
            },
            {
                "id": "preprocessor",
                "name": "LiDAR Preprocessor",
                "stage": "Preprocessing",
                "type": "NumPy / C++ Vectorized",
                "latency_budget": "< 8 ms",
                "description": "Applies vehicle ROI spatial cropping ([-32, 32]m lateral, [-16, 48]m longitudinal), RANSAC ground extraction, and WebGL downsampling.",
            },
            {
                "id": "detector",
                "name": "Simulated object detector",
                "stage": "Perception",
                "type": "Simulation with OpenPCDet output schema",
                "latency_budget": "< 25 ms",
                "description": "Produces simulated bounding boxes from scene actors. Trained-model inference is not implemented.",
            },
            {
                "id": "tracker",
                "name": "Kalman Filter + Hungarian Tracker",
                "stage": "Tracking",
                "type": "SciPy Munkres / 6-DOF Constant Velocity",
                "latency_budget": "< 5 ms",
                "description": "Maintains persistent object identities, estimates velocity [vx, vy], and classifies objects into DYNAMIC vs STATIC states.",
            },
            {
                "id": "elevation_map",
                "name": "2.5D Elevation Map Engine",
                "stage": "Mapping",
                "type": "Dynamic Point Filtering & Geometry",
                "latency_budget": "< 8 ms",
                "description": "Filters out dynamic object points to prevent motion blur and false obstacle trails. Computes mean, variance, min, max, and surface slope.",
            },
            {
                "id": "geometric_features",
                "name": "Geometric Hazard Detector",
                "stage": "Hazard Analysis",
                "type": "Depth & Step Analysis",
                "latency_budget": "< 4 ms",
                "description": "Identifies negative road surface depressions (potholes <= -8cm), positive steps (curbs >= 12cm), and surface inclines (slopes >= 5°).",
            },
            {
                "id": "quadtree",
                "name": "Adaptive Square Quadtree",
                "stage": "Multi-Resolution Representation",
                "type": "7-Factor Importance Tree",
                "latency_budget": "< 12 ms",
                "description": "Dynamically subdivides from Level 0 (4.0m) down to Level 4 (0.25m) based on weighted importance scoring (w1 to w7).",
            },
            {
                "id": "semantic_fusion",
                "name": "Semantic Fusion Engine",
                "stage": "Fusion",
                "type": "Multi-Modal Layer Integration",
                "latency_budget": "< 5 ms",
                "description": "Combines geometric elevation, terrain segmentation, object tracking state, and drivability flags into enriched AdaptiveCell models.",
            },
            {
                "id": "metrics_evaluator",
                "name": "Performance Evaluator",
                "stage": "Telemetry",
                "type": "Memory & Compression Profiler",
                "latency_budget": "< 1 ms",
                "description": "Estimates grid storage from cell counts against a uniform 0.25m 2.5D baseline. Does not measure process memory.",
            },
        ],
        "links": [
            {"source": "sensor", "target": "preprocessor", "label": "Raw 64-Beam Points (N, 4)"},
            {"source": "preprocessor", "target": "detector", "label": "Cropped ROI Points"},
            {"source": "detector", "target": "tracker", "label": "3D Bounding Boxes"},
            {"source": "tracker", "target": "elevation_map", "label": "Dynamic Tracks (Exclude Points)"},
            {"source": "preprocessor", "target": "elevation_map", "label": "Ground & Surface Points"},
            {"source": "elevation_map", "target": "geometric_features", "label": "Static Points & 2.5D Stats"},
            {"source": "geometric_features", "target": "quadtree", "label": "Hazard Coordinates (Potholes, Curbs)"},
            {"source": "tracker", "target": "quadtree", "label": "Object Trajectories & Velocities"},
            {"source": "elevation_map", "target": "quadtree", "label": "Elevation Variance & Slopes"},
            {"source": "quadtree", "target": "semantic_fusion", "label": "Adaptive Leaf Cells"},
            {"source": "semantic_fusion", "target": "metrics_evaluator", "label": "Enriched Map"},
        ],
    }


@router.get("/explain")
async def get_explainability() -> Dict[str, Any]:
    """Explanations of adaptive resolution mechanisms and scoring factors."""
    return {
        "title": "Why Resolution Adapts: 7-Factor Importance Model",
        "formula": "Importance Score S = sum(w_i * f_i) in [0.0, 1.0]",
        "weights": simulation_engine.get_weights(),
        "factors": [
            {
                "code": "w1_distance",
                "name": "Ego Vehicle Proximity",
                "description": "Linear proximity factor f1 = max(0, 1 - d / 40m). Nearby cells receive higher resolution to provide crisp immediate planning data.",
            },
            {
                "code": "w2_elevation_var",
                "name": "Elevation Variance & Surface Roughness",
                "description": "Evaluates vertical point variance f2 = min(1, variance / 0.04). Flat asphalt has near-zero variance; potholes and rubble yield high variance.",
            },
            {
                "code": "w3_semantic",
                "name": "Semantic Class Priority",
                "description": "Assigns categorical importance: potholes (0.95), curbs (0.85), slopes (0.70), road center (0.10). Ensures safety-critical features trigger fine cells.",
            },
            {
                "code": "w4_object_prox",
                "name": "Proximity to Detected 3D Objects",
                "description": "Evaluates distance to nearest tracked object bounding box. Increases resolution around vehicles, pedestrians, and obstacles.",
            },
            {
                "code": "w5_motion",
                "name": "Dynamic Motion Relevance",
                "description": "Activated for cells lying along the trajectory of DYNAMIC tracked actors, ensuring moving obstacles are faithfully monitored.",
            },
            {
                "code": "w6_terrain_complexity",
                "name": "Terrain Complexity & Gradient",
                "description": "Evaluates surface tilt angle f6 = min(1, slope_deg / 15°). Mild slopes adapt to 1m-0.5m; steep slopes adapt to 0.25m.",
            },
            {
                "code": "w7_uncertainty",
                "name": "Measurement Sparsity & Uncertainty",
                "description": "Accounts for sparse LiDAR returns at distance or oblique incidence angles, maintaining appropriate cell size.",
            },
        ],
        "subdivision_thresholds": {
            "Level 0 (4.0m) -> Level 1 (2.0m)": settings.SPLIT_THRESHOLD_L0,
            "Level 1 (2.0m) -> Level 2 (1.0m)": settings.SPLIT_THRESHOLD_L1,
            "Level 2 (1.0m) -> Level 3 (0.5m)": settings.SPLIT_THRESHOLD_L2,
            "Level 3 (0.5m) -> Level 4 (0.25m)": settings.SPLIT_THRESHOLD_L3,
            "Merge Hysteresis": settings.MERGE_HYSTERESIS,
        },
        "dynamic_point_exclusion": {
            "purpose": "Prevents moving vehicles/pedestrians from leaving ghost artifacts or smears in the 2.5D static road elevation map.",
            "rule": "If tracker classifies object as DYNAMIC (speed > 0.5 m/s and displacement > 1.0m), points inside its 3D box + 0.35m margin are excluded from elevation calculations.",
        },
    }


@router.get("/jury-demo/steps")
async def get_jury_demo_steps() -> List[Dict[str, Any]]:
    """Four manual steps. The client awaits scene revisions and restores state."""
    return [
        {"step": 1, "title": "See the scan", "scene_id": "normal_road", "message": "Downsampled browser display of the simulated LiDAR scan."},
        {"step": 2, "title": "Keep the height", "scene_id": "pothole", "message": "Inspect measured cell heights and the same frame with height removed."},
        {"step": 3, "title": "Spend detail where needed", "scene_id": "pothole", "message": "Compare uniform and adaptive 2.5D grids from one captured input."},
        {"step": 4, "title": "Handle movement", "scene_id": "pedestrian", "message": "Follow simulated detections with Kalman tracking and static-point filtering."},
    ]


@router.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket) -> None:
    """
    Real-time bidirectional WebSocket connection.
    Streams 15 Hz LidarFrame payloads to frontend and processes control messages.
    """
    await websocket.accept()
    simulation_engine.register_client(websocket)

    # Immediately push the latest frame so client has state without waiting
    if simulation_engine.last_frame is not None:
        try:
            await websocket.send_text(simulation_engine.last_frame.model_dump_json())
        except Exception:
            pass

    try:
        while True:
            # Receive control commands from frontend
            data = await websocket.receive_json()
            try:
                cmd = ControlMessage(**data)
                response = simulation_engine.handle_command(cmd)
                await websocket.send_json({"type": "command_response", "request_id": cmd.request_id, "result": response})
                for client in tuple(simulation_engine.active_connections):
                    if client is not websocket:
                        try:
                            await client.send_json({"type": "session_state", "state": response["state"]})
                        except Exception:
                            simulation_engine.unregister_client(client)
            except Exception as e:
                await websocket.send_json({"type": "error", "request_id": data.get("request_id"), "message": str(e)})

    except WebSocketDisconnect:
        simulation_engine.unregister_client(websocket)
    except Exception as e:
        simulation_engine.unregister_client(websocket)
