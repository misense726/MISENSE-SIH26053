"""Capture the Python simulation as small, self-contained browser playback files."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.simulation.engine import SimulationEngine
from backend.app.simulation.scenarios import SCENARIOS_CATALOG


OUTPUT = ROOT / "frontend" / "public" / "demo"
FRAMES_PER_SECOND = 6
SCENES = {
    "complex_environment": 70,
    "pedestrian": 80,
    "moving_vehicle": 65,
    "pothole": 12,
    "curb_boundary": 12,
    "normal_road": 12,
}


def compact_frame(frame) -> dict:
    data = frame.model_dump()
    points = data["points"]
    stride = max(1, (len(points) + 1199) // 1200)
    return {
        "scene_id": data["scene_id"],
        "revision": data["revision"],
        "frame_id": data["frame_id"],
        "ego_state": data["ego_state"],
        "points": [
            [round(float(value), 3) for value in point]
            for point in points[::stride][:1200]
        ],
        "adaptive_cells": [
            {
                "cell_id": cell["cell_id"],
                "x": cell["x"],
                "y": cell["y"],
                "size": cell["size"],
                "elevation_mean": round(cell["elevation_mean"], 3),
                "point_count": cell["point_count"],
                "semantic_class": cell["semantic_class"],
                "dynamic_state": cell["dynamic_state"],
            }
            for cell in data["adaptive_cells"]
        ],
        "detections": data["detections"],
        "tracks": [
            {**track, "history": track["history"][-14:]}
            for track in data["tracks"]
        ],
        "terrain_features": data["terrain_features"],
        "metrics": {
            "memory_saved_percent": round(data["metrics"]["memory_saved_percent"], 1),
            "raw_points_count": data["metrics"]["raw_points_count"],
        },
    }


def export() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "kind": "SIMULATED",
        "default_scene": "complex_environment",
        "scenes": [],
    }
    for scene_id, count in SCENES.items():
        engine = SimulationEngine(scene_id)
        engine.playback_speed = 3.0
        frames = [compact_frame(engine.step_frame()) for _ in range(count)]
        payload = json.dumps(
            {"version": 1, "scene_id": scene_id, "fps": FRAMES_PER_SECOND, "frames": frames},
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        filename = f"{scene_id}.json.gz"
        (OUTPUT / filename).write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
        manifest["scenes"].append(
            {"id": scene_id, "name": SCENARIOS_CATALOG[scene_id].name,
             "frames": count, "file": filename}
        )
        print(f"{scene_id}: {count} frames, {len(payload):,} JSON bytes, "
              f"{(OUTPUT / filename).stat().st_size:,} gzip bytes", flush=True)
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    export()
