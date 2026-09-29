import { afterEach, expect, it, vi } from "vitest";
import * as THREE from "three";
import { actorSize, SceneModels } from "./sceneModels";
import type { LidarFrame, TrackedObject } from "../types";

const pools: SceneModels[] = [];
afterEach(() => {
  pools.forEach((pool) => pool.dispose());
  pools.length = 0;
});
function createModels(animate = true) {
  const pool = new SceneModels(animate);
  pools.push(pool);
  return pool;
}
function track(overrides: Partial<TrackedObject> = {}): TrackedObject {
  return {
    track_id: 1,
    class_name: "vehicle",
    position: [2, 10, 0.75],
    dimensions: [4.6, 2, 1.5],
    yaw: 0,
    velocity: [0, 2],
    speed: 2,
    heading: 0,
    dynamic_state: "DYNAMIC",
    age: 3,
    hits: 3,
    confidence: 0.9,
    history: [],
    ...overrides,
  };
}
function frame(tracks: TrackedObject[], revision = 0): LidarFrame {
  return {
    scene_id: "moving_vehicle",
    revision,
    frame_id: 1,
    timestamp: 1,
    ego_state: {
      position: [0, 0, 1.73],
      velocity: [0, 0, 0],
      speed: 0,
      yaw: 0,
      steering_angle: 0,
    },
    points: [],
    adaptive_cells: [],
    detections: [],
    tracks,
    terrain_features: [],
    metrics: {
      frame_id: 1,
      timestamp: 1,
      fps: 15,
      raw_points_count: 0,
      processed_points_count: 0,
      uniform_cells_count: 0,
      adaptive_cells_count: 0,
      cell_reduction_percent: 0,
      uniform_memory_kb: 0,
      adaptive_memory_kb: 0,
      memory_saved_percent: 0,
      fine_cells_count: 0,
      medium_cells_count: 0,
      coarse_cells_count: 0,
      active_tracks_count: tracks.length,
      dynamic_objects_count: tracks.length,
      processing_time_ms: {
        preprocessing: 0,
        detection: 0,
        tracking: 0,
        quadtree: 0,
        total: 0,
      },
    },
    detector_source: "Simulated",
  };
}
function actorRoot(pool: SceneModels) {
  return pool.group.children.find((item) => item.userData.trackId === 1)!;
}

it("fits a vehicle to the LiDAR length/width axes and centers its height on the reported box", () => {
  const pool = createModels();
  const actor = track();
  expect(actorSize(actor)).toEqual([2, 4.6, 1.5]);
  pool.update(frame([actor]), true, false);
  const root = actorRoot(pool);
  pool.group.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(root.children[0]);
  const size = bounds.getSize(new THREE.Vector3());
  expect(size.x).toBeCloseTo(2);
  expect(size.y).toBeCloseTo(4.6);
  expect(size.z).toBeCloseTo(1.5);
  expect(bounds.min.z).toBeCloseTo(0);
  expect(bounds.getCenter(new THREE.Vector3()).x).toBeCloseTo(2);
  const ray = new THREE.Raycaster(
    new THREE.Vector3(2, 10, 8),
    new THREE.Vector3(0, 0, -1),
  );
  expect(pool.pick(ray)).toEqual({ trackId: 1 });
});

it("shares geometry across replacements and releases shared resources exactly once", () => {
  const pool = createModels();
  pool.update(frame([track()]), true, false);
  const paint = actorRoot(pool).getObjectByName("traffic") as THREE.Mesh;
  const dispose = vi.spyOn(paint.geometry, "dispose");
  for (let revision = 1; revision <= 8; revision++) {
    pool.update(
      frame([track({ position: [2, 10 + revision, 0.75] })], revision),
      true,
      false,
    );
    expect(
      (actorRoot(pool).getObjectByName("traffic") as THREE.Mesh).geometry,
    ).toBe(paint.geometry);
    expect(pool.group.children).toHaveLength(2);
  }
  pool.update(frame([]), false, false);
  expect(pool.group.children).toHaveLength(1);
  expect(dispose).not.toHaveBeenCalled();
  pool.dispose();
  expect(dispose).toHaveBeenCalledTimes(1);
});

it("advances the walking pose only when the actor moves and respects reduced motion", () => {
  for (const animate of [true, false]) {
    const pool = createModels(animate);
    const pedestrian = track({
      class_name: "pedestrian",
      dimensions: [0.6, 0.6, 1.7],
      position: [2, 10, 0.85],
    });
    pool.update(frame([pedestrian]), true, false);
    const moved = frame([{ ...pedestrian, position: [2, 10.15, 0.85] }]);
    pool.update(moved, true, false);
    const leg = actorRoot(pool).getObjectByName("left-leg")!;
    const angle = leg.rotation.x;
    expect(Math.abs(angle) > 0).toBe(animate);
    pool.update(moved, true, false);
    expect(leg.rotation.x).toBe(angle);
  }
});
