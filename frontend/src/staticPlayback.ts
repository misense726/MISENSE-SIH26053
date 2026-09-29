import type { LidarFrame } from "./types";

export interface StaticSceneInfo {
  id: string;
  name: string;
  frames: number;
  file: string;
}

export interface StaticManifest {
  version: number;
  kind: "SIMULATED";
  default_scene: string;
  scenes: StaticSceneInfo[];
}

export interface StaticRecording {
  version: number;
  scene_id: string;
  fps: number;
  frames: LidarFrame[];
}

const recordings = new Map<string, Promise<StaticRecording>>();

export function nextFrameIndex(index: number, count: number): number {
  return count > 0 ? (index + 1) % count : 0;
}

export async function loadManifest(): Promise<StaticManifest> {
  const response = await fetch("/demo/manifest.json");
  if (!response.ok) throw new Error("The simulation scenes could not load.");
  const manifest = (await response.json()) as StaticManifest;
  if (
    manifest.version !== 1 ||
    manifest.kind !== "SIMULATED" ||
    !manifest.scenes.length ||
    !manifest.scenes.some((scene) => scene.id === manifest.default_scene)
  ) {
    throw new Error("The simulation scene list is incomplete.");
  }
  return manifest;
}

export function loadRecording(scene: StaticSceneInfo): Promise<StaticRecording> {
  const cached = recordings.get(scene.id);
  if (cached) return cached;
  const promise = fetch(`/demo/${scene.file}`)
    .then(async (response) => {
      if (!response.ok || !response.body)
        throw new Error("This scene could not load. Try again.");
      let recording: StaticRecording;
      if (response.headers.get("content-encoding")?.includes("gzip")) {
        // Fetch has already decoded a gzip-encoded response.
        recording = (await response.json()) as StaticRecording;
      } else {
        if (typeof DecompressionStream === "undefined")
          throw new Error("This browser cannot open the simulation recording.");
        const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
        recording = (await new Response(stream).json()) as StaticRecording;
      }
      if (
        recording.version !== 1 ||
        recording.scene_id !== scene.id ||
        !Number.isFinite(recording.fps) ||
        recording.fps <= 0 ||
        recording.frames.length !== scene.frames
      ) {
        throw new Error("This scene recording is incomplete.");
      }
      return recording;
    })
    .catch((error: unknown) => {
      recordings.delete(scene.id);
      throw error;
    });
  recordings.set(scene.id, promise);
  return promise;
}
