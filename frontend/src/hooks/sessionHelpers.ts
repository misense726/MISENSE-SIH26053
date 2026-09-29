import type {
  AdaptiveCell,
  LayerVisibility,
  LidarFrame,
  MapPreset,
} from "../types";
import { cellIdentity } from "../visualization/palette";
export const TERRAIN_LAYERS: LayerVisibility = {
  surface: true,
  points: false,
  elevation: true,
  adaptiveGrid: true,
  semanticMap: true,
  detections: false,
  tracks: true,
  potholes: true,
  curbs: true,
  slopes: true,
};
export function layersForPreset(preset: MapPreset): LayerVisibility {
  return {
    ...TERRAIN_LAYERS,
    surface: preset !== "scan",
    points: preset === "scan",
    adaptiveGrid: preset !== "scan",
    potholes: preset === "terrain",
    curbs: preset === "terrain",
    slopes: preset === "terrain",
  };
}
export function resolveCell(
  frame: LidarFrame | null,
  identity: string | null,
): AdaptiveCell | null {
  return identity
    ? (frame?.adaptive_cells.find((cell) => cellIdentity(cell) === identity) ??
        null)
    : null;
}
export function hazardCell(frame: LidarFrame | null) {
  return (
    frame?.adaptive_cells
      .filter(
        (cell) =>
          cell.point_count > 0 &&
          ["pothole", "curb", "slope"].includes(cell.semantic_class),
      )
      .sort(
        (a, b) =>
          (a.semantic_class === "pothole" ? -1 : 0) -
            (b.semantic_class === "pothole" ? -1 : 0) ||
          a.elevation_mean - b.elevation_mean ||
          a.size - b.size,
      )[0] ?? null
  );
}
export const SCENE_LABELS: Record<string, string> = {
  normal_road: "Open road",
  pothole: "Pothole ahead",
  curb_boundary: "Curb & sidewalk",
  moving_vehicle: "Passing vehicle",
  pedestrian: "Pedestrian crossing",
  complex_environment: "Mixed terrain",
};
export const SCENE_DESCRIPTIONS: Record<string, string> = {
  normal_road: "A flat road with sparse traffic.",
  pothole: "A depression in the road surface.",
  curb_boundary: "Raised edges along the road.",
  moving_vehicle: "A moving vehicle beside the road.",
  pedestrian: "A pedestrian crosses the road ahead.",
  complex_environment: "Road hazards and moving objects in one scene.",
};
