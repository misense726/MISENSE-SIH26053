import { Color } from "three";
import type { AdaptiveCell } from "../types";

export type MapViewPreset = "terrain" | "resolution" | "scan";
export type MapCameraPreset = "angled" | "top" | "focus";

// Labels, cell boundaries and object shapes accompany categorical colors.
export const MAP_PALETTE = {
  background: "#EDF2F6",
  ink: "#172B40",
  muted: "#526579",
  grid: "#9CAFBF",
  road: "#2a78d6",
  terrain: "#A1B2BF",
  obstacle: "#eb6834",
  dynamic: "#1baf7a",
  unknown: "#E2E8ED",
  selection: "#2858C7",
} as const;

// Ordered blue ramp, with explicit cell sizes in the legend.
// Fine cells are dark. These are backend cell sizes, not distance bands.
export const RESOLUTION_LEGEND = [
  { size: 0.25, color: "#104574" },
  { size: 0.5, color: "#195c93" },
  { size: 1, color: "#2675b4" },
  { size: 2, color: "#488fce" },
  { size: 4, color: "#78a8dd" },
] as const;

export const MAP_FILLS = {
  road: "#9BBCDF",
  obstacle: "#DE865D",
  dynamic: "#307F63",
  terrain: "#C0CBD4",
  unknown: "#EDF2F6",
} as const;
const roadFill = new Color(MAP_FILLS.road);
const obstacleFill = new Color(MAP_FILLS.obstacle);
const dynamicFill = new Color(MAP_FILLS.dynamic);
const neutralFill = new Color(MAP_FILLS.terrain);
const unknownFill = new Color(MAP_FILLS.unknown);
const resolutionColors = RESOLUTION_LEGEND.map(({ color }) => new Color(color));

export function cellIdentity(
  cell: Pick<AdaptiveCell, "x" | "y" | "size">,
): string {
  // Node IDs can be reassigned during subdivision; physical bounds are stable.
  return `${cell.x}:${cell.y}:${cell.size}`;
}

export function cellLabel(cell: AdaptiveCell): string {
  if (cell.point_count === 0) return "Unknown · no returns";
  if (cell.dynamic_state === "DYNAMIC") return "Moving object";
  return cell.semantic_class.replaceAll("_", " ");
}

export function cellColor(
  cell: AdaptiveCell,
  view: MapViewPreset,
  semantic: boolean,
): Color {
  if (cell.point_count === 0) return unknownFill;
  if (view === "resolution") {
    const index = RESOLUTION_LEGEND.findIndex(({ size }) => cell.size <= size);
    return resolutionColors[index < 0 ? resolutionColors.length - 1 : index];
  }
  if (!semantic) return neutralFill;
  if (cell.dynamic_state === "DYNAMIC") return dynamicFill;
  if (cell.semantic_class === "road") return roadFill;
  if (["pothole", "curb", "slope", "obstacle"].includes(cell.semantic_class))
    return obstacleFill;
  if (cell.semantic_class === "unknown") return unknownFill;
  return neutralFill;
}

export function measuredHeight(cell: AdaptiveCell, elevation: boolean): number {
  return elevation &&
    cell.point_count > 0 &&
    Number.isFinite(cell.elevation_mean)
    ? cell.elevation_mean
    : 0;
}
