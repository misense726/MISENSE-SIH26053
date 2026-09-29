/** Bounds are [xmin, xmax, ymin, ymax]; cells are positioned by their center. */
export type ComparisonBounds = readonly [number, number, number, number];
export type ComparisonPoint = readonly [number, number];
export type CellGeometry = readonly [number, number, number, ...unknown[]];

export interface ComparisonCamera {
  x: number;
  y: number;
  zoom: number;
}

export interface ComparisonProjection {
  width: number;
  height: number;
  scale: number;
  camera: ComparisonCamera;
}

export const MIN_COMPARISON_ZOOM = 1;
export const MAX_COMPARISON_ZOOM = 32;

export function fitComparisonCamera(
  bounds: ComparisonBounds,
): ComparisonCamera {
  return {
    x: (bounds[0] + bounds[1]) / 2,
    y: (bounds[2] + bounds[3]) / 2,
    zoom: MIN_COMPARISON_ZOOM,
  };
}

export function createComparisonProjection(
  bounds: ComparisonBounds,
  width: number,
  height: number,
  camera: ComparisonCamera = fitComparisonCamera(bounds),
  padding = 16,
): ComparisonProjection {
  const safeWidth = Math.max(1, width);
  const safeHeight = Math.max(1, height);
  const availableWidth = Math.max(1, safeWidth - padding * 2);
  const availableHeight = Math.max(1, safeHeight - padding * 2);
  const fitScale = Math.min(
    availableWidth / Math.max(Number.EPSILON, bounds[1] - bounds[0]),
    availableHeight / Math.max(Number.EPSILON, bounds[3] - bounds[2]),
  );
  return {
    width: safeWidth,
    height: safeHeight,
    scale: fitScale * camera.zoom,
    camera,
  };
}

export function worldToCanvas(
  point: ComparisonPoint,
  projection: ComparisonProjection,
): ComparisonPoint {
  return [
    projection.width / 2 + (point[0] - projection.camera.x) * projection.scale,
    projection.height / 2 - (point[1] - projection.camera.y) * projection.scale,
  ];
}

export function canvasToWorld(
  point: ComparisonPoint,
  projection: ComparisonProjection,
): ComparisonPoint {
  return [
    projection.camera.x + (point[0] - projection.width / 2) / projection.scale,
    projection.camera.y - (point[1] - projection.height / 2) / projection.scale,
  ];
}

export function cellEdges(cell: CellGeometry): ComparisonBounds {
  const half = cell[2] / 2;
  return [cell[0] - half, cell[0] + half, cell[1] - half, cell[1] + half];
}

export function projectCell(
  cell: CellGeometry,
  projection: ComparisonProjection,
) {
  const [xmin, , , ymax] = cellEdges(cell);
  const [x, y] = worldToCanvas([xmin, ymax], projection);
  const size = cell[2] * projection.scale;
  return { x, y, width: size, height: size };
}

/** Half-open edges assign a shared boundary to exactly one cell. */
export function cellIndexAtPoint(
  cells: readonly CellGeometry[],
  point: ComparisonPoint,
): number {
  return cells.findIndex((cell) => {
    const [xmin, xmax, ymin, ymax] = cellEdges(cell);
    return (
      point[0] >= xmin && point[0] < xmax && point[1] >= ymin && point[1] < ymax
    );
  });
}

export function panComparisonCamera(
  camera: ComparisonCamera,
  dx: number,
  dy: number,
  scale: number,
): ComparisonCamera {
  return { ...camera, x: camera.x - dx / scale, y: camera.y + dy / scale };
}

/** Keep the world point under the cursor fixed when zooming either linked view. */
export function zoomComparisonCamera(
  projection: ComparisonProjection,
  factor: number,
  anchor: ComparisonPoint = [projection.width / 2, projection.height / 2],
): ComparisonCamera {
  const camera = projection.camera;
  const zoom = Math.max(
    MIN_COMPARISON_ZOOM,
    Math.min(MAX_COMPARISON_ZOOM, camera.zoom * factor),
  );
  const [worldX, worldY] = canvasToWorld(anchor, projection);
  const scale = projection.scale * (zoom / camera.zoom);
  return {
    x: worldX - (anchor[0] - projection.width / 2) / scale,
    y: worldY + (anchor[1] - projection.height / 2) / scale,
    zoom,
  };
}
