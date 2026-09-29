import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { MAP_FILLS } from "../visualization/palette";
import type { ComparisonSnapshot } from "../types";
import {
  canvasToWorld,
  cellIndexAtPoint,
  createComparisonProjection,
  fitComparisonCamera,
  MAX_COMPARISON_ZOOM,
  MIN_COMPARISON_ZOOM,
  panComparisonCamera,
  projectCell,
  worldToCanvas,
  zoomComparisonCamera,
} from "../visualization/comparisonProjection";
import type {
  ComparisonBounds,
  ComparisonCamera,
  ComparisonPoint,
} from "../visualization/comparisonProjection";

interface FourPanelComparisonProps {
  snapshot: ComparisonSnapshot | null;
  loading: boolean;
  error: string | null;
  onCapture: () => void;
  enabled: boolean;
}

type Cell = ComparisonSnapshot["uniform"][number];
type GridKind = "uniform" | "adaptive";
interface Selection {
  kind: GridKind;
  index: number;
  point: ComparisonPoint;
}

const GRID_NAMES = {
  uniform: "Uniform 2.5D",
  adaptive: "Adaptive 2.5D",
} as const;
const COLORS = MAP_FILLS;
const number = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });
const count = new Intl.NumberFormat();

function semanticGroup(cell: Cell): keyof typeof COLORS {
  if (cell[4] === 0) return "unknown";
  if (["pothole", "curb", "obstacle", "static_obstacle"].includes(cell[5]))
    return "obstacle";
  if (["dynamic", "vehicle", "pedestrian", "cyclist"].includes(cell[5]))
    return "dynamic";
  if (cell[5] === "road") return "road";
  if (["terrain", "slope", "rough_patch"].includes(cell[5])) return "terrain";
  return "unknown";
}

function semanticLabel(cell: Cell) {
  return cell[4] === 0
    ? "Unknown, no observations"
    : cell[5].replaceAll("_", " ");
}

function CellOutline({
  cell,
  projection,
  hover = false,
}: {
  cell: Cell;
  projection: ReturnType<typeof createComparisonProjection>;
  hover?: boolean;
}) {
  const rect = projectCell(cell, projection);
  return (
    <div
      aria-hidden="true"
      className={
        hover
          ? "comparison-cell-outline is-hovered"
          : "comparison-cell-outline is-selected"
      }
      style={{
        position: "absolute",
        pointerEvents: "none",
        left: rect.x,
        top: rect.y,
        width: rect.width,
        height: rect.height,
        boxSizing: "border-box",
        border: `${hover ? 2 : 3}px ${hover ? "dashed" : "solid"} #172B40`,
        outline: "1px solid white",
      }}
    />
  );
}

function ComparisonPanel({
  kind,
  cells,
  bounds,
  camera,
  onCameraChange,
  onSelect,
  selectedCell,
  textures,
  instructionsId,
}: {
  kind: GridKind;
  cells: readonly Cell[];
  bounds: ComparisonBounds;
  camera: ComparisonCamera;
  onCameraChange: (camera: ComparisonCamera) => void;
  onSelect: (selection: Selection) => void;
  selectedCell: Cell | undefined;
  textures: boolean;
  instructionsId: string;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{
    x: number;
    y: number;
    camera: ComparisonCamera;
    moved: boolean;
  } | null>(null);
  const [size, setSize] = useState({ width: 1, height: 1 });
  const [hoverIndex, setHoverIndex] = useState(-1);
  const [canvasUnavailable, setCanvasUnavailable] = useState(false);
  const projection = useMemo(
    () => createComparisonProjection(bounds, size.width, size.height, camera),
    [bounds, size, camera],
  );
  const hoveredCell = cells[hoverIndex];

  useEffect(() => {
    const element = wrapRef.current;
    if (!element) return;
    const measure = () => {
      const rect = element.getBoundingClientRect();
      setSize({
        width: Math.max(1, rect.width),
        height: Math.max(1, rect.height),
      });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) {
      setCanvasUnavailable(true);
      return;
    }
    const request = requestAnimationFrame(() => {
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.round(size.width * ratio);
      canvas.height = Math.round(size.height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.fillStyle = "#EDF2F6";
      context.fillRect(0, 0, size.width, size.height);

      const patterns: Partial<Record<keyof typeof COLORS, CanvasPattern>> = {};
      if (textures) {
        for (const group of ["obstacle", "dynamic"] as const) {
          const tile = document.createElement("canvas");
          tile.width = tile.height = 8;
          const ink = tile.getContext("2d");
          if (!ink) continue;
          ink.strokeStyle = "rgba(23,43,64,0.65)";
          ink.lineWidth = 1;
          ink.beginPath();
          for (const offset of [-8, 0, 8]) {
            ink.moveTo(offset, group === "obstacle" ? 8 : 0);
            ink.lineTo(offset + 8, group === "obstacle" ? 0 : 8);
          }
          ink.stroke();
          const pattern = context.createPattern(tile, "repeat");
          if (pattern) patterns[group] = pattern;
        }
      }

      // Every returned cell is considered. Only geometrically off-screen cells are culled.
      // At full extent this includes all 65,536 uniform cells, including unknown cells.
      for (const cell of cells) {
        const rect = projectCell(cell, projection);
        if (
          rect.x + rect.width < 0 ||
          rect.x > size.width ||
          rect.y + rect.height < 0 ||
          rect.y > size.height
        )
          continue;
        const group = semanticGroup(cell);
        context.fillStyle = COLORS[group];
        context.fillRect(rect.x, rect.y, rect.width, rect.height);
        const pattern = patterns[group];
        if (pattern) {
          context.fillStyle = pattern;
          context.fillRect(rect.x, rect.y, rect.width, rect.height);
        }
        // Suppress subpixel edge clutter, never the cell itself. Zoom reveals fine edges.
        if (rect.width >= 3) {
          context.strokeStyle = "rgba(23,43,64,0.26)";
          context.lineWidth = 0.6;
          context.strokeRect(rect.x, rect.y, rect.width, rect.height);
        }
      }
      const [left, top] = worldToCanvas([bounds[0], bounds[3]], projection);
      context.strokeStyle = "#526579";
      context.lineWidth = 1;
      context.strokeRect(
        left,
        top,
        (bounds[1] - bounds[0]) * projection.scale,
        (bounds[3] - bounds[2]) * projection.scale,
      );
    });
    return () => cancelAnimationFrame(request);
  }, [cells, bounds, projection, size, textures]);

  // A native non-passive listener prevents wheel zoom from also scrolling the page.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const delta =
        event.deltaY *
        (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? size.height : 1);
      onCameraChange(
        zoomComparisonCamera(
          projection,
          Math.exp(-Math.max(-120, Math.min(120, delta)) * 0.002),
          [event.clientX - rect.left, event.clientY - rect.top],
        ),
      );
      setHoverIndex(-1);
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [projection, onCameraChange, size.height]);

  const pointerPoint = (
    event: ReactPointerEvent<HTMLCanvasElement>,
  ): ComparisonPoint => {
    const rect = event.currentTarget.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  };
  const selectAt = (point: ComparisonPoint) => {
    const world = canvasToWorld(point, projection);
    const index = cellIndexAtPoint(cells, world);
    if (index >= 0) onSelect({ kind, index, point: world });
  };

  return (
    <article className="comparison-panel">
      <header className="comparison-label">
        <h2>{GRID_NAMES[kind]}</h2>
        <span>{count.format(cells.length)} cells</span>
      </header>
      <div
        className="comparison-canvas-wrap"
        ref={wrapRef}
        style={{ position: "relative", minHeight: 280, overflow: "hidden" }}
      >
        <canvas
          ref={canvasRef}
          className="comparison-canvas"
          data-cell-count={cells.length}
          role="img"
          aria-label={`${GRID_NAMES[kind]}, top-down view of all ${count.format(cells.length)} cells. Positive x is right, positive y is up. Cell values are available in the selected-area table.`}
          aria-describedby={instructionsId}
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            display: "block",
            touchAction: "none",
            cursor: "grab",
          }}
          onPointerDown={(event) => {
            if (event.button !== 0 || !event.isPrimary) return;
            const [x, y] = pointerPoint(event);
            drag.current = { x, y, camera, moved: false };
            event.currentTarget.setPointerCapture(event.pointerId);
          }}
          onPointerMove={(event) => {
            const point = pointerPoint(event);
            const active = drag.current;
            if (active) {
              const dx = point[0] - active.x;
              const dy = point[1] - active.y;
              if (Math.hypot(dx, dy) > 4) active.moved = true;
              if (active.moved) {
                onCameraChange(
                  panComparisonCamera(active.camera, dx, dy, projection.scale),
                );
                setHoverIndex(-1);
              }
            } else {
              setHoverIndex(
                cellIndexAtPoint(cells, canvasToWorld(point, projection)),
              );
            }
          }}
          onPointerUp={(event) => {
            const active = drag.current;
            drag.current = null;
            if (event.currentTarget.hasPointerCapture(event.pointerId))
              event.currentTarget.releasePointerCapture(event.pointerId);
            if (active && !active.moved) selectAt(pointerPoint(event));
          }}
          onPointerCancel={() => {
            drag.current = null;
            setHoverIndex(-1);
          }}
          onLostPointerCapture={() => {
            drag.current = null;
          }}
          onPointerLeave={() => setHoverIndex(-1)}
        >
          Use the selected-area table to inspect cell measurements.
        </canvas>
        {selectedCell && (
          <CellOutline cell={selectedCell} projection={projection} />
        )}
        {hoveredCell && (
          <>
            <CellOutline cell={hoveredCell} projection={projection} hover />
            <div
              className="comparison-tooltip"
              aria-hidden="true"
              style={{
                position: "absolute",
                left: 12,
                bottom: 12,
                maxWidth: "calc(100% - 24px)",
                pointerEvents: "none",
              }}
            >
              <strong>{number.format(hoveredCell[2])} m cell</strong>
              <span>
                {semanticLabel(hoveredCell)} ·{" "}
                {hoveredCell[4]
                  ? `${number.format(hoveredCell[3])} m elevation`
                  : "No measured elevation"}
              </span>
              <span>
                {count.format(hoveredCell[4])} points · Click to inspect
              </span>
            </div>
          </>
        )}
        {canvasUnavailable && (
          <p className="comparison-empty" role="status">
            Canvas2D is unavailable. Inspect the snapshot using the cell and
            memory tables below.
          </p>
        )}
      </div>
      <p className="comparison-caption">
        {kind === "uniform"
          ? "One fixed cell size across the captured region."
          : "Actual variable-size cells from the captured map."}{" "}
        Height is retained in the data; this top-down view uses semantic color.
      </p>
    </article>
  );
}

function MemoryComparison({ snapshot }: { snapshot: ComparisonSnapshot }) {
  const rows = [
    {
      kind: "uniform",
      label: GRID_NAMES.uniform,
      cells: snapshot.uniform.length,
      bytes: snapshot.bytes_per_uniform_cell,
    },
    {
      kind: "adaptive",
      label: GRID_NAMES.adaptive,
      cells: snapshot.adaptive.length,
      bytes: snapshot.bytes_per_adaptive_cell,
    },
  ].map((row) => ({ ...row, kib: (row.cells * row.bytes) / 1024 }));
  const maximum = Math.max(...rows.map((row) => row.kib), 1);
  const saving = rows[0].kib > 0 ? (1 - rows[1].kib / rows[0].kib) * 100 : null;
  return (
    <section
      className="memory-comparison"
      aria-label="Estimated grid storage comparison"
    >
      <h2>Estimated grid storage</h2>
      <p>
        {saving === null
          ? "No uniform baseline available."
          : saving >= 0
            ? `${number.format(saving)}% less estimated storage with adaptive cells.`
            : `${number.format(Math.abs(saving))}% more estimated storage with adaptive cells.`}{" "}
        Same captured region, one zero-based scale.
      </p>
      <div
        className="memory-chart"
        role="group"
        aria-label="Estimated grid storage in KiB, bars begin at zero"
      >
        {rows.map((row) => (
          <div
            key={row.kind}
            className="memory-bar-row"
            role="img"
            tabIndex={0}
            title={`${row.label}: ${count.format(row.cells)} cells × ${row.bytes} bytes ÷ 1024 = ${number.format(row.kib)} KiB`}
            aria-label={`${row.label}: ${number.format(row.kib)} estimated KiB, ${count.format(row.cells)} cells at ${row.bytes} bytes per cell`}
          >
            <div className="memory-bar-label">
              <span>{row.label}</span>
              <strong>{number.format(row.kib)} KiB</strong>
            </div>
            <div
              className="memory-bar-track"
              style={{ borderLeft: "1px solid #526579" }}
            >
              <div
                className="memory-bar"
                aria-hidden="true"
                style={{
                  width: `${(row.kib / maximum) * 100}%`,
                  height: 20,
                  backgroundColor: COLORS.road,
                  borderRadius: "0 4px 4px 0",
                }}
              />
            </div>
          </div>
        ))}
        <div
          className="memory-axis"
          aria-hidden="true"
          style={{ display: "flex", justifyContent: "space-between" }}
        >
          <span>0 KiB</span>
          <span>{number.format(maximum)} KiB</span>
        </div>
      </div>
      <details className="comparison-evidence">
        <summary>Storage values and assumptions</summary>
        <div className="evidence-table-wrap">
          <table className="evidence-table">
            <caption>
              Frozen snapshot storage estimates, 1 KiB = 1024 bytes
            </caption>
            <thead>
              <tr>
                <th scope="col">Representation</th>
                <th scope="col">Cells</th>
                <th scope="col">Bytes per cell</th>
                <th scope="col">Estimated KiB</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.kind}>
                  <th scope="row">{row.label}</th>
                  <td>{count.format(row.cells)}</td>
                  <td>{row.bytes}</td>
                  <td>{number.format(row.kib)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>
          Cell count × bytes per cell ÷ 1024. These are grid storage estimates,
          not measured heap or process memory. Point buffers, tracks and
          allocator overhead are excluded. The baseline is uniform 2.5D, not a
          3D voxel map.
        </p>
      </details>
    </section>
  );
}

function SnapshotWorkspace({ snapshot }: { snapshot: ComparisonSnapshot }) {
  const [camera, setCamera] = useState(() =>
    fitComparisonCamera(snapshot.bounds),
  );
  const [textures, setTextures] = useState(false);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [inspectionKind, setInspectionKind] = useState<GridKind>("adaptive");
  const instructionsId = useId();
  const selectedCells = useMemo(() => {
    if (!selection) return { uniform: undefined, adaptive: undefined };
    return {
      uniform:
        snapshot.uniform[cellIndexAtPoint(snapshot.uniform, selection.point)],
      adaptive:
        snapshot.adaptive[cellIndexAtPoint(snapshot.adaptive, selection.point)],
    };
  }, [snapshot, selection]);
  const inspectedCells = snapshot[inspectionKind];
  const inspectedIndex = selection
    ? cellIndexAtPoint(inspectedCells, selection.point)
    : -1;
  const chooseCell = (index: number, kind = inspectionKind) => {
    const cells = snapshot[kind];
    const boundedIndex = Math.min(
      cells.length - 1,
      Math.max(0, Math.trunc(index)),
    );
    const cell = cells[boundedIndex];
    if (!cell) return;
    setSelection({ kind, index: boundedIndex, point: [cell[0], cell[1]] });
  };
  const selectFromCanvas = (next: Selection) => {
    setInspectionKind(next.kind);
    setSelection(next);
  };
  const zoom = (factor: number) =>
    setCamera((current) => ({
      ...current,
      zoom: Math.max(
        MIN_COMPARISON_ZOOM,
        Math.min(MAX_COMPARISON_ZOOM, current.zoom * factor),
      ),
    }));

  return (
    <div
      className="comparison-snapshot"
      data-snapshot-id={snapshot.snapshot_id}
    >
      <p className="comparison-identity">
        Frozen snapshot · {snapshot.scene_id.replaceAll("_", " ")} · frame{" "}
        {count.format(snapshot.frame_id)} · revision {snapshot.revision} ·{" "}
        {new Date(snapshot.timestamp * 1000).toLocaleTimeString()}
      </p>
      <div
        className="compare-toolbar"
        role="group"
        aria-label="Linked comparison view controls"
      >
        <button
          type="button"
          onClick={() => zoom(1 / 1.5)}
          disabled={camera.zoom <= MIN_COMPARISON_ZOOM}
          aria-label="Zoom out both maps"
        >
          Zoom out
        </button>
        <span className="comparison-zoom" aria-live="polite">
          {number.format(camera.zoom)}×
        </span>
        <button
          type="button"
          onClick={() => zoom(1.5)}
          disabled={camera.zoom >= MAX_COMPARISON_ZOOM}
          aria-label="Zoom in both maps"
        >
          Zoom in
        </button>
        <button
          type="button"
          onClick={() => setCamera(fitComparisonCamera(snapshot.bounds))}
        >
          Reset both views
        </button>
        <label className="comparison-texture-toggle">
          <input
            type="checkbox"
            checked={textures}
            onChange={(event) => setTextures(event.target.checked)}
          />{" "}
          Cell textures
        </label>
      </div>
      <p id={instructionsId} className="comparison-instructions">
        Drag either map to pan both. Scroll to zoom, or use the linked buttons.
        Click a cell to compare the same location. Keyboard users can inspect
        every cell with the controls below.
      </p>
      <div
        className="comparison-legend"
        role="group"
        aria-label="Shared semantic legend"
      >
        {(
          [
            ["road", "Road"],
            ["obstacle", "Pothole / curb / obstacle"],
            ["dynamic", "Dynamic object"],
            ["terrain", "Terrain / slope"],
            ["unknown", "Unknown / no data"],
          ] as const
        ).map(([group, label]) => (
          <span key={group}>
            <i
              aria-hidden="true"
              style={{
                display: "inline-block",
                width: 12,
                height: 12,
                backgroundColor: COLORS[group],
                marginRight: 6,
              }}
            />
            {label}
          </span>
        ))}
      </div>
      <div className="compare-pair">
        {(["uniform", "adaptive"] as const).map((kind) => (
          <ComparisonPanel
            key={kind}
            kind={kind}
            cells={snapshot[kind]}
            bounds={snapshot.bounds}
            camera={camera}
            onCameraChange={setCamera}
            onSelect={selectFromCanvas}
            selectedCell={selectedCells[kind]}
            textures={textures}
            instructionsId={instructionsId}
          />
        ))}
      </div>
      <p className="comparison-roi">
        Shared region · x {number.format(snapshot.bounds[0])} to{" "}
        {number.format(snapshot.bounds[1])} m · y{" "}
        {number.format(snapshot.bounds[2])} to{" "}
        {number.format(snapshot.bounds[3])} m · uniform cell size{" "}
        {number.format(snapshot.min_cell_size)} m. Positive y points up in both
        views.
      </p>
      <section
        className="comparison-selection"
        aria-label="Selected area measurements"
      >
        <h2>Inspect the same location</h2>
        <div className="compare-toolbar">
          <label>
            Grid{" "}
            <select
              value={inspectionKind}
              onChange={(event) =>
                setInspectionKind(event.target.value as GridKind)
              }
            >
              <option value="uniform">Uniform 2.5D</option>
              <option value="adaptive">Adaptive 2.5D</option>
            </select>
          </label>
          <label>
            Cell number{" "}
            <input
              type="number"
              min={1}
              max={Math.max(1, inspectedCells.length)}
              step={1}
              value={inspectedIndex < 0 ? "" : inspectedIndex + 1}
              placeholder="Choose a cell"
              disabled={!inspectedCells.length}
              onChange={(event) => {
                if (
                  event.target.value &&
                  Number.isFinite(event.target.valueAsNumber)
                )
                  chooseCell(event.target.valueAsNumber - 1);
              }}
            />
          </label>
          <span>of {count.format(inspectedCells.length)}</span>
          <button
            type="button"
            disabled={inspectedIndex <= 0}
            onClick={() => chooseCell(inspectedIndex - 1)}
          >
            Previous cell
          </button>
          <button
            type="button"
            disabled={
              !inspectedCells.length ||
              inspectedIndex >= inspectedCells.length - 1
            }
            onClick={() => chooseCell(inspectedIndex + 1)}
          >
            Next cell
          </button>
          <button
            type="button"
            disabled={!selection}
            onClick={() => {
              if (selection)
                setCamera((current) => ({
                  ...current,
                  x: selection.point[0],
                  y: selection.point[1],
                  zoom: Math.max(4, current.zoom),
                }));
            }}
          >
            Center selected area
          </button>
        </div>
        {selection ? (
          <div className="evidence-table-wrap">
            <table className="evidence-table">
              <caption>
                Cells containing x {number.format(selection.point[0])} m, y{" "}
                {number.format(selection.point[1])} m. Heights are measured
                means, without vertical exaggeration.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Grid</th>
                  <th scope="col">Center x / y, m</th>
                  <th scope="col">Size, m</th>
                  <th scope="col">Height, m</th>
                  <th scope="col">Points</th>
                  <th scope="col">Semantic label</th>
                  <th scope="col">Drivable</th>
                </tr>
              </thead>
              <tbody>
                {(["uniform", "adaptive"] as const).map((kind) => {
                  const cell = selectedCells[kind];
                  return (
                    <tr key={kind}>
                      <th scope="row">{GRID_NAMES[kind]}</th>
                      {cell ? (
                        <>
                          <td>
                            {number.format(cell[0])} / {number.format(cell[1])}
                          </td>
                          <td>{number.format(cell[2])}</td>
                          <td>
                            {cell[4] > 0
                              ? number.format(cell[3])
                              : "Not observed"}
                          </td>
                          <td>{count.format(cell[4])}</td>
                          <td>{semanticLabel(cell)}</td>
                          <td>
                            {cell[4] > 0 ? (cell[6] ? "Yes" : "No") : "Unknown"}
                          </td>
                        </>
                      ) : (
                        <td colSpan={6}>No cell at this location</td>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="comparison-empty">
            Select a cell on either map or enter a cell number. Both grids will
            show their measurements at that location.
          </p>
        )}
      </section>
      <MemoryComparison snapshot={snapshot} />
    </div>
  );
}

export function FourPanelComparison({
  snapshot,
  loading,
  error,
  onCapture,
  enabled,
}: FourPanelComparisonProps) {
  return (
    <main className="compare-workspace" aria-label="Frozen grid comparison">
      <header className="compare-heading">
        <div>
          <h1>Same scan. Different cell budgets.</h1>
          <p>
            Compare uniform and adaptive 2.5D grids built from one captured
            input, across the full region.
          </p>
        </div>
        <button
          type="button"
          onClick={onCapture}
          disabled={!enabled || loading}
        >
          {loading
            ? "Capturing snapshot…"
            : snapshot
              ? "Recapture snapshot"
              : "Capture comparison"}
        </button>
      </header>
      <div className="comparison-status" role="status" aria-live="polite">
        {loading
          ? snapshot
            ? "Capturing a new snapshot. The previous frozen snapshot remains visible until it is ready."
            : "Building both grids from the captured input. This may take a moment."
          : !enabled
            ? snapshot
              ? "Disconnected. Showing the last frozen snapshot; recapture is unavailable."
              : "Connect to the simulation to capture a comparison."
            : snapshot
              ? "Frozen for inspection. Live playback does not change these maps or values."
              : "Capture a snapshot to compare real cells. No sample or illustrative data is shown."}
      </div>
      {error && (
        <p className="comparison-error" role="alert">
          {error}
          {snapshot ? " The previous frozen snapshot is still shown." : ""}
        </p>
      )}
      {snapshot ? (
        <SnapshotWorkspace key={snapshot.snapshot_id} snapshot={snapshot} />
      ) : (
        <div className="comparison-empty">
          <p>No comparison captured yet.</p>
          <p>
            The capture includes the full uniform baseline, adaptive cells and
            their estimated storage.
          </p>
        </div>
      )}
    </main>
  );
}
