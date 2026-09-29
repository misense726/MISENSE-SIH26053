import { MAP_FILLS } from "../visualization/palette";
import { X, Crosshair, ArrowUpRight } from "lucide-react";
import type { AdaptiveCell, LidarFrame, TrackedObject } from "../types";

interface Props {
  selectedCell: AdaptiveCell | null;
  selectedObject: TrackedObject | null;
  frame: LidarFrame | null;
  onClearSelection: () => void;
  onFocus: () => void;
}
const formatHeight = (value: number) =>
  `${value > 0 ? "+" : ""}${(value * 100).toFixed(1)} cm`;

export function RightInspectorPanel({
  selectedCell: cell,
  selectedObject: object,
  frame,
  onClearSelection,
  onFocus,
}: Props) {
  if (!cell && !object) return null;
  const observed = !!cell?.point_count;
  const reason = cell?.split_reason
    .replace(
      /Pothole detected \(depth:.*?\)/,
      "Scenario pothole overlaps this cell",
    )
    .replace(
      /Curb edge boundary \(step:.*?\)/,
      "Scenario curb boundary overlaps this cell",
    )
    .replace(
      /Terrain slope gradient \(.*?\)/,
      "Scenario slope overlaps this cell",
    );
  const radius = Math.max(2, (cell?.size ?? 1) * 1.5);
  const nearby = cell
    ? (frame?.adaptive_cells.filter(
        (item) =>
          Math.abs(item.x - cell.x) < radius + item.size / 2 &&
          Math.abs(item.y - cell.y) < radius + item.size / 2,
      ) ?? [])
    : [];
  const profile = cell
    ? nearby
        .filter((item) => Math.abs(item.y - cell.y) <= item.size / 2)
        .sort((a, b) => a.x - b.x)
    : [];
  const heights = profile
    .filter((item) => item.point_count > 0)
    .map((item) => item.elevation_mean);
  const extent = Math.max(0.05, ...heights.map(Math.abs));
  return (
    <aside className="selection-inspector" aria-label="Selected map item">
      <div className="inspector-top">
        <span className="eyebrow">
          {cell ? "DETAIL LENS" : "TRACKED OBJECT"}
        </span>
        <button
          className="button icon-button"
          onClick={onClearSelection}
          aria-label="Close selection"
        >
          <X size={17} />
        </button>
      </div>
      <h2>
        {cell
          ? observed
            ? cell.semantic_class.replaceAll("_", " ")
            : "Unobserved cell"
          : object!.class_name.replaceAll("_", " ")}
      </h2>
      {cell && (
        <>
          <div className="inspector-measurement">
            <strong>
              {cell.size < 1
                ? `${Math.round(cell.size * 100)} cm`
                : `${cell.size} m`}
            </strong>
            <span>cell width</span>
          </div>
          <dl className="inspector-stats">
            <div>
              <dt>Map elevation</dt>
              <dd>
                {observed ? formatHeight(cell.elevation_mean) : "No returns"}
              </dd>
            </div>
            <div>
              <dt>Surface</dt>
              <dd>
                {!observed
                  ? "Unknown"
                  : cell.drivable
                    ? "Marked drivable"
                    : "Not drivable"}
              </dd>
            </div>
          </dl>

          <div className="detail-lens">
            <svg
              viewBox="0 0 240 150"
              role="img"
              aria-label={`Magnified ${radius * 2} metre region around the selected cell`}
            >
              {nearby.map((item) => (
                <rect
                  key={item.cell_id}
                  x={120 + ((item.x - item.size / 2 - cell.x) * 120) / radius}
                  y={75 - ((item.y + item.size / 2 - cell.y) * 120) / radius}
                  width={(item.size * 120) / radius}
                  height={(item.size * 120) / radius}
                  fill={
                    item.cell_id === cell.cell_id
                      ? "#2858C7"
                      : !item.point_count
                        ? MAP_FILLS.unknown
                        : item.semantic_class === "road"
                          ? MAP_FILLS.road
                          : ["pothole", "curb", "slope", "obstacle"].includes(
                                item.semantic_class,
                              )
                            ? MAP_FILLS.obstacle
                            : MAP_FILLS.terrain
                  }
                  stroke={item.cell_id === cell.cell_id ? "#172B40" : "#FFFFFF"}
                  strokeWidth={1.3}
                />
              ))}
            </svg>
            <span>
              {(radius * 2).toFixed(0)} m wide · actual cell boundaries
            </span>
          </div>
          <div className="height-profile">
            <svg
              viewBox="0 0 240 65"
              role="img"
              aria-label="Measured height profile across the selected row. Gaps indicate no returns."
            >
              <line
                x1="0"
                y1="32"
                x2="240"
                y2="32"
                stroke="#9CAFBF"
                strokeDasharray="3 3"
              />
              {profile.map((item) => {
                const x =
                  120 + ((item.x - item.size / 2 - cell.x) * 120) / radius;
                const y = 32 - (item.elevation_mean / extent) * 26;
                return item.point_count > 0 ? (
                  <line
                    key={item.cell_id}
                    x1={x}
                    x2={x + (item.size * 120) / radius}
                    y1={y}
                    y2={y}
                    stroke="#2858C7"
                    strokeWidth="3"
                  >
                    <title>
                      {formatHeight(item.elevation_mean)}, {item.size} m cell
                    </title>
                  </line>
                ) : (
                  <text
                    key={item.cell_id}
                    x={x}
                    y="48"
                    fontSize="10"
                    fill="#526579"
                  >
                    ×
                  </text>
                );
              })}
            </svg>
            <small>
              Measured row · ±{(extent * 100).toFixed(0)} cm · dashed line =
              road datum · × no returns
            </small>
          </div>
          <p className="inspector-reason">
            {observed
              ? reason
              : "No LiDAR returns in this cell. No measured height or drivable-surface claim is available."}
          </p>
          <details>
            <summary>More details</summary>
            <dl className="inspector-stats">
              <div>
                <dt>Height range</dt>
                <dd>
                  {observed
                    ? `${formatHeight(cell.elevation_min)} to ${formatHeight(cell.elevation_max)}`
                    : "Not measured"}
                </dd>
              </div>
              <div>
                <dt>Slope</dt>
                <dd>
                  {observed ? `${cell.slope_deg.toFixed(1)}°` : "Not measured"}
                </dd>
              </div>
              <div>
                <dt>Point returns</dt>
                <dd>{cell.point_count}</dd>
              </div>
              <div>
                <dt>Importance score</dt>
                <dd>{cell.importance_score.toFixed(2)}</dd>
              </div>
              <div>
                <dt>Heuristic confidence</dt>
                <dd>{(cell.confidence * 100).toFixed(0)}%</dd>
              </div>
              {Object.entries(cell.importance_factors).map(([key, value]) => (
                <div key={key}>
                  <dt>{key.replaceAll("_", " ")}</dt>
                  <dd>{value.toFixed(2)}</dd>
                </div>
              ))}
            </dl>
            <p className="help-text">
              Height is relative to the map datum, not an independently measured
              pothole depth. Terrain labels come from the prototype's simulation
              and geometry rules. Heuristic confidence is not measured accuracy.
            </p>
          </details>
        </>
      )}
      {object && (
        <>
          <span className="object-badge">
            {object.dynamic_state === "DYNAMIC" ? "Moving" : "Stationary"} ·
            Track {object.track_id}
          </span>
          <div className="inspector-measurement">
            <strong>
              {(object.speed * 3.6).toFixed(1)} <small>km/h</small>
            </strong>
            <span>estimated speed</span>
          </div>
          <dl className="inspector-stats">
            <div>
              <dt>Position</dt>
              <dd>
                {object.position[0].toFixed(1)}, {object.position[1].toFixed(1)}{" "}
                m
              </dd>
            </div>
            <div>
              <dt>Heading</dt>
              <dd>{object.heading.toFixed(0)}°</dd>
            </div>
          </dl>
          <p className="inspector-reason">
            Moving-object points are separated from the static terrain map.
          </p>
          <details>
            <summary>More details</summary>
            <dl className="inspector-stats">
              <div>
                <dt>Simulated confidence</dt>
                <dd>{(object.confidence * 100).toFixed(0)}%</dd>
              </div>
              <div>
                <dt>Track age</dt>
                <dd>{object.age} frames</dd>
              </div>
            </dl>
            <p className="help-text">
              Simulation confidence is not measured classification accuracy.
            </p>
          </details>
        </>
      )}
      <button className="button inspector-focus" onClick={onFocus}>
        <Crosshair size={16} /> Focus on selection <ArrowUpRight size={15} />
      </button>
    </aside>
  );
}
