import { ArchitectureModal } from "./ArchitectureModal";
import type { LidarFrame, SystemConfig } from "../types";
export function TechnicalDossier({
  frame,
  config,
  stale,
  paused,
}: {
  frame: LidarFrame | null;
  config: SystemConfig | null;
  stale: boolean;
  paused: boolean;
}) {
  const metrics = frame?.metrics;
  return (
    <main className="technical-page">
      <div className="page-heading">
        <span className="section-kicker">SIH 26053</span>
        <h1>What this prototype measures</h1>
        <p>
          Adaptive 2.5D mapping from simulated LiDAR, with a traceable
          uniform-grid comparison.
        </p>
      </div>
      <ArchitectureModal />
      <details open>
        <summary>Current capabilities and limits</summary>
        <div className="technical-grid">
          <div>
            <h3>Running now</h3>
            <ul>
              <li>
                Cell widths from{" "}
                {config?.grid_resolutions.slice().reverse().join(", ") ??
                  "0.25, 0.5, 1, 2, 4"}{" "}
                m.
              </li>
              <li>A 64 × 64 m map region and a 50 m simulated sensor range.</li>
              <li>
                Geometric terrain rules, simulated detections, and Kalman
                tracking.
              </li>
              <li>
                Measured cell heights from simulated static point returns.
              </li>
            </ul>
          </div>
          <div>
            <h3>Still to validate</h3>
            <ul>
              <li>The brief's 5 cm near-field and 100 m mapping target.</li>
              <li>A trained neural detector and terrain segmenter.</li>
              <li>Classification accuracy by distance.</li>
              <li>Measured process memory and a uniform 3D voxel benchmark.</li>
            </ul>
          </div>
        </div>
        <p>
          OpenPCDet and CUDA availability do not mean a model is loaded. This
          prototype currently uses simulated detections.
        </p>
      </details>
      <details open>
        <summary>Frame evidence</summary>
        <p>
          {stale
            ? "Stale frame retained"
            : paused
              ? "Paused frame"
              : "Latest delivered frame"}
          {frame
            ? ` · ${frame.scene_id.replaceAll("_", " ")} · revision ${frame.revision} · frame ${frame.frame_id}`
            : ""}
        </p>
        <table className="evidence-table">
          <caption>Pipeline counts and timings</caption>
          <tbody>
            <tr>
              <th>Raw returns</th>
              <td>
                {metrics?.raw_points_count.toLocaleString() ?? "Unavailable"}
              </td>
            </tr>
            <tr>
              <th>Static processed returns</th>
              <td>
                {metrics?.processed_points_count.toLocaleString() ??
                  "Unavailable"}
              </td>
            </tr>
            <tr>
              <th>Browser display sample</th>
              <td>{frame?.points.length.toLocaleString() ?? "Unavailable"}</td>
            </tr>
            <tr>
              <th>Adaptive cells</th>
              <td>
                {metrics?.adaptive_cells_count.toLocaleString() ??
                  "Unavailable"}
              </td>
            </tr>
            {Object.entries(metrics?.processing_time_ms ?? {}).map(
              ([key, value]) => (
                <tr key={key}>
                  <th>{key}</th>
                  <td>{value.toFixed(2)} ms</td>
                </tr>
              ),
            )}
          </tbody>
        </table>
      </details>
      <details>
        <summary>Storage calculation and comparison method</summary>
        <p>
          Uniform 2.5D storage = cell count × 48 bytes. Adaptive storage = cell
          count × 56 bytes, including hierarchy metadata. Divide by 1024 for
          KiB. These estimates exclude point clouds, tracks, allocator overhead
          and application memory.
        </p>
        <p>
          Compare captures the full static input and adaptive cells from one
          frame. An isolated elevation map builds a uniform baseline from those
          points. Both views share the region, camera and semantic rules.
          No-data cells remain unknown.
        </p>
      </details>
      <details>
        <summary>Why cells refine</summary>
        <p>
          Importance S = Σ wᵢ fᵢ combines distance, elevation variation, terrain
          class, nearby objects, motion, terrain complexity and uncertainty.
          Thresholds decide subdivision. A nearby cell is not guaranteed to be
          fine.
        </p>
        <p>
          The detail lens shows the actual cells. Heights use the map datum and
          are not independently validated pothole depths. Confidence describes
          the simulation's own estimate, not measured classification accuracy.
        </p>
      </details>
    </main>
  );
}
