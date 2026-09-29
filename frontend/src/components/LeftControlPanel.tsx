import { useState } from "react";
import { SkipForward } from "lucide-react";
import { Panel } from "./Panel";
import type {
  ImportanceWeights,
  LayerVisibility,
  SessionState,
} from "../types";

const layerNames: Record<keyof LayerVisibility, string> = {
  surface: "Terrain surface",
  points: "LiDAR points",
  elevation: "Retain measured height",
  adaptiveGrid: "Cell boundaries",
  semanticMap: "Terrain classes",
  detections: "Detection boxes",
  tracks: "Tracked objects and trails",
  potholes: "Pothole markers",
  curbs: "Curb markers",
  slopes: "Slope markers",
};
const weightNames: Record<keyof ImportanceWeights, string> = {
  w1_distance: "Distance",
  w2_elevation_var: "Elevation variation",
  w3_semantic: "Terrain class",
  w4_object_prox: "Nearby objects",
  w5_motion: "Movement",
  w6_terrain_complexity: "Terrain complexity",
  w7_uncertainty: "Uncertainty",
};
interface Props {
  session: SessionState;
  layers: LayerVisibility;
  disabled: boolean;
  onClose: () => void;
  onLayer: (key: keyof LayerVisibility) => void;
  onSpeed: (speed: number) => void;
  onStep: () => void;
  onWeights: (weights: ImportanceWeights) => void;
  onAdaptive: (enabled: boolean) => void;
}

export function LeftControlPanel({
  session,
  layers,
  disabled,
  onClose,
  onLayer,
  onSpeed,
  onStep,
  onWeights,
  onAdaptive,
}: Props) {
  const [weights, setWeights] = useState(session.weights);
  return (
    <Panel title="Advanced controls" onClose={onClose}>
      <p className="muted">
        For experiments and technical questions. The guided demo sets these
        controls for you.
      </p>
      <section className="settings-section">
        <h3>Playback</h3>
        <div className="settings-row">
          <label htmlFor="speed">Simulation speed</label>
          <select
            id="speed"
            value={session.playback_speed}
            disabled={disabled}
            onChange={(event) => onSpeed(Number(event.target.value))}
          >
            <option value={0.5}>0.5×</option>
            <option value={1}>1×</option>
            <option value={2}>2×</option>
          </select>
        </div>
        <button className="button" disabled={disabled} onClick={onStep}>
          <SkipForward size={16} /> Pause and advance one frame
        </button>
      </section>
      <section className="settings-section">
        <h3>Map layers</h3>
        {Object.entries(layerNames).map(([key, label]) => (
          <label className="check-row" key={key}>
            <span>{label}</span>
            <input
              type="checkbox"
              checked={layers[key as keyof LayerVisibility]}
              onChange={() => onLayer(key as keyof LayerVisibility)}
            />
          </label>
        ))}
      </section>
      <section className="settings-section">
        <h3>Adaptive mapping</h3>
        <label className="check-row">
          <span>Refine cells by importance</span>
          <input
            type="checkbox"
            disabled={disabled}
            checked={session.adaptive_enabled}
            onChange={(event) => onAdaptive(event.target.checked)}
          />
        </label>
        <p className="help-text">
          Turning this off leaves a coarse grid. Use Compare for the fine
          uniform baseline.
        </p>
        {Object.entries(weightNames).map(([key, label]) => (
          <label className="weight-row" key={key}>
            <span>
              {label}
              <output>
                {weights[key as keyof ImportanceWeights].toFixed(2)}
              </output>
            </span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={weights[key as keyof ImportanceWeights]}
              onChange={(event) =>
                setWeights({ ...weights, [key]: Number(event.target.value) })
              }
            />
          </label>
        ))}
        <button
          className="button primary"
          disabled={disabled}
          onClick={() => onWeights(weights)}
        >
          Apply weights
        </button>
        <p className="help-text">
          Weights are applied together. They change the importance score, not
          the sensor's measurement accuracy.
        </p>
      </section>
    </Panel>
  );
}
