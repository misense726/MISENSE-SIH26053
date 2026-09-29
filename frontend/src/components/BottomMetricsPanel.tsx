import { ArrowUpRight, Pause, Play, RotateCcw } from "lucide-react";
import type { FrameMetrics } from "../types";

interface Props {
  metrics?: FrameMetrics;
  running: boolean;
  disabled: boolean;
  stale: boolean;
  receivedFps: number;
  onToggle: () => void;
  onReset: () => void;
  onEvidence: () => void;
}
export function BottomMetricsPanel({
  metrics,
  running,
  disabled,
  stale,
  receivedFps,
  onToggle,
  onReset,
  onEvidence,
}: Props) {
  return (
    <footer className="playback-strip">
      <div className="playback-controls">
        <button
          className="button primary playback-main"
          disabled={disabled}
          onClick={onToggle}
        >
          {running ? (
            <Pause size={16} fill="currentColor" />
          ) : (
            <Play size={16} fill="currentColor" />
          )}
          {running ? "Pause" : "Play"}
        </button>
        <button
          className="button restart-button"
          disabled={disabled}
          onClick={onReset}
        >
          <RotateCcw size={16} />
          <span>Restart scene</span>
        </button>
      </div>
      <div
        className={`metrics-strip ${stale ? "is-stale" : ""}`}
        role="group"
        aria-label="Simulation measurements"
      >
        <div className="metric">
          <span>Est. grid memory saved</span>
          <strong>
            {metrics ? `${metrics.memory_saved_percent.toFixed(1)}` : "—"}
            <small>{metrics && "%"}</small>
          </strong>
        </div>
        <div className="metric">
          <span>Backend processing</span>
          <strong>
            {metrics ? metrics.processing_time_ms.total.toFixed(0) : "—"}
            <small>{metrics && "ms"}</small>
          </strong>
        </div>
        <div className="metric">
          <span>
            {stale
              ? "Last frame retained"
              : !running
                ? "Playback paused"
                : "Delivered frames"}
          </span>
          <strong>
            {stale || !running || !metrics ? "—" : receivedFps.toFixed(1)}
            <small>{!stale && running && metrics && "fps"}</small>
          </strong>
        </div>
      </div>
      <button className="text-button evidence-link" onClick={onEvidence}>
        View evidence <ArrowUpRight size={16} />
      </button>
    </footer>
  );
}
