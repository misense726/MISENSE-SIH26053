import { Grid2X2, Play, Maximize2, Minimize2 } from "lucide-react";
import type { ConnectionState } from "../types";

interface Props {
  page: "demo" | "compare" | "details";
  onPage: (page: "demo" | "compare" | "details") => void;
  connection: ConnectionState;
  paused: boolean;
  stale: boolean;
  onDemo: () => void;
  disabled: boolean;
  presentation: boolean;
  onPresentation: () => void;
}

export function TopBar({
  page,
  onPage,
  connection,
  paused,
  stale,
  onDemo,
  disabled,
  presentation,
  onPresentation,
}: Props) {
  const label =
    connection === "connected"
      ? stale
        ? "Waiting for frames"
        : paused
          ? "Simulation paused"
          : "Simulation connected"
      : connection === "connecting"
        ? "Connecting"
        : connection === "reconnecting"
          ? "Reconnecting"
          : "Disconnected";
  return (
    <header className="app-header">
      <a
        className="brand"
        href="#"
        onClick={(event) => {
          event.preventDefault();
          onPage("demo");
        }}
        aria-label="MI Sense demo"
      >
        <span className="brand-mark">
          <Grid2X2 size={24} strokeWidth={1.7} />
        </span>
        <span className="brand-name">
          MI SENSE<span className="brand-subtitle">ADAPTIVE PERCEPTION</span>
        </span>
      </a>
      {!presentation && (
        <nav className="view-nav" aria-label="Workspace">
          {(
            [
              ["demo", "Demo"],
              ["compare", "Compare"],
              ["details", "Technical details"],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              onClick={() => onPage(value)}
              className={`view-tab ${page === value ? "is-active" : ""}`}
              aria-current={page === value ? "page" : undefined}
            >
              {label}
            </button>
          ))}
        </nav>
      )}
      <div className="header-actions">
        <span
          className={`connection-status ${connection === "connected" && !stale ? "online" : ""}`}
          role="status"
        >
          <i />
          {label}
        </span>
        <button
          aria-label="Start guided demo"
          className="button primary demo-launch"
          onClick={onDemo}
          disabled={disabled}
        >
          <Play size={15} fill="currentColor" />
          <span className="demo-label-long">Start guided demo</span>
          <span className="demo-label-short">Guided demo</span>
        </button>
        <button
          className="button icon-button"
          onClick={onPresentation}
          aria-label={
            presentation ? "Exit presentation mode" : "Enter presentation mode"
          }
          title={presentation ? "Exit presentation mode" : "Presentation mode"}
        >
          {presentation ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
        </button>
      </div>
    </header>
  );
}
