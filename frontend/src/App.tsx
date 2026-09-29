import { useEffect, useRef, useState } from "react";
import {
  Crosshair,
  RotateCcw,
  SlidersHorizontal,
  List,
  Maximize,
} from "lucide-react";
import { TopBar } from "./components/TopBar";
import { LeftControlPanel } from "./components/LeftControlPanel";
import { RightInspectorPanel } from "./components/RightInspectorPanel";
import { BottomMetricsPanel } from "./components/BottomMetricsPanel";
import {
  AdaptiveMapCanvas,
  type CameraRequest,
  type CameraState,
} from "./visualization/AdaptiveMapCanvas";
import { JuryDemoOverlay } from "./components/JuryDemoOverlay";
import { FourPanelComparison } from "./components/FourPanelComparison";
import { TechnicalDossier } from "./components/TechnicalDossier";
import { Panel } from "./components/Panel";
import {
  usePerceptionSession,
  errorMessage,
} from "./hooks/usePerceptionSession";
import { useGuidedDemo } from "./hooks/useGuidedDemo";
import {
  hazardCell,
  layersForPreset,
  resolveCell,
  SCENE_LABELS,
  SCENE_DESCRIPTIONS,
} from "./hooks/sessionHelpers";
import { cellIdentity } from "./visualization/palette";
import { captureComparison } from "./services/api";
import type {
  AdaptiveCell,
  ComparisonSnapshot,
  ControlCommand,
  LidarFrame,
  MapPreset,
  TrackedObject,
} from "./types";

export default function App() {
  const perception = usePerceptionSession();
  const { frame, session, ready, busy, stale, connection } = perception;
  const [page, setPage] = useState<"demo" | "compare" | "details">("demo");
  const [preset, setPreset] = useState<MapPreset>("terrain");
  const [layers, setLayers] = useState(() => layersForPreset("terrain"));
  const [camera, setCamera] = useState<CameraRequest>({
    preset: "angled",
    serial: 0,
  });
  const cameraState = useRef<CameraState | undefined>(undefined);
  const [selection, setSelection] = useState<{
    revision: number;
    cell?: string;
    track?: number;
  } | null>(null);
  const [drawer, setDrawer] = useState<"advanced" | "features" | null>(null);
  const [presentation, setPresentation] = useState(false);
  const [snapshot, setSnapshot] = useState<ComparisonSnapshot | null>(null);
  const [capturing, setCapturing] = useState(false),
    [captureError, setCaptureError] = useState<string | null>(null);
  const captureRef = useRef<Promise<ComparisonSnapshot> | null>(null);
  const selectedCell =
    selection?.revision === frame?.revision
      ? resolveCell(frame, selection?.cell ?? null)
      : null;
  const selectedObject =
    selection?.revision === frame?.revision
      ? (frame?.tracks.find((t) => t.track_id === selection?.track) ?? null)
      : null;
  useEffect(() => {
    if (selection && frame && !selectedCell && !selectedObject) {
      queueMicrotask(() =>
        setSelection((current) => (current === selection ? null : current)),
      );
    }
  }, [frame, selection, selectedCell, selectedObject]);
  const selectCell = (cell: AdaptiveCell | null) =>
    setSelection(
      cell && frame
        ? { revision: frame.revision, cell: cellIdentity(cell) }
        : null,
    );
  const selectObject = (track: TrackedObject | null) =>
    setSelection(
      track && frame
        ? { revision: frame.revision, track: track.track_id }
        : null,
    );
  const choosePreset = (next: MapPreset) => {
    setPreset(next);
    setLayers(layersForPreset(next));
  };
  const capture = () => {
    if (captureRef.current) return captureRef.current;
    setCapturing(true);
    setCaptureError(null);
    const promise = captureComparison()
      .then((result) => {
        setSnapshot(result);
        return result;
      })
      .catch((cause) => {
        setCaptureError(errorMessage(cause));
        throw cause;
      })
      .finally(() => {
        captureRef.current = null;
        setCapturing(false);
      });
    captureRef.current = promise;
    return promise;
  };
  const focus = (source: LidarFrame, actor = false) => {
    const object = actor
      ? (source.tracks.find((t) => t.class_name === "pedestrian") ??
        source.tracks[0])
      : null;
    const cell = object ? null : hazardCell(source);
    if (object) {
      setSelection({ revision: source.revision, track: object.track_id });
      setCamera({
        preset: "focus",
        serial: Date.now(),
        target: [object.position[0], object.position[1]],
      });
    } else if (cell) {
      setSelection({ revision: source.revision, cell: cellIdentity(cell) });
      setCamera({
        preset: "focus",
        serial: Date.now(),
        target: [cell.x, cell.y],
      });
    }
  };
  const guide = useGuidedDemo({
    session,
    preset,
    layers,
    camera,
    cameraState,
    command: perception.command,
    waitForFrame: perception.waitForFrame,
    capture,
    onPreset: choosePreset,
    onLayers: setLayers,
    onCamera: setCamera,
    onPage: setPage,
    onFocus: focus,
    onError: perception.setError,
  });
  const disabled = !ready || busy || guide.busy;
  const command = (cmd: ControlCommand) => {
    void perception.command(cmd).catch(() => {});
  };
  const feature = hazardCell(frame),
    actor =
      frame?.tracks.find((t) => t.class_name === "pedestrian") ??
      frame?.tracks[0];
  const preferActor =
    !!actor &&
    ["pedestrian", "moving_vehicle"].includes(session?.scene_id ?? "");
  const focusSelection = () => {
    const target = selectedCell
      ? [selectedCell.x, selectedCell.y]
      : selectedObject?.position;
    if (target)
      setCamera({
        preset: "focus",
        serial: Date.now(),
        target: [target[0], target[1]],
      });
  };
  const inspectTruck = () => {
    setSelection(null);
    setCamera({
      preset: "focus",
      target: [0, 0],
      restore: { position: [5, 7, 4.8], target: [0, 0, 0.7] },
      serial: Date.now(),
    });
  };
  return (
    <div className={`app-shell ${presentation ? "presentation" : ""}`}>
      <TopBar
        page={page}
        onPage={setPage}
        connection={connection}
        paused={!session?.is_running}
        stale={stale}
        onDemo={() => void guide.start()}
        disabled={disabled || guide.step !== null}
        presentation={presentation}
        onPresentation={() => setPresentation((value) => !value)}
      />
      {perception.error && (
        <div className="notice" role="status">
          {perception.error}
          <button
            onClick={() => perception.setError(null)}
            aria-label="Dismiss message"
          >
            Dismiss
          </button>
        </div>
      )}
      {connection !== "connected" && (
        <div className="notice" role="status">
          {frame
            ? "Connection lost. The last frame is retained and marked stale."
            : "Connecting to the local simulation."}
          <button onClick={perception.retry}>Retry connection</button>
        </div>
      )}
      {page === "demo" && (
        <main className="demo-workspace">
          <div className="scene-toolbar">
            <div className="scenario-control">
              <label htmlFor="scenario">Scenario</label>
              <select
                id="scenario"
                disabled={disabled || guide.step !== null}
                value={session?.scene_id ?? ""}
                onChange={(event) => {
                  setSelection(null);
                  setCamera({ preset: "angled", serial: Date.now() });
                  command({
                    action: "set_scene",
                    scene_id: event.target.value,
                  });
                }}
              >
                {!session && <option value="">Connecting…</option>}
                {perception.scenes.map((scene) => (
                  <option key={scene.id} value={scene.id}>
                    {SCENE_LABELS[scene.id] ?? scene.name}
                  </option>
                ))}
              </select>
              <p>
                {SCENE_DESCRIPTIONS[session?.scene_id ?? ""] ??
                  "Choose a scene once connected."}
              </p>
            </div>
            <div className="segmented" role="group" aria-label="Map display">
              {(["terrain", "resolution", "scan"] as const).map((value) => (
                <button
                  key={value}
                  aria-pressed={preset === value}
                  onClick={() => choosePreset(value)}
                >
                  {value === "scan"
                    ? "Raw scan"
                    : value === "terrain"
                      ? "Terrain"
                      : "Resolution"}
                </button>
              ))}
            </div>
            <button onClick={() => setDrawer("features")}>
              <List size={17} /> Map features
            </button>
            <button
              onClick={() => setDrawer("advanced")}
              disabled={!session || guide.step !== null}
            >
              <SlidersHorizontal size={17} /> Advanced
            </button>
          </div>
          <section className="map-stage" aria-label="Simulation map">
            <AdaptiveMapCanvas
              frame={frame}
              layers={layers}
              preset={preset}
              camera={camera}
              selectedCell={selectedCell}
              selectedObject={selectedObject}
              onSelectCell={selectCell}
              onSelectObject={selectObject}
              onInspectTruck={inspectTruck}
              onCamera={(state) => {
                cameraState.current = state;
              }}
            />
            <div className="map-caption">
              <span>SIH 26053</span>
              <h1>Adaptive 2.5D perception</h1>
              <small>
                {stale
                  ? "Stale frame"
                  : !session?.is_running
                    ? "Paused frame"
                    : "Simulation"}{" "}
                · {frame?.adaptive_cells.length.toLocaleString() ?? "0"} cells
              </small>
            </div>
            <div className="camera-controls">
              <div className="segmented">
                <button
                  aria-pressed={camera.preset === "top"}
                  onClick={() =>
                    setCamera({ preset: "top", serial: Date.now() })
                  }
                >
                  Top
                </button>
                <button
                  aria-pressed={camera.preset === "angled"}
                  onClick={() =>
                    setCamera({ preset: "angled", serial: Date.now() })
                  }
                >
                  Angled
                </button>
                <button
                  aria-label="Inspect survey truck"
                  aria-pressed={
                    camera.preset === "focus" &&
                    camera.target?.[0] === 0 &&
                    camera.target?.[1] === 0 &&
                    !selectedObject &&
                    !selectedCell
                  }
                  onClick={inspectTruck}
                >
                  Truck
                </button>
              </div>
              <button
                aria-label="Reset camera"
                title="Reset camera"
                onClick={() =>
                  setCamera({ preset: "angled", serial: Date.now() })
                }
              >
                <RotateCcw size={17} />
              </button>
              {presentation && (
                <button
                  aria-label="Enter fullscreen"
                  onClick={() => {
                    void document.documentElement
                      .requestFullscreen()
                      .catch(() =>
                        perception.setError(
                          "Fullscreen is unavailable in this browser.",
                        ),
                      );
                  }}
                >
                  <Maximize size={17} />
                </button>
              )}
            </div>
            <div className="context-action">
              {feature || actor ? (
                <button
                  onClick={() => frame && focus(frame, preferActor || !feature)}
                >
                  <Crosshair size={17} />
                  {feature && !preferActor
                    ? `Focus on ${feature.semantic_class}`
                    : `Follow ${actor!.class_name}`}
                </button>
              ) : (
                <span>No hazard or tracked actor in this frame</span>
              )}
            </div>
            {stale && (
              <div className="stale-tag">Stale · last received frame</div>
            )}
            <RightInspectorPanel
              selectedCell={selectedCell}
              selectedObject={selectedObject}
              frame={frame}
              onClearSelection={() => setSelection(null)}
              onFocus={focusSelection}
            />
            {guide.step === 1 && (
              <div className="height-toggle">
                <button
                  onClick={() =>
                    setLayers((value) => ({
                      ...value,
                      elevation: !value.elevation,
                    }))
                  }
                >
                  {layers.elevation
                    ? "Remove height"
                    : "Restore measured height"}
                </button>
                <span>
                  Same frozen frame ·{" "}
                  {layers.elevation ? "height retained" : "height removed"}
                </span>
              </div>
            )}
          </section>
          <BottomMetricsPanel
            metrics={frame?.metrics}
            running={session?.is_running ?? false}
            disabled={disabled || guide.step !== null}
            stale={stale}
            receivedFps={perception.receivedFps}
            onToggle={() =>
              command({ action: session?.is_running ? "pause" : "play" })
            }
            onReset={() => command({ action: "reset" })}
            onEvidence={() => setPage("details")}
          />
        </main>
      )}
      {page === "compare" && (
        <FourPanelComparison
          snapshot={snapshot}
          loading={capturing}
          error={captureError}
          onCapture={() => {
            void capture().catch(() => {});
          }}
          enabled={ready && !guide.busy}
        />
      )}
      {page === "details" && (
        <TechnicalDossier
          frame={frame}
          config={perception.config}
          stale={stale}
          paused={!session?.is_running}
        />
      )}
      {guide.step !== null && (
        <JuryDemoOverlay
          step={guide.step}
          busy={guide.busy || !ready}
          error={guide.error}
          onStep={(next) => void guide.transition(next)}
          onExit={() => void guide.exit()}
          onEvidence={() => void guide.exit("details")}
        />
      )}
      {drawer === "advanced" && session && (
        <LeftControlPanel
          session={session}
          layers={layers}
          disabled={disabled}
          onClose={() => setDrawer(null)}
          onLayer={(key) =>
            setLayers((value) => ({ ...value, [key]: !value[key] }))
          }
          onSpeed={(speed) => command({ action: "set_speed", speed })}
          onStep={() => command({ action: "step" })}
          onWeights={(weights) =>
            command({ action: "update_weights", weights })
          }
          onAdaptive={(enabled) =>
            command({ action: "set_adaptive_enabled", enabled })
          }
        />
      )}
      {drawer === "features" && (
        <Panel title="Map features" onClose={() => setDrawer(null)}>
          <p>Select an object or a measured cell to inspect it on the map.</p>
          <h3>Tracked objects</h3>
          {frame?.tracks.length ? (
            frame.tracks.map((track) => (
              <button
                className="feature-row"
                key={track.track_id}
                onClick={() => {
                  selectObject(track);
                  setDrawer(null);
                }}
              >
                {track.class_name} · Track {track.track_id} ·{" "}
                {(track.speed * 3.6).toFixed(1)} km/h
              </button>
            ))
          ) : (
            <p>No confirmed tracks in this frame.</p>
          )}
          <h3>Map cells</h3>
          <label className="cell-picker">
            Select a cell
            <select
              aria-label="Select a map cell"
              value={selectedCell ? cellIdentity(selectedCell) : ""}
              onChange={(event) => {
                selectCell(resolveCell(frame, event.target.value));
                setDrawer(null);
              }}
            >
              <option value="">Choose a location</option>
              {frame?.adaptive_cells.map((cell) => (
                <option key={cellIdentity(cell)} value={cellIdentity(cell)}>
                  {cell.point_count ? cell.semantic_class : "Unknown"} · x{" "}
                  {cell.x}, y {cell.y} · {cell.size} m
                </option>
              ))}
            </select>
          </label>
        </Panel>
      )}
    </div>
  );
}
