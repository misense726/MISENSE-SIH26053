import { useEffect, useState } from "react";
import { Crosshair, Grid2X2, Pause, Play, RotateCcw } from "lucide-react";
import { AdaptiveMapCanvas, type CameraRequest } from "./visualization/AdaptiveMapCanvas";
import { cellIdentity, cellLabel } from "./visualization/palette";
import { layersForPreset, SCENE_DESCRIPTIONS, SCENE_LABELS } from "./hooks/sessionHelpers";
import { loadManifest, loadRecording, nextFrameIndex } from "./staticPlayback";
import type { StaticManifest, StaticRecording } from "./staticPlayback";
import type { AdaptiveCell, MapPreset, TrackedObject } from "./types";
import "./StaticDemo.css";

type Selection = { kind: "cell"; id: string } | { kind: "track"; id: number };

export default function StaticDemo() {
  const [manifest, setManifest] = useState<StaticManifest | null>(null);
  const [sceneId, setSceneId] = useState("");
  const [recording, setRecording] = useState<StaticRecording | null>(null);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [preset, setPreset] = useState<MapPreset>("terrain");
  const [camera, setCamera] = useState<CameraRequest>({ preset: "angled", serial: 0 });
  const [selection, setSelection] = useState<Selection | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    let active = true;
    loadManifest()
      .then((result) => {
        if (!active) return;
        setManifest(result);
        setSceneId(result.default_scene);
      })
      .catch((cause: unknown) => {
        if (active)
          setError(cause instanceof Error ? cause.message : "The scenes could not load.");
      });
    return () => {
      active = false;
    };
  }, [retry]);

  const scene = manifest?.scenes.find((item) => item.id === sceneId);
  useEffect(() => {
    if (!scene) return;
    let active = true;
    loadRecording(scene)
      .then((result) => {
        if (active) setRecording(result);
      })
      .catch((cause: unknown) => {
        if (active)
          setError(cause instanceof Error ? cause.message : "This scene could not load.");
      });
    return () => {
      active = false;
    };
  }, [scene, retry]);

  useEffect(() => {
    if (!recording || !playing) return;
    const timer = window.setInterval(() => {
      if (!document.hidden)
        setIndex((current) => nextFrameIndex(current, recording.frames.length));
    }, 1000 / (recording.fps * speed));
    return () => window.clearInterval(timer);
  }, [recording, playing, speed]);

  const frame = recording?.frames[index] ?? null;
  const layers = layersForPreset(preset);
  const selectedCell =
    selection?.kind === "cell"
      ? frame?.adaptive_cells.find((cell) => cellIdentity(cell) === selection.id) ?? null
      : null;
  const selectedObject =
    selection?.kind === "track"
      ? frame?.tracks.find((track) => track.track_id === selection.id) ?? null
      : null;

  const chooseScene = (next: string) => {
    setSceneId(next);
    setRecording(null);
    setIndex(0);
    setError(null);
    setSelection(null);
    setPlaying(true);
    setCamera({ preset: "angled", serial: Date.now() });
  };
  const chooseCell = (cell: AdaptiveCell | null) =>
    setSelection(cell ? { kind: "cell", id: cellIdentity(cell) } : null);
  const chooseObject = (object: TrackedObject | null) => {
    setSelection(object ? { kind: "track", id: object.track_id } : null);
    if (object)
      setCamera({
        preset: "focus",
        target: [object.position[0], object.position[1]],
        serial: Date.now(),
      });
  };
  const reset = () => {
    setIndex(0);
    setPlaying(true);
    setSelection(null);
  };
  const retryScene = () => {
    setError(null);
    setRecording(null);
    setIndex(0);
    setRetry((value) => value + 1);
  };

  return (
    <div className="app-shell static-demo">
      <header className="app-header static-header">
        <a className="brand" href="/" aria-label="MI Sense simulation home">
          <span className="brand-mark"><Grid2X2 size={24} strokeWidth={1.7} /></span>
          <span className="brand-name">MI SENSE<span className="brand-subtitle">SIH 26053</span></span>
        </a>
        <div className="static-header-right">
          <span className="connection-status online" role="status">
            <i /> {error ? "Scene unavailable" : !recording ? "Loading scene" : playing ? "Simulation looping" : "Simulation paused"}
          </span>
          <a href="https://github.com/misense726/MISENSE-SIH26053" target="_blank" rel="noreferrer">View project</a>
        </div>
      </header>

      <main className="demo-workspace" id="main-content">
        <div className="scene-toolbar static-toolbar">
          <div className="scenario-control">
            <label htmlFor="scene">Scenario</label>
            <select id="scene" value={sceneId} disabled={!manifest} onChange={(event) => chooseScene(event.target.value)}>
              {!manifest && <option value="">Loading scenes</option>}
              {manifest?.scenes.map((item) => (
                <option key={item.id} value={item.id}>{SCENE_LABELS[item.id] ?? item.name}</option>
              ))}
            </select>
            <p>{SCENE_DESCRIPTIONS[sceneId] ?? "Simulated terrain and movement."}</p>
          </div>
          <div className="segmented" role="group" aria-label="Map display">
            {(["terrain", "resolution", "scan"] as const).map((value) => (
              <button key={value} aria-pressed={preset === value} onClick={() => setPreset(value)}>
                {value === "scan" ? "Raw scan" : value === "resolution" ? "Resolution" : "Terrain"}
              </button>
            ))}
          </div>
        </div>

        <section className="map-stage static-map" aria-label="Simulation map">
          <AdaptiveMapCanvas
            frame={frame}
            layers={layers}
            preset={preset}
            camera={camera}
            selectedCell={selectedCell}
            selectedObject={selectedObject}
            onSelectCell={chooseCell}
            onSelectObject={chooseObject}
            onInspectTruck={() => setCamera({ preset: "focus", target: [0, 0], serial: Date.now() })}
            onCamera={() => {}}
          />
          <div className="map-caption">
            <span>SIMULATED · SIH 26053</span>
            <h1>Adaptive 2.5D perception</h1>
            <small>{recording ? `${playing ? "Looping" : "Paused"} · Frame ${index + 1} of ${recording.frames.length}` : "Preparing the map"}</small>
          </div>
          <div className="camera-controls">
            <div className="segmented" role="group" aria-label="Camera angle">
              {(["top", "angled"] as const).map((value) => (
                <button key={value} aria-pressed={camera.preset === value} onClick={() => setCamera({ preset: value, serial: Date.now() })}>
                  {value === "top" ? "Top" : "Angled"}
                </button>
              ))}
              <button aria-pressed={camera.preset === "focus" && camera.target?.[0] === 0 && camera.target?.[1] === 0} onClick={() => setCamera({ preset: "focus", target: [0, 0], serial: Date.now() })}>Truck</button>
            </div>
          </div>
          {error && (
            <div className="static-error" role="alert">
              <p>{error}</p>
              <button className="button primary" onClick={retryScene}>Retry scene</button>
            </div>
          )}
          {(selectedCell || selectedObject) && (
            <div className="static-inspector">
              <button className="static-close" onClick={() => setSelection(null)} aria-label="Close selection">×</button>
              <span>Map selection</span>
              <strong>{selectedCell ? cellLabel(selectedCell) : selectedObject?.class_name.replaceAll("_", " ")}</strong>
              {selectedCell ? (
                <p>{selectedCell.size} m cell · {selectedCell.point_count} returns<br />Height {selectedCell.elevation_mean.toFixed(2)} m</p>
              ) : (
                <p>Track {selectedObject?.track_id} · {((selectedObject?.speed ?? 0) * 3.6).toFixed(1)} km/h</p>
              )}
              <button onClick={() => {
                const target = selectedCell ? [selectedCell.x, selectedCell.y] : selectedObject?.position;
                if (target) setCamera({ preset: "focus", target: [target[0], target[1]], serial: Date.now() });
              }}><Crosshair size={15} /> Focus here</button>
            </div>
          )}
        </section>

        <footer className="static-playback-strip">
          <div className="playback-controls">
            <button className="button primary" disabled={!recording} onClick={() => setPlaying((value) => !value)}>
              {playing ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}
              {playing ? "Pause" : "Play"}
            </button>
            <button className="button" disabled={!recording} onClick={reset}><RotateCcw size={16} /> Restart</button>
          </div>
          <div className="static-progress">
            <label htmlFor="scene-progress">{playing ? "Playing continuously" : "Playback paused"}</label>
            <progress id="scene-progress" value={recording ? index + 1 : 0} max={recording?.frames.length ?? 1} />
          </div>
          <div className="static-measures" aria-label="Frame measurements">
            <div><span>Adaptive cells</span><strong>{frame?.adaptive_cells.length.toLocaleString() ?? "—"}</strong></div>
            <div><span>Est. grid storage saved</span><strong>{frame ? `${frame.metrics.memory_saved_percent.toFixed(1)}%` : "—"}</strong></div>
          </div>
          <label className="static-speed">Speed
            <select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
              <option value={0.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option>
            </select>
          </label>
        </footer>
      </main>
    </div>
  );
}
