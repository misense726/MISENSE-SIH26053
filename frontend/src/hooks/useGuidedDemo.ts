import { useRef, useState } from "react";
import type {
  CameraRequest,
  CameraState,
} from "../visualization/AdaptiveMapCanvas";
import type {
  ComparisonSnapshot,
  ControlCommand,
  LayerVisibility,
  LidarFrame,
  MapPreset,
  SessionState,
} from "../types";
import { hazardCell } from "./sessionHelpers";
export const GUIDE_STEPS = [
  {
    title: "See the scan",
    text: "The pipeline processes the full cropped scan. This view draws a smaller point sample so you can explore it smoothly.",
  },
  {
    title: "Keep the height",
    text: "Inspect a measured road hazard. Remove height from this frozen frame, then restore it. The lens shows observed cells and missing returns.",
  },
  {
    title: "Spend detail where needed",
    text: "Both grids use the same captured input. Inspect the real cell boundaries and compare their estimated storage.",
  },
  {
    title: "Handle movement",
    text: "Follow the pedestrian while the terrain stays readable. Moving-object returns are filtered from the static elevation map.",
  },
];
interface Options {
  session: SessionState | null;
  preset: MapPreset;
  layers: LayerVisibility;
  camera: CameraRequest;
  cameraState: React.RefObject<CameraState | undefined>;
  command: (cmd: ControlCommand) => Promise<SessionState>;
  waitForFrame: (state: SessionState) => Promise<LidarFrame>;
  capture: () => Promise<ComparisonSnapshot>;
  onPreset: (preset: MapPreset) => void;
  onLayers: (layers: LayerVisibility) => void;
  onCamera: (camera: CameraRequest) => void;
  onPage: (page: "demo" | "compare" | "details") => void;
  onFocus: (frame: LidarFrame, actor?: boolean) => void;
  onError: (message: string | null) => void;
}
export function useGuidedDemo(options: Options) {
  const [step, setStep] = useState<number | null>(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState<string | null>(null);
  const saved = useRef<{
    session: SessionState;
    preset: MapPreset;
    layers: LayerVisibility;
    camera: CameraRequest;
    cameraState?: CameraState;
  } | null>(null);
  const active = useRef(false);
  const transition = async (next: number) => {
    if (active.current) return;
    active.current = true;
    setBusy(true);
    setStep(next);
    setError(null);
    options.onError(null);
    try {
      await options.command({ action: "pause" });
      await options.command({ action: "set_adaptive_enabled", enabled: true });
      if (next === 0 || next === 1 || next === 3) {
        const state = await options.command({
          action: "set_scene",
          scene_id:
            next === 0 ? "normal_road" : next === 1 ? "pothole" : "pedestrian",
        });
        let frame = await options.waitForFrame(state);
        if (next === 3) {
          // Confirm a track before asking the camera to follow it.
          for (let i = 0; i < 8 && !frame.tracks.length; i++) {
            const stepped = await options.command({ action: "step" });
            frame = await options.waitForFrame(stepped);
          }
        }
        options.onPage("demo");
        options.onPreset(next === 0 ? "scan" : "terrain");
        options.onCamera({
          preset: next === 0 ? "angled" : "focus",
          serial: Date.now(),
          target:
            next === 1
              ? (() => {
                  const cell = hazardCell(frame);
                  return cell
                    ? ([cell.x, cell.y] as [number, number])
                    : ([0, 16] as [number, number]);
                })()
              : undefined,
        });
        if (next !== 0) options.onFocus(frame, next === 3);
        if (next === 3) await options.command({ action: "play" });
      } else {
        if (options.session?.scene_id !== "pothole") {
          const state = await options.command({
            action: "set_scene",
            scene_id: "pothole",
          });
          await options.waitForFrame(state);
        }
        options.onPreset("resolution");
        await options.capture();
        options.onPage("compare");
      }
      setStep(next);
    } catch (cause) {
      const message =
        cause instanceof Error ? cause.message : "Could not prepare this step.";
      setError(message);
    } finally {
      setBusy(false);
      active.current = false;
    }
  };
  const start = async () => {
    if (!options.session || saved.current) return;
    saved.current = {
      session: { ...options.session, weights: { ...options.session.weights } },
      preset: options.preset,
      layers: { ...options.layers },
      camera: options.camera,
      cameraState: options.cameraState.current,
    };
    setStep(0);
    await transition(0);
  };
  const exit = async (page: "demo" | "details" = "demo") => {
    if (active.current || !saved.current) return;
    active.current = true;
    setBusy(true);
    setError(null);
    try {
      const previous = saved.current;
      await options.command({ action: "pause" });
      await options.command({
        action: "update_weights",
        weights: previous.session.weights,
      });
      await options.command({
        action: "set_adaptive_enabled",
        enabled: previous.session.adaptive_enabled,
      });
      await options.command({
        action: "set_speed",
        speed: previous.session.playback_speed,
      });
      const restored = await options.command({
        action: "set_scene",
        scene_id: previous.session.scene_id,
      });
      await options.waitForFrame(restored);
      await options.command({
        action: previous.session.is_running ? "play" : "pause",
      });
      options.onPreset(previous.preset);
      options.onLayers(previous.layers);
      options.onCamera({
        ...previous.camera,
        serial: Date.now(),
        restore: previous.cameraState,
      });
      options.onPage(page);
      options.onError(
        "Previous scenario restarted. Your view, mapping settings and playback state have been restored.",
      );
      saved.current = null;
      setStep(null);
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Could not restore the previous session. Reconnect and retry Exit.",
      );
    } finally {
      active.current = false;
      setBusy(false);
    }
  };
  return { step, busy, error, start, transition, exit };
}
