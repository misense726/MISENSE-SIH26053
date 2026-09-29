import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { useGuidedDemo } from "./useGuidedDemo";
import { TERRAIN_LAYERS } from "./sessionHelpers";
import type {
  ComparisonSnapshot,
  ControlCommand,
  LidarFrame,
  SessionState,
} from "../types";
afterEach(cleanup);
function harness() {
  const session = {
    scene_id: "curb_boundary",
    revision: 2,
    frame_id: 5,
    is_running: true,
    playback_speed: 2,
    adaptive_enabled: false,
    weights: { w1_distance: 0.8 },
  } as SessionState;
  let state = { ...session };
  const command = vi.fn(async (cmd: ControlCommand) => {
    if (cmd.action === "set_scene")
      state = {
        ...state,
        scene_id: cmd.scene_id!,
        revision: state.revision + 1,
        frame_id: 0,
      };
    if (cmd.action === "step")
      state = { ...state, frame_id: state.frame_id + 1, is_running: false };
    if (cmd.action === "play" || cmd.action === "pause")
      state = { ...state, is_running: cmd.action === "play" };
    if (cmd.action === "set_speed")
      state = { ...state, playback_speed: cmd.speed! };
    if (cmd.action === "set_adaptive_enabled")
      state = { ...state, adaptive_enabled: cmd.enabled! };
    return { ...state };
  });
  const waitForFrame = vi.fn(
    async (s: SessionState) =>
      ({
        ...s,
        adaptive_cells: [],
        tracks: [
          { track_id: 1, class_name: "pedestrian", position: [1, 20, 0] },
        ],
      }) as unknown as LidarFrame,
  );
  const cameraState = {
    current: {
      position: [2, 3, 4] as [number, number, number],
      target: [0, 1, 0] as [number, number, number],
    },
  };
  const options = {
    session,
    preset: "resolution" as const,
    layers: { ...TERRAIN_LAYERS, tracks: false },
    camera: { preset: "top" as const, serial: 8 },
    cameraState,
    command,
    waitForFrame,
    capture: vi.fn(async () => ({ snapshot_id: "one" }) as ComparisonSnapshot),
    onPreset: vi.fn(),
    onLayers: vi.fn(),
    onCamera: vi.fn(),
    onPage: vi.fn(),
    onFocus: vi.fn(),
    onError: vi.fn(),
  };
  return { options, getState: () => state };
}
it("waits for the matching frame before completing a guide step", async () => {
  const { options } = harness();
  let release: (frame: LidarFrame) => void = () => {};
  options.waitForFrame.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        release = resolve;
      }),
  );
  const { result } = renderHook(() => useGuidedDemo(options));
  let started: Promise<void>;
  await act(async () => {
    started = result.current.start();
    await Promise.resolve();
  });
  expect(result.current.busy).toBe(true);
  expect(options.onPreset).not.toHaveBeenCalled();
  await act(async () => {
    release({ adaptive_cells: [], tracks: [] } as unknown as LidarFrame);
    await started;
  });
  expect(result.current.busy).toBe(false);
  expect(options.onPreset).toHaveBeenCalledWith("scan");
});
it("restores settings, camera and playback and explicitly restarts the previous scene", async () => {
  const { options, getState } = harness();
  const { result } = renderHook(() => useGuidedDemo(options));
  await act(async () => {
    await result.current.start();
  });
  await act(async () => {
    await result.current.transition(1);
  });
  await act(async () => {
    await result.current.transition(2);
  });
  expect(options.capture).toHaveBeenCalledTimes(1);
  expect(options.onPage).toHaveBeenCalledWith("compare");
  await act(async () => {
    await result.current.transition(3);
  });
  expect(getState().is_running).toBe(true);
  await act(async () => {
    await result.current.exit();
  });
  expect(result.current.step).toBeNull();
  expect(getState().scene_id).toBe("curb_boundary");
  expect(getState().playback_speed).toBe(2);
  expect(getState().adaptive_enabled).toBe(false);
  expect(options.onLayers).toHaveBeenLastCalledWith(options.layers);
  expect(options.onCamera).toHaveBeenLastCalledWith(
    expect.objectContaining({ restore: options.cameraState.current }),
  );
  expect(options.onError).toHaveBeenLastCalledWith(
    expect.stringContaining("Previous scenario restarted"),
  );
});
it("keeps the guide recoverable when a command fails", async () => {
  const { options } = harness();
  options.command.mockRejectedValueOnce(new Error("Connection lost"));
  const { result } = renderHook(() => useGuidedDemo(options));
  await act(async () => {
    await result.current.start();
  });
  expect(result.current.error).toBe("Connection lost");
  expect(result.current.busy).toBe(false);
  await act(async () => {
    await result.current.exit();
  });
  expect(result.current.step).toBeNull();
});
