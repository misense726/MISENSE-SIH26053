import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { SessionState } from "../types";
const mock = vi.hoisted(() => ({
  frame: null as ((value: unknown) => void) | null,
  connection: null as ((value: string) => void) | null,
  session: null as ((value: unknown) => void) | null,
  config: vi.fn(),
  scenes: vi.fn(),
  send: vi.fn(),
}));
vi.mock("../services/api", () => ({
  fetchConfig: mock.config,
  fetchScenes: mock.scenes,
}));
vi.mock("../services/websocket", () => ({
  wsService: {
    onFrame: (cb: typeof mock.frame) => {
      mock.frame = cb;
      return () => {};
    },
    onConnectionChange: (cb: typeof mock.connection) => {
      mock.connection = cb;
      return () => {};
    },
    onSession: (cb: typeof mock.session) => {
      mock.session = cb;
      return () => {};
    },
    connect: vi.fn(),
    disconnect: vi.fn(),
    retry: vi.fn(),
    send: mock.send,
  },
}));
import { usePerceptionSession } from "./usePerceptionSession";
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});
it("rehydrates authoritative state after reconnecting to a restarted server", async () => {
  const initial = {
    scene_id: "pothole",
    revision: 20,
    frame_id: 80,
    is_running: false,
  } as SessionState;
  mock.config.mockResolvedValue({ session: initial });
  mock.scenes.mockResolvedValue([]);
  const { result } = renderHook(usePerceptionSession);
  await act(async () => {
    await mock.connection?.("connected");
  });
  await waitFor(() => expect(result.current.session?.revision).toBe(20));
  act(() => mock.frame?.({ ...initial, adaptive_cells: [], points: [] }));
  act(() =>
    mock.frame?.({ scene_id: "normal_road", revision: 19, frame_id: 100 }),
  );
  expect(result.current.frame?.scene_id).toBe("pothole");
  const restarted = {
    scene_id: "normal_road",
    revision: 0,
    frame_id: 0,
    is_running: true,
  };
  mock.config.mockResolvedValue({ session: restarted });
  await act(async () => {
    mock.connection?.("disconnected");
    await mock.connection?.("connected");
  });
  act(() => mock.frame?.({ ...restarted, adaptive_cells: [], points: [] }));
  expect(result.current.frame?.revision).toBe(0);
  expect(result.current.session?.is_running).toBe(true);
});
it("does not accept an old frame as completion of a single step", async () => {
  const state = {
    scene_id: "pothole",
    revision: 2,
    frame_id: 4,
    is_running: false,
  } as SessionState;
  mock.config.mockResolvedValue({ session: state });
  mock.scenes.mockResolvedValue([]);
  const { result } = renderHook(usePerceptionSession);
  await act(async () => {
    await mock.connection?.("connected");
  });
  act(() => mock.frame?.({ ...state }));
  let resolved = false;
  const pending = result.current
    .waitForFrame({ ...state, frame_id: 5 })
    .then(() => {
      resolved = true;
    });
  await new Promise((resolve) => setTimeout(resolve, 80));
  expect(resolved).toBe(false);
  act(() => mock.frame?.({ ...state, frame_id: 5 }));
  await pending;
  expect(resolved).toBe(true);
});
it("reports command errors without applying optimistic state", async () => {
  const state = {
    scene_id: "normal_road",
    revision: 0,
    frame_id: 1,
    is_running: true,
  } as SessionState;
  mock.config.mockResolvedValue({ session: state });
  mock.scenes.mockResolvedValue([]);
  mock.send.mockRejectedValue(new Error("Command rejected"));
  const { result } = renderHook(usePerceptionSession);
  await act(async () => {
    await mock.connection?.("connected");
  });
  await act(async () => {
    await expect(result.current.command({ action: "pause" })).rejects.toThrow(
      "Command rejected",
    );
  });
  expect(result.current.session?.is_running).toBe(true);
  expect(result.current.error).toBe("Command rejected");
  expect(result.current.busy).toBe(false);
});
