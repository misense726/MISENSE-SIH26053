import { describe, it, expect, vi, afterEach } from "vitest";
import { acceptsFrame } from "./hooks/usePerceptionSession";
import { resolveCell } from "./hooks/sessionHelpers";
import { cellIdentity, measuredHeight } from "./visualization/palette";
import {
  canvasToWorld,
  worldToCanvas,
  projectCell,
  cellIndexAtPoint,
  createComparisonProjection,
  zoomComparisonCamera,
} from "./visualization/comparisonProjection";
import { PerceptionWebSocket } from "./services/websocket";
import type { AdaptiveCell, LidarFrame, SessionState } from "./types";
const state = { scene_id: "pothole", revision: 4, frame_id: 9 } as SessionState;
const frame = (revision: number, scene_id = "pothole", frame_id = 9) =>
  ({ revision, scene_id, frame_id }) as LidarFrame;
describe("session identity", () => {
  it("rejects old scenes, revisions and frames", () => {
    expect(acceptsFrame(frame(3), state)).toBe(false);
    expect(acceptsFrame(frame(4, "normal_road"), state)).toBe(false);
    expect(acceptsFrame(frame(4, "pothole", 8), state)).toBe(false);
    expect(acceptsFrame(frame(4), state)).toBe(true);
  });
  it("refreshes selections by bounds and clears missing cells", () => {
    const cell = {
      x: 1,
      y: 2,
      size: 0.25,
      cell_id: "a",
      elevation_mean: -0.1,
      point_count: 3,
    } as AdaptiveCell;
    const next = { ...cell, cell_id: "b", elevation_mean: -0.15 };
    expect(
      resolveCell({ adaptive_cells: [next] } as LidarFrame, cellIdentity(cell)),
    ).toBe(next);
    expect(
      resolveCell(
        { adaptive_cells: [] } as unknown as LidarFrame,
        cellIdentity(cell),
      ),
    ).toBeNull();
    expect(measuredHeight({ ...cell, point_count: 0 }, true)).toBe(0);
  });
});
describe("shared comparison projection", () => {
  it("maps cell centers and edges consistently and includes the full ROI", () => {
    const projection = createComparisonProjection([-32, 32, -16, 48], 672, 672);
    expect(
      canvasToWorld(worldToCanvas([-32, -16], projection), projection),
    ).toEqual([-32, -16]);
    expect(projectCell([-31.875, 47.875, 0.25], projection)).toEqual({
      x: 16,
      y: 16,
      width: 2.5,
      height: 2.5,
    });
    expect(
      cellIndexAtPoint(
        [
          [0, 0, 2],
          [2, 0, 2],
        ],
        [1, 0],
      ),
    ).toBe(1);
  });
  it("keeps the point under the cursor fixed while zooming", () => {
    const projection = createComparisonProjection([-32, 32, -16, 48], 500, 400);
    const anchor = [123, 75] as const,
      before = canvasToWorld(anchor, projection);
    const camera = zoomComparisonCamera(projection, 2, anchor);
    const after = canvasToWorld(
      anchor,
      createComparisonProjection([-32, 32, -16, 48], 500, 400, camera),
    );
    expect(after[0]).toBeCloseTo(before[0]);
    expect(after[1]).toBeCloseTo(before[1]);
  });
});
class FakeSocket {
  static OPEN = 1;
  static instances: FakeSocket[] = [];
  readyState = 0;
  onopen?: () => void;
  onclose?: () => void;
  onmessage?: (e: { data: string }) => void;
  sent: string[] = [];
  constructor() {
    FakeSocket.instances.push(this);
  }
  send(value: string) {
    this.sent.push(value);
  }
  close() {
    this.readyState = 3;
    this.onclose?.();
  }
  open() {
    this.readyState = 1;
    this.onopen?.();
  }
  message(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }
}
afterEach(() => {
  vi.unstubAllGlobals();
  FakeSocket.instances = [];
});
it("requires a matching acknowledgement and rejects server errors and disconnects", async () => {
  vi.stubGlobal("WebSocket", FakeSocket);
  const client = new PerceptionWebSocket("ws://localhost/test");
  client.connect();
  const socket = FakeSocket.instances[0];
  socket.open();
  const request = client.send({ action: "pause" });
  const id = JSON.parse(socket.sent[0]).request_id;
  socket.message({ request_id: id, result: { status: "success", state } });
  expect((await request).state).toEqual(state);
  const failure = client.send({ action: "set_scene", scene_id: "bad" });
  const assertion = expect(failure).rejects.toThrow("Bad scene");
  socket.message({
    request_id: JSON.parse(socket.sent[1]).request_id,
    result: { status: "error", message: "Bad scene" },
  });
  await assertion;
  const pending = client.send({ action: "play" });
  const disconnected = expect(pending).rejects.toThrow("Connection closed");
  client.disconnect();
  await disconnected;
  await expect(client.send({ action: "play" })).rejects.toThrow(
    "Connect to the simulation",
  );
});
