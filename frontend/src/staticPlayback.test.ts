import { expect, it } from "vitest";
import { nextFrameIndex } from "./staticPlayback";

it("returns to the first captured frame after the last one", () => {
  expect(nextFrameIndex(69, 70)).toBe(0);
  expect(nextFrameIndex(0, 70)).toBe(1);
  expect(nextFrameIndex(0, 0)).toBe(0);
});
