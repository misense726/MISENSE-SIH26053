import * as THREE from "three";
import type { AdaptiveCell } from "../types";
import { cellColor, cellIdentity, measuredHeight } from "./palette";
import type { MapViewPreset } from "./palette";

/** Grow geometrically, not once per frame. Dispose replaced GPU attributes too. */
export class VertexBatch {
  readonly object: THREE.LineSegments | THREE.Points;
  private capacity = 0;
  private used = 0;
  private positions = new Float32Array(0);
  private colors = new Float32Array(0);

  constructor(kind: "lines" | "points", opacity = 1) {
    const geometry = new THREE.BufferGeometry();
    this.object =
      kind === "lines"
        ? new THREE.LineSegments(
            geometry,
            new THREE.LineBasicMaterial({
              vertexColors: true,
              transparent: opacity < 1,
              opacity,
            }),
          )
        : new THREE.Points(
            geometry,
            new THREE.PointsMaterial({
              vertexColors: true,
              size: 2.2,
              sizeAttenuation: false,
              transparent: true,
              opacity,
              depthWrite: false,
            }),
          );
    this.object.frustumCulled = false;
  }

  begin(requiredVertices: number): void {
    this.used = 0;
    if (requiredVertices <= this.capacity) return;
    this.capacity = Math.max(128, 2 ** Math.ceil(Math.log2(requiredVertices)));
    this.positions = new Float32Array(this.capacity * 3);
    this.colors = new Float32Array(this.capacity * 3);
    this.object.geometry.dispose();
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(this.positions, 3).setUsage(
        THREE.DynamicDrawUsage,
      ),
    );
    geometry.setAttribute(
      "color",
      new THREE.BufferAttribute(this.colors, 3).setUsage(
        THREE.DynamicDrawUsage,
      ),
    );
    this.object.geometry = geometry;
  }

  vertex(x: number, y: number, z: number, color: THREE.Color): void {
    const offset = this.used++ * 3;
    this.positions[offset] = x;
    this.positions[offset + 1] = y;
    this.positions[offset + 2] = z;
    this.colors[offset] = color.r;
    this.colors[offset + 1] = color.g;
    this.colors[offset + 2] = color.b;
  }

  segment(
    a: readonly number[],
    b: readonly number[],
    color: THREE.Color,
  ): void {
    this.vertex(a[0], a[1], a[2], color);
    this.vertex(b[0], b[1], b[2], color);
  }

  rectangle(
    x: number,
    y: number,
    width: number,
    height: number,
    z: number,
    color: THREE.Color,
  ): void {
    const left = x - width / 2,
      right = x + width / 2;
    const near = y - height / 2,
      far = y + height / 2;
    this.segment([left, near, z], [right, near, z], color);
    this.segment([right, near, z], [right, far, z], color);
    this.segment([right, far, z], [left, far, z], color);
    this.segment([left, far, z], [left, near, z], color);
  }

  end(): void {
    this.object.geometry.setDrawRange(0, this.used);
    if (this.used === 0) return;
    for (const name of ["position", "color"]) {
      const attribute = this.object.geometry.getAttribute(
        name,
      ) as THREE.BufferAttribute;
      attribute.clearUpdateRanges();
      attribute.addUpdateRange(0, this.used * 3);
      attribute.needsUpdate = true;
    }
  }

  dispose(): void {
    this.object.geometry.dispose();
    (this.object.material as THREE.Material).dispose();
  }
}

/** Spatial identities retain their instance slots even if the frame array reorders. */
export class CellInstances {
  mesh: THREE.InstancedMesh;
  private capacity = 0;
  private slots = new Map<string, number>();
  private cells: (AdaptiveCell | null)[] = [];
  private freeSlots: number[] = [];
  private readonly matrix = new THREE.Matrix4();
  private readonly hiddenMatrix = new THREE.Matrix4().makeScale(0, 0, 0);
  private readonly material = new THREE.MeshBasicMaterial({
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });

  constructor(private readonly scene: THREE.Scene) {
    this.mesh = this.createMesh(1);
    this.mesh.count = 0;
    scene.add(this.mesh);
  }

  private createMesh(capacity: number): THREE.InstancedMesh {
    this.capacity = capacity;
    const mesh = new THREE.InstancedMesh(
      new THREE.PlaneGeometry(1, 1),
      this.material,
      capacity,
    );
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.frustumCulled = false;
    return mesh;
  }

  update(
    cells: AdaptiveCell[],
    view: MapViewPreset,
    semantic: boolean,
    elevation: boolean,
  ): void {
    const live = new Set(cells.map(cellIdentity));
    for (const [key, slot] of this.slots) {
      if (!live.has(key)) {
        this.slots.delete(key);
        this.cells[slot] = null;
        this.freeSlots.push(slot);
        this.mesh.setMatrixAt(slot, this.hiddenMatrix);
      }
    }
    const required =
      this.cells.length +
      Math.max(0, cells.length - this.slots.size - this.freeSlots.length);
    if (required > this.capacity) {
      const old = this.mesh;
      this.mesh = this.createMesh(
        Math.max(256, 2 ** Math.ceil(Math.log2(required))),
      );
      this.mesh.visible = old.visible;
      this.scene.remove(old);
      old.geometry.dispose();
      old.dispose();
      this.scene.add(this.mesh);
      // Vacant slots must remain non-pickable after reallocating the instance buffer.
      for (let i = 0; i < this.cells.length; i++)
        this.mesh.setMatrixAt(i, this.hiddenMatrix);
    }
    for (const cell of cells) {
      const key = cellIdentity(cell);
      let slot = this.slots.get(key);
      if (slot === undefined) {
        slot = this.freeSlots.pop() ?? this.cells.length;
        this.slots.set(key, slot);
      }
      this.cells[slot] = cell;
      this.matrix.makeScale(cell.size, cell.size, 1);
      this.matrix.setPosition(cell.x, cell.y, measuredHeight(cell, elevation));
      this.mesh.setMatrixAt(slot, this.matrix);
      this.mesh.setColorAt(slot, cellColor(cell, view, semantic));
    }
    this.mesh.count = this.cells.length;
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
    this.mesh.boundingSphere = null;
  }

  cellAt(instanceId: number): AdaptiveCell | null {
    return this.cells[instanceId] ?? null;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.mesh.dispose();
    this.material.dispose();
    this.slots.clear();
    this.cells = [];
  }
}
