import * as THREE from "three";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import type { Detection3D, LidarFrame, TrackedObject } from "../types";

type Actor = Detection3D | TrackedObject;
type ModelKind = "truck" | "car" | "van" | "person" | "barrier";
type Position = [number, number, number];

// Vehicle dimensions arrive as length, width, height. Their forward axis is +Y.
export function actorSize(
  actor: Pick<Actor, "class_name" | "dimensions">,
): Position {
  const [length, width, height] = actor.dimensions;
  return [width, length, height];
}

const FINISHES = {
  paint: { color: "#D9A13D", roughness: 0.32, metalness: 0.18 },
  traffic: { color: "#40766F", roughness: 0.3, metalness: 0.2 },
  white: { color: "#D9E2E7", roughness: 0.4, metalness: 0.15 },
  trim: { color: "#253240", roughness: 0.65, metalness: 0.1 },
  rubber: { color: "#161F28", roughness: 0.95 },
  glass: { color: "#34596D", roughness: 0.15, metalness: 0.35 },
  reflection: { color: "#89B7C9", roughness: 0.22, metalness: 0.25 },
  alloy: { color: "#8FA1AD", roughness: 0.3, metalness: 0.65 },
  lamp: { color: "#E7F7FF", emissive: "#A6CFE3", emissiveIntensity: 0.25 },
  brake: { color: "#A72720", emissive: "#D83823", emissiveIntensity: 0.2 },
  amber: { color: "#F3AE2C", emissive: "#FF9D16", emissiveIntensity: 0.2 },
  skin: { color: "#BB8468", roughness: 0.8 },
  fabric: { color: "#294D65", roughness: 1 },
} satisfies Record<string, THREE.MeshStandardMaterialParameters>;
type Finish = keyof typeof FINISHES;

/** Merge fixed parts by finish. Instances share these meshes and their materials. */
class ModelBuilder {
  private parts = new Map<Finish, THREE.BufferGeometry[]>();
  constructor(private materials: Record<Finish, THREE.MeshStandardMaterial>) {}

  add(
    geometry: THREE.BufferGeometry,
    finish: Finish,
    at: Position = [0, 0, 0],
    rotation: Position = [0, 0, 0],
  ) {
    const unindexed = geometry.index ? geometry.toNonIndexed() : geometry;
    if (unindexed !== geometry) geometry.dispose();
    const matrix = new THREE.Matrix4().compose(
      new THREE.Vector3(...at),
      new THREE.Quaternion().setFromEuler(new THREE.Euler(...rotation)),
      new THREE.Vector3(1, 1, 1),
    );
    unindexed.applyMatrix4(matrix);
    if (!unindexed.getAttribute("uv"))
      unindexed.setAttribute(
        "uv",
        new THREE.Float32BufferAttribute(
          new Float32Array(unindexed.getAttribute("position").count * 2),
          2,
        ),
      );
    const entries = this.parts.get(finish) ?? [];
    entries.push(unindexed);
    this.parts.set(finish, entries);
  }

  box(size: Position, at: Position, finish: Finish, radius = 0.025) {
    this.add(
      new RoundedBoxGeometry(
        ...size,
        2,
        Math.min(radius, ...size.map((v) => v / 3)),
      ),
      finish,
      at,
    );
  }

  cylinder(
    radius: number,
    height: number,
    at: Position,
    finish: Finish,
    axis: "x" | "z" = "z",
  ) {
    this.add(
      new THREE.CylinderGeometry(radius, radius, height, 20),
      finish,
      at,
      axis === "x" ? [0, 0, Math.PI / 2] : [Math.PI / 2, 0, 0],
    );
  }

  panel(vertices: Position[], finish: Finish) {
    const points: number[] = [];
    for (let i = 1; i < vertices.length - 1; i++)
      points.push(...vertices[0], ...vertices[i], ...vertices[i + 1]);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(points, 3),
    );
    geometry.computeVertexNormals();
    this.add(geometry, finish);
  }

  hull(
    width: number,
    profile: [number, number][],
    finish: Finish,
    at: Position = [0, 0, 0],
  ) {
    const shape = new THREE.Shape(
      profile.map(([y, z]) => new THREE.Vector2(y, z)),
    );
    const geometry = new THREE.ExtrudeGeometry(shape, {
      depth: width,
      bevelEnabled: false,
      steps: 1,
    });
    geometry.applyMatrix4(
      new THREE.Matrix4().makeBasis(
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3(0, 0, 1),
        new THREE.Vector3(1, 0, 0),
      ),
    );
    geometry.translate(-width / 2, 0, 0);
    this.add(geometry, finish, at);
  }

  finish(name: string): THREE.Group {
    const group = new THREE.Group();
    group.name = name;
    for (const [finish, geometries] of this.parts) {
      const combined = mergeGeometries(geometries)!;
      geometries.forEach((geometry) => geometry.dispose());
      const mesh = new THREE.Mesh(combined, this.materials[finish]);
      mesh.name = finish;
      group.add(mesh);
    }
    this.parts.clear();
    return group;
  }
}

function wheel(
  materials: Record<Finish, THREE.MeshStandardMaterial>,
): THREE.Group {
  const b = new ModelBuilder(materials);
  b.cylinder(0.355, 0.25, [0, 0, 0], "rubber", "x");
  b.add(
    new THREE.TorusGeometry(0.285, 0.085, 8, 24),
    "rubber",
    [0, 0, 0],
    [0, Math.PI / 2, 0],
  );
  b.cylinder(0.205, 0.264, [0, 0, 0], "alloy", "x");
  b.cylinder(0.142, 0.27, [0, 0, 0], "trim", "x");
  b.cylinder(0.072, 0.29, [0, 0, 0], "alloy", "x");
  for (const side of [-1, 1]) {
    for (let i = 0; i < 6; i++) {
      const angle = (i * Math.PI) / 3;
      b.add(
        new THREE.BoxGeometry(0.014, 0.033, 0.14),
        "alloy",
        [side * 0.139, Math.sin(angle) * 0.115, Math.cos(angle) * 0.115],
        [-angle, 0, 0],
      );
    }
  }
  return b.finish("wheel");
}

function vehicle(
  kind: "truck" | "car" | "van",
  materials: Record<Finish, THREE.MeshStandardMaterial>,
  tire: THREE.Group,
): THREE.Group {
  const b = new ModelBuilder(materials);
  const truck = kind === "truck",
    van = kind === "van";
  const paint: Finish = truck ? "paint" : van ? "white" : "traffic";
  b.box([1.45, 3.95, 0.19], [0, 0, 0.45], "trim");
  b.box([1.78, 4.1, 0.29], [0, 0, 0.69], paint, 0.075);
  b.box([1.7, 0.18, 0.21], [0, 2.06, 0.6], "trim", 0.045);
  b.box([1.7, 0.14, 0.19], [0, -2.09, 0.6], "trim", 0.035);
  if (truck) {
    b.hull(
      1.76,
      [
        [-0.55, 0.79],
        [-0.55, 1.27],
        [-0.34, 1.49],
        [0.68, 1.49],
        [1.1, 1.13],
        [2, 1.04],
        [2.09, 0.79],
      ],
      paint,
    );
    b.box([1.63, 1.04, 0.075], [0, 0.15, 1.51], paint);
    b.box([1.71, 0.91, 0.055], [0, 1.55, 1.06], paint);
    // Open bed, liner, rails and tailgate.
    b.box([1.53, 1.51, 0.075], [0, -1.34, 0.84], "trim");
    for (let x = -0.61; x <= 0.62; x += 0.15)
      b.box([0.025, 1.35, 0.018], [x, -1.34, 0.885], "alloy", 0.004);
    for (const side of [-1, 1]) {
      b.box([0.13, 1.63, 0.41], [side * 0.82, -1.32, 1], paint);
      b.box([0.16, 1.64, 0.045], [side * 0.82, -1.32, 1.215], "trim");
      b.cylinder(0.035, 0.43, [side * 0.7, -0.65, 1.31], "trim");
    }
    b.cylinder(0.038, 1.4, [0, -0.65, 1.52], "trim", "x");
    b.box([1.55, 0.12, 0.38], [0, -2.08, 0.99], paint);
    b.box([0.35, 0.021, 0.065], [0, -2.15, 1.06], "trim");
    b.box([1.41, 0.25, 0.18], [0, -0.71, 0.98], "alloy");
    b.panel(
      [
        [-0.8, 1.092, 1.15],
        [0.8, 1.092, 1.15],
        [0.78, 0.726, 1.465],
        [-0.78, 0.726, 1.465],
      ],
      "glass",
    );
    b.panel(
      [
        [-0.77, 0.998, 1.235],
        [0.77, 0.998, 1.235],
        [0.76, 0.947, 1.278],
        [-0.76, 0.947, 1.278],
      ],
      "reflection",
    );
    b.box([1.44, 0.018, 0.25], [0, -0.556, 1.23], "glass", 0.015);
    for (const side of [-1, 1]) {
      const x = side * 0.889;
      b.panel(
        [
          [x, -0.43, 1.1],
          [x, -0.03, 1.1],
          [x, -0.03, 1.435],
          [x, -0.32, 1.435],
        ],
        "glass",
      );
      b.panel(
        [
          [x, 0.04, 1.1],
          [x, 1.02, 1.1],
          [x, 0.64, 1.445],
          [x, 0.04, 1.445],
        ],
        "glass",
      );
      b.box([0.025, 0.05, 0.61], [x, 0, 1.04], "trim", 0.005);
      b.box([0.035, 0.16, 0.037], [x, 0.15, 0.985], "trim");
    }
    // LiDAR sits on the cab roof; no decorative scan rays obscure the map.
    b.cylinder(0.19, 0.07, [0, 0.14, 1.59], "alloy");
    b.cylinder(0.155, 0.12, [0, 0.14, 1.68], "glass");
    b.cylinder(0.16, 0.035, [0, 0.14, 1.755], "trim");
    for (const x of [-0.54, 0.54]) {
      b.box([0.2, 0.19, 0.045], [x, 0.18, 1.575], "trim");
      b.box([0.15, 0.14, 0.09], [x, 0.18, 1.64], "amber", 0.03);
    }
  } else {
    const roof = van ? 1.94 : 1.48;
    const rear = van ? -1.87 : -0.95;
    b.hull(
      1.73,
      [
        [-2.02, 0.79],
        ...(van
          ? [[rear, roof - 0.1] as [number, number]]
          : ([
              [-2.02, 0.93],
              [-1.47, 0.99],
            ] as [number, number][])),
        [rear + 0.18, roof],
        [0.52, roof],
        [1.18, 1.02],
        [2.03, 0.97],
        [2.1, 0.79],
      ],
      paint,
    );
    b.box(
      [1.6, 0.43 - (rear + 0.18), 0.065],
      [0, (0.43 + rear + 0.18) / 2, roof + 0.007],
      paint,
    );
    const windscreenY = (z: number) =>
      0.52 + ((roof - z) * 0.66) / (roof - 1.02) + 0.012;
    b.panel(
      [
        [-0.8, windscreenY(1.06), 1.06],
        [0.8, windscreenY(1.06), 1.06],
        [0.76, windscreenY(roof - 0.05), roof - 0.05],
        [-0.76, windscreenY(roof - 0.05), roof - 0.05],
      ],
      "glass",
    );
    for (const side of [-1, 1]) {
      const x = side * 0.875;
      b.panel(
        [
          [x, -0.32, 1.05],
          [x, 1.08, 1.05],
          [x, 0.47, roof - 0.075],
          [x, -0.32, roof - 0.075],
        ],
        "glass",
      );
      if (!van)
        b.panel(
          [
            [x, -1.31, 1.04],
            [x, -0.4, 1.04],
            [x, -0.4, roof - 0.075],
            [x, -0.77, roof - 0.075],
          ],
          "glass",
        );
      b.box([0.022, 0.048, 0.53], [x, -0.36, 1.12], "trim", 0.005);
      b.box([0.024, 0.18, 0.035], [x, -0.18, 0.94], "alloy");
      if (van) {
        b.box([0.024, 1.45, 0.035], [x, -1.04, 1.1], "alloy");
        b.box([0.024, 0.035, 1.0], [x, -1.73, 1.3], "alloy");
      }
    }
    if (van) {
      b.box([0.04, 0.025, 1.14], [0, -1.92, 1.31], "trim");
      for (const x of [-0.39, 0.39])
        b.box([0.65, 0.035, 0.49], [x, -1.93, 1.53], "glass");
    } else {
      b.panel(
        [
          [-0.75, -1.425, 1.03],
          [-0.74, -0.832, 1.445],
          [0.74, -0.832, 1.445],
          [0.75, -1.425, 1.03],
        ],
        "glass",
      );
    }
  }
  for (const side of [-1, 1]) {
    b.box([0.12, 0.29, 0.1], [side * 0.91, 0.88, 1.12], "trim");
    b.box([0.15, 0.26, 0.15], [side * 1.0, 0.83, 1.18], paint);
    b.box([0.16, 0.03, 0.1], [side * 1.0, 0.686, 1.18], "reflection");
    b.box([0.26, 0.066, 0.13], [side * 0.7, 2.075, 0.9], "lamp");
    b.box([0.1, 0.07, 0.095], [side * 0.85, 2.06, 0.89], "amber");
    b.box([0.16, 0.07, 0.23], [side * 0.76, -2.13, 0.88], "brake");
    b.box([0.16, 0.075, 0.055], [side * 0.76, -2.135, 0.86], "lamp");
    b.box([0.13, 1.82, 0.065], [side * 0.91, 0.05, 0.46], "alloy");
    for (const y of [-1.39, 1.39]) {
      b.hull(
        0.14,
        [
          [-0.49, 0.05],
          [-0.47, 0.27],
          [-0.3, 0.45],
          [0.3, 0.45],
          [0.47, 0.27],
          [0.49, 0.05],
          [0.38, 0.05],
          [0.35, 0.23],
          [0.24, 0.34],
          [-0.24, 0.34],
          [-0.35, 0.23],
          [-0.38, 0.05],
        ],
        "trim",
        [side * 0.87, y, 0.37],
      );
    }
  }
  b.box([0.94, 0.04, 0.18], [0, 2.105, 0.86], "trim");
  for (let z = 0.79; z <= 0.94; z += 0.045)
    b.box([0.87, 0.027, 0.017], [0, 2.13, z], "alloy", 0.004);
  b.box([0.28, 0.027, 0.095], [0, 2.16, 0.62], "white");
  const root = b.finish(kind);
  for (const x of [-0.86, 0.86])
    for (const y of [-1.39, 1.39]) {
      const copy = tire.clone(true);
      copy.position.set(x, y, 0.375);
      root.add(copy);
    }
  return root;
}

function person(
  materials: Record<Finish, THREE.MeshStandardMaterial>,
): THREE.Group {
  const b = new ModelBuilder(materials);
  b.box([0.35, 0.23, 0.4], [0, 0, 1.14], "paint", 0.055);
  b.box([0.29, 0.21, 0.16], [0, 0, 0.9], "fabric", 0.035);
  b.cylinder(0.059, 0.1, [0, 0, 1.38], "skin");
  const head = new THREE.SphereGeometry(0.115, 16, 12);
  head.scale(0.89, 0.9, 1.16);
  b.add(head, "skin", [0, 0, 1.51]);
  b.add(
    new THREE.SphereGeometry(0.122, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2),
    "white",
    [0, 0, 1.565],
    [Math.PI / 2, 0, 0],
  );
  b.cylinder(0.145, 0.022, [0, 0.017, 1.565], "white");
  b.box([0.12, 0.03, 0.03], [0, 0.109, 1.53], "glass", 0.008);
  for (const side of [-1, 1]) {
    b.box([0.32, 0.016, 0.036], [0, side * 0.124, 1.035], "white", 0.004);
    for (const x of [-0.1, 0.1])
      b.box([0.032, 0.016, 0.27], [x, side * 0.124, 1.17], "white", 0.004);
  }
  const root = b.finish("person");
  for (const side of [-1, 1]) {
    const leg = new ModelBuilder(materials);
    leg.box([0.125, 0.145, 0.37], [0, 0, -0.17], "fabric", 0.042);
    leg.box([0.11, 0.13, 0.35], [0, 0.018, -0.51], "fabric", 0.035);
    leg.box([0.14, 0.27, 0.11], [0, 0.065, -0.725], "trim", 0.032);
    const joint = leg.finish(side === -1 ? "left-leg" : "right-leg");
    joint.position.set(side * 0.092, 0, 0.8);
    root.add(joint);
    const arm = new ModelBuilder(materials);
    arm.box([0.095, 0.12, 0.25], [0, 0, -0.115], "fabric", 0.036);
    arm.box([0.085, 0.105, 0.25], [0, 0.035, -0.335], "fabric", 0.034);
    arm.box([0.09, 0.11, 0.105], [0, 0.044, -0.48], "trim", 0.028);
    const shoulder = arm.finish(side === -1 ? "left-arm" : "right-arm");
    shoulder.position.set(side * 0.223, 0, 1.27);
    shoulder.rotation.y = side * -0.1;
    root.add(shoulder);
  }
  return root;
}

function barrier(
  materials: Record<Finish, THREE.MeshStandardMaterial>,
): THREE.Group {
  const b = new ModelBuilder(materials);
  b.hull(
    1.2,
    [
      [-0.21, 0],
      [0.21, 0],
      [0.14, 0.22],
      [0.07, 0.73],
      [-0.07, 0.73],
      [-0.14, 0.22],
    ],
    "white",
  );
  for (const x of [-0.4, 0, 0.4])
    for (const side of [-1, 1])
      b.box([0.19, 0.027, 0.22], [x, side * 0.11, 0.46], "paint", 0.003);
  b.box([0.08, 0.1, 0.035], [0, 0, 0.75], "amber");
  return b.finish("barrier");
}

interface ModelInstance {
  root: THREE.Group;
  wheels: THREE.Object3D[];
  leftLeg?: THREE.Object3D;
  rightLeg?: THREE.Object3D;
  leftArm?: THREE.Object3D;
  rightArm?: THREE.Object3D;
  previous?: THREE.Vector2;
  phase: number;
  kind: ModelKind;
  extent: THREE.Vector3;
}

export class SceneModels {
  readonly group = new THREE.Group();
  private materials = Object.fromEntries(
    Object.entries(FINISHES).map(([name, params]) => [
      name,
      new THREE.MeshStandardMaterial({ ...params, side: THREE.DoubleSide }),
    ]),
  ) as Record<Finish, THREE.MeshStandardMaterial>;
  private templates = new Map<ModelKind, THREE.Group>();
  private instances = new Map<string, ModelInstance>();
  private epoch = "";
  private ego: ModelInstance;
  private shadowGeometry = new THREE.PlaneGeometry(1, 1);
  private shadowMaterial = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    vertexShader:
      "varying vec2 p; void main(){p=uv*2.0-1.0;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}",
    fragmentShader:
      "varying vec2 p; void main(){float a=(1.0-smoothstep(0.2,1.0,length(p)))*0.22;gl_FragColor=vec4(0.07,0.12,0.17,a);}",
  });

  constructor(private animate = true) {
    const tire = wheel(this.materials);
    for (const kind of ["truck", "car", "van"] as const)
      this.templates.set(kind, vehicle(kind, this.materials, tire));
    this.templates.set("person", person(this.materials));
    this.templates.set("barrier", barrier(this.materials));
    this.ego = this.instantiate("truck", [2.05, 4.6, 1.73]);
    this.ego.root.name = "Survey truck";
    this.ego.root.userData.ego = true;
    this.group.add(this.ego.root);
  }

  private instantiate(kind: ModelKind, dimensions: Position): ModelInstance {
    const model = this.templates.get(kind)!.clone(true);
    const bounds = new THREE.Box3().setFromObject(model);
    const extent = bounds.getSize(new THREE.Vector3());
    const root = new THREE.Group();
    model.position.z = -bounds.min.z;
    root.add(model);
    root.scale.set(
      dimensions[0] / extent.x,
      dimensions[1] / extent.y,
      dimensions[2] / extent.z,
    );
    const shadow = new THREE.Mesh(this.shadowGeometry, this.shadowMaterial);
    shadow.scale.set(extent.x * 1.3, extent.y * 1.18, 1);
    shadow.position.z = 0.021;
    shadow.name = "contact-shadow";
    root.add(shadow);
    const wheels: THREE.Object3D[] = [];
    model.traverse((part) => {
      if (part.name === "wheel") wheels.push(part);
    });
    return {
      root,
      wheels,
      kind,
      extent,
      phase: 0,
      leftLeg: model.getObjectByName("left-leg"),
      rightLeg: model.getObjectByName("right-leg"),
      leftArm: model.getObjectByName("left-arm"),
      rightArm: model.getObjectByName("right-arm"),
    };
  }

  private pose(
    instance: ModelInstance,
    position: THREE.Vector2,
    walking: boolean,
  ) {
    if (instance.previous && this.animate) {
      const distance = position.distanceTo(instance.previous);
      // Scene resets and actors wrapping to the start are not travelled distance.
      if (distance < 5) instance.phase += distance;
    }
    instance.previous = position;
    for (const tire of instance.wheels)
      tire.rotation.x = -instance.phase / 0.36;
    if (
      instance.leftLeg &&
      instance.rightLeg &&
      instance.leftArm &&
      instance.rightArm
    ) {
      const stride = walking ? Math.sin(instance.phase * 5.4) * 0.38 : 0;
      instance.leftLeg.rotation.x = stride;
      instance.rightLeg.rotation.x = -stride;
      instance.leftArm.rotation.x = -stride * 0.75;
      instance.rightArm.rotation.x = stride * 0.75;
    }
  }

  update(frame: LidarFrame, showTracks: boolean, showDetections: boolean) {
    const epoch = `${frame.scene_id}:${frame.revision}`;
    if (epoch !== this.epoch) {
      for (const instance of this.instances.values())
        this.group.remove(instance.root);
      this.instances.clear();
      this.ego.previous = undefined;
      this.ego.phase = 0;
      this.epoch = epoch;
    }
    const [egoX, egoY] = frame.ego_state.position;
    this.pose(this.ego, new THREE.Vector2(egoX, egoY), false);
    const actors: Actor[] = showTracks
      ? frame.tracks
      : showDetections
        ? frame.detections
        : [];
    const retained = new Set<string>();
    for (const actor of actors) {
      const key =
        "track_id" in actor
          ? `track:${actor.track_id}`
          : `detection:${actor.id}`;
      const size = actorSize(actor);
      const kind: ModelKind =
        actor.class_name === "pedestrian"
          ? "person"
          : actor.class_name === "vehicle" ||
              (actor.class_name === "static_obstacle" &&
                size[1] > 3 &&
                size[0] > 1.3 &&
                size[2] > 1.3)
            ? size[2] >= 1.8
              ? "van"
              : "car"
            : "barrier";
      let instance = this.instances.get(key);
      if (instance && instance.kind !== kind) {
        this.group.remove(instance.root);
        this.instances.delete(key);
        instance = undefined;
      }
      if (!instance) {
        instance = this.instantiate(kind, size);
        this.instances.set(key, instance);
        this.group.add(instance.root);
      }
      instance.root.position.set(
        actor.position[0],
        actor.position[1],
        actor.position[2] - size[2] / 2,
      );
      instance.root.scale.set(
        size[0] / instance.extent.x,
        size[1] / instance.extent.y,
        size[2] / instance.extent.z,
      );
      instance.root.rotation.z = actor.yaw;
      instance.root.userData.trackId =
        "track_id" in actor ? actor.track_id : undefined;
      const velocity = actor.velocity ?? [0, 0];
      const moving =
        "dynamic_state" in actor
          ? actor.dynamic_state === "DYNAMIC"
          : Math.hypot(velocity[0], velocity[1]) > 0.15;
      if (kind === "person" && Math.hypot(velocity[0], velocity[1]) > 0.15)
        instance.root.rotation.z = Math.atan2(-velocity[0], velocity[1]);
      this.pose(
        instance,
        new THREE.Vector2(actor.position[0] + egoX, actor.position[1] + egoY),
        moving,
      );
      retained.add(key);
    }
    for (const [key, instance] of this.instances)
      if (!retained.has(key)) {
        this.group.remove(instance.root);
        this.instances.delete(key);
      }
  }

  /** Ignore translucent contact patches when picking a model. */
  pick(raycaster: THREE.Raycaster): { ego: true } | { trackId: number } | null {
    for (const hit of raycaster.intersectObject(this.group, true)) {
      if (hit.object.name === "contact-shadow") continue;
      let part: THREE.Object3D | null = hit.object;
      while (part && part !== this.group) {
        if (part.userData.ego) return { ego: true };
        if (part.userData.trackId !== undefined)
          return { trackId: part.userData.trackId };
        part = part.parent;
      }
    }
    return null;
  }

  dispose() {
    const geometries = new Set<THREE.BufferGeometry>();
    for (const template of this.templates.values())
      template.traverse((part) => {
        if (part instanceof THREE.Mesh) geometries.add(part.geometry);
      });
    geometries.forEach((geometry) => geometry.dispose());
    Object.values(this.materials).forEach((material) => material.dispose());
    this.shadowGeometry.dispose();
    this.shadowMaterial.dispose();
    this.group.clear();
    this.templates.clear();
    this.instances.clear();
  }
}
