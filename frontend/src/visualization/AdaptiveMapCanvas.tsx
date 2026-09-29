import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { CellInstances, VertexBatch } from "./mapBuffers";
import { SceneModels, actorSize } from "./sceneModels";
import {
  MAP_PALETTE,
  MAP_FILLS,
  RESOLUTION_LEGEND,
  measuredHeight,
} from "./palette";
import type {
  AdaptiveCell,
  CameraPreset,
  LayerVisibility,
  LidarFrame,
  MapPreset,
  TrackedObject,
} from "../types";

export interface CameraState {
  position: [number, number, number];
  target: [number, number, number];
}
export interface CameraRequest {
  preset: CameraPreset;
  serial: number;
  target?: [number, number];
  restore?: CameraState;
}
interface Props {
  frame: LidarFrame | null;
  layers: LayerVisibility;
  preset: MapPreset;
  camera: CameraRequest;
  selectedCell: AdaptiveCell | null;
  selectedObject: TrackedObject | null;
  onSelectCell: (cell: AdaptiveCell | null) => void;
  onSelectObject: (object: TrackedObject | null) => void;
  onCamera: (state: CameraState) => void;
  onInspectTruck: () => void;
}
export function AdaptiveMapCanvas(props: Props) {
  const container = useRef<HTMLDivElement>(null);
  const latest = useRef(props);
  useEffect(() => {
    latest.current = props;
  });
  const [failure, setFailure] = useState(false);
  const runtime = useRef<{ update: () => void; camera: () => void } | null>(
    null,
  );
  useEffect(() => {
    const host = container.current!;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      queueMicrotask(() => setFailure(true));
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(MAP_PALETTE.background);
    renderer.domElement.setAttribute(
      "aria-label",
      "Interactive adaptive terrain map. Drag to orbit; scroll to zoom. Use Map features for keyboard selection.",
    );
    renderer.domElement.setAttribute("role", "img");
    host.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const objectScene = new THREE.Scene();
    const sky = new THREE.HemisphereLight("#FFFFFF", "#8A9AAA", 2.1);
    sky.position.set(0, 0, 30);
    objectScene.add(sky);
    const sun = new THREE.DirectionalLight("#FFF2DA", 2.5);
    sun.position.set(-12, -18, 28);
    objectScene.add(sun);
    const fill = new THREE.DirectionalLight("#C5DEFA", 0.7);
    fill.position.set(15, 12, 10);
    objectScene.add(fill);
    const models = new SceneModels(
      !window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    );
    objectScene.add(models.group);
    const camera = new THREE.PerspectiveCamera(43, 1, 0.1, 500);
    camera.up.set(0, 0, 1);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;
    controls.maxPolarAngle = Math.PI / 2 - 0.02;
    controls.minDistance = 3;
    controls.maxDistance = 160;
    const cells = new CellInstances(scene);
    const edges = new VertexBatch("lines", 0.48),
      points = new VertexBatch("points", 0.8);
    const actors = new VertexBatch("lines"),
      hazards = new VertexBatch("lines"),
      selected = new VertexBatch("lines");
    const guides = new VertexBatch("lines", 0.5);
    [edges, points, hazards, guides].forEach((batch) =>
      scene.add(batch.object),
    );
    objectScene.add(actors.object, selected.object);
    const ink = new THREE.Color(MAP_PALETTE.ink),
      grid = new THREE.Color(MAP_PALETTE.grid);
    const orange = new THREE.Color(MAP_PALETTE.obstacle),
      green = new THREE.Color(MAP_PALETTE.dynamic);
    const pointColor = new THREE.Color();
    guides.begin(100);
    guides.rectangle(0, 16, 64, 64, -0.07, grid);
    guides.segment([-27, -12, 0], [-22, -12, 0], ink);
    guides.segment([-27, -12, 0], [-27, -11.5, 0], ink);
    guides.segment([-22, -12, 0], [-22, -11.5, 0], ink);
    guides.end();
    const box = (
      batch: VertexBatch,
      position: number[],
      dimensions: number[],
      yaw: number,
      color: THREE.Color,
    ) => {
      const [x, y, centerZ] = position,
        [w, h, d] = dimensions;
      const z = centerZ - d / 2;
      const corners = [
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2],
      ].map(([a, b]) => [
        x + a * Math.cos(yaw) - b * Math.sin(yaw),
        y + a * Math.sin(yaw) + b * Math.cos(yaw),
      ]);
      for (let i = 0; i < 4; i++) {
        const a = corners[i],
          b = corners[(i + 1) % 4];
        batch.segment([a[0], a[1], z], [b[0], b[1], z], color);
        batch.segment([a[0], a[1], z + d], [b[0], b[1], z + d], color);
        batch.segment([a[0], a[1], z], [a[0], a[1], z + d], color);
      }
    };
    const applyCamera = () => {
      const request = latest.current.camera;
      if (request.restore) {
        camera.position.fromArray(request.restore.position);
        controls.target.fromArray(request.restore.target);
      } else {
        const [x, y] = request.target ?? [
          0,
          request.preset === "top" ? 16 : 10,
        ];
        controls.target.set(x, y, 0);
        if (request.preset === "top") camera.position.set(x, y - 0.01, 90);
        else if (request.preset === "focus")
          camera.position.set(x + 5, y - 8, 6.8);
        else camera.position.set(x + 27, y - 40, 37);
      }
      controls.update();
    };
    applyCamera();
    const onCameraChange = () =>
      latest.current.onCamera({
        position: camera.position.toArray() as CameraState["position"],
        target: controls.target.toArray() as CameraState["target"],
      });
    controls.addEventListener("change", onCameraChange);
    onCameraChange();
    const update = () => {
      const { frame, layers, preset, selectedCell, selectedObject } =
        latest.current;
      if (!frame) return;
      models.update(frame, layers.tracks, layers.detections);
      cells.update(
        frame.adaptive_cells,
        preset,
        layers.semanticMap,
        layers.elevation,
      );
      cells.mesh.visible = layers.surface;
      edges.begin(frame.adaptive_cells.length * 12);
      for (const cell of frame.adaptive_cells) {
        const z = measuredHeight(cell, layers.elevation) + 0.015;
        edges.rectangle(cell.x, cell.y, cell.size, cell.size, z, grid);
        if (!cell.point_count) {
          const d = cell.size * 0.09;
          edges.segment(
            [cell.x - d, cell.y - d, z],
            [cell.x + d, cell.y + d, z],
            grid,
          );
          edges.segment(
            [cell.x - d, cell.y + d, z],
            [cell.x + d, cell.y - d, z],
            grid,
          );
        }
      }
      edges.end();
      edges.object.visible = layers.adaptiveGrid;
      points.begin(frame.points.length);
      for (const [x, y, z] of frame.points)
        points.vertex(
          x,
          y,
          z,
          pointColor.setHSL(
            0.57,
            0.35,
            Math.max(0.22, Math.min(0.65, 0.32 + z * 0.12)),
          ),
        );
      points.end();
      points.object.visible = layers.points;
      actors.begin(
        frame.detections.length * 24 +
          frame.tracks.reduce((n, t) => n + 32 + t.history.length * 2, 0),
      );
      if (layers.detections)
        for (const item of frame.detections)
          box(actors, item.position, actorSize(item), item.yaw, orange);
      if (layers.tracks)
        for (const item of frame.tracks) {
          // The model provides the object silhouette; retain the ground footprint and trail.
          const size = actorSize(item);
          box(
            actors,
            [item.position[0], item.position[1], 0.035],
            [size[0], size[1], 0],
            item.yaw,
            green,
          );
          for (let i = 1; i < item.history.length; i++)
            actors.segment(
              [...item.history[i - 1], 0.2],
              [...item.history[i], 0.2],
              green,
            );
        }
      actors.end();
      hazards.begin(frame.terrain_features.length * 8);
      for (const feature of frame.terrain_features) {
        if (
          (feature.feature_type === "pothole" && !layers.potholes) ||
          (feature.feature_type === "curb" && !layers.curbs) ||
          (feature.feature_type === "slope" && !layers.slopes)
        )
          continue;
        const [a, b, c, d] = feature.bounds;
        hazards.rectangle((a + b) / 2, (c + d) / 2, b - a, d - c, 0.06, orange);
      }
      hazards.end();
      selected.begin(24);
      if (selectedCell)
        selected.rectangle(
          selectedCell.x,
          selectedCell.y,
          selectedCell.size,
          selectedCell.size,
          measuredHeight(selectedCell, layers.elevation) + 0.04,
          ink,
        );
      if (selectedObject && layers.tracks)
        box(
          selected,
          selectedObject.position,
          actorSize(selectedObject),
          selectedObject.yaw,
          ink,
        );
      selected.end();
      if (
        selectedObject &&
        layers.tracks &&
        latest.current.camera.preset === "focus"
      ) {
        const target = new THREE.Vector3(
          selectedObject.position[0],
          selectedObject.position[1],
          0,
        );
        camera.position.add(target.clone().sub(controls.target));
        controls.target.copy(target);
        controls.update();
      }
    };
    runtime.current = { update, camera: applyCamera };
    update();
    const resize = () => {
      const { width, height } = host.getBoundingClientRect();
      renderer.setSize(width, height);
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();
    const raycaster = new THREE.Raycaster();
    let down: [number, number] | null = null;
    const onDown = (event: PointerEvent) => {
      down = [event.clientX, event.clientY];
    };
    const onUp = (event: PointerEvent) => {
      if (
        !down ||
        Math.hypot(event.clientX - down[0], event.clientY - down[1]) > 5
      )
        return;
      down = null;
      const rect = host.getBoundingClientRect();
      raycaster.setFromCamera(
        new THREE.Vector2(
          ((event.clientX - rect.left) / rect.width) * 2 - 1,
          (-(event.clientY - rect.top) / rect.height) * 2 + 1,
        ),
        camera,
      );
      const model = models.pick(raycaster);
      if (model && "trackId" in model) {
        const actor = latest.current.frame?.tracks.find(
          (t) => t.track_id === model.trackId,
        );
        if (actor) latest.current.onSelectObject(actor);
        return;
      }
      if (model && "ego" in model) {
        latest.current.onInspectTruck();
        return;
      }
      const hit = raycaster.intersectObject(cells.mesh)[0];
      if (hit?.instanceId !== undefined) {
        const cell = cells.cellAt(hit.instanceId);
        const actor = latest.current.layers.tracks
          ? latest.current.frame?.tracks.find(
              (t) =>
                cell &&
                Math.abs(t.position[0] - cell.x) <
                  actorSize(t)[0] / 2 + cell.size / 2 &&
                Math.abs(t.position[1] - cell.y) <
                  actorSize(t)[1] / 2 + cell.size / 2,
            )
          : null;
        if (actor) latest.current.onSelectObject(actor);
        else latest.current.onSelectCell(cell);
      }
    };
    renderer.domElement.addEventListener("pointerdown", onDown);
    renderer.domElement.addEventListener("pointerup", onUp);
    let animation = 0;
    renderer.autoClear = false;
    renderer.info.autoReset = false;
    const draw = () => {
      renderer.info.reset();
      renderer.clear();
      renderer.render(scene, camera);
      // Object models are perception annotations. Elevated scan cells must not slice their bodies.
      renderer.clearDepth();
      renderer.render(objectScene, camera);
      host.dataset.geometries = String(renderer.info.memory.geometries);
      host.dataset.textures = String(renderer.info.memory.textures);
      host.dataset.drawCalls = String(renderer.info.render.calls);
      animation = requestAnimationFrame(draw);
    };
    draw();
    return () => {
      cancelAnimationFrame(animation);
      observer.disconnect();
      controls.dispose();
      runtime.current = null;
      renderer.domElement.removeEventListener("pointerdown", onDown);
      renderer.domElement.removeEventListener("pointerup", onUp);
      cells.dispose();
      [edges, points, actors, hazards, selected, guides].forEach((batch) =>
        batch.dispose(),
      );
      models.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
    };
  }, []);
  useEffect(
    () => runtime.current?.update(),
    [
      props.frame,
      props.layers,
      props.preset,
      props.selectedCell,
      props.selectedObject,
    ],
  );
  useEffect(() => runtime.current?.camera(), [props.camera]);
  return (
    <>
      <div className="map-canvas" ref={container} />
      {failure && (
        <div className="map-empty" role="alert">
          <h2>3D view unavailable</h2>
          <p>
            Enable browser hardware acceleration or use Compare and the Map
            features list.
          </p>
        </div>
      )}
      {!failure && !props.frame && (
        <div className="map-empty">
          <h2>Preparing the terrain map</h2>
          <p>Waiting for the simulation's first scan.</p>
        </div>
      )}
      <div className="map-legend">
        <strong>
          {props.preset === "resolution"
            ? "Actual cell widths"
            : props.preset === "scan"
              ? "Raw LiDAR scan"
              : "Terrain map"}
        </strong>
        {props.preset === "resolution" ? (
          <div className="legend-items">
            {RESOLUTION_LEGEND.map((item) => (
              <span key={item.size}>
                <i style={{ background: item.color }} />
                {item.size} m
              </span>
            ))}
          </div>
        ) : props.preset === "scan" ? (
          <span>
            Downsampled display ·{" "}
            {props.frame?.points.length.toLocaleString() ?? 0} points
          </span>
        ) : (
          <div className="legend-items">
            <span>
              <i style={{ background: MAP_FILLS.road }} />
              Road
            </span>
            <span>
              <i style={{ background: MAP_FILLS.obstacle }} />
              Hazard
            </span>
            <span>
              <i style={{ background: MAP_FILLS.dynamic }} />
              Object
            </span>
            <span>
              <i style={{ background: MAP_FILLS.terrain }} />
              Terrain
            </span>
            <span>× No returns</span>
          </div>
        )}
        <small>
          {props.layers.elevation
            ? "Measured height · 1× vertical scale"
            : "Height removed"}{" "}
          · 5 m world ruler
        </small>
      </div>
    </>
  );
}
