# MI Sense architecture

## Running pipeline

The FastAPI application starts one simulation engine. A frame advances the world, generates simulated LiDAR returns, crops points, simulates actor detections, updates tracking, filters moving-object returns, builds an adaptive elevation map, fuses terrain labels and emits a frame.

```text
Simulation world and LiDAR
    |
Road-datum and ego-coordinate conversion
    |
ROI crop -> simulated detections -> Kalman / Hungarian tracking
    |                                      |
Static point filtering <-------------------+
    |
Elevation statistics and scenario-assisted terrain features
    |
Importance-weighted quadtree -> semantic fusion -> frame metrics
    |
WebSocket -> session hook -> Three.js map / contextual inspector
```

The target stream rate is 15 Hz; the measured delivered rate depends on processing and transport. Playback speed changes the simulated time step once per frame. It does not multiply both frame frequency and the time step.

## Coordinate contract

The LiDAR simulator emits sensor-relative points. At the engine boundary, z is translated by the 1.73 m sensor height so the nominal road datum is zero. Actor boxes are converted from world coordinates into the ego frame before entering the simulated detector. Terrain markers use the same height datum. Quadtree ego proximity uses the local origin.

This boundary matters: interpreting sensor-relative road returns as road-relative heights marks a flat road as a pothole. A regression test checks the road datum and semantic label.

## Session contract

HTTP health/config responses include the scene, revision, last frame ID, playback state, speed, adaptive setting, weights, ROI and storage assumptions. Model metadata distinguishes simulated detections from loaded inference.

Commands carry request IDs. Acknowledgements contain the corresponding ID, status and authoritative session state. Scene changes, resets and mapping configuration changes advance the revision. Frames carry scene, revision and frame identity. The client rejects stale frames, waits for the acknowledged frame during guide transitions, and rehydrates after reconnecting. Other connected clients receive session-state updates.

Play, pause and speed changes preserve frame identity. Step pauses and produces one new frame. Tuning while paused rebuilds the captured input without advancing simulation time. Invalid scene names and weights return errors.

## Comparison contract

The engine keeps the latest immutable pipeline snapshot, including all cropped static input, feature and track copies, compact adaptive cells, metrics and geometry settings. `POST /api/comparison` builds a separate uniform elevation map in a worker thread. It never changes the live quadtree.

Concurrent requests share the same pending comparison. Only one completed result is retained. The response identifies its scene, revision, frame and snapshot and includes both grids. Each compact row contains center x/y, cell size, measured height, return count, semantic label and drivability.

Canvas2D renders the full ROI with linked cameras. Fine cell borders are hidden only when subpixel; the cells themselves are never arbitrarily dropped. A location selection inspects both representations at the same position. Missing returns have no measured height or drivability claim.

## Storage estimates

Uniform cells use an assumed 48 bytes each; adaptive cells use 56 bytes. Divide by 1024 for KiB. The 64 × 64 m uniform grid at 0.25 m has 65,536 cells and an estimated 3,072 KiB. Adaptive storage uses the actual captured cell count.

These figures exclude point buffers, tracks, allocators and application overhead. They compare two 2.5D representations, not measured process memory or a uniform 3D voxel map.

## Frontend ownership

`usePerceptionSession` owns connection, authoritative session, incoming frames and command errors. `useGuidedDemo` owns guide transitions and restoration. Selection uses spatial cell identity or track ID. Renderer buffers grow geometrically and are disposed on teardown. Points and line edges use reusable batches; cells use instanced geometry. ResizeObserver follows viewport, drawer and presentation changes.

Presets choose coherent layer combinations. Surface, edges, detections and tracks can also be changed independently in Advanced. A click is distinguished from an orbit drag. Keyboard selection and the comparison table expose map measurements without pointer picking.

## Prototype limits

The current detector always uses simulated actor data. CUDA and OpenPCDet dependency probes do not load a trained model. Terrain segmentation combines geometric heuristics with simulated scenario information. Confidence is not accuracy.

Cells range from 0.25 to 4 m; the simulated range is 50 m. The brief's 5 cm / 100 m policy, trained perception, classification accuracy by distance, physical testing and a uniform 3D benchmark remain separate engineering work.
