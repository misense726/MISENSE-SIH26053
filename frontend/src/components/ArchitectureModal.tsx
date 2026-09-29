import {
  ArrowRight,
  ScanLine,
  LocateFixed,
  Mountain,
  Grid2X2,
} from "lucide-react";
const stages = [
  {
    name: "Scan",
    icon: ScanLine,
    detail: "Simulated LiDAR returns are cropped to the map region.",
  },
  {
    name: "Detect & track",
    icon: LocateFixed,
    detail:
      "Simulated detections feed Kalman tracking. Moving-object returns are filtered from terrain.",
  },
  {
    name: "Measure",
    icon: Mountain,
    detail:
      "Static points provide elevation statistics and geometric terrain features.",
  },
  {
    name: "Refine",
    icon: Grid2X2,
    detail:
      "The quadtree splits cells by importance and combines terrain and object information.",
  },
];
export function ArchitectureModal() {
  return (
    <div
      className="pipeline"
      role="group"
      aria-label="Current prototype pipeline"
    >
      {stages.map(({ name, icon: Icon, detail }, index) => (
        <div className="pipeline-stage" key={name}>
          <div>
            <Icon size={23} />
            <span>{index + 1}</span>
          </div>
          <h3>{name}</h3>
          <p>{detail}</p>
          {index < stages.length - 1 && (
            <ArrowRight
              className="pipeline-arrow"
              size={18}
              aria-hidden="true"
            />
          )}
        </div>
      ))}
    </div>
  );
}
