import { GUIDE_STEPS } from "../hooks/useGuidedDemo";
interface Props {
  step: number;
  busy: boolean;
  error: string | null;
  onStep: (step: number) => void;
  onExit: () => void;
  onEvidence: () => void;
}
export function JuryDemoOverlay({
  step,
  busy,
  error,
  onStep,
  onExit,
  onEvidence,
}: Props) {
  return (
    <section className="guide-dock" aria-label="Guided demo">
      <div
        className="guide-progress"
        role="group"
        aria-label={`Step ${step + 1} of 4`}
      >
        {GUIDE_STEPS.map((s, i) => (
          <span className={i === step ? "current" : ""} key={s.title}>
            {i + 1}
          </span>
        ))}
      </div>
      <div className="guide-copy">
        <h2>{GUIDE_STEPS[step].title}</h2>
        <p>
          {busy
            ? "Preparing the scene and waiting for confirmation…"
            : GUIDE_STEPS[step].text}
        </p>
        {error && (
          <p role="alert">
            {error}
            <button onClick={() => onStep(step)} disabled={busy}>
              Retry step
            </button>
          </p>
        )}
      </div>
      <div className="guide-actions">
        <button disabled={busy || step === 0} onClick={() => onStep(step - 1)}>
          Back
        </button>
        {step < 3 ? (
          <button
            className="primary"
            disabled={busy}
            onClick={() => onStep(step + 1)}
          >
            Next
          </button>
        ) : (
          <>
            <button className="primary" disabled={busy} onClick={onExit}>
              Explore scenarios
            </button>
            <button disabled={busy} onClick={onEvidence}>
              View evidence
            </button>
          </>
        )}
        <button disabled={busy} onClick={() => onStep(0)}>
          Restart guide
        </button>
        <button disabled={busy} onClick={onExit}>
          Exit guide
        </button>
      </div>
    </section>
  );
}
