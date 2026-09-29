import { useEffect, useRef, type ReactNode } from "react";
import { X } from "lucide-react";

export function Panel({
  title,
  onClose,
  children,
  wide = false,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  wide?: boolean;
}) {
  const ref = useRef<HTMLElement>(null);
  const closeRef = useRef(onClose);
  useEffect(() => {
    closeRef.current = onClose;
  }, [onClose]);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    ref.current?.querySelector<HTMLButtonElement>("button")?.focus();
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeRef.current();
      }
      if (event.key !== "Tab") return;
      const items = Array.from(
        ref.current?.querySelectorAll<HTMLElement>(
          'button:not(:disabled), input:not(:disabled), select:not(:disabled), summary, a[href], [tabindex="0"]',
        ) ?? [],
      ).filter((node) => node.getClientRects().length);
      const first = items[0],
        last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("keydown", handleKey);
      previous?.focus();
    };
  }, []);
  return (
    <div
      className="panel-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        ref={ref}
        className={`drawer ${wide ? "drawer-wide" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        <div className="drawer-heading">
          <h2>{title}</h2>
          <button
            className="button icon-button"
            onClick={onClose}
            aria-label={`Close ${title.toLowerCase()}`}
          >
            <X size={20} />
          </button>
        </div>
        <div className="drawer-content">{children}</div>
      </section>
    </div>
  );
}
