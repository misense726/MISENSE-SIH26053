import type {
  ComparisonSnapshot,
  ImportanceWeights,
  SceneInfo,
  SystemConfig,
} from "../types";

export const API_BASE = import.meta.env.VITE_API_URL || "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    signal: init?.signal ?? AbortSignal.timeout(60000),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.detail || `Request failed (${response.status}). Try again.`,
    );
  }
  return response.json() as Promise<T>;
}

export const fetchHealth = () =>
  request<{ status: string; scene: string; is_running: boolean }>("/health");
export const fetchScenes = () => request<SceneInfo[]>("/scenes");
export const fetchConfig = () => request<SystemConfig>("/config");
export const captureComparison = () =>
  request<ComparisonSnapshot>("/comparison", { method: "POST" });
export const updateWeights = (weights: Partial<ImportanceWeights>) =>
  request("/config/weights", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(weights),
  });
export const fetchArchitectureData = () => request("/architecture");
export const fetchJuryDemoSteps = () => request<unknown[]>("/jury-demo/steps");
