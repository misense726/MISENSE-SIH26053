import { useCallback, useEffect, useRef, useState } from "react";
import { fetchConfig, fetchScenes } from "../services/api";
import { wsService } from "../services/websocket";
import type {
  ConnectionState,
  ControlCommand,
  LidarFrame,
  SceneInfo,
  SessionState,
  SystemConfig,
} from "../types";

export function acceptsFrame(
  frame: LidarFrame,
  state: SessionState | null,
): boolean {
  return (
    !state ||
    frame.revision > state.revision ||
    (frame.revision === state.revision &&
      frame.scene_id === state.scene_id &&
      frame.frame_id >= state.frame_id)
  );
}

export function usePerceptionSession() {
  const [frame, setFrame] = useState<LidarFrame | null>(null);
  const [session, setSession] = useState<SessionState | null>(null);
  const [config, setConfig] = useState<SystemConfig | null>(null);
  const [scenes, setScenes] = useState<SceneInfo[]>([]);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [receivedFps, setReceivedFps] = useState(0);
  const [stale, setStale] = useState(false);
  const frameRef = useRef<LidarFrame | null>(null);
  const sessionRef = useRef<SessionState | null>(null);
  const lastArrival = useRef(0);
  const arrivals = useRef<number[]>([]);
  const mounted = useRef(false);
  const connectionRef = useRef<ConnectionState>("connecting");

  useEffect(() => {
    mounted.current = true;
    let generation = 0;
    const unsubscribeSession = wsService.onSession((next) => {
      sessionRef.current = next;
      setSession(next);
    });
    const unsubscribeFrame = wsService.onFrame((next) => {
      if (!acceptsFrame(next, sessionRef.current)) return;
      frameRef.current = next;
      setFrame(next);
      lastArrival.current = performance.now();
      arrivals.current.push(lastArrival.current);
      setStale(false);
    });
    const unsubscribeConnection = wsService.onConnectionChange(
      async (state) => {
        connectionRef.current = state;
        setConnection(state);
        const current = ++generation;
        if (state !== "connected") return;
        // A restarted backend can begin again at revision zero.
        sessionRef.current = null;
        frameRef.current = null;
        setSession(null);
        arrivals.current = [];
        try {
          const [nextConfig, nextScenes] = await Promise.all([
            fetchConfig(),
            fetchScenes(),
          ]);
          if (!mounted.current || current !== generation) return;
          sessionRef.current = nextConfig.session;
          setSession(nextConfig.session);
          setConfig(nextConfig);
          setScenes(nextScenes);
          setError(null);
        } catch (cause) {
          if (mounted.current && current === generation)
            setError(errorMessage(cause));
        }
      },
    );
    wsService.connect();
    const tick = window.setInterval(() => {
      const now = performance.now();
      arrivals.current = arrivals.current.filter((time) => now - time < 2000);
      setReceivedFps(arrivals.current.length / 2);
      setStale(
        !!frameRef.current &&
          (connectionRef.current !== "connected" ||
            (!!sessionRef.current?.is_running &&
              now - lastArrival.current > 5000)),
      );
    }, 1000);
    return () => {
      mounted.current = false;
      ++generation;
      clearInterval(tick);
      unsubscribeFrame();
      unsubscribeSession();
      unsubscribeConnection();
      wsService.disconnect();
    };
  }, []);

  const command = useCallback(
    async (action: ControlCommand): Promise<SessionState> => {
      setBusy(true);
      setError(null);
      try {
        const result = await wsService.send(action);
        if (mounted.current) {
          sessionRef.current = result.state;
          setSession(result.state);
        }
        return result.state;
      } catch (cause) {
        if (mounted.current) setError(errorMessage(cause));
        throw cause;
      } finally {
        if (mounted.current) setBusy(false);
      }
    },
    [],
  );

  const waitForFrame = useCallback(
    async (state: SessionState): Promise<LidarFrame> => {
      const deadline = performance.now() + 20000;
      while (mounted.current && performance.now() < deadline) {
        const latest = frameRef.current;
        if (
          latest &&
          latest.scene_id === state.scene_id &&
          latest.revision === state.revision &&
          latest.frame_id >= state.frame_id
        )
          return latest;
        if (connectionRef.current !== "connected")
          throw new Error(
            "The simulation disconnected while preparing this scene.",
          );
        await new Promise((resolve) => setTimeout(resolve, 60));
      }
      throw new Error(
        "The scene is taking longer than expected. Retry this step.",
      );
    },
    [],
  );

  return {
    frame,
    frameRef,
    session,
    config,
    scenes,
    connection,
    busy,
    stale,
    receivedFps,
    error,
    setError,
    command,
    waitForFrame,
    retry: () => wsService.retry(),
    ready: connection === "connected" && !!session,
  };
}

export function errorMessage(cause: unknown): string {
  return cause instanceof Error
    ? cause.message
    : "The action could not finish. Try again.";
}
