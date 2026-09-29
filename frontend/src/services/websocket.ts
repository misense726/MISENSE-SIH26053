import type {
  CommandResult,
  ConnectionState,
  ControlCommand,
  LidarFrame,
  SessionState,
} from "../types";

type Pending = {
  resolve: (result: CommandResult) => void;
  reject: (error: Error) => void;
  timer: number;
};

export class PerceptionWebSocket {
  private ws: WebSocket | null = null;
  private frameCallbacks = new Set<(frame: LidarFrame) => void>();
  private connectionCallbacks = new Set<(state: ConnectionState) => void>();
  private sessionCallbacks = new Set<(state: SessionState) => void>();
  private pending = new Map<string, Pending>();
  private reconnectTimer: number | null = null;
  private explicitlyClosed = false;
  private connectedBefore = false;
  private sequence = 0;

  constructor(
    private url = import.meta.env.VITE_WS_URL ||
      `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws/stream`,
  ) {}

  connect(): void {
    this.explicitlyClosed = false;
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) return;
    this.notifyConnection(this.connectedBefore ? "reconnecting" : "connecting");
    const socket = new WebSocket(this.url);
    this.ws = socket;
    socket.onopen = () => {
      if (this.ws !== socket) return;
      this.connectedBefore = true;
      this.notifyConnection("connected");
    };
    socket.onmessage = (event) => {
      if (this.ws !== socket) return;
      try {
        const message = JSON.parse(event.data);
        if (message.type === "session_state")
          this.sessionCallbacks.forEach((callback) => callback(message.state));
        if (isLidarFrame(message)) {
          this.frameCallbacks.forEach((callback) => callback(message));
        } else if (message.request_id && this.pending.has(message.request_id)) {
          const entry = this.pending.get(message.request_id)!;
          clearTimeout(entry.timer);
          this.pending.delete(message.request_id);
          if (message.type === "error" || message.result?.status === "error")
            entry.reject(
              new Error(
                message.message ||
                  message.result?.message ||
                  "The command failed.",
              ),
            );
          else if (message.result?.state) entry.resolve(message.result);
          else
            entry.reject(
              new Error(
                "The server did not confirm its state. Reconnect and try again.",
              ),
            );
        }
      } catch {
        console.warn("Received an unreadable frame from the simulation.");
      }
    };
    socket.onclose = () => {
      if (this.ws !== socket) return;
      this.ws = null;
      this.rejectPending("Connection lost. The command was not confirmed.");
      this.notifyConnection("disconnected");
      if (!this.explicitlyClosed)
        this.reconnectTimer = window.setTimeout(() => this.connect(), 1800);
    };
    socket.onerror = () => socket.close();
  }

  send(command: ControlCommand): Promise<CommandResult> {
    if (this.ws?.readyState !== WebSocket.OPEN)
      return Promise.reject(
        new Error("Connect to the simulation before using this control."),
      );
    const request_id = `ui-${Date.now()}-${++this.sequence}`;
    return new Promise((resolve, reject) => {
      const timer = window.setTimeout(() => {
        this.pending.delete(request_id);
        reject(
          new Error(
            "The simulation did not confirm this action. Retry when connected.",
          ),
        );
      }, 20000);
      this.pending.set(request_id, { resolve, reject, timer });
      this.ws!.send(JSON.stringify({ ...command, request_id }));
    });
  }

  onFrame(callback: (frame: LidarFrame) => void): () => void {
    this.frameCallbacks.add(callback);
    return () => {
      this.frameCallbacks.delete(callback);
    };
  }
  onConnectionChange(callback: (state: ConnectionState) => void): () => void {
    this.connectionCallbacks.add(callback);
    return () => {
      this.connectionCallbacks.delete(callback);
    };
  }
  onSession(callback: (state: SessionState) => void): () => void {
    this.sessionCallbacks.add(callback);
    return () => {
      this.sessionCallbacks.delete(callback);
    };
  }
  private notifyConnection(state: ConnectionState): void {
    this.connectionCallbacks.forEach((callback) => callback(state));
  }
  private rejectPending(message: string): void {
    this.pending.forEach(({ reject, timer }) => {
      clearTimeout(timer);
      reject(new Error(message));
    });
    this.pending.clear();
  }
  disconnect(): void {
    this.explicitlyClosed = true;
    if (this.reconnectTimer !== null) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
    const oldSocket = this.ws;
    this.ws = null;
    oldSocket?.close();
    this.rejectPending("Connection closed.");
  }
  retry(): void {
    this.disconnect();
    this.connect();
  }
}

export function isLidarFrame(value: unknown): value is LidarFrame {
  if (!value || typeof value !== "object") return false;
  const frame = value as Partial<LidarFrame>;
  return (
    typeof frame.scene_id === "string" &&
    typeof frame.revision === "number" &&
    !!frame.ego_state &&
    !!frame.metrics &&
    Array.isArray(frame.points) &&
    Array.isArray(frame.adaptive_cells)
  );
}
export const wsService = new PerceptionWebSocket();
