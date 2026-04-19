/**
 * Heli.OS API client.
 *
 * All backend calls go through the API gateway at NEXT_PUBLIC_API_URL.
 * Policy denials (400/403) are surfaced via custom window events so
 * PolicyNotifications can render them globally.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8001/ws';

export interface ApiErrorDetail {
  message: string;
  policy_violations?: string[];
}

export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const url = path.startsWith('http') ? path : `${API_BASE}${path}`;
  const headers: HeadersInit = { 'Content-Type': 'application/json', ...(init?.headers || {}) };
  if (typeof window !== 'undefined') {
    const token = localStorage.getItem('access_token');
    if (token) (headers as Record<string, string>)['Authorization'] = `Bearer ${token}`;
  }
  const res = await fetch(url, { ...init, headers });
  if (!res.ok) {
    try {
      const data = await res.json();
      // Check for policy violation structure
      const detail = (data?.detail || data) as { message?: string; policy_violations?: string[]; deny_reasons?: string[] };
      const violations: string[] | undefined = detail?.policy_violations || detail?.deny_reasons;
      if (res.status === 400 || res.status === 403) {
        if (typeof window !== 'undefined') {
          window.dispatchEvent(new CustomEvent('policy-denied', {
            detail: {
              status: res.status,
              message: detail?.message || 'Policy denied',
              violations: violations || [],
            }
          }));
        }
      }
    } catch {
      // ignore parse errors
    }
  }
  return res;
}

async function apiJson<T = unknown>(path: string, init?: RequestInit): Promise<T> {
  const res = await apiFetch(path, init);
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json() as Promise<T>;
}

// --- Geofences ---

export interface GeofenceAPI {
  id?: number;
  name: string;
  type?: string;
  props?: Record<string, unknown>;
  coordinates?: { lat: number; lon: number }[];
  altitude_min?: number;
  altitude_max?: number;
  active?: boolean;
}

export async function fetchGeofences(): Promise<{ geofences: GeofenceAPI[] }> {
  return apiJson('/v1/geofences');
}

export async function createGeofence(geofence: GeofenceAPI): Promise<Record<string, unknown>> {
  return apiJson('/v1/geofences', { method: 'POST', body: JSON.stringify(geofence) });
}

export async function deleteGeofence(id: number): Promise<Record<string, unknown>> {
  return apiJson(`/v1/geofences/${id}`, { method: 'DELETE' });
}

// --- Alerts ---

export interface AlertAPI {
  alert_id: string;
  severity: string;
  description: string;
  source: string;
  ts_iso: string;
}

export async function fetchAlerts(limit = 100): Promise<{ alerts: AlertAPI[] }> {
  return apiJson(`/v1/alerts?limit=${limit}`);
}

// --- Missions ---

export interface MissionAPI {
  mission_id: string;
  name: string | null;
  objectives: string[];
  status: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export async function fetchMissions(limit = 50): Promise<MissionAPI[]> {
  return apiJson(`/v1/missions?limit=${limit}`);
}

export async function createMission(payload: Record<string, unknown>) {
  return apiJson('/v1/missions', { method: 'POST', body: JSON.stringify(payload) });
}

// --- Tasks / Dispatch ---

export interface TaskDispatch {
  asset_id: string;
  action: string;
  risk_level?: string;
  waypoints?: unknown[];
}

export async function dispatchTask(req: TaskDispatch): Promise<{ task_id: string; status: string }> {
  return apiJson('/v1/tasks', {
    method: 'POST',
    body: JSON.stringify({ risk_level: 'LOW', waypoints: [], ...req }),
  });
}

// --- Agent Commands (HALT / RTB / ACTIVATE_CAMERA) ---

export interface AgentCommand {
  entity_id: string;
  command: string;
  mission_objective?: string;
}

export async function sendAgentCommand(req: AgentCommand): Promise<unknown> {
  return apiJson('/agents', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

// --- Alert Acknowledgement ---

export async function acknowledgeAlert(alertId: string): Promise<unknown> {
  return apiJson(`/v1/alerts/${alertId}/acknowledge`, { method: 'POST' });
}

// --- Task Approval ---

export async function approveTask(taskId: string, approvedBy: string): Promise<unknown> {
  return apiJson(`/v1/tasks/${taskId}/approve`, {
    method: 'POST',
    body: JSON.stringify({ approved_by: approvedBy }),
  });
}

export async function fetchPendingApprovals(): Promise<{ task_id: string; asset_id: string; action: string; risk_level: string }[]> {
  return apiJson('/v1/tasks/pending');
}

// --- Mission Replay ---

export interface ReplayTimeline {
  mission_id: string;
  count: number;
  start: string | null;
  end: string | null;
  timestamps: string[];
}

export async function fetchReplayTimeline(missionId: string): Promise<ReplayTimeline> {
  return apiJson(`/v1/missions/${missionId}/replay/timeline`);
}

export async function fetchReplaySnapshot(missionId: string, ts?: string, index?: number) {
  const params = new URLSearchParams();
  if (ts) params.set('ts', ts);
  if (index !== undefined) params.set('index', String(index));
  return apiJson(`/v1/missions/${missionId}/replay/snapshot?${params}`);
}

// --- HLS Video ---

export async function startHLSStream(streamId: string, rtspUrl: string) {
  return apiJson(`/v1/video/hls/${streamId}/start`, {
    method: 'POST',
    body: JSON.stringify({ rtsp_url: rtspUrl }),
  });
}

export async function stopHLSStream(streamId: string) {
  return apiJson(`/v1/video/hls/${streamId}`, { method: 'DELETE' });
}

// --- Assets ---

export interface AssetAPI {
  asset_id: string;
  type: string;
  capabilities?: string[];
  battery?: number;
  link?: string;
}

export async function fetchAssets(): Promise<{ assets: AssetAPI[] }> {
  return apiJson('/v1/assets');
}

// --- Mission Builder: NLP parse ---

export interface NlpParseResponse {
  mission_type: string;
  pattern: string;
  altitude_m: number;
  asset_hint: string | null;
  objectives: string[];
  confidence: number;
  interpretation: string;
}

export async function parseMissionNlp(text: string): Promise<NlpParseResponse> {
  return apiJson('/v1/missions/parse', {
    method: 'POST',
    body: JSON.stringify({ text }),
  });
}

// --- Mission Builder: waypoint preview ---

export interface WaypointPreview {
  lat: number;
  lon: number;
  alt: number;
}

export interface PreviewResponse {
  waypoints: WaypointPreview[];
  pattern: string;
  count: number;
}

export async function previewMissionWaypoints(params: {
  pattern: string;
  altitude_m: number;
  area: { lat: number; lon: number }[];
}): Promise<PreviewResponse> {
  return apiJson('/v1/missions/preview', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

// --- World State ---

export async function fetchWorldState() {
  return apiJson('/v1/worldstate');
}

// --- WebSocket ---

export function connectWebSocket(onMessage: (data: unknown) => void): WebSocket | null {
  if (typeof window === 'undefined') return null;
  try {
    const ws = new WebSocket(WS_BASE);
    ws.onmessage = (event) => {
      try { onMessage(JSON.parse(event.data)); } catch { /* non-JSON */ }
    };
    return ws;
  } catch {
    return null;
  }
}

export { API_BASE, WS_BASE };

// --- Tasks (full list) ---

export interface TaskAPI {
  task_id: string;
  asset_id: string;
  asset_name?: string;
  action: string;
  status: 'pending' | 'active' | 'completed' | 'failed';
  risk_level: string;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  waypoints?: { lat: number; lon: number; alt?: number }[];
  mission_id?: string | null;
  objective?: string | null;
}

export async function fetchTasks(limit = 100): Promise<TaskAPI[]> {
  try {
    return await apiJson<TaskAPI[]>(`/v1/tasks?limit=${limit}`);
  } catch {
    return [];
  }
}
