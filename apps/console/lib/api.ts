/**
 * Summit.OS API client.
 *
 * All backend calls go through the API gateway at NEXT_PUBLIC_API_URL.
 * Policy denials (400/403) are surfaced via custom window events so
 * PolicyNotifications can render them globally.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8001';

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

// --- World State ---

export async function fetchWorldState() {
  return apiJson('/v1/worldstate');
}

// --- WebSocket ---

export function connectWebSocket(onMessage: (data: unknown) => void): WebSocket | null {
  if (typeof window === 'undefined') return null;
  try {
    const ws = new WebSocket(`${WS_BASE}/ws`);
    ws.onmessage = (event) => {
      try { onMessage(JSON.parse(event.data)); } catch { /* non-JSON */ }
    };
    return ws;
  } catch {
    return null;
  }
}

export { API_BASE, WS_BASE };
