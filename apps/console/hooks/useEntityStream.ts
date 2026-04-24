'use client';

import { useEffect, useState, useCallback, useRef } from 'react';

function getWsUrl(): string {
  if (process.env.NEXT_PUBLIC_WS_URL) return process.env.NEXT_PUBLIC_WS_URL;
  if (typeof window === 'undefined') return 'ws://localhost:8001/ws';
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws`;
}

// ── Types ──────────────────────────────────────────────────

export interface EntityPosition {
  lat: number;
  lon: number;
  alt: number;
  heading_deg: number;
}

export interface EntityData {
  entity_id: string;
  entity_type: 'active' | 'alert' | 'neutral' | 'unknown';
  domain: 'aerial' | 'ground' | 'maritime' | 'fixed' | 'sensor';
  classification: string;
  position: EntityPosition;
  speed_mps: number;
  confidence: number;
  last_seen: number;
  source_sensors: string[];
  track_state?: 'tentative' | 'confirmed' | 'coasting';
  callsign?: string;
  battery_pct?: number;
  mission_id?: string;
  properties?: Record<string, unknown>;
}

export interface TrackData {
  track_id: string;
  state: 'tentative' | 'confirmed' | 'coasting';
  position: EntityPosition;
  velocity: { north: number; east: number; down: number };
  uncertainty_m: number;
  confidence: number;
  class_label: string;
  contributing_sensors: string[];
  first_seen: number;
  last_seen: number;
  hits: number;
  misses: number;
}

export interface MeshStatus {
  node_id: string;
  peers_alive: number;
  peers_suspect: number;
  peers_dead: number;
  partitioned: boolean;
  crdt_keys: number;
}

export type ConnectionState = 'connected' | 'reconnecting' | 'offline';

// ── Backoff config ─────────────────────────────────────────

const BACKOFF_BASE_MS   = 1_000;
const BACKOFF_MAX_MS    = 30_000;
const PING_INTERVAL_MS  = 30_000;
const OFFLINE_AFTER_MS  = 15_000; // no reconnect for 15s → show "offline"

// ── Hook ───────────────────────────────────────────────────

export function useEntityStream() {
  const [entities,        setEntities]        = useState<Map<string, EntityData>>(new Map());
  const [tracks,          setTracks]          = useState<Map<string, TrackData>>(new Map());
  const [meshStatus,      setMeshStatus]      = useState<MeshStatus | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>('reconnecting');
  const [lastUpdate,      setLastUpdate]      = useState<number>(0);
  const [reconnectCount,  setReconnectCount]  = useState<number>(0);

  const wsRef              = useRef<WebSocket | null>(null);
  const backoffRef         = useRef<number>(BACKOFF_BASE_MS);
  const reconnectTimerRef  = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pingTimerRef       = useRef<ReturnType<typeof setInterval> | null>(null);
  const offlineTimerRef    = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastUpdateRef      = useRef<number>(0);
  const mountedRef         = useRef<boolean>(true);

  const handleMessage = useCallback((raw: string) => {
    try {
      const msg = JSON.parse(raw);

      switch (msg.type) {
        case 'entity_update': {
          const entity = msg.data as EntityData;
          setEntities(prev => { const n = new Map(prev); n.set(entity.entity_id, entity); return n; });
          setLastUpdate(Date.now());
          lastUpdateRef.current = Date.now();
          break;
        }
        case 'entity_batch': {
          const batch = msg.data as EntityData[];
          setEntities(prev => {
            const n = new Map(prev);
            for (const e of batch) n.set(e.entity_id, e);
            return n;
          });
          setLastUpdate(Date.now());
          lastUpdateRef.current = Date.now();
          break;
        }
        case 'track_update': {
          const track = msg.data as TrackData;
          setTracks(prev => { const n = new Map(prev); n.set(track.track_id, track); return n; });
          break;
        }
        case 'track_deleted': {
          const trackId = msg.data.track_id as string;
          setTracks(prev => { const n = new Map(prev); n.delete(trackId); return n; });
          break;
        }
        case 'mesh_status':
          setMeshStatus(msg.data as MeshStatus);
          break;
        case 'entity_removed': {
          const entityId = msg.data.entity_id as string;
          setEntities(prev => { const n = new Map(prev); n.delete(entityId); return n; });
          break;
        }
        case 'pong':
          break;
      }
    } catch {
      // non-JSON — skip
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;

    const clearTimers = () => {
      if (reconnectTimerRef.current)  clearTimeout(reconnectTimerRef.current);
      if (pingTimerRef.current)       clearInterval(pingTimerRef.current);
      if (offlineTimerRef.current)    clearTimeout(offlineTimerRef.current);
    };

    const connect = () => {
      if (!mountedRef.current) return;

      // If we've been trying for a while, show "offline"
      offlineTimerRef.current = setTimeout(() => {
        if (mountedRef.current) setConnectionState('offline');
      }, OFFLINE_AFTER_MS);

      let ws: WebSocket;
      try {
        ws = new WebSocket(getWsUrl());
      } catch {
        // URL parse error — back off and retry
        scheduleReconnect();
        return;
      }
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) { ws.close(); return; }

        if (offlineTimerRef.current) clearTimeout(offlineTimerRef.current);
        setConnectionState('connected');
        setReconnectCount(c => c + 1);
        backoffRef.current = BACKOFF_BASE_MS; // reset on success

        // Subscribe, asking backend to replay events since our last update
        ws.send(JSON.stringify({
          type: 'subscribe',
          channels: ['entities', 'tracks', 'mesh'],
          since: lastUpdateRef.current || undefined,
        }));

        pingTimerRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
          }
        }, PING_INTERVAL_MS);
      };

      ws.onmessage = (event) => handleMessage(event.data);

      ws.onclose = () => {
        if (!mountedRef.current) return;
        if (offlineTimerRef.current) clearTimeout(offlineTimerRef.current);
        if (pingTimerRef.current)    clearInterval(pingTimerRef.current);
        setConnectionState('reconnecting');
        scheduleReconnect();
      };

      ws.onerror = () => ws.close();
    };

    const scheduleReconnect = () => {
      if (!mountedRef.current) return;
      const delay = backoffRef.current;
      backoffRef.current = Math.min(backoffRef.current * 2, BACKOFF_MAX_MS);
      reconnectTimerRef.current = setTimeout(connect, delay);
    };

    connect();

    return () => {
      mountedRef.current = false;
      clearTimers();
      wsRef.current?.close();
    };
  }, [handleMessage]);

  // Derived
  const entityList     = Array.from(entities.values());
  const trackList      = Array.from(tracks.values());
  const connected      = connectionState === 'connected';

  return {
    entities,
    tracks,
    meshStatus,
    connected,
    connectionState,
    reconnectCount,
    lastUpdate,
    entityList,
    trackList,
    friendlyEntities:  entityList.filter(e => e.entity_type === 'active'),
    hostileEntities:   entityList.filter(e => e.entity_type === 'alert'),
    unknownEntities:   entityList.filter(e => e.entity_type === 'unknown'),
    confirmedTracks:   trackList.filter(t => t.state === 'confirmed'),
    entityCount:       entities.size,
    trackCount:        tracks.size,
  };
}
