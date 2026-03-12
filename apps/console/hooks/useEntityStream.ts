'use client';

import { useEffect, useState, useCallback, useRef } from 'react';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8001/ws';

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
  // Domain-specific
  callsign?: string;
  battery_pct?: number;
  mission_id?: string;
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

// ── Hook ───────────────────────────────────────────────────

export function useEntityStream() {
  const [entities, setEntities] = useState<Map<string, EntityData>>(new Map());
  const [tracks, setTracks] = useState<Map<string, TrackData>>(new Map());
  const [meshStatus, setMeshStatus] = useState<MeshStatus | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<number>(0);
  const wsRef = useRef<WebSocket | null>(null);

  // Process incoming WebSocket messages
  const handleMessage = useCallback((raw: string) => {
    try {
      const msg = JSON.parse(raw);

      switch (msg.type) {
        case 'entity_update': {
          const entity = msg.data as EntityData;
          setEntities(prev => {
            const next = new Map(prev);
            next.set(entity.entity_id, entity);
            return next;
          });
          setLastUpdate(Date.now());
          break;
        }

        case 'entity_batch': {
          const batch = msg.data as EntityData[];
          setEntities(prev => {
            const next = new Map(prev);
            for (const e of batch) {
              next.set(e.entity_id, e);
            }
            return next;
          });
          setLastUpdate(Date.now());
          break;
        }

        case 'track_update': {
          const track = msg.data as TrackData;
          setTracks(prev => {
            const next = new Map(prev);
            next.set(track.track_id, track);
            return next;
          });
          break;
        }

        case 'track_deleted': {
          const trackId = msg.data.track_id as string;
          setTracks(prev => {
            const next = new Map(prev);
            next.delete(trackId);
            return next;
          });
          break;
        }

        case 'mesh_status': {
          setMeshStatus(msg.data as MeshStatus);
          break;
        }

        case 'entity_removed': {
          const entityId = msg.data.entity_id as string;
          setEntities(prev => {
            const next = new Map(prev);
            next.delete(entityId);
            return next;
          });
          break;
        }
      }
    } catch {
      // non-JSON or malformed — skip
    }
  }, []);

  // WebSocket lifecycle
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout>;
    let pingTimer: ReturnType<typeof setInterval>;

    const connect = () => {
      try {
        ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => {
          setConnected(true);
          // Subscribe to entity stream
          ws?.send(JSON.stringify({ type: 'subscribe', channels: ['entities', 'tracks', 'mesh'] }));
          // Periodic ping
          pingTimer = setInterval(() => {
            if (ws?.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({ type: 'ping' }));
            }
          }, 30000);
        };

        ws.onmessage = (event) => handleMessage(event.data);

        ws.onclose = () => {
          setConnected(false);
          clearInterval(pingTimer);
          reconnectTimer = setTimeout(connect, 3000);
        };

        ws.onerror = () => {
          ws?.close();
        };
      } catch {
        reconnectTimer = setTimeout(connect, 5000);
      }
    };

    connect();

    return () => {
      ws?.close();
      clearTimeout(reconnectTimer);
      clearInterval(pingTimer);
    };
  }, [handleMessage]);

  // Derived data
  const entityList = Array.from(entities.values());
  const trackList = Array.from(tracks.values());

  const friendlyEntities = entityList.filter(e => e.entity_type === 'active');
  const hostileEntities = entityList.filter(e => e.entity_type === 'alert');
  const unknownEntities = entityList.filter(e => e.entity_type === 'unknown');
  const confirmedTracks = trackList.filter(t => t.state === 'confirmed');

  return {
    // Raw data
    entities,
    tracks,
    meshStatus,
    connected,
    lastUpdate,

    // Lists
    entityList,
    trackList,
    friendlyEntities,
    hostileEntities,
    unknownEntities,
    confirmedTracks,

    // Counts
    entityCount: entities.size,
    trackCount: tracks.size,
  };
}
