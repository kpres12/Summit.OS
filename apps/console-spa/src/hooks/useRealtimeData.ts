import { useEffect, useState, useCallback } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';

interface Asset {
  id: string;
  type: string;
  battery: number;
  temp: number;
  signal: number;
  status: 'ACTIVE' | 'IDLE' | 'WARNING' | 'OFFLINE';
  position?: { x: number; y: number };
}

interface FeedEvent {
  id: string;
  timestamp: string;
  type: 'INFO' | 'ALERT' | 'THREAT' | 'TASK' | 'AI';
  message: string;
}

interface SystemStats {
  cpu: number;
  network: number;
  power: number;
}

export function useRealtimeData() {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [events, setEvents] = useState<FeedEvent[]>([]);
  const [stats, setStats] = useState<SystemStats>({ cpu: 0, network: 0, power: 0 });
  const [connected, setConnected] = useState(false);

  const fetchAssets = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/assets`);
      if (response.ok) setAssets(await response.json());
    } catch (e) { console.error('Failed to fetch assets:', e); }
  }, []);

  const fetchEvents = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/events`);
      if (response.ok) setEvents(await response.json());
    } catch (e) { console.error('Failed to fetch events:', e); }
  }, []);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnect: number | undefined;

    const connect = () => {
      ws = new WebSocket(WS_URL);
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        reconnect = window.setTimeout(connect, 5000);
      };
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          switch (msg.type) {
            case 'asset_update':
              setAssets(prev => {
                const i = prev.findIndex(a => a.id === msg.data.id);
                if (i >= 0) { const next = [...prev]; next[i] = { ...next[i], ...msg.data }; return next; }
                return [...prev, msg.data];
              });
              break;
            case 'new_event':
              setEvents(prev => [...prev, msg.data].slice(-50));
              break;
            case 'stats_update':
              setStats(msg.data);
              break;
          }
        } catch (e) { console.error('WS parse error', e); }
      };
    };

    connect();
    return () => { if (ws) ws.close(); if (reconnect) clearTimeout(reconnect); };
  }, []);

  useEffect(() => { fetchAssets(); fetchEvents(); }, [fetchAssets, fetchEvents]);

  const sendCommand = useCallback(async (command: string) => {
    const res = await fetch(`${API_URL}/api/command`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ command })
    });
    if (!res.ok) throw new Error('Command failed');
    return res.json();
  }, []);

  return { assets, events, stats, connected, sendCommand, refreshAssets: fetchAssets, refreshEvents: fetchEvents };
}
