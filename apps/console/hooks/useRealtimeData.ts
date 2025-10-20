'use client';

import { useEffect, useState, useCallback } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws';

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

  // Fetch initial data from REST API
  const fetchAssets = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/assets`);
      if (response.ok) {
        const data = await response.json();
        setAssets(data);
      }
    } catch (error) {
      console.error('Failed to fetch assets:', error);
    }
  }, []);

  const fetchEvents = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/events`);
      if (response.ok) {
        const data = await response.json();
        setEvents(data);
      }
    } catch (error) {
      console.error('Failed to fetch events:', error);
    }
  }, []);

  // WebSocket connection for real-time updates
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimeout: NodeJS.Timeout;

    const connect = () => {
      ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        console.log('WebSocket connected');
        setConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          switch (message.type) {
            case 'asset_update':
              setAssets(prev => {
                const index = prev.findIndex(a => a.id === message.data.id);
                if (index >= 0) {
                  const newAssets = [...prev];
                  newAssets[index] = { ...newAssets[index], ...message.data };
                  return newAssets;
                }
                return [...prev, message.data];
              });
              break;
              
            case 'new_event':
              setEvents(prev => [...prev, message.data].slice(-50)); // Keep last 50 events
              break;
              
            case 'stats_update':
              setStats(message.data);
              break;
              
            default:
              console.log('Unknown message type:', message.type);
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setConnected(false);
        
        // Attempt to reconnect after 5 seconds
        reconnectTimeout = setTimeout(() => {
          console.log('Attempting to reconnect...');
          connect();
        }, 5000);
      };
    };

    connect();

    return () => {
      if (ws) {
        ws.close();
      }
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
    };
  }, []);

  // Fetch initial data on mount
  useEffect(() => {
    fetchAssets();
    fetchEvents();
  }, [fetchAssets, fetchEvents]);

  const sendCommand = useCallback(async (command: string) => {
    try {
      const response = await fetch(`${API_URL}/api/command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command }),
      });
      
      if (response.ok) {
        return await response.json();
      } else {
        throw new Error('Command failed');
      }
    } catch (error) {
      console.error('Failed to send command:', error);
      throw error;
    }
  }, []);

  return {
    assets,
    events,
    stats,
    connected,
    sendCommand,
    refreshAssets: fetchAssets,
    refreshEvents: fetchEvents,
  };
}
