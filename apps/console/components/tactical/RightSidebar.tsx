import React, { useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import { fetchAlerts, AlertAPI, connectWebSocket } from '../../lib/api';

const DynamicMissionTimeline = dynamic(() => import('./MissionTimeline'), { ssr: false });

interface FeedEvent {
  id: string;
  timestamp: string;
  type: 'INFO' | 'ALERT' | 'THREAT' | 'TASK' | 'AI';
  message: string;
}

const mockEvents: FeedEvent[] = [
  { id: '001', timestamp: '03:42:15', type: 'TASK', message: 'UAV-01 DEPLOYED TO SECTOR B7' },
  { id: '002', timestamp: '03:42:18', type: 'INFO', message: 'PERIMETER SCAN INITIATED' },
  { id: '003', timestamp: '03:42:22', type: 'AI', message: 'PATTERN RECOGNITION: ANOMALY DETECTED' },
  { id: '004', timestamp: '03:42:29', type: 'ALERT', message: 'GND-01 BATTERY THRESHOLD WARNING' },
  { id: '005', timestamp: '03:42:35', type: 'INFO', message: 'MESH NETWORK OPTIMIZATION COMPLETE' },
];

function severityToType(severity: string): FeedEvent['type'] {
  switch (severity?.toUpperCase()) {
    case 'CRITICAL':
    case 'HIGH': return 'THREAT';
    case 'MEDIUM':
    case 'WARNING': return 'ALERT';
    case 'AI':
    case 'ADVISORY': return 'AI';
    default: return 'INFO';
  }
}

function alertToEvent(a: AlertAPI): FeedEvent {
  return {
    id: a.alert_id,
    timestamp: a.ts_iso
      ? new Date(a.ts_iso).toLocaleTimeString('en-GB', { hour12: false })
      : '—',
    type: severityToType(a.severity),
    message: a.description || `[${a.source}] ${a.severity}`,
  };
}

function typeColor(type: FeedEvent['type']): string {
  switch (type) {
    case 'THREAT': return '#FF3333';
    case 'ALERT': return '#FF9933';
    case 'AI': return '#00DDFF';
    case 'TASK': return '#00FF91';
    default: return '#00FF91';
  }
}

export default function RightSidebar() {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [events, setEvents] = useState<FeedEvent[]>(mockEvents);
  const [live, setLive] = useState(false);

  // Fetch alerts from backend on mount
  useEffect(() => {
    fetchAlerts(50)
      .then(({ alerts }) => {
        if (alerts && alerts.length > 0) {
          setEvents(alerts.map(alertToEvent));
          setLive(true);
        }
      })
      .catch(() => { /* keep mock */ });
  }, []);

  // Subscribe to WebSocket for realtime events
  useEffect(() => {
    const ws = connectWebSocket((data) => {
      const msg = data as Record<string, unknown>;
      if (msg.alert_id || msg.type === 'alert') {
        const ev: FeedEvent = {
          id: String(msg.alert_id || Date.now()),
          timestamp: new Date().toLocaleTimeString('en-GB', { hour12: false }),
          type: severityToType(String(msg.severity || 'INFO')),
          message: String(msg.description || msg.message || 'New event'),
        };
        setEvents((prev) => [...prev.slice(-99), ev]);
        setLive(true);
      }
    });
    return () => { ws?.close(); };
  }, []);

  // Auto-scroll to bottom on new events
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events]);

  return (
    <div className="w-80 bg-[#0F0F0F] border-l-2 border-[#00FF91]/20 flex flex-col overflow-hidden">
      {/* Mission Feed */}
      <div className="flex-1 flex flex-col">
        <div className="h-10 border-b border-[#00FF91]/20 flex items-center px-4 bg-[#0A0A0A]">
          <div className="text-[#00FF91] text-sm font-semibold tracking-wider uppercase">
            MISSION FEED
          </div>
          <div className="ml-auto text-[10px] text-[#006644] font-mono">
            {live ? '[LIVE]' : '[SIMULATED]'}
          </div>
        </div>
        <div ref={scrollRef} className="flex-1 overflow-y-auto overflow-x-hidden p-2 space-y-1">
          {events.map((e) => {
            const color = typeColor(e.type);
            return (
              <div key={e.id} className="p-2 border-l-2 hover:bg-[#00FF91]/5 transition-colors" style={{ borderLeftColor: color }}>
                <div className="flex items-center gap-2 mb-1">
                  <div className="text-[10px] text-[#006644] font-mono tracking-wider">{e.timestamp}</div>
                  <div className="text-[8px] px-1.5 py-0.5 font-semibold tracking-wider border" style={{ color, borderColor: `${color}40`, backgroundColor: `${color}10` }}>{e.type}</div>
                </div>
                <div className="text-xs text-[#00CC74] font-mono leading-relaxed">{e.message}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Mission Timeline */}
      <div className="h-64">
        {/* Client-only timeline to avoid SSR hydration mismatch */}
        <DynamicMissionTimeline />
      </div>
    </div>
  );
}
