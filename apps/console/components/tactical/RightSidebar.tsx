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
      : '\u2014',
    type: severityToType(a.severity),
    message: a.description || `[${a.source}] ${a.severity}`,
  };
}

function typeStyles(type: FeedEvent['type']): { color: string; icon: string } {
  switch (type) {
    case 'THREAT': return { color: '#FF3333', icon: '\u2B22' };
    case 'ALERT':  return { color: '#FF9933', icon: '\u25B2' };
    case 'AI':     return { color: '#00DDFF', icon: '\u25C6' };
    case 'TASK':   return { color: '#00FF91', icon: '\u25B6' };
    default:       return { color: '#00CC74', icon: '\u25CF' };
  }
}

function nowTs(): string {
  return new Date().toLocaleTimeString('en-GB', { hour12: false });
}

export default function RightSidebar() {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [events, setEvents] = useState<FeedEvent[]>([]);
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
      .catch(() => { /* backend unreachable */ });
  }, []);

  // Subscribe to WebSocket for realtime events (alerts, entities, missions)
  useEffect(() => {
    const ws = connectWebSocket((data) => {
      const msg = data as Record<string, unknown>;
      let ev: FeedEvent | null = null;

      if (msg.alert_id || msg.type === 'alert') {
        ev = {
          id: String(msg.alert_id || Date.now()),
          timestamp: nowTs(),
          type: severityToType(String(msg.severity || 'INFO')),
          message: String(msg.description || msg.message || 'New alert'),
        };
      } else if (msg.type === 'entity_update') {
        const d = msg.data as Record<string, unknown> | undefined;
        if (d) {
          ev = {
            id: `ent-${Date.now()}`,
            timestamp: nowTs(),
            type: 'INFO',
            message: `ENTITY ${String(d.callsign || d.entity_id || '').slice(0, 16).toUpperCase()} UPDATED`,
          };
        }
      } else if (msg.type === 'mission_update') {
        ev = {
          id: `msn-${Date.now()}`,
          timestamp: nowTs(),
          type: 'TASK',
          message: `MISSION ${String(msg.name || msg.mission_id || '').slice(0, 20).toUpperCase()} \u2192 ${String(msg.status || '').toUpperCase()}`,
        };
      } else if (msg.type === 'entity_removed') {
        const d = msg.data as Record<string, unknown> | undefined;
        ev = {
          id: `rem-${Date.now()}`,
          timestamp: nowTs(),
          type: 'ALERT',
          message: `ENTITY ${String(d?.entity_id || '').slice(0, 16).toUpperCase()} REMOVED`,
        };
      }

      if (ev) {
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
      <div className="flex-1 flex flex-col">
        <div className="h-10 border-b border-[#00FF91]/20 flex items-center px-4 bg-[#0A0A0A]">
          <div className="text-[#00FF91] text-sm font-semibold tracking-wider uppercase">
            MISSION FEED
          </div>
          <div className="ml-auto text-[10px] text-[#006644] font-mono">
            {live ? '[LIVE]' : events.length > 0 ? '[CACHED]' : '[IDLE]'}
          </div>
        </div>
        <div ref={scrollRef} className="flex-1 overflow-y-auto overflow-x-hidden p-2 space-y-1">
          {events.length === 0 ? (
            <div className="px-4 py-8 text-center">
              <div className="text-[11px] text-[#006644] font-mono">NO EVENTS</div>
              <div className="text-[10px] text-[#004422] font-mono mt-1">Awaiting data\u2026</div>
            </div>
          ) : (
            events.map((e, idx) => {
              const { color, icon } = typeStyles(e.type);
              return (
                <div
                  key={e.id}
                  className="p-2 border-l-2 hover:bg-[#00FF91]/5 transition-colors"
                  style={{ borderLeftColor: color, animationDelay: `${idx * 0.05}s` }}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <div className="text-[10px] text-[#006644] font-mono tracking-wider">
                      {e.timestamp}
                    </div>
                    <div
                      className="text-[8px] px-1.5 py-0.5 font-semibold tracking-wider border"
                      style={{ color, borderColor: `${color}40`, backgroundColor: `${color}10` }}
                    >
                      {e.type}
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="text-xs mt-0.5" style={{ color }}>{icon}</div>
                    <div className="text-xs text-[#00CC74] font-mono leading-relaxed">
                      {e.message}
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
      <div className="h-56 border-t border-[#00FF91]/20">
        <DynamicMissionTimeline />
      </div>
    </div>
  );
}
