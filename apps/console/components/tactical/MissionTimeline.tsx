'use client';

import React, { useEffect, useState } from 'react';
import { fetchMissions, MissionAPI, connectWebSocket } from '../../lib/api';

interface TimelineEvent {
  id: string;
  ts: string; // ISO or HH:MM:SS
  phase: 'PLANNING' | 'APPROVAL' | 'DISPATCHED' | 'ACTIVE' | 'COMPLETED' | 'FAILED';
  label: string;
  meta?: string;
}

const mockTimeline: TimelineEvent[] = [
  { id: 'e1', ts: '03:42:12', phase: 'PLANNING', label: 'Mission created', meta: 'GRID pattern, 2 assets' },
  { id: 'e2', ts: '03:42:25', phase: 'APPROVAL', label: 'Supervisor approved', meta: 'risk: MEDIUM' },
  { id: 'e3', ts: '03:42:40', phase: 'DISPATCHED', label: 'Assignments dispatched', meta: 'UAV-01, UAV-02' },
  { id: 'e4', ts: '03:43:05', phase: 'ACTIVE', label: 'UAV-01 on station', meta: 'alt 100m' },
  { id: 'e5', ts: '03:44:10', phase: 'ACTIVE', label: 'UAV-02 on station', meta: 'alt 120m' },
];

const STATUS_TO_PHASE: Record<string, TimelineEvent['phase']> = {
  PLANNING: 'PLANNING',
  PENDING_APPROVAL: 'APPROVAL',
  APPROVED: 'DISPATCHED',
  ACTIVE: 'ACTIVE',
  RUNNING: 'ACTIVE',
  COMPLETED: 'COMPLETED',
  FAILED: 'FAILED',
};

function missionToEvents(m: MissionAPI): TimelineEvent {
  const phase = STATUS_TO_PHASE[m.status?.toUpperCase()] || 'PLANNING';
  const ts = m.created_at
    ? new Date(m.created_at).toLocaleTimeString('en-GB', { hour12: false })
    : '—';
  return {
    id: m.mission_id,
    ts,
    phase,
    label: m.name || `Mission ${m.mission_id.slice(0, 8)}`,
    meta: m.objectives?.join(', ') || undefined,
  };
}

function phaseColor(phase: TimelineEvent['phase']): string {
  switch (phase) {
    case 'PLANNING': return '#00CC74';
    case 'APPROVAL': return '#00DDFF';
    case 'DISPATCHED': return '#00FF91';
    case 'ACTIVE': return '#FF9933';
    case 'COMPLETED': return '#00FF91';
    case 'FAILED': return '#FF3333';
    default: return '#006644';
  }
}

const PHASE_ORDER: TimelineEvent['phase'][] = ['PLANNING', 'APPROVAL', 'DISPATCHED', 'ACTIVE', 'COMPLETED'];

function progressPct(events: TimelineEvent[]): number {
  if (!events.length) return 0;
  const latest = events.reduce((best, ev) => {
    const idx = PHASE_ORDER.indexOf(ev.phase);
    return idx > PHASE_ORDER.indexOf(best.phase) ? ev : best;
  }, events[0]);
  const idx = PHASE_ORDER.indexOf(latest.phase);
  return idx < 0 ? 0 : ((idx + 1) / PHASE_ORDER.length) * 100;
}

export default function MissionTimeline() {
  const [events, setEvents] = useState<TimelineEvent[]>(mockTimeline);
  const [live, setLive] = useState(false);

  // Fetch missions from backend on mount
  useEffect(() => {
    fetchMissions(20)
      .then((missions) => {
        const list = Array.isArray(missions) ? missions : [];
        if (list.length > 0) {
          setEvents(list.map(missionToEvents));
          setLive(true);
        }
      })
      .catch(() => { /* keep mock */ });
  }, []);

  // Subscribe to WebSocket for realtime mission updates
  useEffect(() => {
    const ws = connectWebSocket((data) => {
      const msg = data as Record<string, unknown>;
      if (msg.type === 'mission_update' && msg.mission_id) {
        const phase = STATUS_TO_PHASE[String(msg.status || '').toUpperCase()] || 'ACTIVE';
        setEvents((prev) => {
          const exists = prev.find((e) => e.id === msg.mission_id);
          if (exists) {
            return prev.map((e) =>
              e.id === msg.mission_id ? { ...e, phase, label: String(msg.name || e.label) } : e
            );
          }
          return [
            ...prev,
            {
              id: String(msg.mission_id),
              ts: new Date().toLocaleTimeString('en-GB', { hour12: false }),
              phase,
              label: String(msg.name || msg.mission_id),
            },
          ];
        });
        setLive(true);
      }
    });
    return () => { ws?.close(); };
  }, []);

  return (
    <div className="bg-[#0F0F0F] border-t-2 border-[#00FF91]/20 h-64 flex flex-col">
      {/* Header */}
      <div className="h-10 border-b border-[#00FF91]/20 flex items-center px-4 bg-[#0A0A0A]">
        <div className="text-[#00FF91] text-sm font-semibold tracking-wider uppercase">
          MISSION TIMELINE
        </div>
        <div className="ml-auto text-[10px] text-[#006644] font-mono">
          {live ? '[LIVE]' : '[SIMULATED]'}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {/* Progress Bar */}
        <div className="w-full">
          <div className="h-2 bg-[#006644] relative">
            <div className="absolute top-0 left-0 h-2 bg-[#00FF91] transition-all" style={{ width: `${progressPct(events)}%` }} />
          </div>
          <div className="flex justify-between text-[10px] text-[#006644] font-mono mt-1">
            <span>PLANNING</span>
            <span>APPROVAL</span>
            <span>ASSIGN</span>
            <span>ACTIVE</span>
            <span>COMPLETE</span>
          </div>
        </div>

        {/* Events */}
        <div className="space-y-2">
          {events.map((ev) => (
            <div key={ev.id} className="flex items-start gap-3">
              {/* Marker */}
              <div
                className="w-2 h-2 mt-1 rounded-full"
                style={{ backgroundColor: phaseColor(ev.phase), boxShadow: `0 0 6px ${phaseColor(ev.phase)}80` }}
              />
              {/* Details */}
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <div className="text-[10px] text-[#006644] font-mono">{ev.ts}</div>
                  <div
                    className="text-[9px] px-1.5 py-0.5 font-semibold tracking-wider border"
                    style={{ color: phaseColor(ev.phase), borderColor: `${phaseColor(ev.phase)}40`, backgroundColor: `${phaseColor(ev.phase)}10` }}
                  >
                    {ev.phase}
                  </div>
                </div>
                <div className="text-xs text-[#00CC74] font-mono">{ev.label}</div>
                {ev.meta && (
                  <div className="text-[10px] text-[#006644] font-mono">{ev.meta}</div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
