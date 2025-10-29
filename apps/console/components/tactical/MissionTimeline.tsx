'use client';

import React from 'react';

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

export default function MissionTimeline() {
  return (
    <div className="bg-[#0F0F0F] border-t-2 border-[#00FF91]/20 h-64 flex flex-col">
      {/* Header */}
      <div className="h-10 border-b border-[#00FF91]/20 flex items-center px-4 bg-[#0A0A0A]">
        <div className="text-[#00FF91] text-sm font-semibold tracking-wider uppercase">
          MISSION TIMELINE
        </div>
        <div className="ml-auto text-[10px] text-[#006644] font-mono">[SIMULATED]</div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {/* Progress Bar */}
        <div className="w-full">
          <div className="h-2 bg-[#006644] relative">
            <div className="absolute top-0 left-0 h-2 bg-[#00FF91]" style={{ width: '65%' }} />
          </div>
          <div className="flex justify-between text-[10px] text-[#006644] font-mono mt-1">
            <span>PLANNING</span>
            <span>APPROVAL</span>
            <span>DISPATCH</span>
            <span>ACTIVE</span>
            <span>COMPLETE</span>
          </div>
        </div>

        {/* Events */}
        <div className="space-y-2">
          {mockTimeline.map((ev) => (
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
