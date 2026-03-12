'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { fetchMissions, connectWebSocket, MissionAPI } from '@/lib/api';

function ageString(isoString: string | null): string {
  if (!isoString) return '—';
  const ts = new Date(isoString).getTime();
  const diff = Math.floor((Date.now() - ts) / 1000);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function statusColor(status: string): string {
  switch (status.toUpperCase()) {
    case 'ACTIVE': return '#00FF9C';
    case 'PLANNING': return 'rgba(200,230,201,0.45)';
    case 'FAILED': return '#FF3B3B';
    case 'COMPLETED': return '#4FC3F7';
    default: return 'rgba(200,230,201,0.45)';
  }
}

const PHASE_ORDER = ['PLANNING', 'STAGING', 'ACTIVE', 'RETURNING', 'COMPLETED'];

function MissionTimeline({ status }: { status: string }) {
  const s = status.toUpperCase();
  const currentIdx = PHASE_ORDER.indexOf(s);
  const activeIdx = currentIdx >= 0 ? currentIdx : s === 'FAILED' ? -1 : 0;
  const isFailed = s === 'FAILED';

  return (
    <div className="flex items-center gap-0 mt-2 mb-1" style={{ height: '18px' }}>
      {PHASE_ORDER.map((phase, i) => {
        const done = !isFailed && i < activeIdx;
        const current = !isFailed && i === activeIdx;
        const failed = isFailed && i === (currentIdx >= 0 ? currentIdx : 2);
        const color = failed ? '#FF3B3B' : done || current ? '#00FF9C' : 'rgba(200,230,201,0.15)';
        const textColor = failed ? '#FF3B3B' : done || current ? '#00FF9C' : 'rgba(200,230,201,0.25)';
        return (
          <React.Fragment key={phase}>
            <div
              title={phase}
              style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                background: current ? color : done ? color : 'transparent',
                border: `1px solid ${color}`,
                flexShrink: 0,
                boxShadow: current ? `0 0 4px ${color}` : 'none',
              }}
            />
            {i < PHASE_ORDER.length - 1 && (
              <div style={{ flex: 1, height: '1px', background: done ? '#00FF9C40' : 'rgba(200,230,201,0.1)', minWidth: '4px' }} />
            )}
          </React.Fragment>
        );
      })}
      <span
        className="ml-2 text-[8px]"
        style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: isFailed ? '#FF3B3B' : 'rgba(200,230,201,0.35)', whiteSpace: 'nowrap' }}
      >
        {isFailed ? 'FAILED' : PHASE_ORDER[activeIdx] || s}
      </span>
    </div>
  );
}

export default function OpsMissions() {
  const [missions, setMissions] = useState<MissionAPI[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchMissions(20)
      .then((data) => setMissions(data || []))
      .catch(() => setMissions([]))
      .finally(() => setLoading(false));
  }, []);

  const handleWsMessage = useCallback((data: unknown) => {
    const msg = data as { type?: string; data?: MissionAPI };
    if (msg.type === 'mission_update' && msg.data) {
      const updated = msg.data as MissionAPI;
      setMissions((prev) => {
        const idx = prev.findIndex((m) => m.mission_id === updated.mission_id);
        if (idx >= 0) {
          const next = [...prev];
          next[idx] = updated;
          return next;
        }
        return [updated, ...prev];
      });
    }
  }, []);

  useEffect(() => {
    const ws = connectWebSocket(handleWsMessage);
    return () => { ws?.close(); };
  }, [handleWsMessage]);

  return (
    <div className="flex flex-col h-full">
      {/* Panel header */}
      <div
        className="flex-none flex items-center justify-between px-3 py-2"
        style={{ borderBottom: '1px solid rgba(0,255,156,0.15)' }}
      >
        <span
          className="text-xs font-bold tracking-widest"
          style={{ fontFamily: 'var(--font-orbitron), Orbitron, sans-serif', color: '#00FF9C' }}
        >
          MISSIONS
        </span>
        <span
          className="text-[10px] px-1.5 py-0.5"
          style={{
            background: 'rgba(0,255,156,0.1)',
            color: '#00FF9C',
            border: '1px solid rgba(0,255,156,0.3)',
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
          }}
        >
          {missions.length}
        </span>
      </div>

      {/* Mission list */}
      <div className="flex-1 overflow-y-auto">
        {loading && (
          <div
            className="flex items-center justify-center h-20 text-[10px]"
            style={{ color: 'rgba(200,230,201,0.35)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
          >
            LOADING...
          </div>
        )}
        {!loading && missions.length === 0 && (
          <div
            className="flex items-center justify-center h-full text-[10px] tracking-widest"
            style={{ color: 'rgba(200,230,201,0.35)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
          >
            NO MISSIONS
          </div>
        )}

        {missions.map((mission) => {
          const color = statusColor(mission.status);
          const name = mission.name || mission.mission_id.slice(0, 8);
          return (
            <div
              key={mission.mission_id}
              className="mx-2 my-1.5 px-3 py-2"
              style={{
                background: '#111916',
                border: '1px solid rgba(0,255,156,0.15)',
              }}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-1.5">
                <span
                  className="text-xs font-bold truncate flex-1 mr-2"
                  style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: '#00FF9C' }}
                >
                  {name}
                </span>
                <span
                  className="text-[9px] px-1.5 py-0.5 font-bold tracking-widest flex-none"
                  style={{
                    color,
                    border: `1px solid ${color}`,
                    background: `${color}15`,
                    fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  }}
                >
                  {mission.status.toUpperCase()}
                </span>
              </div>

              {/* ID */}
              <div
                className="text-[9px] mb-1"
                style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.35)' }}
              >
                {mission.mission_id.slice(0, 8)}
              </div>

              {/* Objectives */}
              {mission.objectives && mission.objectives.length > 0 && (
                <div className="mb-1">
                  {mission.objectives.slice(0, 2).map((obj, i) => (
                    <div
                      key={i}
                      className="text-[10px] leading-tight"
                      style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.45)' }}
                    >
                      · {obj.slice(0, 50)}
                    </div>
                  ))}
                </div>
              )}

              {/* Timeline */}
              <MissionTimeline status={mission.status} />

              {/* Timestamp */}
              <div
                className="text-[9px]"
                style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.3)' }}
              >
                {ageString(mission.created_at)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
