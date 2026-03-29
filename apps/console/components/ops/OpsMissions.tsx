'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { fetchMissions, connectWebSocket, MissionAPI } from '@/lib/api';
import PanelHeader from '@/components/ui/PanelHeader';
import StatusBadge from '@/components/ui/StatusBadge';
import EmptyState from '@/components/ui/EmptyState';
import { ageFromISO, statusColor } from '@/lib/format';

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

interface OpsMissionsProps {
  onReplay?: (missionId: string) => void;
}

export default function OpsMissions({ onReplay }: OpsMissionsProps) {
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
    <div className="flex flex-col h-full panel-scanline">
      <PanelHeader title="MISSIONS" count={missions.length} />

      {/* Mission list */}
      <div className="flex-1 overflow-y-auto">
        {loading && <EmptyState message="LOADING..." />}
        {!loading && missions.length === 0 && <EmptyState message="NO MISSIONS" />}

        {missions.map((mission) => {
          const color = statusColor(mission.status);
          const name = mission.name || mission.mission_id.slice(0, 8);
          return (
            <div
              key={mission.mission_id}
              className="mx-2 my-1.5 px-3 py-2"
              style={{
                background: 'var(--background-card)',
                border: '1px solid var(--border)',
              }}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-1.5">
                <span
                  className="text-xs font-bold truncate flex-1 mr-2"
                  style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--accent)' }}
                >
                  {name}
                </span>
                <StatusBadge label={mission.status.toUpperCase()} color={color} />
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

              {/* Timestamp + Replay */}
              <div className="flex items-center justify-between mt-1">
                <div
                  className="text-[9px]"
                  style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.3)' }}
                >
                {ageFromISO(mission.created_at)}
                </div>
                {onReplay && (mission.status === 'COMPLETED' || mission.status === 'FAILED') && (
                  <button
                    onClick={() => onReplay(mission.mission_id)}
                    style={{
                      fontFamily: 'var(--font-ibm-plex-mono), monospace',
                      fontSize: 9,
                      color: '#FFB300',
                      background: 'transparent',
                      border: '1px solid rgba(255,179,0,0.3)',
                      padding: '2px 6px',
                      cursor: 'pointer',
                      letterSpacing: '0.1em',
                    }}
                  >
                    REPLAY
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
