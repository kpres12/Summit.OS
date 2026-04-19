'use client';

import React, { useMemo } from 'react';
import { TaskAPI } from '@/lib/api';
import { EntityData } from '@/hooks/useEntityStream';

const TASK_COLORS: Record<string, string> = {
  VOLUME_SEARCH: '#4FC3F7',
  AREA_SEARCH:   '#4FC3F7',
  TRANSIT:       '#FFB300',
  SURVEILLANCE:  '#00FF9C',
  PATROL:        '#00FF9C',
  INTERCEPT:     '#FF3B3B',
  DISPATCH:      '#FFB300',
  RTB:           '#4A5568',
  HALT:          '#FF3B3B',
};

const WINDOW_MINUTES = 120; // show last 2 hours

function pct(ts: string, windowStart: number, windowMs: number): number {
  return Math.max(0, Math.min(100, (new Date(ts).getTime() - windowStart) / windowMs * 100));
}

interface OpsMissionTimelineProps {
  tasks: TaskAPI[];
  entityList: EntityData[];
  isOpen: boolean;
  onToggle: () => void;
}

export default function OpsMissionTimeline({ tasks, entityList, isOpen, onToggle }: OpsMissionTimelineProps) {
  const now = Date.now();
  const windowMs = WINDOW_MINUTES * 60 * 1000;
  const windowStart = now - windowMs;

  // Build asset → tasks map
  const assetTasks = useMemo(() => {
    const map = new Map<string, { name: string; tasks: TaskAPI[] }>();

    // Seed from active entity list
    for (const e of entityList) {
      if (e.entity_type === 'active') {
        map.set(e.entity_id, {
          name: (e.callsign || e.entity_id.slice(0, 8)).toUpperCase(),
          tasks: [],
        });
      }
    }

    // Attach tasks
    for (const t of tasks) {
      if (!map.has(t.asset_id)) {
        map.set(t.asset_id, {
          name: (t.asset_name || t.asset_id.slice(0, 10)).toUpperCase(),
          tasks: [],
        });
      }
      map.get(t.asset_id)!.tasks.push(t);
    }

    return Array.from(map.entries()).filter(([, v]) => v.tasks.length > 0);
  }, [tasks, entityList]);

  // Time axis labels (every 30 min)
  const timeLabels = useMemo(() => {
    const labels: { label: string; pctPos: number }[] = [];
    const step = 30 * 60 * 1000;
    const start = Math.ceil(windowStart / step) * step;
    for (let t = start; t <= now; t += step) {
      const p = (t - windowStart) / windowMs * 100;
      labels.push({
        label: new Date(t).toISOString().slice(11, 16) + 'Z',
        pctPos: p,
      });
    }
    return labels;
  }, [windowStart, windowMs, now]);

  const nowPct = 100; // current time is always at right edge

  return (
    <div
      style={{
        background: 'var(--background-panel)',
        borderTop: '1px solid var(--border)',
        flexShrink: 0,
        transition: 'height 0.2s ease-out',
        height: isOpen ? (assetTasks.length === 0 ? '60px' : `${Math.min(200, 36 + assetTasks.length * 28 + 24)}px`) : '28px',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Toggle header */}
      <div
        className="flex items-center justify-between flex-none"
        style={{
          height: '28px',
          paddingLeft: '12px',
          paddingRight: '12px',
          borderBottom: isOpen ? '1px solid var(--border)' : 'none',
          cursor: 'pointer',
        }}
        onClick={onToggle}
      >
        <div className="flex items-center gap-3">
          <span style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            fontSize: '9px',
            fontWeight: 'bold',
            letterSpacing: '0.15em',
            color: 'var(--text-dim)',
          }}>
            MISSION TIMELINE
          </span>
          <span style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            fontSize: '8px',
            color: 'var(--text-muted)',
          }}>
            {tasks.filter(t => t.status === 'active').length} ACTIVE · LAST {WINDOW_MINUTES}MIN
          </span>
        </div>
        <span style={{
          fontFamily: 'var(--font-ibm-plex-mono), monospace',
          fontSize: '9px',
          color: 'var(--text-muted)',
          transform: isOpen ? 'rotate(180deg)' : 'none',
          transition: 'transform 0.2s',
          display: 'inline-block',
        }}>
          ▲
        </span>
      </div>

      {/* Timeline content */}
      {isOpen && (
        <div className="flex-1 overflow-hidden flex flex-col">
          {assetTasks.length === 0 ? (
            <div className="flex items-center justify-center flex-1"
              style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', fontSize: '9px', color: 'var(--text-muted)', letterSpacing: '0.12em' }}>
              NO ACTIVE MISSIONS
            </div>
          ) : (
            <>
              {/* Time axis */}
              <div className="relative flex-none" style={{ height: '18px', marginLeft: '96px', marginRight: '8px' }}>
                {timeLabels.map((l, i) => (
                  <span
                    key={i}
                    style={{
                      position: 'absolute',
                      left: `${l.pctPos}%`,
                      transform: 'translateX(-50%)',
                      fontFamily: 'var(--font-ibm-plex-mono), monospace',
                      fontSize: '7px',
                      color: 'var(--text-muted)',
                      letterSpacing: '0.04em',
                      top: '4px',
                    }}
                  >
                    {l.label}
                  </span>
                ))}
                {/* NOW marker */}
                <div style={{
                  position: 'absolute',
                  left: `${nowPct}%`,
                  top: 0,
                  height: '18px',
                  width: '1px',
                  background: 'var(--accent)',
                  opacity: 0.8,
                }} />
              </div>

              {/* Asset rows */}
              {assetTasks.map(([assetId, { name, tasks: aTasks }]) => (
                <div key={assetId} className="flex items-center flex-none" style={{ height: '26px', paddingRight: '8px' }}>
                  {/* Asset name */}
                  <div style={{
                    width: '96px',
                    flexShrink: 0,
                    paddingLeft: '12px',
                    fontFamily: 'var(--font-ibm-plex-mono), monospace',
                    fontSize: '8px',
                    color: 'var(--text-dim)',
                    letterSpacing: '0.06em',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}>
                    {name}
                  </div>

                  {/* Timeline track */}
                  <div className="flex-1 relative" style={{ height: '18px', background: 'var(--accent-5)', borderRadius: '2px' }}>
                    {/* Grid lines */}
                    {timeLabels.map((l, i) => (
                      <div key={i} style={{
                        position: 'absolute',
                        left: `${l.pctPos}%`,
                        top: 0, bottom: 0, width: '1px',
                        background: 'var(--border)',
                        opacity: 0.5,
                      }} />
                    ))}

                    {/* Task blocks */}
                    {aTasks.map(t => {
                      const startPct = pct(t.started_at || t.created_at, windowStart, windowMs);
                      const endPct = t.completed_at ? pct(t.completed_at, windowStart, windowMs) : nowPct;
                      const width = Math.max(0.5, endPct - startPct);
                      const color = TASK_COLORS[t.action] || '#8A8A8A';
                      return (
                        <div
                          key={t.task_id}
                          title={`${t.action.replace(/_/g, ' ')} · ${t.status}`}
                          style={{
                            position: 'absolute',
                            left: `${startPct}%`,
                            width: `${width}%`,
                            top: '3px',
                            height: '12px',
                            background: color,
                            opacity: t.status === 'active' ? 0.85 : 0.4,
                            borderRadius: '1px',
                          }}
                        />
                      );
                    })}

                    {/* NOW line */}
                    <div style={{
                      position: 'absolute',
                      left: `${nowPct}%`,
                      top: 0, bottom: 0, width: '1px',
                      background: 'var(--accent)',
                      opacity: 0.9,
                    }} />
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}
