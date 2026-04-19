'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { fetchTasks, TaskAPI } from '@/lib/api';

const TASK_TYPE_COLORS: Record<string, string> = {
  VOLUME_SEARCH:  '#4FC3F7',
  AREA_SEARCH:    '#4FC3F7',
  TRANSIT:        '#FFB300',
  SURVEILLANCE:   '#00FF9C',
  PATROL:         '#00FF9C',
  INTERCEPT:      '#FF3B3B',
  DISPATCH:       '#FFB300',
  RTB:            '#8A8A8A',
  HALT:           '#FF3B3B',
};

const STATUS_COLOR: Record<string, string> = {
  active:    '#00FF9C',
  pending:   '#FFB300',
  completed: '#4A5568',
  failed:    '#FF3B3B',
};

function taskTypeLabel(action: string): string {
  return action.replace(/_/g, ' ').toUpperCase();
}

function elapsed(iso: string): string {
  const secs = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}

function createdAt(iso: string): string {
  return new Date(iso).toISOString().slice(11, 19) + 'Z';
}

// Mock tasks so the panel is useful when backend isn't live
function generateMockTasks(): TaskAPI[] {
  const now = Date.now();
  return [
    {
      task_id: 'task-mock-001',
      asset_id: 'sim-drone-001',
      asset_name: 'FALCON-01',
      action: 'VOLUME_SEARCH',
      status: 'active',
      risk_level: 'LOW',
      created_at: new Date(now - 9 * 60000).toISOString(),
      started_at:  new Date(now - 8 * 60000).toISOString(),
      waypoints: [{ lat: 34.052, lon: -118.243 }],
      objective: 'Search Zone Alpha',
    },
    {
      task_id: 'task-mock-002',
      asset_id: 'sim-drone-002',
      asset_name: 'RAVEN-03',
      action: 'SURVEILLANCE',
      status: 'active',
      risk_level: 'LOW',
      created_at: new Date(now - 14 * 60000).toISOString(),
      started_at:  new Date(now - 13 * 60000).toISOString(),
      waypoints: [{ lat: 34.071, lon: -118.261 }],
      objective: 'Overwatch Staging Area',
    },
    {
      task_id: 'task-mock-003',
      asset_id: 'sim-ugv-001',
      asset_name: 'ROVER-01',
      action: 'TRANSIT',
      status: 'active',
      risk_level: 'LOW',
      created_at: new Date(now - 4 * 60000).toISOString(),
      started_at:  new Date(now - 3 * 60000).toISOString(),
      waypoints: [{ lat: 34.038, lon: -118.255 }],
      objective: 'Transit to Sector B',
    },
    {
      task_id: 'task-mock-004',
      asset_id: 'sim-drone-003',
      asset_name: 'HAWK-05',
      action: 'AREA_SEARCH',
      status: 'pending',
      risk_level: 'MEDIUM',
      created_at: new Date(now - 60000).toISOString(),
      waypoints: [{ lat: 34.065, lon: -118.230 }],
      objective: 'Zone B sweep',
    },
  ];
}

type FilterType = 'ALL' | 'ACTIVE' | 'PENDING' | 'DONE';

interface OpsActiveTasksPanelProps {
  onFlyTo?: (lat: number, lon: number) => void;
}

export default function OpsActiveTasksPanel({ onFlyTo }: OpsActiveTasksPanelProps) {
  const [tasks, setTasks] = useState<TaskAPI[]>([]);
  const [filter, setFilter] = useState<FilterType>('ALL');
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const load = useCallback(async () => {
    const data = await fetchTasks();
    setTasks(data.length > 0 ? data : generateMockTasks());
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [load]);

  const filtered = tasks.filter(t => {
    if (filter === 'ALL') return t.status !== 'completed' && t.status !== 'failed';
    if (filter === 'ACTIVE') return t.status === 'active';
    if (filter === 'PENDING') return t.status === 'pending';
    if (filter === 'DONE') return t.status === 'completed' || t.status === 'failed';
    return true;
  });

  const activeCnt  = tasks.filter(t => t.status === 'active').length;
  const pendingCnt = tasks.filter(t => t.status === 'pending').length;

  const toggleExpand = (id: string) => {
    setExpanded(prev => {
      const n = new Set(prev);
      if (n.has(id)) n.delete(id); else n.add(id);
      return n;
    });
  };

  return (
    <div className="h-full flex flex-col" style={{ background: 'var(--background-panel)' }}>
      <div className="flex-none flex items-center justify-between px-3 py-2"
        style={{ borderBottom: '1px solid var(--border)' }}>
        <span style={{
          fontFamily: 'var(--font-ibm-plex-mono), monospace',
          fontSize: '11px',
          fontWeight: 'bold',
          letterSpacing: '0.15em',
          color: 'rgba(200,230,201,0.9)',
        }}>
          ACTIVE TASKS
        </span>
        <div className="flex items-center gap-2">
          <span style={{
            fontSize: '9px',
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: 'var(--accent)',
            background: 'var(--accent-10)',
            border: '1px solid var(--accent-20)',
            padding: '1px 6px',
          }}>
            {activeCnt} LIVE
          </span>
          {pendingCnt > 0 && (
            <span style={{
              fontSize: '9px',
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: 'var(--warning)',
              background: 'color-mix(in srgb, var(--warning) 10%, transparent)',
              border: '1px solid color-mix(in srgb, var(--warning) 30%, transparent)',
              padding: '1px 6px',
            }}>
              {pendingCnt} QUEUED
            </span>
          )}
        </div>
      </div>

      {/* Filter tabs */}
      <div className="flex-none flex px-3 py-2 gap-1"
        style={{ borderBottom: '1px solid var(--border)' }}>
        {(['ALL', 'ACTIVE', 'PENDING', 'DONE'] as FilterType[]).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              fontSize: '9px',
              letterSpacing: '0.1em',
              padding: '2px 8px',
              cursor: 'pointer',
              border: `1px solid ${filter === f ? 'var(--accent-50)' : 'var(--border)'}`,
              background: filter === f ? 'var(--accent-10)' : 'transparent',
              color: filter === f ? 'var(--accent)' : 'var(--text-muted)',
            }}
          >
            {f}
          </button>
        ))}
      </div>

      {/* Task list */}
      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center h-32"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', fontSize: '10px', color: 'var(--text-muted)', letterSpacing: '0.1em' }}>
            NO {filter === 'ALL' ? 'ACTIVE' : filter} TASKS
          </div>
        ) : (
          filtered.map(task => {
            const typeColor = TASK_TYPE_COLORS[task.action] || '#8A8A8A';
            const statusColor = STATUS_COLOR[task.status] || '#8A8A8A';
            const isExpanded = expanded.has(task.task_id);
            const wp = task.waypoints?.[0];

            return (
              <div
                key={task.task_id}
                style={{
                  borderBottom: '1px solid var(--border)',
                  borderLeft: `2px solid ${typeColor}`,
                  cursor: 'pointer',
                }}
                onClick={() => toggleExpand(task.task_id)}
              >
                {/* Task header row */}
                <div className="flex items-start justify-between px-3 pt-2 pb-1">
                  <div className="flex flex-col gap-0.5 flex-1 min-w-0">
                    {/* Type badge + created time */}
                    <div className="flex items-center gap-2">
                      <span style={{
                        fontFamily: 'var(--font-ibm-plex-mono), monospace',
                        fontSize: '9px',
                        fontWeight: 'bold',
                        color: typeColor,
                        letterSpacing: '0.08em',
                      }}>
                        {taskTypeLabel(task.action)}
                      </span>
                      <span style={{
                        fontFamily: 'var(--font-ibm-plex-mono), monospace',
                        fontSize: '8px',
                        color: 'var(--text-muted)',
                      }}>
                        {createdAt(task.created_at)}
                      </span>
                    </div>

                    {/* Asset + objective */}
                    <div className="flex items-center gap-2">
                      <span style={{
                        fontFamily: 'var(--font-ibm-plex-mono), monospace',
                        fontSize: '9px',
                        color: 'rgba(200,230,201,0.85)',
                        letterSpacing: '0.04em',
                      }}>
                        {task.asset_name || task.asset_id.slice(0, 10).toUpperCase()}
                      </span>
                      {task.objective && (
                        <span style={{
                          fontFamily: 'var(--font-ibm-plex-mono), monospace',
                          fontSize: '8px',
                          color: 'var(--text-dim)',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}>
                          ⚡ {task.objective}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Status dot + elapsed */}
                  <div className="flex flex-col items-end gap-0.5 flex-none ml-2">
                    <div className="flex items-center gap-1">
                      <div style={{
                        width: '6px', height: '6px', borderRadius: '50%',
                        background: statusColor,
                        boxShadow: task.status === 'active' ? `0 0 5px ${statusColor}` : 'none',
                      }} />
                      <span style={{
                        fontFamily: 'var(--font-ibm-plex-mono), monospace',
                        fontSize: '8px',
                        color: statusColor,
                        letterSpacing: '0.08em',
                      }}>
                        {task.status.toUpperCase()}
                      </span>
                    </div>
                    <span style={{
                      fontFamily: 'var(--font-ibm-plex-mono), monospace',
                      fontSize: '8px',
                      color: 'var(--text-muted)',
                    }}>
                      +{elapsed(task.started_at || task.created_at)}
                    </span>
                  </div>
                </div>

                {/* Expanded details */}
                {isExpanded && wp && (
                  <div className="px-3 pb-2" style={{ borderTop: '1px solid var(--accent-10)' }}>
                    <div className="flex items-center justify-between mt-1.5">
                      <span style={{
                        fontFamily: 'var(--font-ibm-plex-mono), monospace',
                        fontSize: '8px',
                        color: 'var(--text-muted)',
                        letterSpacing: '0.06em',
                      }}>
                        ◎ {wp.lat.toFixed(4)}, {wp.lon.toFixed(4)}
                        {wp.alt != null ? ` · ${Math.round(wp.alt * 3.28084)}ft` : ''}
                      </span>
                      <button
                        onClick={(e) => { e.stopPropagation(); onFlyTo?.(wp.lat, wp.lon); }}
                        style={{
                          fontFamily: 'var(--font-ibm-plex-mono), monospace',
                          fontSize: '8px',
                          letterSpacing: '0.1em',
                          padding: '1px 6px',
                          background: 'var(--accent-10)',
                          border: '1px solid var(--accent-20)',
                          color: 'var(--accent)',
                          cursor: 'pointer',
                        }}
                      >
                        GO TO
                      </button>
                    </div>
                    <div className="mt-1" style={{
                      fontFamily: 'var(--font-ibm-plex-mono), monospace',
                      fontSize: '8px',
                      color: 'var(--text-muted)',
                    }}>
                      RISK: {task.risk_level}  ·  ID: {task.task_id.slice(0, 12)}
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
