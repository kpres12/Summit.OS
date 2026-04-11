'use client';

import React, { useState, useEffect, useCallback } from 'react';
import Map, { Marker, NavigationControl } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useEntityStream, EntityData } from '@/hooks/useEntityStream';
import { fetchMissions, fetchAlerts, connectWebSocket, fetchPendingApprovals, approveTask, MissionAPI, AlertAPI } from '@/lib/api';
import ErrorBoundary from '@/components/ErrorBoundary';

const DARK_STYLE =
  process.env.NEXT_PUBLIC_TILE_URL ||
  'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

interface CommandLayoutProps {
  onSwitchRole: () => void;
}

// ─── Utilities ────────────────────────────────────────────────

function utcString(d: Date): string {
  const y = d.getUTCFullYear();
  const mo = String(d.getUTCMonth() + 1).padStart(2, '0');
  const day = String(d.getUTCDate()).padStart(2, '0');
  const h = String(d.getUTCHours()).padStart(2, '0');
  const m = String(d.getUTCMinutes()).padStart(2, '0');
  const s = String(d.getUTCSeconds()).padStart(2, '0');
  return `${y}-${mo}-${day} // ${h}:${m}:${s}Z`;
}

function ageString(isoOrEpoch: string | number): string {
  const ts = typeof isoOrEpoch === 'number'
    ? isoOrEpoch * 1000
    : new Date(isoOrEpoch).getTime();
  const diff = Math.floor((Date.now() - ts) / 1000);
  if (diff < 60) return `${diff}s`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
  return `${Math.floor(diff / 86400)}d`;
}

function markerColor(e: EntityData): string {
  switch (e.entity_type) {
    case 'active': return '#00E896';
    case 'alert': return '#FF3B3B';
    case 'neutral': return 'rgba(200,230,201,0.6)';
    default: return '#FFB300';
  }
}

function batteryColor(pct: number): string {
  if (pct > 40) return '#00E896';
  if (pct > 20) return '#FFB300';
  return '#FF3B3B';
}

// ─── Event types ──────────────────────────────────────────────

type EventType = 'ALL' | 'DETECTIONS' | 'MISSIONS' | 'ASSETS' | 'ALERTS';

interface SituationEvent {
  id: string;
  type: Exclude<EventType, 'ALL'>;
  timestamp: number;
  description: string;
}

const EVENT_TYPE_COLOR: Record<Exclude<EventType, 'ALL'>, string> = {
  DETECTIONS: '#00E896',
  MISSIONS: '#4FC3F7',
  ASSETS: '#FFB300',
  ALERTS: '#FF3B3B',
};

// ─── Top Bar ─────────────────────────────────────────────────

function CommandTopBar({
  onSwitchRole,
  pendingApprovals,
  onToggleApprovals,
}: {
  onSwitchRole: () => void;
  pendingApprovals: number;
  onToggleApprovals: () => void;
}) {
  const { connected } = useEntityStream();
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  return (
    <div
      className="flex-none flex items-center px-4 relative"
      style={{ height: '40px', background: '#0D1210', borderBottom: '1px solid rgba(0,232,150,0.15)' }}
    >
      {/* Left */}
      <div className="flex items-center gap-3 z-10">
        <span
          className="text-sm font-bold tracking-widest"
          style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: '#00E896' }}
        >
          SUMMIT.OS
        </span>
        <span style={{ color: 'rgba(0,232,150,0.3)' }}>|</span>
        <span
          className="text-xs tracking-widest"
          style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.45)', fontSize: '10px' }}
        >
          COMMAND
        </span>
        {pendingApprovals > 0 && (
          <button
            onClick={onToggleApprovals}
            className="text-[10px] px-2 py-0.5 font-bold"
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: '#FF3B3B',
              border: '1px solid rgba(255,59,59,0.5)',
              background: 'rgba(255,59,59,0.1)',
              cursor: 'pointer',
            }}
          >
            [APPROVALS: {pendingApprovals}]
          </button>
        )}
      </div>

      {/* Center */}
      <div
        className="absolute left-1/2 -translate-x-1/2 text-xs"
        style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.45)' }}
      >
        {utcString(now)}
      </div>

      {/* Right */}
      <div className="ml-auto flex items-center gap-3 z-10">
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full" style={{ background: connected ? '#00E896' : '#FF3B3B' }} />
          <span className="text-[10px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: connected ? '#00E896' : '#FF3B3B' }}>
            {connected ? 'WS LIVE' : 'WS DOWN'}
          </span>
        </div>
        <button
          onClick={onSwitchRole}
          className="text-[10px] tracking-widest px-2 py-0.5 transition-colors"
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: 'rgba(200,230,201,0.45)',
            border: '1px solid rgba(0,232,150,0.15)',
            background: 'transparent',
            cursor: 'pointer',
          }}
        >
          ⊕ ROLE
        </button>
      </div>
    </div>
  );
}

// ─── Situation Feed ───────────────────────────────────────────

function SituationFeed() {
  const [events, setEvents] = useState<SituationEvent[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeFilters, setActiveFilters] = useState<Set<EventType>>(new Set(['ALL']));

  const pushEvent = useCallback((ev: SituationEvent) => {
    setEvents((prev) => [ev, ...prev].slice(0, 200));
  }, []);

  const handleMsg = useCallback((data: unknown) => {
    const msg = data as { type?: string; data?: unknown };
    const id = `${Date.now()}-${Math.random()}`;
    if (msg.type === 'entity_update') {
      const e = msg.data as EntityData;
      pushEvent({
        id,
        type: 'DETECTIONS',
        timestamp: Date.now(),
        description: `DETECTION: ${e.callsign || e.entity_id.slice(0, 8)} [${e.entity_type.toUpperCase()}]`,
      });
    } else if (msg.type === 'mission_update') {
      const m = msg.data as MissionAPI;
      pushEvent({
        id,
        type: 'MISSIONS',
        timestamp: Date.now(),
        description: `MISSION UPDATE: ${m.name || m.mission_id.slice(0, 8)} → ${m.status}`,
      });
    } else if (msg.type === 'entity_removed') {
      pushEvent({
        id,
        type: 'ASSETS',
        timestamp: Date.now(),
        description: `ASSET REMOVED: ${(msg.data as { entity_id?: string })?.entity_id?.slice(0, 8) || '?'}`,
      });
    } else if (msg.type === 'alert') {
      const a = msg.data as AlertAPI;
      pushEvent({
        id,
        type: 'ALERTS',
        timestamp: Date.now(),
        description: `ALERT [${a.severity}]: ${a.description.slice(0, 50)}`,
      });
    }
  }, [pushEvent]);

  useEffect(() => {
    const ws = connectWebSocket(handleMsg);
    return () => { ws?.close(); };
  }, [handleMsg]);

  const toggleFilter = (f: EventType) => {
    if (f === 'ALL') {
      setActiveFilters(new Set(['ALL']));
    } else {
      setActiveFilters((prev) => {
        const next = new Set(prev);
        next.delete('ALL');
        if (next.has(f)) {
          next.delete(f);
          if (next.size === 0) next.add('ALL');
        } else {
          next.add(f);
        }
        return next;
      });
    }
  };

  const isAll = activeFilters.has('ALL');
  const filtered = events.filter((e) => isAll || activeFilters.has(e.type));

  const FILTERS: EventType[] = ['ALL', 'DETECTIONS', 'MISSIONS', 'ASSETS', 'ALERTS'];

  return (
    <div
      className="flex flex-col h-full"
      style={{ background: '#0D1210', borderRight: '1px solid rgba(0,232,150,0.15)' }}
    >
      {/* Header */}
      <div
        className="flex-none px-3 py-2"
        style={{ borderBottom: '1px solid rgba(0,232,150,0.15)' }}
      >
        <span
          className="text-xs font-bold tracking-widest"
          style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: '#00E896' }}
        >
          SITUATION FEED
        </span>
      </div>

      {/* Filter bar */}
      <div
        className="flex-none flex flex-wrap gap-1 px-2 py-2"
        style={{ borderBottom: '1px solid rgba(0,232,150,0.1)' }}
      >
        {FILTERS.map((f) => {
          const active = activeFilters.has(f);
          const color = f === 'ALL' ? '#00E896' : EVENT_TYPE_COLOR[f as Exclude<EventType, 'ALL'>];
          return (
            <button
              key={f}
              onClick={() => toggleFilter(f)}
              className="text-[9px] px-1.5 py-0.5 tracking-wider transition-colors"
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color: active ? color : 'rgba(200,230,201,0.35)',
                border: `1px solid ${active ? color + '60' : 'rgba(0,232,150,0.1)'}`,
                background: active ? `${color}10` : 'transparent',
                cursor: 'pointer',
              }}
            >
              {f}
            </button>
          );
        })}
      </div>

      {/* Events */}
      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 && (
          <div
            className="flex items-center justify-center h-full text-[10px]"
            style={{ color: 'rgba(200,230,201,0.25)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
          >
            MONITORING...
          </div>
        )}
        {filtered.map((ev) => {
          const color = EVENT_TYPE_COLOR[ev.type];
          const isSelected = ev.id === selectedId;
          return (
            <div
              key={ev.id}
              onClick={() => setSelectedId(isSelected ? null : ev.id)}
              className="px-3 py-2 cursor-pointer transition-colors"
              style={{
                borderLeft: `3px solid ${color}`,
                borderBottom: '1px solid rgba(0,232,150,0.05)',
                background: isSelected ? `${color}0A` : 'transparent',
              }}
            >
              <div
                className="text-[9px] mb-0.5"
                style={{ color: 'rgba(200,230,201,0.35)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
              >
                {ageString(ev.timestamp)} · {ev.type}
              </div>
              <div
                className="text-[10px] leading-tight"
                style={{ color: isSelected ? color : 'rgba(200,230,201,0.7)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
              >
                {ev.description}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Resource Status ──────────────────────────────────────────

function ResourceStatus() {
  const { entityList } = useEntityStream();
  const [missions, setMissions] = useState<MissionAPI[]>([]);
  const [alerts, setAlerts] = useState<AlertAPI[]>([]);
  const [showBrief, setShowBrief] = useState(false);
  const [briefText, setBriefText] = useState('');
  const [copied, setCopied] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    fetchMissions(20).then((data) => setMissions(data || [])).catch((e: Error) => setFetchError(e.message));
    fetchAlerts(100).then((r) => setAlerts(r.alerts || [])).catch((e: Error) => setFetchError(e.message));
  }, []);

  const activeEntities = entityList.filter((e) => e.speed_mps > 0.5);
  const idleEntities = entityList.filter((e) => e.speed_mps <= 0.5);
  const activeMissions = missions.filter((m) => m.status.toUpperCase() === 'ACTIVE');

  const domainGroups: Record<string, EntityData[]> = {
    Aerial: entityList.filter((e) => e.domain === 'aerial'),
    Ground: entityList.filter((e) => e.domain === 'ground'),
    Maritime: entityList.filter((e) => e.domain === 'maritime'),
    Sensors: entityList.filter((e) => e.domain === 'sensor' || e.domain === 'fixed'),
  };

  const generateBrief = () => {
    const ts = new Date().toISOString().replace('T', ' ').slice(0, 19) + 'Z';
    const missionLines = activeMissions.map((m) => `  · ${m.name || m.mission_id.slice(0, 8)} [${m.status}]`).join('\n') || '  None';
    const assetLines = entityList.map((e) => {
      const bat = e.battery_pct !== undefined ? ` BAT:${Math.round(e.battery_pct)}%` : '';
      return `  · ${e.callsign || e.entity_id.slice(0, 8)} [${e.entity_type.toUpperCase()}]${bat}`;
    }).join('\n') || '  None';

    const brief = `SUMMIT.OS — SITUATION BRIEF
Generated: ${ts}
─────────────────────────────────
ACTIVE ASSETS: ${activeEntities.length}
ACTIVE MISSIONS: ${activeMissions.length}
ALERTS (last 4h): ${alerts.length}
─────────────────────────────────
MISSIONS:
${missionLines}
─────────────────────────────────
ASSETS:
${assetLines}
─────────────────────────────────
Generated by Summit.OS Command Console`;

    setBriefText(brief);
    setShowBrief(true);
  };

  const copyBrief = async () => {
    try {
      await navigator.clipboard.writeText(briefText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {}
  };

  return (
    <div
      className="flex flex-col h-full"
      style={{ background: '#0D1210', borderLeft: '1px solid rgba(0,232,150,0.15)' }}
    >
      {/* Header */}
      <div
        className="flex-none px-3 py-2"
        style={{ borderBottom: '1px solid rgba(0,232,150,0.15)' }}
      >
        <span
          className="text-xs font-bold tracking-widest"
          style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: '#00E896' }}
        >
          RESOURCES
        </span>
      </div>

      {/* Fetch error badge */}
      {fetchError && (
        <div className="flex-none px-3 py-1 text-[9px]" style={{ background: 'rgba(255,59,59,0.08)', color: '#FF3B3B', fontFamily: 'var(--font-ibm-plex-mono), monospace', borderBottom: '1px solid rgba(255,59,59,0.2)' }}>
          DATA ERR: {fetchError}
        </div>
      )}

      {/* Summary row */}
      <div
        className="flex-none px-3 py-2 text-[10px]"
        style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.45)', borderBottom: '1px solid rgba(0,232,150,0.1)' }}
      >
        {entityList.length} ASSETS // {activeEntities.length} ACTIVE // {idleEntities.length} IDLE // 0 OFFLINE
      </div>

      {/* Entity list by domain */}
      <div className="flex-1 overflow-y-auto">
        {Object.entries(domainGroups).map(([domain, entities]) => {
          if (entities.length === 0) return null;
          return (
            <div key={domain}>
              <div
                className="px-3 py-1 text-[9px] tracking-widest"
                style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  color: 'rgba(0,232,150,0.4)',
                  background: 'rgba(0,232,150,0.03)',
                  borderBottom: '1px solid rgba(0,232,150,0.08)',
                }}
              >
                {domain.toUpperCase()} ({entities.length})
              </div>
              {entities.map((e) => {
                const isActive = e.speed_mps > 0.5;
                return (
                  <div
                    key={e.entity_id}
                    className="px-3 py-1.5 flex items-center gap-2"
                    style={{ borderBottom: '1px solid rgba(0,232,150,0.05)' }}
                  >
                    <div
                      className="w-1.5 h-1.5 rounded-full flex-none"
                      style={{ background: isActive ? '#00E896' : 'rgba(200,230,201,0.3)' }}
                    />
                    <span
                      className="flex-1 text-[10px] truncate"
                      style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.7)' }}
                    >
                      {e.callsign || e.entity_id.slice(0, 8)}
                    </span>
                    {e.battery_pct !== undefined && (
                      <div className="flex items-center gap-1 flex-none">
                        <div
                          className="h-1 rounded-full overflow-hidden"
                          style={{ width: '24px', background: 'rgba(0,232,150,0.1)' }}
                        >
                          <div
                            className="h-full"
                            style={{ width: `${e.battery_pct}%`, background: batteryColor(e.battery_pct) }}
                          />
                        </div>
                      </div>
                    )}
                    <span
                      className="text-[9px] px-1 flex-none"
                      style={{
                        fontFamily: 'var(--font-ibm-plex-mono), monospace',
                        color: isActive ? '#00E896' : 'rgba(200,230,201,0.35)',
                        border: `1px solid ${isActive ? 'rgba(0,232,150,0.3)' : 'rgba(200,230,201,0.1)'}`,
                      }}
                    >
                      {isActive ? 'ACT' : 'IDL'}
                    </span>
                  </div>
                );
              })}
            </div>
          );
        })}
        {entityList.length === 0 && (
          <div
            className="flex items-center justify-center h-20 text-[10px]"
            style={{ color: 'rgba(200,230,201,0.25)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
          >
            NO ASSETS
          </div>
        )}

        {/* Mission status */}
        <div
          className="px-3 py-1 text-[9px] tracking-widest mt-1"
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: 'rgba(0,232,150,0.4)',
            background: 'rgba(0,232,150,0.03)',
            borderTop: '1px solid rgba(0,232,150,0.08)',
            borderBottom: '1px solid rgba(0,232,150,0.08)',
          }}
        >
          MISSION STATUS
        </div>
        {missions.length === 0 && (
          <div
            className="px-3 py-3 text-[10px]"
            style={{ color: 'rgba(200,230,201,0.25)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
          >
            NO MISSIONS
          </div>
        )}
        {missions.slice(0, 10).map((m) => {
          const statusColor = m.status.toUpperCase() === 'ACTIVE' ? '#00E896'
            : m.status.toUpperCase() === 'FAILED' ? '#FF3B3B'
            : m.status.toUpperCase() === 'COMPLETED' ? '#4FC3F7'
            : 'rgba(200,230,201,0.45)';
          return (
            <div
              key={m.mission_id}
              className="px-3 py-2 flex items-center justify-between"
              style={{ borderBottom: '1px solid rgba(0,232,150,0.05)' }}
            >
              <span
                className="text-[10px] truncate flex-1"
                style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.7)' }}
              >
                {m.name || m.mission_id.slice(0, 8)}
              </span>
              <span
                className="text-[9px] px-1 flex-none ml-2"
                style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  color: statusColor,
                  border: `1px solid ${statusColor}50`,
                }}
              >
                {m.status.toUpperCase()}
              </span>
            </div>
          );
        })}
      </div>

      {/* Handoff Brief button */}
      <div
        className="flex-none p-3"
        style={{ borderTop: '1px solid rgba(0,232,150,0.15)' }}
      >
        <button
          onClick={generateBrief}
          className="w-full py-2 text-xs font-bold tracking-widest transition-colors"
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: '#080C0A',
            background: '#00E896',
            border: 'none',
            cursor: 'pointer',
          }}
          onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.background = '#00CC74')}
          onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.background = '#00E896')}
        >
          HANDOFF BRIEF
        </button>
      </div>

      {/* Brief modal */}
      {showBrief && (
        <div
          className="fixed inset-0 flex items-center justify-center z-50"
          style={{ background: 'rgba(8,12,10,0.9)' }}
          onClick={() => setShowBrief(false)}
        >
          <div
            className="flex flex-col"
            style={{
              width: '540px',
              maxHeight: '80vh',
              background: '#0D1210',
              border: '1px solid rgba(0,232,150,0.3)',
              boxShadow: '0 0 40px rgba(0,232,150,0.1)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div
              className="flex items-center justify-between px-4 py-3"
              style={{ borderBottom: '1px solid rgba(0,232,150,0.15)' }}
            >
              <span
                className="text-sm font-bold tracking-widest"
                style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: '#00E896' }}
              >
                HANDOFF BRIEF
              </span>
              <button
                onClick={() => setShowBrief(false)}
                style={{ color: 'rgba(200,230,201,0.4)', background: 'none', border: 'none', cursor: 'pointer' }}
              >
                ✕
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              <pre
                className="text-[11px] leading-relaxed whitespace-pre-wrap"
                style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.8)' }}
              >
                {briefText}
              </pre>
            </div>
            <div
              className="flex-none p-3"
              style={{ borderTop: '1px solid rgba(0,232,150,0.15)' }}
            >
              <button
                onClick={copyBrief}
                className="w-full py-2 text-xs tracking-widest transition-colors"
                style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  color: copied ? '#00E896' : 'rgba(200,230,201,0.7)',
                  border: `1px solid ${copied ? 'rgba(0,232,150,0.5)' : 'rgba(0,232,150,0.2)'}`,
                  background: 'transparent',
                  cursor: 'pointer',
                }}
              >
                {copied ? 'COPIED!' : 'COPY TO CLIPBOARD'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Command Layout ───────────────────────────────────────────

export default function CommandLayout({ onSwitchRole }: CommandLayoutProps) {
  const { entityList } = useEntityStream();
  const [pendingApprovals, setPendingApprovals] = useState(0);
  const [approvalQueue, setApprovalQueue] = useState<{ task_id: string; asset_id: string; action: string; risk_level: string }[]>([]);
  const [showApprovals, setShowApprovals] = useState(false);
  const [approvingId, setApprovingId] = useState<string | null>(null);

  const refreshApprovals = useCallback(() => {
    fetchPendingApprovals()
      .then((tasks) => {
        setApprovalQueue(tasks);
        setPendingApprovals(tasks.length);
      })
      .catch(() => {});
    fetchMissions(20)
      .then((data) => {
        const missionPending = (data || []).filter((m) => m.status.toUpperCase() === 'PENDING_APPROVAL').length;
        setPendingApprovals((n) => Math.max(n, missionPending));
      })
      .catch((e: Error) => console.warn('[CommandLayout] missions fetch failed:', e.message));
  }, []);

  useEffect(() => {
    refreshApprovals();
    const interval = setInterval(refreshApprovals, 15_000);
    return () => clearInterval(interval);
  }, [refreshApprovals]);

  const handleApprove = useCallback(async (taskId: string) => {
    setApprovingId(taskId);
    try {
      await approveTask(taskId, 'commander');
      refreshApprovals();
    } catch {
      // keep queue visible, operator can retry
    } finally {
      setApprovingId(null);
    }
  }, [refreshApprovals]);

  return (
    <ErrorBoundary>
    <div className="fixed inset-0 flex flex-col" style={{ background: '#080C0A', position: 'relative' }}>
      {/* Top bar */}
      <CommandTopBar
        onSwitchRole={onSwitchRole}
        pendingApprovals={pendingApprovals}
        onToggleApprovals={() => setShowApprovals((v) => !v)}
      />
      {showApprovals && (
        <div
          style={{
            position: 'absolute',
            top: '40px',
            left: '12px',
            zIndex: 200,
            background: '#0D1210',
            border: '1px solid rgba(255,59,59,0.4)',
            width: '340px',
            maxHeight: '400px',
            overflowY: 'auto',
          }}
        >
          <div style={{ padding: '8px 12px', borderBottom: '1px solid rgba(255,59,59,0.2)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: '#FF3B3B', fontSize: '10px', letterSpacing: '0.15em' }}>
              PENDING APPROVALS
            </span>
            <button onClick={() => setShowApprovals(false)} style={{ color: 'rgba(200,230,201,0.45)', background: 'none', border: 'none', cursor: 'pointer', fontSize: '11px' }}>✕</button>
          </div>
          {approvalQueue.length === 0 ? (
            <div style={{ padding: '12px', fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.45)', fontSize: '10px' }}>
              No tasks awaiting approval.
            </div>
          ) : approvalQueue.map((task) => (
            <div key={task.task_id} style={{ padding: '8px 12px', borderBottom: '1px solid rgba(255,59,59,0.1)', display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: '#FFB300', fontSize: '10px' }}>
                  {task.asset_id}
                </span>
                <span style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.45)', fontSize: '9px' }}>
                  {task.risk_level}
                </span>
              </div>
              <div style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.7)', fontSize: '9px' }}>
                ACTION: {task.action}
              </div>
              <button
                onClick={() => handleApprove(task.task_id)}
                disabled={approvingId === task.task_id}
                style={{
                  marginTop: '4px',
                  padding: '4px 12px',
                  background: 'transparent',
                  border: '1px solid rgba(0,232,150,0.4)',
                  color: '#00E896',
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  fontSize: '9px',
                  letterSpacing: '0.1em',
                  cursor: approvingId === task.task_id ? 'not-allowed' : 'pointer',
                  opacity: approvingId === task.task_id ? 0.5 : 1,
                  alignSelf: 'flex-start',
                }}
              >
                {approvingId === task.task_id ? 'APPROVING...' : 'APPROVE'}
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Main row */}
      <div className="flex flex-row flex-1 overflow-hidden">
        {/* LEFT 25% — Situation feed */}
        <div className="flex-none overflow-hidden" style={{ width: '25%' }}>
          <SituationFeed />
        </div>

        {/* CENTER 50% — Map */}
        <div className="flex-1 relative overflow-hidden">
          <Map
            initialViewState={{ longitude: -98.5, latitude: 39.8, zoom: 4 }}
            style={{ width: '100%', height: '100%' }}
            mapStyle={DARK_STYLE}
          >
            <NavigationControl position="top-right" />
            {entityList.map((entity) =>
              entity.position ? (
                <Marker
                  key={entity.entity_id}
                  longitude={entity.position.lon}
                  latitude={entity.position.lat}
                >
                  <div
                    title={entity.callsign || entity.entity_id}
                    style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      background: markerColor(entity),
                      border: '1px solid rgba(8,12,10,0.8)',
                      boxShadow: `0 0 4px ${markerColor(entity)}60`,
                    }}
                  />
                </Marker>
              ) : null
            )}
          </Map>
        </div>

        {/* RIGHT 25% — Resource status */}
        <div className="flex-none overflow-hidden" style={{ width: '25%' }}>
          <ResourceStatus />
        </div>
      </div>
    </div>
    </ErrorBoundary>
  );
}
