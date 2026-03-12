'use client';

import React, { useState, useEffect } from 'react';
import { EntityData } from '@/hooks/useEntityStream';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface OpsEntityDetailProps {
  entity: EntityData | null;
  onClose: () => void;
  onDispatch?: (entity: EntityData) => void;
}

function ageString(lastSeen: number): string {
  const diff = Math.floor((Date.now() / 1000) - lastSeen);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

function entityTypeColor(type: string): string {
  switch (type) {
    case 'active': return '#00FF9C';
    case 'alert': return '#FF3B3B';
    case 'neutral': return 'rgba(200,230,201,0.45)';
    default: return '#FFB300';
  }
}

function batteryColor(pct: number): string {
  if (pct > 40) return '#00FF9C';
  if (pct > 20) return '#FFB300';
  return '#FF3B3B';
}

function DataRow({ label, value, valueColor }: { label: string; value: string; valueColor?: string }) {
  return (
    <div className="flex items-baseline justify-between py-0.5">
      <span
        className="text-[10px]"
        style={{ color: 'rgba(200,230,201,0.45)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
      >
        {label}
      </span>
      <span
        className="text-[11px] font-bold"
        style={{ color: valueColor || '#00FF9C', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
      >
        {value}
      </span>
    </div>
  );
}

function SectionHeader({ title }: { title: string }) {
  return (
    <div
      className="text-[9px] font-bold tracking-widest pt-3 pb-1"
      style={{
        fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
        color: 'rgba(0,255,156,0.5)',
        borderBottom: '1px solid rgba(0,255,156,0.1)',
        marginBottom: '4px',
      }}
    >
      {title}
    </div>
  );
}

// Generate mock AI reasoning based on entity state
function buildThoughts(entity: EntityData): { ts: string; msg: string; confidence: number }[] {
  const thoughts = [];
  const now = Date.now();

  if (entity.entity_type === 'alert') {
    thoughts.push({ ts: `${new Date(now - 8000).toISOString().slice(11,19)}Z`, msg: `Anomalous velocity detected: ${entity.speed_mps.toFixed(1)} m/s exceeds baseline`, confidence: 0.91 });
    thoughts.push({ ts: `${new Date(now - 5000).toISOString().slice(11,19)}Z`, msg: 'Cross-referencing against known flight corridors — no match found', confidence: 0.87 });
    thoughts.push({ ts: `${new Date(now - 2000).toISOString().slice(11,19)}Z`, msg: 'Flagging for operator review. Recommend visual verification.', confidence: 0.84 });
  } else if (entity.battery_pct !== undefined && entity.battery_pct < 25) {
    thoughts.push({ ts: `${new Date(now - 6000).toISOString().slice(11,19)}Z`, msg: `Battery critical at ${entity.battery_pct.toFixed(0)}% — estimating 4 min flight time remaining`, confidence: 0.96 });
    thoughts.push({ ts: `${new Date(now - 3000).toISOString().slice(11,19)}Z`, msg: 'Initiating RTB evaluation. Current position within return range.', confidence: 0.94 });
  } else {
    thoughts.push({ ts: `${new Date(now - 10000).toISOString().slice(11,19)}Z`, msg: `Tracking ${entity.classification || 'entity'} on nominal trajectory`, confidence: 0.97 });
    thoughts.push({ ts: `${new Date(now - 4000).toISOString().slice(11,19)}Z`, msg: `Speed ${entity.speed_mps.toFixed(1)} m/s, heading ${entity.position.heading_deg.toFixed(0)}° — consistent with mission profile`, confidence: 0.95 });
  }
  return thoughts;
}

export default function OpsEntityDetail({ entity, onClose, onDispatch }: OpsEntityDetailProps) {
  const [dispatched, setDispatched] = useState(false);
  const [overrideStatus, setOverrideStatus] = useState<string | null>(null);
  const [thoughts, setThoughts] = useState<{ ts: string; msg: string; confidence: number }[]>([]);

  useEffect(() => {
    if (!entity) return;
    setDispatched(false);
    setOverrideStatus(null);
    // Try real reasoning endpoint first, fall back to local generation
    fetch(`${API}/reasoning/${entity.entity_id}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data?.thoughts?.length) {
          setThoughts(data.thoughts);
        } else {
          setThoughts(buildThoughts(entity));
        }
      })
      .catch(() => setThoughts(buildThoughts(entity)));
  }, [entity?.entity_id]);

  if (!entity) return null;

  const typeColor = entityTypeColor(entity.entity_type);
  const shortId = entity.entity_id.slice(0, 12);
  const displayName = entity.callsign || shortId;

  const handleDispatch = () => {
    setDispatched(true);
    onDispatch?.(entity);
    setTimeout(() => {
      onClose();
    }, 600);
  };

  return (
    <div
      className="h-full flex flex-col overflow-hidden"
      style={{ background: '#0D1210' }}
    >
      {/* Header */}
      <div
        className="flex-none px-4 py-3 flex items-start justify-between"
        style={{ borderBottom: '1px solid rgba(0,255,156,0.15)' }}
      >
        <div>
          <div
            className="text-sm font-bold tracking-wide"
            style={{ fontFamily: 'var(--font-orbitron), Orbitron, sans-serif', color: '#00FF9C' }}
          >
            {displayName}
          </div>
          <div
            className="text-[10px] mt-0.5"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.35)' }}
          >
            {entity.entity_id.slice(0, 16)}
          </div>
        </div>
        <button
          onClick={onClose}
          className="text-base transition-colors"
          style={{ color: 'rgba(200,230,201,0.4)', background: 'none', border: 'none', cursor: 'pointer' }}
          onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.color = '#FF3B3B')}
          onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.color = 'rgba(200,230,201,0.4)')}
        >
          ✕
        </button>
      </div>

      {/* ASSIGN — primary action */}
      <div
        className="flex-none px-4 py-3"
        style={{ borderBottom: '1px solid rgba(0,255,156,0.15)' }}
      >
        <button
          onClick={handleDispatch}
          disabled={dispatched}
          className="w-full py-3 text-sm font-bold tracking-widest transition-all"
          style={{
            fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
            color: dispatched ? '#080C0A' : '#080C0A',
            background: dispatched ? '#00CC74' : '#00FF9C',
            border: 'none',
            cursor: dispatched ? 'default' : 'pointer',
            letterSpacing: '0.2em',
          }}
          onMouseEnter={(e) => {
            if (!dispatched) (e.currentTarget as HTMLButtonElement).style.background = '#00CC74';
          }}
          onMouseLeave={(e) => {
            if (!dispatched) (e.currentTarget as HTMLButtonElement).style.background = '#00FF9C';
          }}
        >
          {dispatched ? 'ASSIGNED' : 'ASSIGN TASK'}
        </button>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto px-4 pb-4">
        {/* Status section */}
        <SectionHeader title="STATUS" />
        <DataRow label="TYPE" value={entity.entity_type.toUpperCase()} valueColor={typeColor} />
        <DataRow label="DOMAIN" value={entity.domain.toUpperCase()} />
        <DataRow label="CLASS" value={entity.classification || '—'} />
        {entity.track_state && (
          <DataRow label="TRACK" value={entity.track_state.toUpperCase()} />
        )}

        {/* Position section */}
        <SectionHeader title="POSITION" />
        <DataRow label="LAT" value={entity.position.lat.toFixed(6)} />
        <DataRow label="LON" value={entity.position.lon.toFixed(6)} />
        <DataRow label="ALT" value={`${entity.position.alt.toFixed(0)} m`} />
        <DataRow label="HDG" value={`${entity.position.heading_deg.toFixed(1)}°`} />
        <DataRow label="SPD" value={`${entity.speed_mps.toFixed(1)} m/s`} />

        {/* Battery section */}
        {entity.battery_pct !== undefined && (
          <>
            <SectionHeader title="BATTERY" />
            <div className="mb-1">
              <div className="flex items-center justify-between mb-1">
                <span
                  className="text-[10px]"
                  style={{ color: 'rgba(200,230,201,0.45)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
                >
                  CHARGE
                </span>
                <span
                  className="text-[11px] font-bold"
                  style={{ color: batteryColor(entity.battery_pct), fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
                >
                  {Math.round(entity.battery_pct)}%
                </span>
              </div>
              {/* Battery bar */}
              <div
                className="h-1.5 w-full rounded-full overflow-hidden"
                style={{ background: 'rgba(0,255,156,0.1)' }}
              >
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${entity.battery_pct}%`,
                    background: batteryColor(entity.battery_pct),
                  }}
                />
              </div>
            </div>
          </>
        )}

        {/* Confidence section */}
        <SectionHeader title="CONFIDENCE" />
        <div className="mb-1">
          <div className="flex items-center justify-between mb-1">
            <span
              className="text-[10px]"
              style={{ color: 'rgba(200,230,201,0.45)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
            >
              SCORE
            </span>
            <span
              className="text-[11px] font-bold"
              style={{ color: '#00FF9C', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
            >
              {Math.round(entity.confidence * 100)}%
            </span>
          </div>
          <div
            className="h-1.5 w-full rounded-full overflow-hidden"
            style={{ background: 'rgba(0,255,156,0.1)' }}
          >
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{ width: `${entity.confidence * 100}%`, background: '#00FF9C' }}
            />
          </div>
        </div>

        {/* Meta section */}
        <SectionHeader title="META" />
        <DataRow label="LAST SEEN" value={ageString(entity.last_seen)} />
        <DataRow
          label="MISSION"
          value={entity.mission_id ? entity.mission_id.slice(0, 12) : 'UNASSIGNED'}
          valueColor={entity.mission_id ? '#4FC3F7' : 'rgba(200,230,201,0.35)'}
        />
        {entity.source_sensors && entity.source_sensors.length > 0 && (
          <DataRow label="SENSORS" value={entity.source_sensors.join(', ')} />
        )}

        {/* Brain Reasoning */}
        <SectionHeader title="BRAIN REASONING" />
        <div
          className="flex flex-col gap-1 mb-1"
          style={{
            background: 'rgba(0,255,156,0.02)',
            border: '1px solid rgba(0,255,156,0.08)',
            padding: '8px',
          }}
        >
          {thoughts.length === 0 ? (
            <span style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.25)', fontSize: 9 }}>
              No reasoning available
            </span>
          ) : (
            thoughts.map((t, i) => (
              <div key={i} className="flex flex-col gap-0.5">
                <div className="flex items-center justify-between">
                  <span style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.3)', fontSize: 8 }}>
                    {t.ts}
                  </span>
                  <span style={{
                    fontFamily: 'var(--font-ibm-plex-mono), monospace',
                    fontSize: 8,
                    color: t.confidence > 0.9 ? '#00FF9C' : t.confidence > 0.8 ? '#FFB300' : '#FF3B3B',
                  }}>
                    {Math.round(t.confidence * 100)}%
                  </span>
                </div>
                <span style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.65)', fontSize: 9, lineHeight: 1.4 }}>
                  {t.msg}
                </span>
              </div>
            ))
          )}
        </div>

        {/* Manual Overrides */}
        <SectionHeader title="MANUAL OVERRIDES" />
        {overrideStatus && (
          <div style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            fontSize: 9,
            color: '#00FF9C',
            border: '1px solid rgba(0,255,156,0.2)',
            padding: '4px 8px',
            marginBottom: 8,
          }}>
            ✓ {overrideStatus}
          </div>
        )}
        <div className="flex flex-col gap-2 mt-1">
          <button
            className="w-full text-[10px] py-2 tracking-widest transition-colors"
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: '#FF3B3B',
              border: '1px solid rgba(255,59,59,0.4)',
              background: 'transparent',
              cursor: 'pointer',
            }}
            onClick={() => {
              fetch(`${API}/agents`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mission_objective: `HALT ${entity.entity_id}`, entity_id: entity.entity_id, command: 'halt' }) }).catch(() => {});
              setOverrideStatus(`HALT sent to ${entity.callsign || entity.entity_id.slice(0,8)}`);
            }}
            onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,59,59,0.08)')}
            onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.background = 'transparent')}
          >
            HALT
          </button>
          <button
            className="w-full text-[10px] py-2 tracking-widest transition-colors"
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: '#FFB300',
              border: '1px solid rgba(255,179,0,0.4)',
              background: 'transparent',
              cursor: 'pointer',
            }}
            onClick={() => {
              fetch(`${API}/agents`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mission_objective: `Return ${entity.entity_id} to base`, entity_id: entity.entity_id, command: 'rtb' }) }).catch(() => {});
              setOverrideStatus(`RTB sent to ${entity.callsign || entity.entity_id.slice(0,8)}`);
            }}
            onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.background = 'rgba(255,179,0,0.08)')}
            onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.background = 'transparent')}
          >
            RETURN TO BASE
          </button>
          <button
            className="w-full text-[10px] py-2 tracking-widest transition-colors"
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: '#4FC3F7',
              border: '1px solid rgba(79,195,247,0.4)',
              background: 'transparent',
              cursor: 'pointer',
            }}
            onClick={() => {
              fetch(`${API}/agents`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mission_objective: `Activate camera on ${entity.entity_id}`, entity_id: entity.entity_id, command: 'activate_camera' }) }).catch(() => {});
              setOverrideStatus(`Camera activated on ${entity.callsign || entity.entity_id.slice(0,8)}`);
            }}
            onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.background = 'rgba(79,195,247,0.08)')}
            onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.background = 'transparent')}
          >
            ACTIVATE CAMERA
          </button>
        </div>
      </div>
    </div>
  );
}
