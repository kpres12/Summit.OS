'use client';

import React from 'react';
import { useEntityStream, EntityData } from '@/hooks/useEntityStream';

function ageString(lastSeen: number): string {
  const diff = Math.floor((Date.now() / 1000) - lastSeen);
  if (diff < 60) return `${diff}s`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m`;
  return `${Math.floor(diff / 3600)}h`;
}

function domainTag(domain: string): string {
  switch (domain) {
    case 'aerial': return 'UAV';
    case 'ground': return 'GND';
    case 'maritime': return 'MAR';
    case 'fixed': return 'FIX';
    case 'sensor': return 'SEN';
    default: return domain.slice(0, 3).toUpperCase();
  }
}

function entityTypeColor(type: string): string {
  switch (type) {
    case 'friendly': return '#00FF9C';
    case 'hostile': return '#FF3B3B';
    case 'neutral': return 'rgba(200,230,201,0.45)';
    default: return '#FFB300';
  }
}

function batteryColor(pct: number): string {
  if (pct > 40) return '#00FF9C';
  if (pct > 20) return '#FFB300';
  return '#FF3B3B';
}

export default function OpsEntityList() {
  const { entityList, entityCount } = useEntityStream();

  // Group by domain
  const grouped = entityList.reduce<Record<string, EntityData[]>>((acc, e) => {
    const key = e.domain || 'unknown';
    if (!acc[key]) acc[key] = [];
    acc[key].push(e);
    return acc;
  }, {});

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
          ENTITY LIST
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
          {entityCount}
        </span>
      </div>

      {/* Entity list */}
      <div className="flex-1 overflow-y-auto">
        {entityList.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-3 px-4">
            <div
              className="text-[10px] tracking-widest text-center"
              style={{ color: 'rgba(200,230,201,0.35)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
            >
              AWAITING CONNECTIONS
            </div>
            <div
              className="text-[9px] text-center leading-relaxed"
              style={{ color: 'rgba(200,230,201,0.2)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
            >
              pip install summit-os-sdk
            </div>
          </div>
        )}

        {Object.entries(grouped).map(([domain, entities]) => (
          <div key={domain}>
            {/* Domain header */}
            <div
              className="px-3 py-1 text-[9px] tracking-widest uppercase"
              style={{
                fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
                color: 'rgba(0,255,156,0.4)',
                background: 'rgba(0,255,156,0.03)',
                borderBottom: '1px solid rgba(0,255,156,0.08)',
              }}
            >
              {domain} ({entities.length})
            </div>

            {entities.map((e) => (
              <EntityRow key={e.entity_id} entity={e} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function EntityRow({ entity }: { entity: EntityData }) {
  const color = entityTypeColor(entity.entity_type);
  const callsign = entity.callsign || entity.entity_id.slice(0, 8);

  return (
    <div
      className="px-3 py-2 flex flex-col gap-1 transition-colors"
      style={{ borderBottom: '1px solid rgba(0,255,156,0.05)', cursor: 'pointer' }}
      onMouseEnter={(e) => ((e.currentTarget as HTMLDivElement).style.background = 'rgba(0,255,156,0.04)')}
      onMouseLeave={(e) => ((e.currentTarget as HTMLDivElement).style.background = 'transparent')}
    >
      {/* Row 1: dot + callsign + domain + age */}
      <div className="flex items-center gap-2">
        <div
          className="w-1.5 h-1.5 rounded-full flex-none"
          style={{ background: color }}
        />
        <span
          className="flex-1 text-[11px] font-bold truncate"
          style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color }}
        >
          {callsign}
        </span>
        <span
          className="text-[9px] px-1"
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: 'rgba(200,230,201,0.4)',
            border: '1px solid rgba(0,255,156,0.1)',
          }}
        >
          {domainTag(entity.domain)}
        </span>
        <span
          className="text-[9px]"
          style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.35)' }}
        >
          {ageString(entity.last_seen)}
        </span>
      </div>

      {/* Row 2: battery + speed */}
      {(entity.battery_pct !== undefined || (entity.speed_mps && entity.speed_mps > 0.5)) && (
        <div className="flex items-center gap-3 pl-4">
          {entity.battery_pct !== undefined && (
            <span
              className="text-[10px]"
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color: batteryColor(entity.battery_pct),
              }}
            >
              BAT {Math.round(entity.battery_pct)}%
            </span>
          )}
          {entity.speed_mps !== undefined && entity.speed_mps > 0.5 && (
            <span
              className="text-[10px]"
              style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(200,230,201,0.45)' }}
            >
              {entity.speed_mps.toFixed(1)} m/s
            </span>
          )}
        </div>
      )}
    </div>
  );
}
