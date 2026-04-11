'use client';

import React from 'react';
import { useEntityStream, EntityData } from '@/hooks/useEntityStream';
import PanelHeader from '@/components/ui/PanelHeader';
import { ageTerse, entityTypeColor, batteryColor, domainTag } from '@/lib/format';

// How long before we consider data stale
const STALE_WARN_S = 60;   // amber
const STALE_DEAD_S = 300;  // grey out, stop showing on map

function staleness(lastSeen: number): 'live' | 'warn' | 'stale' {
  const age = (Date.now() / 1000) - lastSeen;
  if (age > STALE_DEAD_S) return 'stale';
  if (age > STALE_WARN_S) return 'warn';
  return 'live';
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
    <div className="flex flex-col h-full panel-scanline">
      <PanelHeader title="ASSETS" count={entityCount} />

      <div className="flex-1 overflow-y-auto">
        {entityList.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-3 px-6">
            <div style={{ fontSize: '24px', opacity: 0.15 }} aria-hidden="true">◉</div>
            <span
              className="text-[10px] tracking-widest"
              style={{ color: 'var(--accent-30)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
            >
              NO ASSETS ONLINE
            </span>
            <span
              className="text-[9px] text-center leading-relaxed"
              style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
            >
              Connect adapters in Hardware ›
            </span>
          </div>
        )}

        {Object.entries(grouped).map(([domain, entities]) => (
          <div key={domain}>
            <div
              className="px-4 py-2 text-[10px] tracking-[0.15em] uppercase"
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color: 'var(--text-dim)',
                background: 'var(--accent-5)',
                borderBottom: '1px solid var(--accent-10)',
              }}
            >
              {domain} <span style={{ color: 'var(--text-muted)' }}>({entities.length})</span>
            </div>
            {entities.map((e) => (
              <AssetRow key={e.entity_id} entity={e} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function AssetRow({ entity }: { entity: EntityData }) {
  const color = entityTypeColor(entity.entity_type);
  const callsign = entity.callsign || entity.entity_id.slice(0, 8);
  const age = staleness(entity.last_seen);

  const dotColor = age === 'stale' ? 'var(--text-muted)' : age === 'warn' ? 'var(--warning)' : color;

  return (
    <div
      className="summit-btn px-4 py-3 flex flex-col gap-1.5"
      style={{
        borderBottom: '1px solid var(--accent-5)',
      }}
    >
      {/* Primary row: dot + callsign + age */}
      <div className="flex items-center gap-2">
        <div
          className="w-2 h-2 rounded-full flex-none"
          style={{ background: dotColor, boxShadow: age === 'live' ? `0 0 4px ${dotColor}` : 'none' }}
        />
        <span
          className="flex-1 text-[13px] font-bold truncate"
          style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: age === 'stale' ? 'var(--text-muted)' : color }}
        >
          {callsign}
        </span>
        <span
          className="text-[10px]"
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: age === 'warn' ? 'var(--warning)' : age === 'stale' ? 'var(--critical)' : 'var(--text-muted)',
          }}
        >
          {ageTerse(entity.last_seen)}
        </span>
      </div>

      {/* Secondary row: domain tag + battery + speed */}
      <div className="flex items-center gap-2 pl-4">
        <span
          className="text-[10px] px-1.5 py-0.5"
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: 'var(--text-dim)',
            border: '1px solid var(--accent-10)',
          }}
        >
          {domainTag(entity.domain)}
        </span>
        {age === 'warn' && (
          <span className="text-[9px] tracking-widest px-1" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--warning)', border: '1px solid color-mix(in srgb, var(--warning) 30%, transparent)' }}>
            STALE
          </span>
        )}
        {age === 'stale' && (
          <span className="text-[9px] tracking-widest px-1" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--critical)', border: '1px solid color-mix(in srgb, var(--critical) 30%, transparent)' }}>
            NO SIGNAL
          </span>
        )}
        {age !== 'stale' && entity.battery_pct != null && (
          <span className="text-[10px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: batteryColor(entity.battery_pct) }}>
            BAT {Math.round(entity.battery_pct)}%
          </span>
        )}
        {age !== 'stale' && entity.speed_mps != null && entity.speed_mps > 0.5 && (
          <span className="text-[10px]" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-dim)' }}>
            {entity.speed_mps.toFixed(1)} m/s
          </span>
        )}
      </div>
    </div>
  );
}
