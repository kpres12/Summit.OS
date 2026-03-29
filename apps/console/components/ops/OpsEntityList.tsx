'use client';

import React from 'react';
import { useEntityStream, EntityData } from '@/hooks/useEntityStream';
import PanelHeader from '@/components/ui/PanelHeader';
import EmptyState from '@/components/ui/EmptyState';
import { ageTerse, entityTypeColor, batteryColor, domainTag } from '@/lib/format';

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
      <PanelHeader title="ENTITY LIST" count={entityCount} />

      {/* Entity list */}
      <div className="flex-1 overflow-y-auto">
        {entityList.length === 0 && (
          <EmptyState message="AWAITING CONNECTIONS" hint="pip install summit-os-sdk" />
        )}

        {Object.entries(grouped).map(([domain, entities]) => (
          <div key={domain}>
            {/* Domain header */}
            <div
              className="px-3 py-1 text-[9px] tracking-widest uppercase"
              style={{
                fontFamily: 'var(--font-orbitron), Orbitron, sans-serif',
                color: 'var(--accent-30)',
                background: 'var(--accent-5)',
                borderBottom: '1px solid var(--accent-5)',
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
      className="summit-btn px-3 py-2 flex flex-col gap-1"
      style={{ borderBottom: '1px solid var(--accent-5)' }}
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
              color: 'var(--text-dim)',
              border: '1px solid var(--accent-10)',
            }}
          >
            {domainTag(entity.domain)}
          </span>
          <span
            className="text-[9px]"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-muted)' }}
          >
            {ageTerse(entity.last_seen)}
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
