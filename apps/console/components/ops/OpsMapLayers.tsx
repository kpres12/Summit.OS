'use client';

import React, { useState } from 'react';
import PanelHeader from '@/components/ui/PanelHeader';

interface Layer {
  id: string;
  label: string;
  enabled: boolean;
}

interface OpsMapLayersProps {
  onDrawGeofence?: () => void;
}

export default function OpsMapLayers({ onDrawGeofence }: OpsMapLayersProps) {
  const [layers, setLayers] = useState<Layer[]>([
    { id: 'entities', label: 'ASSETS', enabled: true },
    { id: 'tracks', label: 'TRACKS', enabled: true },
    { id: 'geofences', label: 'GEOFENCES', enabled: false },
    { id: 'grid', label: 'GRID OVERLAY', enabled: false },
    { id: 'threat', label: 'ALERT ZONES', enabled: false },
  ]);

  const toggle = (id: string) => {
    setLayers((prev) => prev.map((l) => l.id === id ? { ...l, enabled: !l.enabled } : l));
  };

  return (
    <div className="flex flex-col h-full panel-scanline">
      <PanelHeader title="MAP LAYERS" />
      <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-2">
        {layers.map((l) => (
          <button
            key={l.id}
            onClick={() => toggle(l.id)}
            aria-pressed={l.enabled}
            aria-label={`Toggle ${l.label} layer`}
            className="summit-btn flex items-center gap-3 text-left"
            style={{
              background: l.enabled ? 'var(--accent-5)' : 'transparent',
              border: 'none',
              padding: '8px 12px',
              borderLeft: `2px solid ${l.enabled ? 'var(--accent)' : 'var(--accent-15)'}`,
            }}
          >
            <div
              className="w-3 h-3 border flex-none"
              aria-hidden="true"
              style={{
                border: `1px solid ${l.enabled ? 'var(--accent)' : 'var(--accent-30)'}`,
                background: l.enabled ? 'var(--accent)' : 'transparent',
              }}
            />
            <span
              className="text-xs tracking-wider"
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color: l.enabled ? 'var(--accent)' : 'var(--text-dim)',
              }}
            >
              {l.label}
            </span>
          </button>
        ))}

        {/* Geofence draw — action, not a layer toggle */}
        <div style={{ marginTop: '8px', borderTop: '1px solid var(--accent-5)', paddingTop: '12px' }}>
          <button
            onClick={onDrawGeofence}
            aria-label="Draw geofence on map"
            className="summit-btn w-full text-left flex items-center gap-3"
            style={{
              padding: '8px 12px',
              background: 'transparent',
              border: '1px solid color-mix(in srgb, var(--warning) 30%, transparent)',
              borderLeft: '2px solid var(--warning)',
            }}
          >
            <span aria-hidden="true" style={{ fontSize: '12px', color: 'var(--warning)' }}>⬡</span>
            <span
              className="text-xs tracking-wider"
              style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--warning)' }}
            >
              DRAW GEOFENCE
            </span>
          </button>
        </div>
      </div>
    </div>
  );
}
