'use client';

import React from 'react';
import PanelHeader from '@/components/ui/PanelHeader';
import SectionHeader from '@/components/ui/SectionHeader';

// ── Layer definitions ─────────────────────────────────────────────────────────

interface LayerDef {
  id: string;
  label: string;
  description: string;
  group: 'tactical' | 'satellite' | 'airspace';
}

const LAYERS: LayerDef[] = [
  // Tactical overlays
  { id: 'entities',   label: 'ASSETS',              description: 'Tracked entities + UAV fleet',         group: 'tactical' },
  { id: 'tracks',     label: 'FUSED TRACKS',         description: 'Kalman-filtered multi-sensor tracks',  group: 'tactical' },
  { id: 'geofences',  label: 'GEOFENCES',            description: 'Active exclusion / inclusion zones',   group: 'tactical' },
  { id: 'satellites', label: 'ORBITAL ASSETS',       description: 'Live satellite positions (TLE)',        group: 'tactical' },
  { id: 'gpsjam',     label: 'GPS JAMMING',          description: 'ELINT-derived GPS interference zones', group: 'tactical' },
  { id: 'maritime',   label: 'MARITIME (AIS)',        description: 'AIS vessel tracks',                    group: 'tactical' },
  { id: 'fires',      label: 'FIRE HOTSPOTS',        description: 'NASA FIRMS VIIRS 375 m — 24 h',        group: 'tactical' },
  { id: 'grid',       label: 'COORD GRID',           description: 'Lat/lon reference grid',               group: 'tactical' },

  // Free satellite imagery layers (NASA GIBS — no key)
  { id: 'gibs-modis',      label: 'MODIS TRUE-COLOR',    description: 'NASA MODIS Terra — daily 250 m optical',  group: 'satellite' },
  { id: 'gibs-viirs',      label: 'VIIRS TRUE-COLOR',    description: 'NASA VIIRS NOAA-20 — daily 375 m optical', group: 'satellite' },
  { id: 'gibs-fires',      label: 'MODIS FIRE THERMAL',  description: 'MODIS Terra thermal anomalies — daily',    group: 'satellite' },
  { id: 'gibs-nightlights',label: 'VIIRS NIGHT LIGHTS',  description: 'VIIRS nighttime lights — monthly composite', group: 'satellite' },

  // Airspace
  { id: 'noflyzones', label: 'AIRSPACE CLOSURES',    description: 'NOTAM-sourced airspace restrictions',  group: 'airspace' },
];

const GROUP_LABELS: Record<string, string> = {
  tactical:  'TACTICAL OVERLAYS',
  satellite: 'FREE SATELLITE (NASA GIBS)',
  airspace:  'AIRSPACE',
};

interface OpsMapLayersProps {
  onDrawGeofence?: () => void;
  activeLayers?: Set<string>;
  onToggleLayer?: (id: string) => void;
}

export default function OpsMapLayers({ onDrawGeofence, activeLayers, onToggleLayer }: OpsMapLayersProps) {
  const isEnabled = (id: string) => activeLayers?.has(id) ?? false;

  const groups = ['tactical', 'satellite', 'airspace'] as const;

  return (
    <div className="flex flex-col h-full panel-scanline">
      <PanelHeader title="MAP LAYERS" />
      <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-1">

        {groups.map(group => (
          <div key={group}>
            <SectionHeader title={GROUP_LABELS[group]} />
            {LAYERS.filter(l => l.group === group).map(layer => {
              const on = isEnabled(layer.id);
              return (
                <button
                  key={layer.id}
                  onClick={() => onToggleLayer?.(layer.id)}
                  aria-pressed={on}
                  aria-label={`Toggle ${layer.label} layer`}
                  title={layer.description}
                  className="summit-btn flex items-center gap-3 text-left w-full mb-1"
                  style={{
                    background:  on ? 'var(--accent-5)' : 'transparent',
                    border:      'none',
                    padding:     '7px 12px',
                    borderLeft:  `2px solid ${on ? 'var(--accent)' : 'var(--accent-15)'}`,
                  }}
                >
                  <div
                    aria-hidden="true"
                    style={{
                      width: 11, height: 11, flexShrink: 0,
                      border:     `1px solid ${on ? 'var(--accent)' : 'var(--accent-30)'}`,
                      background: on ? 'var(--accent)' : 'transparent',
                    }}
                  />
                  <div className="flex flex-col gap-0">
                    <span className="text-[10px] tracking-wider"
                      style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace',
                        color: on ? 'var(--accent)' : 'var(--text-dim)' }}>
                      {layer.label}
                    </span>
                    <span className="text-[8px]"
                      style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace',
                        color: 'var(--text-muted)', lineHeight: 1.4 }}>
                      {layer.description}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        ))}

        {/* Geofence draw action */}
        <div style={{ marginTop: '8px', borderTop: '1px solid var(--accent-5)', paddingTop: '12px' }}>
          <button
            onClick={onDrawGeofence}
            aria-label="Draw geofence on map"
            className="summit-btn w-full text-left flex items-center gap-3"
            style={{
              padding:    '8px 12px',
              background: 'transparent',
              border:     '1px solid color-mix(in srgb, var(--warning) 30%, transparent)',
              borderLeft: '2px solid var(--warning)',
            }}
          >
            <span aria-hidden="true" style={{ fontSize: '12px', color: 'var(--warning)' }}>⬡</span>
            <span className="text-xs tracking-wider"
              style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--warning)' }}>
              DRAW GEOFENCE
            </span>
          </button>
        </div>
      </div>
    </div>
  );
}
