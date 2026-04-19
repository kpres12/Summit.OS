'use client';

/**
 * DomainProvider — locked to Heli.OS baseline config.
 *
 * Domain packs (fire, pipeline, SAR, etc.) are available in lib/domains/
 * for Forward Deployed Engineers to configure per customer deployment.
 * The picker has been removed from the UI — one codebase, one brand.
 */

import React, { createContext, useContext, useEffect } from 'react';
import type { DomainConfig } from '@/lib/domains/types';

const DEFAULT_CONFIG: DomainConfig = {
  id: 'default',
  name: 'Heli.OS',
  description: 'Autonomous systems coordination platform',
  palette: {
    accent: '#00FF9C',
    accentDim: '#00CC74',
    accentDark: '#009862',
    warning: '#FFB300',
    critical: '#FF3B3B',
    nominal: '#4AEDC4',
    active: '#4FC3F7',
    backgroundTint: '#080C0A',
    panelBg: '#0D1210',
    border: 'rgba(0,255,156,0.15)',
    scanline: 'rgba(0,255,156,0.03)',
  },
  entityLabels: {
    friendly: { displayName: 'Asset',   icon: '●', color: '#4AEDC4' },
    alert:    { displayName: 'Alert',   icon: '▲', color: '#FF3B3B' },
    neutral:  { displayName: 'Tracked', icon: '◆', color: '#a1a1aa' },
    unknown:  { displayName: 'Unknown', icon: '?', color: '#FFB300' },
  },
  assetTypes: [
    { type: 'DRONE',  label: 'DRONE',  icon: '○' },
    { type: 'UGV',    label: 'UGV',    icon: '○' },
    { type: 'TOWER',  label: 'TOWER',  icon: '○' },
    { type: 'SENSOR', label: 'SENSOR', icon: '○' },
  ],
  mapLayers: [
    { id: 'entities',  name: 'Entities',  enabled: true,  color: '#4AEDC4', icon: '●' },
    { id: 'geofences', name: 'Geofences', enabled: true,  color: '#FFB300', icon: '⬢' },
    { id: 'tracks',    name: 'Tracks',    enabled: false, color: '#60a5fa', icon: '〰' },
    { id: 'orbits',    name: 'Coverage',  enabled: true,  color: '#818cf8', icon: '◌' },
  ],
  commandExamples: [
    'patrol sector 4',
    'return all assets',
    'survey grid alpha',
    'status drone-01',
    'deploy sensor array bravo',
  ],
  alertTypes: [
    { id: 'detection', label: 'Detection',    icon: '◉', color: '#FFB300' },
    { id: 'anomaly',   label: 'Anomaly',      icon: '▲', color: '#FF3B3B' },
    { id: 'status',    label: 'Status Change', icon: '●', color: '#4AEDC4' },
  ],
  missionTemplates: [
    { id: 'survey',  label: 'Survey Area',       intent: 'survey',  description: 'Grid coverage of a target area' },
    { id: 'patrol',  label: 'Patrol Perimeter',  intent: 'monitor', description: 'Continuous perimeter patrol' },
    { id: 'observe', label: 'Observe Point',     intent: 'observe', description: 'Loiter and observe a location' },
  ],
  terminology: {
    mission:       'Mission',
    asset:         'Asset',
    alert:         'Alert',
    entity:        'Entity',
    operatorView:  'Operator',
    supervisorView: 'Supervisor',
  },
};

interface DomainContextValue {
  config: DomainConfig;
  setDomain: (id: string) => void;
  domains: DomainConfig[];
  loading: boolean;
}

const DomainContext = createContext<DomainContextValue>({
  config:    DEFAULT_CONFIG,
  setDomain: () => {},
  domains:   [DEFAULT_CONFIG],
  loading:   false,
});

export function useDomain() {
  return useContext(DomainContext);
}

function applyPalette(palette: DomainConfig['palette']) {
  const root = document.documentElement;
  root.style.setProperty('--accent',           palette.accent);
  root.style.setProperty('--accent-dim',       palette.accentDim);
  root.style.setProperty('--accent-dark',      palette.accentDark);
  root.style.setProperty('--warning',          palette.warning);
  root.style.setProperty('--critical',         palette.critical);
  root.style.setProperty('--nominal',          palette.nominal);
  root.style.setProperty('--color-active',     palette.active);
  root.style.setProperty('--background',       palette.backgroundTint);
  root.style.setProperty('--background-panel', palette.panelBg);
  root.style.setProperty('--border',           palette.border);
  root.style.setProperty('--scanline',         palette.scanline);
  root.style.setProperty('--accent-5',  `${palette.accent}0D`);
  root.style.setProperty('--accent-10', `${palette.accent}1A`);
  root.style.setProperty('--accent-15', `${palette.accent}26`);
  root.style.setProperty('--accent-30', `${palette.accent}4D`);
  root.style.setProperty('--accent-50', `${palette.accent}80`);
}

export default function DomainProvider({ children }: { children: React.ReactNode }) {
  useEffect(() => { applyPalette(DEFAULT_CONFIG.palette); }, []);

  return (
    <DomainContext.Provider value={{
      config:    DEFAULT_CONFIG,
      setDomain: () => {},
      domains:   [DEFAULT_CONFIG],
      loading:   false,
    }}>
      {children}
    </DomainContext.Provider>
  );
}
