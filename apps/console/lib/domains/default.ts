import type { DomainConfig } from './types';

const defaultDomain: DomainConfig = {
  id: 'default',
  name: 'Summit.OS',
  description: 'General-purpose operations platform',
  palette: {
    accent: '#4AEDC4',
    accentDim: '#2BA88A',
    accentDark: '#1A6B58',
    warning: '#FFB300',
    critical: '#FF3B3B',
    nominal: '#4AEDC4',
    active: '#4FC3F7',
    backgroundTint: '#080C0A',
    panelBg: '#0D1210',
    border: 'rgba(74, 237, 196, 0.15)',
    scanline: 'rgba(74, 237, 196, 0.03)',
  },
  entityLabels: {
    friendly: { displayName: 'Asset', icon: '●', color: '#4AEDC4' },
    alert: { displayName: 'Hazard', icon: '▲', color: '#FF3B3B' },
    neutral: { displayName: 'Tracked', icon: '◆', color: '#a1a1aa' },
    unknown: { displayName: 'Unknown', icon: '?', color: '#FFB300' },
  },
  assetTypes: [
    { type: 'DRONE', label: 'DRONE', icon: '○' },
    { type: 'UGV', label: 'UGV', icon: '○' },
    { type: 'TOWER', label: 'TOWER', icon: '○' },
    { type: 'SENSOR', label: 'SENSOR', icon: '○' },
  ],
  mapLayers: [
    { id: 'entities', name: 'Entities', enabled: true, color: '#4AEDC4', icon: '●' },
    { id: 'geofences', name: 'Geofences', enabled: true, color: '#FFB300', icon: '⬢' },
    { id: 'tracks', name: 'Tracks', enabled: false, color: '#60a5fa', icon: '〰' },
    { id: 'orbits', name: 'Coverage', enabled: true, color: '#818cf8', icon: '◌' },
  ],
  commandExamples: [
    'patrol sector 4',
    'return all assets',
    'survey grid alpha',
    'status drone-01',
    'deploy sensor array bravo',
  ],
  alertTypes: [
    { id: 'detection', label: 'Detection', icon: '◉', color: '#FFB300' },
    { id: 'anomaly', label: 'Anomaly', icon: '▲', color: '#FF3B3B' },
    { id: 'status', label: 'Status Change', icon: '●', color: '#4AEDC4' },
  ],
  missionTemplates: [
    { id: 'survey', label: 'Survey Area', intent: 'survey', description: 'Grid coverage of a target area' },
    { id: 'patrol', label: 'Patrol Perimeter', intent: 'monitor', description: 'Continuous perimeter patrol' },
    { id: 'observe', label: 'Observe Point', intent: 'observe', description: 'Loiter and observe a location' },
  ],
  terminology: {
    mission: 'Mission',
    asset: 'Asset',
    alert: 'Alert',
    entity: 'Entity',
    operatorView: 'Operator',
    supervisorView: 'Supervisor',
  },
};

export default defaultDomain;
