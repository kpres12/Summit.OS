'use client';

import React, { useState, useCallback } from 'react';
import { EntityData, useEntityStream } from '@/hooks/useEntityStream';
import { AlertAPI } from '@/lib/api';
import ErrorBoundary from '@/components/ErrorBoundary';
import OpsTopBar from './OpsTopBar';
import OpsNavRail from './OpsNavRail';
import OpsAlertQueue from './OpsAlertQueue';
import OpsEntityList from './OpsEntityList';
import OpsMissions from './OpsMissions';
import OpsEntityDetail from './OpsEntityDetail';
import OpsBottomBar from './OpsBottomBar';
import OpsMapView from './OpsMapView';
import OpsHardware from './OpsHardware';
import OpsVideoPane from './OpsVideoPane';
import OpsReplayControls from './OpsReplayControls';
import { useReplay } from '@/hooks/useReplay';

type PanelId = 'alerts' | 'entities' | 'missions' | 'layers' | 'hardware' | 'system';

interface OpsLayoutProps {
  onSwitchRole: () => void;
}

// Inline layer toggles panel
function OpsMapLayers() {
  const [layers, setLayers] = useState([
    { id: 'entities', label: 'ENTITIES', enabled: true },
    { id: 'tracks', label: 'TRACKS', enabled: true },
    { id: 'geofences', label: 'GEOFENCES', enabled: false },
    { id: 'grid', label: 'GRID OVERLAY', enabled: false },
    { id: 'threat', label: 'ALERT ZONES', enabled: false },
  ]);

  const toggle = (id: string) => {
    setLayers((prev) => prev.map((l) => l.id === id ? { ...l, enabled: !l.enabled } : l));
  };

  return (
    <div className="flex flex-col h-full">
      <div
        className="flex-none px-3 py-2"
        style={{ borderBottom: '1px solid rgba(0,255,156,0.15)' }}
      >
        <span
          className="text-xs font-bold tracking-widest"
          style={{ fontFamily: 'var(--font-orbitron), Orbitron, sans-serif', color: '#00FF9C' }}
        >
          MAP LAYERS
        </span>
      </div>
      <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-2">
        {layers.map((l) => (
          <button
            key={l.id}
            onClick={() => toggle(l.id)}
            className="flex items-center gap-3 text-left transition-colors"
            style={{
              background: l.enabled ? 'rgba(0,255,156,0.05)' : 'transparent',
              border: 'none',
              cursor: 'pointer',
              padding: '8px 12px',
              borderLeft: `2px solid ${l.enabled ? '#00FF9C' : 'rgba(0,255,156,0.2)'}`,
            }}
          >
            <div
              className="w-3 h-3 border flex-none"
              style={{
                border: `1px solid ${l.enabled ? '#00FF9C' : 'rgba(0,255,156,0.3)'}`,
                background: l.enabled ? '#00FF9C' : 'transparent',
              }}
            />
            <span
              className="text-xs tracking-wider"
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color: l.enabled ? '#00FF9C' : 'rgba(200,230,201,0.45)',
              }}
            >
              {l.label}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

// Inline system info panel
function OpsSystem() {
  return (
    <div className="flex flex-col h-full">
      <div
        className="flex-none px-3 py-2"
        style={{ borderBottom: '1px solid rgba(0,255,156,0.15)' }}
      >
        <span
          className="text-xs font-bold tracking-widest"
          style={{ fontFamily: 'var(--font-orbitron), Orbitron, sans-serif', color: '#00FF9C' }}
        >
          SYSTEM
        </span>
      </div>
      <div className="flex-1 overflow-y-auto p-3">
        {[
          { label: 'NODE', value: 'CONSOLE-01' },
          { label: 'VERSION', value: '1.0.0' },
          { label: 'ENV', value: process.env.NODE_ENV?.toUpperCase() || 'PRODUCTION' },
          { label: 'API', value: process.env.NEXT_PUBLIC_API_URL || 'localhost:8000' },
          { label: 'WS', value: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8001' },
        ].map((row) => (
          <div
            key={row.label}
            className="flex items-baseline justify-between py-1.5"
            style={{ borderBottom: '1px solid rgba(0,255,156,0.06)' }}
          >
            <span
              className="text-[10px]"
              style={{ color: 'rgba(200,230,201,0.45)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
            >
              {row.label}
            </span>
            <span
              className="text-[10px] font-bold"
              style={{ color: '#00FF9C', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
            >
              {row.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function OpsLayout({ onSwitchRole }: OpsLayoutProps) {
  const { entityList } = useEntityStream();
  const [activePanel, setActivePanel] = useState<PanelId | null>(null);
  const [selectedEntity, setSelectedEntity] = useState<EntityData | null>(null);
  const [flyToLocation, setFlyToLocation] = useState<{ lat: number; lon: number } | null>(null);
  const [alertEntityIds, setAlertEntityIds] = useState<Set<string>>(new Set());
  // Live video overlay (Gap 4)
  const [videoStreamId, setVideoStreamId] = useState<string | null>(null);
  // Mission replay (Gap 5)
  const [replayMissionId, setReplayMissionId] = useState<string | null>(null);
  const replay = useReplay(replayMissionId);

  // Core investigate flow — the one interaction that has to be perfect.
  // Finds the entity matching the alert source, zooms the map, opens entity detail.
  const investigateAlert = useCallback((alert: AlertAPI) => {
    const source = alert.source || '';
    const match = entityList.find((e) =>
      e.callsign === source ||
      e.entity_id === source ||
      e.entity_id.startsWith(source.slice(0, 8)) ||
      source.toLowerCase().includes(e.entity_id.slice(0, 8).toLowerCase()) ||
      source.toLowerCase().includes((e.callsign || '').toLowerCase())
    ) || (entityList.length > 0 ? entityList[0] : null);

    if (match) {
      setSelectedEntity(match);
      setFlyToLocation({ lat: match.position.lat, lon: match.position.lon });
      // Mark this entity for alert pulse on the map
      setAlertEntityIds((prev) => {
        const next = new Set(prev);
        next.add(match.entity_id);
        return next;
      });
      // Clear pulse after animation completes (3 pulses × 600ms = ~1.8s)
      setTimeout(() => {
        setAlertEntityIds((prev) => {
          const next = new Set(prev);
          next.delete(match.entity_id);
          return next;
        });
      }, 2000);
    }
    setActivePanel(null); // close the alert panel
  }, [entityList]);

  // Investigate by callsign — triggered from detection chips in bottom bar
  const investigateEntity = useCallback((callsign: string) => {
    const match = entityList.find((e) =>
      e.callsign === callsign || e.entity_id.startsWith(callsign.slice(0, 8))
    );
    if (match) {
      setSelectedEntity(match);
      setFlyToLocation({ lat: match.position.lat, lon: match.position.lon });
    }
  }, [entityList]);

  const renderPanel = () => {
    switch (activePanel) {
      case 'alerts': return <OpsAlertQueue onInvestigate={investigateAlert} />;
      case 'entities': return <OpsEntityList />;
      case 'missions': return <OpsMissions onReplay={(id) => setReplayMissionId(id)} />;
      case 'layers': return <OpsMapLayers />;
      case 'hardware': return <OpsHardware />;
      case 'system': return <OpsSystem />;
      default: return null;
    }
  };

  return (
    <ErrorBoundary>
    <div
      className="fixed inset-0 flex flex-col"
      style={{ background: '#080C0A' }}
    >
      {/* Top bar */}
      <OpsTopBar onSwitchRole={onSwitchRole} />

      {/* Middle row */}
      <div className="flex flex-row flex-1 overflow-hidden">
        {/* Nav rail */}
        <OpsNavRail
          activePanel={activePanel}
          onSelect={(p) => setActivePanel(p as PanelId | null)}
        />

        {/* Sliding side panel */}
        <div
          className="flex-none overflow-hidden transition-[width] duration-150 ease-out"
          style={{
            width: activePanel ? '320px' : '0px',
            borderRight: activePanel ? '1px solid rgba(0,255,156,0.15)' : 'none',
            background: '#0D1210',
          }}
        >
          <div style={{ width: '320px', height: '100%' }}>
            {renderPanel()}
          </div>
        </div>

        {/* Map area */}
        <div className="flex-1 relative overflow-hidden">
          <OpsMapView
            onSelectEntity={setSelectedEntity}
            flyToLocation={flyToLocation}
            alertEntityIds={alertEntityIds}
          />
          {/* Live video overlay */}
          {videoStreamId && (
            <OpsVideoPane
              streamId={videoStreamId}
              onClose={() => setVideoStreamId(null)}
            />
          )}
          {/* Mission replay controls */}
          {replayMissionId && (
            <div style={{ position: 'absolute', bottom: 64, left: 8, width: 320, zIndex: 200 }}>
              <OpsReplayControls
                replay={replay}
                onClose={() => setReplayMissionId(null)}
              />
            </div>
          )}
        </div>

        {/* Entity detail panel */}
        <div
          className="flex-none overflow-hidden transition-[width] duration-150 ease-out"
          style={{
            width: selectedEntity ? '384px' : '0px',
            borderLeft: selectedEntity ? '1px solid rgba(0,255,156,0.15)' : 'none',
            background: '#0D1210',
          }}
        >
          <div style={{ width: '384px', height: '100%' }}>
            <OpsEntityDetail
              entity={selectedEntity}
              onClose={() => setSelectedEntity(null)}
              onDispatch={() => setSelectedEntity(null)}
              onLiveFeed={(streamId) => setVideoStreamId(streamId)}
            />
          </div>
        </div>
      </div>

      {/* Bottom bar */}
      <OpsBottomBar onInvestigateEntity={investigateEntity} />
    </div>
    </ErrorBoundary>
  );
}
