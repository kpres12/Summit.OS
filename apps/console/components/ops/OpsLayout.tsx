'use client';

import React, { useState } from 'react';
import { useEntityStream } from '@/hooks/useEntityStream';
import { useInvestigation } from '@/hooks/useInvestigation';
import { useMissionDraw } from '@/hooks/useMissionDraw';
import { useReplay } from '@/hooks/useReplay';
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
import OpsMissionBuilder from './OpsMissionBuilder';
import OpsMapLayers from './OpsMapLayers';
import OpsSystem from './OpsSystem';

type PanelId = 'alerts' | 'entities' | 'missions' | 'layers' | 'hardware' | 'system' | 'mission-builder';

interface OpsLayoutProps {
  onSwitchRole: () => void;
}

export default function OpsLayout({ onSwitchRole }: OpsLayoutProps) {
  const { entityList } = useEntityStream();
  // Default to alert panel — the most urgent thing should be visible immediately.
  // User can close it; after that their preference is respected.
  const [activePanel, setActivePanel] = useState<PanelId | null>('alerts');

  // Investigation flow (alert → zoom → entity detail)
  const investigation = useInvestigation(entityList);

  // Live video overlay
  const [videoStreamId, setVideoStreamId] = useState<string | null>(null);

  // Mission replay
  const [replayMissionId, setReplayMissionId] = useState<string | null>(null);
  const replay = useReplay(replayMissionId);

  // Mission builder draw state
  const missionDraw = useMissionDraw();

  // Geofence draw (triggered from Layers panel)
  const [geofenceDrawMode, setGeofenceDrawMode] = useState(false);

  const handleInvestigateAlert = (alert: Parameters<typeof investigation.investigateAlert>[0]) => {
    investigation.investigateAlert(alert);
    setActivePanel(null); // close the alert panel after investigating
  };

  const renderPanel = () => {
    switch (activePanel) {
      case 'alerts': return <OpsAlertQueue onInvestigate={handleInvestigateAlert} />;
      case 'entities': return <OpsEntityList />;
      case 'missions': return <OpsMissions onReplay={(id) => setReplayMissionId(id)} />;
      case 'layers': return (
        <OpsMapLayers
          onDrawGeofence={() => {
            setGeofenceDrawMode(true);
            setActivePanel(null); // close panel so the map is fully visible for drawing
          }}
        />
      );
      case 'hardware': return <OpsHardware />;
      case 'system': return <OpsSystem />;
      case 'mission-builder': return (
        <OpsMissionBuilder
          onMissionLaunched={() => {
            setActivePanel(null);
            missionDraw.resetMissionDraw();
          }}
          onRequestDrawArea={() => {
            missionDraw.setMissionPolygon(null);
            missionDraw.setMissionDrawMode(true);
          }}
          missionPolygon={missionDraw.missionPolygon}
          onWaypointsChanged={missionDraw.setMissionWaypoints}
        />
      );
      default: return null;
    }
  };

  return (
    <ErrorBoundary>
    <div
      className="fixed inset-0 flex flex-col"
      style={{ background: 'var(--background)' }}
    >
      {/* Top bar — ARIA banner */}
      <OpsTopBar onSwitchRole={onSwitchRole} />

      {/* Middle row */}
      <div className="flex flex-row flex-1 overflow-hidden">
        {/* Nav rail — ARIA navigation */}
        <OpsNavRail
          activePanel={activePanel}
          onSelect={(p) => setActivePanel(p as PanelId | null)}
        />

        {/* Sliding side panel */}
        <aside
          role="complementary"
          aria-label="Side panel"
          className="flex-none overflow-hidden transition-[width] duration-150 ease-out"
          style={{
            width: activePanel ? '320px' : '0px',
            borderRight: activePanel ? '1px solid var(--border)' : 'none',
            background: 'var(--background-panel)',
          }}
        >
          <div style={{ width: '320px', height: '100%' }}>
            {renderPanel()}
          </div>
        </aside>

        {/* Map area — ARIA main */}
        <main role="main" aria-label="Map view" className="flex-1 relative overflow-hidden">
          <OpsMapView
            onSelectEntity={investigation.setSelectedEntity}
            flyToLocation={investigation.flyToLocation}
            alertEntityIds={investigation.alertEntityIds}
            selectedEntityId={investigation.selectedEntity?.entity_id ?? null}
            missionDrawMode={missionDraw.missionDrawMode}
            onMissionArea={(coords) => {
              missionDraw.setMissionPolygon(coords);
              missionDraw.setMissionDrawMode(false);
            }}
            missionWaypoints={missionDraw.missionWaypoints}
            geofenceDrawMode={geofenceDrawMode}
            onGeofenceDrawEnd={() => setGeofenceDrawMode(false)}
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
        </main>

        {/* Entity detail panel */}
        <aside
          role="complementary"
          aria-label="Entity detail"
          className="flex-none overflow-hidden transition-[width] duration-150 ease-out"
          style={{
            width: investigation.selectedEntity ? '384px' : '0px',
            borderLeft: investigation.selectedEntity ? '1px solid var(--border)' : 'none',
            background: 'var(--background-panel)',
          }}
        >
          <div style={{ width: '384px', height: '100%' }}>
            <OpsEntityDetail
              entity={investigation.selectedEntity}
              onClose={() => investigation.setSelectedEntity(null)}
              onDispatch={() => investigation.setSelectedEntity(null)}
              onLiveFeed={(streamId) => setVideoStreamId(streamId)}
            />
          </div>
        </aside>
      </div>

      {/* Bottom bar — ARIA contentinfo */}
      <OpsBottomBar onInvestigateEntity={investigation.investigateEntity} />
    </div>
    </ErrorBoundary>
  );
}
