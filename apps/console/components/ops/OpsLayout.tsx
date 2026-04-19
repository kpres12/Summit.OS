'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useEntityStream } from '@/hooks/useEntityStream';
import { useInvestigation } from '@/hooks/useInvestigation';
import { useMissionDraw } from '@/hooks/useMissionDraw';
import { useReplay } from '@/hooks/useReplay';
import { useToast } from '@/contexts/ToastContext';
import ErrorBoundary from '@/components/ErrorBoundary';
import OpsTopBar from './OpsTopBar';
import OpsNavRail from './OpsNavRail';
import OpsAlertQueue from './OpsAlertQueue';
import OpsEntityList from './OpsEntityList';
import OpsMissions from './OpsMissions';
import OpsEntityDetail from './OpsEntityDetail';
import OpsBottomBar from './OpsBottomBar';
import OpsMapView, { RouteOverlay } from './OpsMapView';
import OpsHardware from './OpsHardware';
import OpsVideoPane from './OpsVideoPane';
import OpsReplayControls from './OpsReplayControls';
import OpsMissionBuilder from './OpsMissionBuilder';
import OpsMapLayers from './OpsMapLayers';
import OpsSystem from './OpsSystem';
import OpsIntel from './OpsIntel';
import OpsActionLog from './OpsActionLog';
import OpsActiveTasksPanel from './OpsActiveTasksPanel';
import OpsMissionTimeline from './OpsMissionTimeline';
import ToastContainer from '@/components/ui/ToastContainer';
import SessionExpiryBanner from '@/components/auth/SessionExpiryBanner';
import { fetchTasks, TaskAPI } from '@/lib/api';

type PanelId = 'alerts' | 'entities' | 'missions' | 'layers' | 'hardware' | 'system' | 'mission-builder' | 'intel' | 'log' | 'tasks';

interface OpsLayoutProps {
  onSwitchRole: () => void;
}

export default function OpsLayout({ onSwitchRole }: OpsLayoutProps) {
  const { entityList, connected } = useEntityStream();
  const { addToast } = useToast();
  const prevConnected = useRef<boolean | null>(null);

  // Incident name — read from env var or localStorage, editable at runtime
  const [incidentName, setIncidentName] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('summit_incident_name')
        ?? process.env.NEXT_PUBLIC_INCIDENT_NAME
        ?? '';
    }
    return process.env.NEXT_PUBLIC_INCIDENT_NAME ?? '';
  });

  const handleIncidentNameChange = (name: string) => {
    setIncidentName(name);
    if (typeof window !== 'undefined') localStorage.setItem('summit_incident_name', name);
  };

  // Toast on WS connect/disconnect
  useEffect(() => {
    if (prevConnected.current === null) {
      prevConnected.current = connected;
      return;
    }
    if (connected && !prevConnected.current) {
      addToast({ message: 'DATA FEED RESTORED — stream reconnected', severity: 'success' });
    } else if (!connected && prevConnected.current) {
      addToast({ message: 'DATA FEED LOST — attempting reconnect', severity: 'critical', persistent: true });
    }
    prevConnected.current = connected;
  }, [connected, addToast]);

  // Default to alert panel — the most urgent thing should be visible immediately.
  // User can close it; after that their preference is respected.
  const [activePanel, setActivePanel] = useState<PanelId | null>('alerts');

  // Investigation flow (alert → zoom → entity detail)
  const investigation = useInvestigation(entityList);

  // Live video overlay
  const [videoStreamId, setVideoStreamId] = useState<string | null>(null);

  // Active tasks (polled, used by panel + map route overlays)
  const [activeTasks, setActiveTasks] = useState<TaskAPI[]>([]);
  useEffect(() => {
    const load = async () => { const d = await fetchTasks(); setActiveTasks(d); };
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  // Build route overlays from active tasks
  const taskRoutes: RouteOverlay[] = activeTasks
    .filter(t => t.status === 'active' && t.waypoints?.length)
    .map(t => ({
      entityId: t.asset_id,
      targetLat: t.waypoints![0].lat,
      targetLon: t.waypoints![0].lon,
      taskType: t.action,
    }));

  // Timeline visibility
  const [timelineOpen, setTimelineOpen] = useState(true);

  // Mission replay
  const [replayMissionId, setReplayMissionId] = useState<string | null>(null);
  const replay = useReplay(replayMissionId);

  // Mission builder draw state
  const missionDraw = useMissionDraw();

  // Geofence draw (triggered from Layers panel)
  const [geofenceDrawMode, setGeofenceDrawMode] = useState(false);

  // Map layer visibility
  const [activeLayers, setActiveLayers] = useState<Set<string>>(new Set(['entities', 'tracks']));
  const toggleLayer = (id: string) => setActiveLayers(prev => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });

  // Auto-reopen mission builder when user finishes drawing the area on the map
  useEffect(() => {
    if (missionDraw.missionPolygon && missionDraw.missionPolygon.length >= 3) {
      setActivePanel('mission-builder');
    }
  }, [missionDraw.missionPolygon]);

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
          activeLayers={activeLayers}
          onToggleLayer={toggleLayer}
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
            setActivePanel(null); // collapse panel so map is fully visible
          }}
          missionPolygon={missionDraw.missionPolygon}
          onWaypointsChanged={missionDraw.setMissionWaypoints}
        />
      );
      case 'intel': return <OpsIntel />;
      case 'log':   return <OpsActionLog />;
      case 'tasks': return (
        <OpsActiveTasksPanel
          onFlyTo={(lat, lon) => investigation.investigateAlert({ alert_id: '', severity: '', description: '', source: '', ts_iso: '', _lat: lat, _lon: lon } as never)}
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
      <OpsTopBar
        onSwitchRole={onSwitchRole}
        missionName={incidentName}
        onMissionNameChange={handleIncidentNameChange}
      />

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
            showSatellites={activeLayers.has('satellites')}
            showGpsJam={activeLayers.has('gpsjam')}
            showMaritime={activeLayers.has('maritime')}
            showNoFlyZones={activeLayers.has('noflyzones')}
            showGrid={activeLayers.has('grid')}
            taskRoutes={taskRoutes}
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

      {/* Mission timeline */}
      <OpsMissionTimeline
        tasks={activeTasks}
        entityList={entityList}
        isOpen={timelineOpen}
        onToggle={() => setTimelineOpen(o => !o)}
      />

      {/* Bottom bar — ARIA contentinfo */}
      <OpsBottomBar onInvestigateEntity={investigation.investigateEntity} />
    </div>
      <ToastContainer />
      <SessionExpiryBanner />
    </ErrorBoundary>
  );
}
