'use client';

import React, { useCallback, useEffect, useState, lazy, Suspense } from 'react';
import Map, { Marker, NavigationControl, ScaleControl } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import MapLayerControls, { MapLayer } from './MapLayerControls';
import GeofenceEditor, { Geofence } from './GeofenceEditor';
import EntityDetail from './EntityDetail';
import { useEntityStream, EntityData } from '../../hooks/useEntityStream';
import { fetchGeofences, createGeofence, deleteGeofence } from '../../lib/api';
import ShaderPipeline, { ShaderMode } from './ShaderPipeline';

// Lazy-load CesiumMap to avoid SSR + large bundle in 2D mode
const CesiumMap = lazy(() => import('./CesiumMap'));

const DARK_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

export type ViewMode = '2d' | '3d';

export default function TacticalMap() {
  const { entityList, connected, entityCount, trackCount } = useEntityStream();
  const [selectedEntity, setSelectedEntity] = useState<EntityData | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('3d');
  const [shaderMode, setShaderMode] = useState<ShaderMode>('NORMAL');

  const [layers, setLayers] = useState<MapLayer[]>([
    { id: 'entities', name: 'Entities', enabled: true, color: '#34d399', icon: '●' },
    { id: 'geofences', name: 'Geofences', enabled: true, color: '#f59e0b', icon: '⬢' },
    { id: 'tracks', name: 'Tracks', enabled: false, color: '#60a5fa', icon: '〰' },
    { id: 'orbits', name: 'Orbits', enabled: true, color: '#818cf8', icon: '◌' },
  ]);

  const [geofences, setGeofences] = useState<Geofence[]>([]);
  const [editingGeofence, setEditingGeofence] = useState<Geofence | null>(null);

  useEffect(() => {
    fetchGeofences()
      .then(({ geofences: remote }) => {
        if (remote && remote.length > 0) {
          setGeofences(
            remote.map((g) => ({
              id: String(g.id ?? `gf${Date.now()}${Math.random()}`),
              name: g.name,
              points: (g.coordinates || []).map((c) => ({ lat: c.lat, lon: c.lon })),
              type: (g.type as Geofence['type']) || 'warning',
              altitude_min: g.altitude_min,
              altitude_max: g.altitude_max,
              active: g.active ?? true,
            }))
          );
        }
      })
      .catch(() => {});
  }, []);

  // Keyboard shortcuts for shader modes + view toggle
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Don't capture if user is typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      switch (e.key) {
        case '1': setShaderMode('NORMAL'); break;
        case '2': setShaderMode('NVG'); break;
        case '3': setShaderMode('FLIR'); break;
        case '4': setShaderMode('CRT'); break;
        case 'v': setViewMode(prev => prev === '2d' ? '3d' : '2d'); break;
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const handleToggleLayer = (layerId: string) => {
    setLayers(prev =>
      prev.map(layer =>
        layer.id === layerId ? { ...layer, enabled: !layer.enabled } : layer
      )
    );
  };

  const handleCreateGeofence = () => {
    setEditingGeofence({
      id: 'new',
      name: '',
      points: [],
      type: 'warning',
      active: true,
    });
  };

  const handleSaveGeofence = useCallback(async (geofence: Geofence) => {
    const localId = geofence.id === 'new' ? `gf${Date.now()}` : geofence.id;
    const saved = { ...geofence, id: localId };
    if (geofence.id === 'new') {
      setGeofences(prev => [...prev, saved]);
    } else {
      setGeofences(prev => prev.map(g => (g.id === geofence.id ? saved : g)));
    }
    setEditingGeofence(null);

    try {
      await createGeofence({
        name: geofence.name,
        type: geofence.type,
        coordinates: geofence.points.map(p => ({ lat: p.lat, lon: p.lon })),
        altitude_min: geofence.altitude_min,
        altitude_max: geofence.altitude_max,
        active: geofence.active,
      });
    } catch {
      // backend unavailable
    }
  }, []);

  const handleDeleteGeofence = useCallback(async (id: string) => {
    setGeofences(prev => prev.filter(g => g.id !== id));
    try {
      const numId = parseInt(id.replace(/^gf/, ''), 10);
      if (!isNaN(numId)) await deleteGeofence(numId);
    } catch {
      // backend unavailable
    }
  }, []);

  const handleToggleGeofenceActive = (id: string) => {
    setGeofences(prev =>
      prev.map(g => (g.id === id ? { ...g, active: !g.active } : g))
    );
  };

  const markerColor = (e: EntityData) => {
    switch (e.entity_type) {
      case 'friendly': return '#34d399';
      case 'hostile': return '#ef4444';
      case 'neutral': return '#a1a1aa';
      default: return '#fbbf24';
    }
  };

  const entitiesLayer = layers.find(l => l.id === 'entities');
  const orbitsLayer = layers.find(l => l.id === 'orbits');

  // Count aircraft and satellites
  const aircraftCount = entityList.filter(e => e.classification === 'aircraft').length;
  const satelliteCount = entityList.filter(e => e.classification === 'satellite').length;

  return (
    <div className="flex-1 relative overflow-hidden">
      <ShaderPipeline mode={shaderMode}>
        {/* 3D Globe (CesiumJS) */}
        {viewMode === '3d' && (
          <Suspense fallback={
            <div className="w-full h-full bg-[#0A0A0A] flex items-center justify-center">
              <span className="text-zinc-600 font-mono text-xs">INITIALIZING 3D GLOBE...</span>
            </div>
          }>
            <CesiumMap
              entities={entityList}
              onSelectEntity={setSelectedEntity}
              showEntities={entitiesLayer?.enabled ?? true}
              showOrbits={orbitsLayer?.enabled ?? true}
            />
          </Suspense>
        )}

        {/* 2D Map (MapLibre) */}
        {viewMode === '2d' && (
          <Map
            initialViewState={{
              longitude: -98.5,
              latitude: 39.8,
              zoom: 4,
            }}
            style={{ width: '100%', height: '100%' }}
            mapStyle={DARK_STYLE}
          >
            <NavigationControl position="top-right" />
            <ScaleControl position="bottom-left" />

            {entitiesLayer?.enabled && entityList.map((entity) => (
              entity.position && (
                <Marker
                  key={entity.entity_id}
                  longitude={entity.position.lon}
                  latitude={entity.position.lat}
                  onClick={(e) => {
                    e.originalEvent.stopPropagation();
                    setSelectedEntity(entity);
                  }}
                >
                  <div
                    className="w-3 h-3 rounded-full border-2 border-zinc-900 cursor-pointer hover:scale-150 transition-transform"
                    style={{ backgroundColor: markerColor(entity) }}
                    title={entity.callsign || entity.entity_id}
                  />
                </Marker>
              )
            ))}
          </Map>
        )}
      </ShaderPipeline>

      {/* Status bar */}
      <div className="absolute top-3 left-3 flex gap-2 text-[10px] font-mono z-10">
        <span className={`px-2 py-1 rounded ${connected ? 'bg-zinc-800 text-emerald-400' : 'bg-red-900/50 text-red-400'}`}>
          {connected ? 'CONNECTED' : 'DISCONNECTED'}
        </span>
        <span className="px-2 py-1 rounded bg-zinc-800 text-zinc-400">
          {entityCount} entities
        </span>
        {aircraftCount > 0 && (
          <span className="px-2 py-1 rounded bg-zinc-800 text-emerald-400/80">
            ✈ {aircraftCount}
          </span>
        )}
        {satelliteCount > 0 && (
          <span className="px-2 py-1 rounded bg-zinc-800 text-indigo-400/80">
            ◉ {satelliteCount}
          </span>
        )}
        <span className="px-2 py-1 rounded bg-zinc-800 text-zinc-400">
          {trackCount} tracks
        </span>
        {shaderMode !== 'NORMAL' && (
          <span className="px-2 py-1 rounded bg-zinc-800 text-amber-400">
            {shaderMode}
          </span>
        )}
      </div>

      {/* View mode toggle */}
      <div className="absolute top-3 left-1/2 -translate-x-1/2 flex gap-0.5 text-[10px] font-mono z-10">
        <button
          onClick={() => setViewMode('3d')}
          className={`px-3 py-1 rounded-l transition-colors ${
            viewMode === '3d'
              ? 'bg-zinc-700 text-zinc-100'
              : 'bg-zinc-900/80 text-zinc-500 hover:text-zinc-300'
          }`}
        >
          3D GLOBE
        </button>
        <button
          onClick={() => setViewMode('2d')}
          className={`px-3 py-1 rounded-r transition-colors ${
            viewMode === '2d'
              ? 'bg-zinc-700 text-zinc-100'
              : 'bg-zinc-900/80 text-zinc-500 hover:text-zinc-300'
          }`}
        >
          2D MAP
        </button>
      </div>

      {/* Shader mode selector */}
      <div className="absolute bottom-3 left-3 flex gap-0.5 text-[10px] font-mono z-10">
        {(['NORMAL', 'NVG', 'FLIR', 'CRT'] as ShaderMode[]).map((mode, i) => (
          <button
            key={mode}
            onClick={() => setShaderMode(mode)}
            className={`px-2 py-1 transition-colors ${
              shaderMode === mode
                ? mode === 'NVG' ? 'bg-green-900/80 text-green-400'
                  : mode === 'FLIR' ? 'bg-orange-900/80 text-orange-400'
                  : mode === 'CRT' ? 'bg-cyan-900/80 text-cyan-400'
                  : 'bg-zinc-700 text-zinc-100'
                : 'bg-zinc-900/80 text-zinc-500 hover:text-zinc-300'
            } ${i === 0 ? 'rounded-l' : ''} ${i === 3 ? 'rounded-r' : ''}`}
            title={`${mode} (${i + 1})`}
          >
            {mode}
          </button>
        ))}
      </div>

      {/* Layer Controls */}
      <MapLayerControls
        layers={layers}
        onToggleLayer={handleToggleLayer}
      />

      {/* Geofence Editor */}
      <GeofenceEditor
        geofences={geofences}
        editingGeofence={editingGeofence}
        onCreateNew={handleCreateGeofence}
        onSelectGeofence={(id) => setEditingGeofence(geofences.find(g => g.id === id) || null)}
        onDeleteGeofence={handleDeleteGeofence}
        onSaveGeofence={handleSaveGeofence}
        onCancelEdit={() => setEditingGeofence(null)}
        onToggleActive={handleToggleGeofenceActive}
      />

      {/* Entity Detail Panel */}
      {selectedEntity && (
        <EntityDetail
          entity={selectedEntity}
          onClose={() => setSelectedEntity(null)}
        />
      )}
    </div>
  );
}
