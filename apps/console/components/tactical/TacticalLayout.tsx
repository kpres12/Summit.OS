'use client';

import React, { useCallback, useEffect, useState } from 'react';
import TopOverlay from './TopOverlay';
import LeftSidebar from './LeftSidebar';
import TacticalMap from './TacticalMap';
import CommandBar from './CommandBar';
import RightSidebar from './RightSidebar';
import { MapLayer } from './MapLayerControls';
import { Geofence } from './GeofenceEditor';
import { fetchGeofences, createGeofence, deleteGeofence } from '../../lib/api';

export default function TacticalLayout() {
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

  return (
    <div className="fixed inset-0 flex flex-col bg-zinc-950 overflow-hidden">
      <TopOverlay />
      <div className="flex flex-1 overflow-hidden relative">
        <LeftSidebar
          layers={layers}
          onToggleLayer={handleToggleLayer}
          geofences={geofences}
          editingGeofence={editingGeofence}
          onCreateGeofence={handleCreateGeofence}
          onSelectGeofence={(id) => setEditingGeofence(geofences.find(g => g.id === id) || null)}
          onDeleteGeofence={handleDeleteGeofence}
          onSaveGeofence={handleSaveGeofence}
          onCancelEdit={() => setEditingGeofence(null)}
          onToggleGeofenceActive={handleToggleGeofenceActive}
        />
        <TacticalMap layers={layers} />
        <RightSidebar />
      </div>
      <CommandBar />
    </div>
  );
}
