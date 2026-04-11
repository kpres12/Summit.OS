'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import Map, { Marker, NavigationControl, ScaleControl, MapRef, Source, Layer, MapLayerMouseEvent } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useEntityStream, EntityData } from '@/hooks/useEntityStream';
import { apiFetch, createGeofence } from '@/lib/api';

// Tile source priority:
// 1. NEXT_PUBLIC_TILE_URL — operator-configured (local tile server, PMTiles, etc.)
// 2. Carto dark matter — default online basemap
// See infra/tiles/ for offline tile server setup instructions
const DARK_STYLE =
  process.env.NEXT_PUBLIC_TILE_URL ||
  'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

interface OpsMapViewProps {
  onSelectEntity?: (entity: EntityData | null) => void;
  flyToLocation?: { lat: number; lon: number } | null;
  alertEntityIds?: Set<string>;
  selectedEntityId?: string | null;
  // Mission builder integration
  missionDrawMode?: boolean;
  onMissionArea?: (coords: { lat: number; lon: number }[]) => void;
  missionWaypoints?: { lat: number; lon: number; alt: number }[];
  // Geofence draw (triggered from Layers panel)
  geofenceDrawMode?: boolean;
  onGeofenceDrawEnd?: () => void;
}

function markerColor(e: EntityData): string {
  switch (e.entity_type) {
    case 'active': return 'var(--accent)';
    case 'alert': return 'var(--critical)';
    case 'neutral': return 'var(--text-dim)';
    default: return 'var(--warning)';
  }
}

const STALE_WARN_S = 60;
const STALE_DEAD_S = 300;

function markerOpacity(lastSeen: number): number {
  const age = (Date.now() / 1000) - lastSeen;
  if (age > STALE_DEAD_S) return 0.3;
  if (age > STALE_WARN_S) return 0.55;
  return 1;
}

type DrawVertex = { lat: number; lon: number };

export default function OpsMapView({
  onSelectEntity, flyToLocation, alertEntityIds, selectedEntityId,
  missionDrawMode, onMissionArea, missionWaypoints,
  geofenceDrawMode, onGeofenceDrawEnd,
}: OpsMapViewProps) {
  const { entityList } = useEntityStream();
  const mapRef = useRef<MapRef>(null);
  const [drawMode, setDrawMode] = useState(false);
  const [vertices, setVertices] = useState<DrawVertex[]>([]);
  const [geoName, setGeoName] = useState('');
  const [savedFeedback, setSavedFeedback] = useState<string | null>(null);
  // Mission area draw (controlled externally via missionDrawMode prop)
  const [missionVertices, setMissionVertices] = useState<DrawVertex[]>([]);
  // Entity trail
  const [trailCoords, setTrailCoords] = useState<[number, number][]>([]);

  // Fly to location when prop changes
  useEffect(() => {
    if (!flyToLocation || !mapRef.current) return;
    mapRef.current.flyTo({
      center: [flyToLocation.lon, flyToLocation.lat],
      zoom: 13,
      duration: 800,
    });
  }, [flyToLocation]);

  // Clear mission vertices when missionDrawMode turns off externally
  useEffect(() => {
    if (!missionDrawMode) setMissionVertices([]);
  }, [missionDrawMode]);

  // Sync geofence draw mode from Layers panel
  useEffect(() => {
    if (geofenceDrawMode) {
      setDrawMode(true);
      setVertices([]);
      setGeoName('');
    }
  }, [geofenceDrawMode]);

  // Fetch position trail when selected entity changes
  useEffect(() => {
    if (!selectedEntityId) {
      setTrailCoords([]);
      return;
    }
    let cancelled = false;
    apiFetch(`/v1/entities/${selectedEntityId}/trail`)
      .then((res) => res.json())
      .then((data: { trail: { lat: number; lon: number }[] }) => {
        if (!cancelled && data?.trail) {
          setTrailCoords(data.trail.map((p) => [p.lon, p.lat]));
        }
      })
      .catch(() => setTrailCoords([]));
    return () => { cancelled = true; };
  }, [selectedEntityId]);

  const handleMapClick = useCallback((e: MapLayerMouseEvent) => {
    if (missionDrawMode) {
      setMissionVertices((prev) => [...prev, { lat: e.lngLat.lat, lon: e.lngLat.lng }]);
      return;
    }
    if (!drawMode) return;
    setVertices((prev) => [...prev, { lat: e.lngLat.lat, lon: e.lngLat.lng }]);
  }, [drawMode, missionDrawMode]);

  const handleMapDblClick = useCallback((_evt: MapLayerMouseEvent) => {
    if (missionDrawMode && missionVertices.length >= 3) {
      onMissionArea?.(missionVertices.map((v) => ({ lat: v.lat, lon: v.lon })));
      setMissionVertices([]);
      return;
    }
    if (!drawMode || vertices.length < 3) return;
    finishPolygon();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drawMode, vertices, missionDrawMode, missionVertices]);

  const finishPolygon = async () => {
    if (vertices.length < 3) return;
    const coords = vertices.map((v) => [v.lon, v.lat]);
    coords.push(coords[0]); // close ring
    const name = geoName.trim() || `GEOFENCE-${Date.now().toString(36).toUpperCase()}`;
    try {
      await createGeofence({ name, type: 'exclusion', coordinates: coords.map(([lon, lat]) => ({ lat, lon })) });
      setSavedFeedback(`SAVED: ${name}`);
    } catch {
      setSavedFeedback('SAVE FAILED');
    }
    setVertices([]);
    setDrawMode(false);
    setGeoName('');
    onGeofenceDrawEnd?.();
    setTimeout(() => setSavedFeedback(null), 3000);
  };

  // Build GeoJSON for in-progress polygon
  const drawGeoJSON = vertices.length >= 2 ? {
    type: 'Feature' as const,
    geometry: {
      type: 'LineString' as const,
      coordinates: [
        ...vertices.map((v): [number, number] => [v.lon, v.lat]),
        vertices.length >= 3 ? [vertices[0].lon, vertices[0].lat] as [number, number] : [vertices[vertices.length - 1].lon, vertices[vertices.length - 1].lat] as [number, number],
      ],
    },
    properties: {},
  } : null;

  return (
    <div className="w-full h-full relative">
      <Map
        ref={mapRef}
        initialViewState={{ longitude: -98.5, latitude: 39.8, zoom: 4 }}
        style={{ width: '100%', height: '100%' }}
        mapStyle={DARK_STYLE}
        cursor={drawMode || missionDrawMode ? 'crosshair' : 'grab'}
        onClick={handleMapClick}
        onDblClick={handleMapDblClick}
      >
        <NavigationControl position="top-right" />
        <ScaleControl position="bottom-left" />

        {/* Mission area in-progress polygon (blue) */}
        {missionDrawMode && missionVertices.length >= 2 && (() => {
          const coords = missionVertices.map((v): [number, number] => [v.lon, v.lat]);
          if (missionVertices.length >= 3) coords.push(coords[0]);
          return (
            <Source id="mission-draw-line" type="geojson" data={{ type: 'Feature', geometry: { type: 'LineString', coordinates: coords }, properties: {} }}>
              <Layer id="mission-draw-line-layer" type="line"
                paint={{ 'line-color': '#4FC3F7', 'line-width': 2, 'line-dasharray': [4, 2] }} />
            </Source>
          );
        })()}
        {missionDrawMode && missionVertices.map((v, i) => (
          <Marker key={`mv-${i}`} longitude={v.lon} latitude={v.lat}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: i === 0 ? 'var(--color-active)' : 'color-mix(in srgb, var(--color-active) 50%, transparent)', border: '1px solid var(--color-active)' }} />
          </Marker>
        ))}

        {/* Mission waypoint preview dots */}
        {missionWaypoints && missionWaypoints.length > 0 && (() => {
          const geojson = {
            type: 'FeatureCollection' as const,
            features: missionWaypoints.map((wp) => ({
              type: 'Feature' as const,
              geometry: { type: 'Point' as const, coordinates: [wp.lon, wp.lat] },
              properties: { alt: wp.alt },
            })),
          };
          return (
            <Source id="mission-waypoints" type="geojson" data={geojson}>
              <Layer id="mission-waypoints-layer" type="circle"
                paint={{ 'circle-radius': 3, 'circle-color': '#4FC3F7', 'circle-opacity': 0.6, 'circle-stroke-width': 0 }} />
            </Source>
          );
        })()}

        {/* In-progress draw polygon */}
        {drawGeoJSON && (
          <Source id="draw-line" type="geojson" data={drawGeoJSON}>
            <Layer
              id="draw-line-layer"
              type="line"
              paint={{ 'line-color': '#FFB300', 'line-width': 2, 'line-dasharray': [4, 2] }}
            />
          </Source>
        )}

        {/* Draw vertices */}
        {drawMode && vertices.map((v, i) => (
          <Marker key={`v-${i}`} longitude={v.lon} latitude={v.lat}>
            <div style={{
              width: '8px', height: '8px', borderRadius: '50%',
              background: i === 0 ? 'var(--warning)' : 'color-mix(in srgb, var(--warning) 50%, transparent)',
              border: '1px solid var(--warning)',
            }} />
          </Marker>
        ))}

        {/* Selected entity position trail */}
        {trailCoords.length >= 2 && (
          <Source
            id="entity-trail"
            type="geojson"
            data={{ type: 'Feature', geometry: { type: 'LineString', coordinates: trailCoords }, properties: {} }}
          >
            <Layer
              id="entity-trail-layer"
              type="line"
              paint={{ 'line-color': 'var(--accent)', 'line-width': 2, 'line-opacity': 0.7 }}
            />
          </Source>
        )}

        {/* Entity markers */}
        {entityList.map((entity) =>
          entity.position ? (
            <Marker
              key={entity.entity_id}
              longitude={entity.position.lon}
              latitude={entity.position.lat}
              onClick={(e) => {
                if (drawMode) return;
                e.originalEvent.stopPropagation();
                onSelectEntity?.(entity);
              }}
            >
              <div
                title={entity.callsign || entity.entity_id}
                className={alertEntityIds?.has(entity.entity_id) ? 'alert-pulse' : undefined}
                style={{
                  width: '10px', height: '10px', borderRadius: '50%',
                  background: markerColor(entity),
                  border: '2px solid rgba(8,12,10,0.8)',
                  cursor: drawMode ? 'crosshair' : 'pointer',
                  boxShadow: `0 0 6px ${markerColor(entity)}80`,
                  opacity: markerOpacity(entity.last_seen),
                  transition: 'transform 0.1s, opacity 0.3s',
                }}
                onMouseEnter={(e) => { if (!drawMode) (e.currentTarget as HTMLDivElement).style.transform = 'scale(1.6)'; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLDivElement).style.transform = 'scale(1)'; }}
              />
            </Marker>
          ) : null
        )}
      </Map>

      {/* Draw controls overlay */}
      <div
        className="absolute bottom-8 right-3 flex flex-col items-end gap-2"
        style={{ zIndex: 10 }}
      >
        {savedFeedback && (
          <div
            className="text-[10px] px-2 py-1"
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: savedFeedback.startsWith('SAVE FAILED') ? 'var(--critical)' : 'var(--accent)',
              background: 'var(--background-panel)',
              border: `1px solid ${savedFeedback.startsWith('SAVE FAILED') ? 'color-mix(in srgb, var(--critical) 40%, transparent)' : 'var(--accent-30)'}`,
            }}
          >
            {savedFeedback}
          </div>
        )}

        {drawMode && (
          <>
            <input
              type="text"
              value={geoName}
              onChange={(e) => setGeoName(e.target.value)}
              placeholder="Geofence name..."
              className="text-[10px] px-2 py-1 outline-none"
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                background: 'var(--background-panel)',
                border: '1px solid color-mix(in srgb, var(--warning) 40%, transparent)',
                color: 'var(--warning)',
                width: '140px',
              }}
            />
            <div className="text-[9px] px-2" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'color-mix(in srgb, var(--warning) 70%, transparent)' }}>
              {vertices.length < 3 ? `CLICK TO ADD POINTS (${vertices.length}/3 min)` : 'DOUBLE-CLICK TO CLOSE'}
            </div>
            <div className="flex gap-1">
              {vertices.length >= 3 && (
                <button
                  onClick={finishPolygon}
                  className="text-[10px] px-2 py-1 tracking-wider"
                  style={{
                    fontFamily: 'var(--font-ibm-plex-mono), monospace',
                    color: 'var(--background)', background: 'var(--warning)', border: 'none', cursor: 'pointer',
                  }}
                >
                  SAVE
                </button>
              )}
              <button
                onClick={() => { setDrawMode(false); setVertices([]); setGeoName(''); onGeofenceDrawEnd?.(); }}
                className="text-[10px] px-2 py-1 tracking-wider"
                style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  color: 'var(--critical)', background: 'transparent', border: '1px solid color-mix(in srgb, var(--critical) 40%, transparent)', cursor: 'pointer',
                }}
              >
                CANCEL
              </button>
            </div>
          </>
        )}

      </div>
    </div>
  );
}
