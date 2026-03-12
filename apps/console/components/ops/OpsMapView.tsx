'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import Map, { Marker, NavigationControl, ScaleControl, MapRef, Source, Layer, MapLayerMouseEvent } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useEntityStream, EntityData } from '@/hooks/useEntityStream';
import { createGeofence } from '@/lib/api';

const DARK_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

interface OpsMapViewProps {
  onSelectEntity?: (entity: EntityData | null) => void;
  flyToLocation?: { lat: number; lon: number } | null;
  alertEntityIds?: Set<string>;
}

function markerColor(e: EntityData): string {
  switch (e.entity_type) {
    case 'active': return '#00FF9C';
    case 'alert': return '#FF3B3B';
    case 'neutral': return 'rgba(200,230,201,0.6)';
    default: return '#FFB300';
  }
}

type DrawVertex = { lat: number; lon: number };

export default function OpsMapView({ onSelectEntity, flyToLocation, alertEntityIds }: OpsMapViewProps) {
  const { entityList } = useEntityStream();
  const mapRef = useRef<MapRef>(null);
  const [drawMode, setDrawMode] = useState(false);
  const [vertices, setVertices] = useState<DrawVertex[]>([]);
  const [geoName, setGeoName] = useState('');
  const [savedFeedback, setSavedFeedback] = useState<string | null>(null);

  // Fly to location when prop changes
  useEffect(() => {
    if (!flyToLocation || !mapRef.current) return;
    mapRef.current.flyTo({
      center: [flyToLocation.lon, flyToLocation.lat],
      zoom: 13,
      duration: 800,
    });
  }, [flyToLocation]);

  const handleMapClick = useCallback((e: MapLayerMouseEvent) => {
    if (!drawMode) return;
    setVertices((prev) => [...prev, { lat: e.lngLat.lat, lon: e.lngLat.lng }]);
  }, [drawMode]);

  const handleMapDblClick = useCallback((_e: MapLayerMouseEvent) => {
    if (!drawMode || vertices.length < 3) return;
    finishPolygon();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drawMode, vertices]);

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
        cursor={drawMode ? 'crosshair' : 'grab'}
        onClick={handleMapClick}
        onDblClick={handleMapDblClick}
      >
        <NavigationControl position="top-right" />
        <ScaleControl position="bottom-left" />

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
              background: i === 0 ? '#FFB300' : '#FFB30080',
              border: '1px solid #FFB300',
            }} />
          </Marker>
        ))}

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
                  transition: 'transform 0.1s',
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
              color: savedFeedback.startsWith('SAVE FAILED') ? '#FF3B3B' : '#00FF9C',
              background: '#0D1210',
              border: `1px solid ${savedFeedback.startsWith('SAVE FAILED') ? 'rgba(255,59,59,0.4)' : 'rgba(0,255,156,0.4)'}`,
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
                background: '#0D1210',
                border: '1px solid rgba(255,179,0,0.4)',
                color: '#FFB300',
                width: '140px',
              }}
            />
            <div className="text-[9px] px-2" style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'rgba(255,179,0,0.7)' }}>
              {vertices.length < 3 ? `CLICK TO ADD POINTS (${vertices.length}/3 min)` : 'DOUBLE-CLICK TO CLOSE'}
            </div>
            <div className="flex gap-1">
              {vertices.length >= 3 && (
                <button
                  onClick={finishPolygon}
                  className="text-[10px] px-2 py-1 tracking-wider"
                  style={{
                    fontFamily: 'var(--font-ibm-plex-mono), monospace',
                    color: '#080C0A', background: '#FFB300', border: 'none', cursor: 'pointer',
                  }}
                >
                  SAVE
                </button>
              )}
              <button
                onClick={() => { setDrawMode(false); setVertices([]); setGeoName(''); }}
                className="text-[10px] px-2 py-1 tracking-wider"
                style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  color: '#FF3B3B', background: 'transparent', border: '1px solid rgba(255,59,59,0.4)', cursor: 'pointer',
                }}
              >
                CANCEL
              </button>
            </div>
          </>
        )}

        {!drawMode && (
          <button
            onClick={() => setDrawMode(true)}
            className="text-[10px] px-2 py-1 tracking-wider"
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: '#FFB300',
              background: '#0D1210',
              border: '1px solid rgba(255,179,0,0.3)',
              cursor: 'pointer',
            }}
            title="Draw geofence polygon"
          >
            ⬡ DRAW GEOFENCE
          </button>
        )}
      </div>
    </div>
  );
}
