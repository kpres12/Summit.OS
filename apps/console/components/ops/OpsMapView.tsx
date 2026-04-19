'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import Map, { Marker, NavigationControl, ScaleControl, MapRef, Source, Layer, MapLayerMouseEvent } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useEntityStream, EntityData } from '@/hooks/useEntityStream';
import { apiFetch, createGeofence } from '@/lib/api';

interface SatellitePos { entity_id: string; name: string; sat_type: string; position: { lat: number; lon: number; alt: number }; }
interface JamZone { id: string; name: string; lat: number; lon: number; radius_km: number; intensity: number; }
interface Vessel { id: string; name: string; type: string; lat: number; lon: number; heading: number; speed_kts: number; status?: string; }
interface NoFlyZone { id: string; name: string; severity: string; active: boolean; coordinates: { lat: number; lon: number }[]; }

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
  // OSINT / WorldView layers
  showSatellites?: boolean;
  showGpsJam?: boolean;
  showMaritime?: boolean;
  showNoFlyZones?: boolean;
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

// Generate a GeoJSON circle polygon for a geographic radius
function circlePolygon(lat: number, lon: number, radiusKm: number, steps = 32): [number, number][] {
  const coords: [number, number][] = [];
  for (let i = 0; i <= steps; i++) {
    const angle = (i / steps) * 2 * Math.PI;
    const dLat = (radiusKm / 111) * Math.cos(angle);
    const dLon = (radiusKm / (111 * Math.cos((lat * Math.PI) / 180))) * Math.sin(angle);
    coords.push([lon + dLon, lat + dLat]);
  }
  return coords;
}

const SAT_COLOR: Record<string, string> = {
  station: '#00FF9C',
  sar: '#4FC3F7',
  reconnaissance: '#FFB300',
  optical: '#B39DDB',
  comms: '#8A8A8A',
};

export default function OpsMapView({
  onSelectEntity, flyToLocation, alertEntityIds, selectedEntityId,
  missionDrawMode, onMissionArea, missionWaypoints,
  geofenceDrawMode, onGeofenceDrawEnd,
  showSatellites, showGpsJam, showMaritime, showNoFlyZones,
}: OpsMapViewProps) {
  const { entityList } = useEntityStream();
  const mapRef = useRef<MapRef>(null);
  // geofenceDrawMode is owned by OpsLayout — no intermediate state, use prop directly
  const [vertices, setVertices] = useState<DrawVertex[]>([]);
  const [geoName, setGeoName] = useState('');
  const [cursorPos, setCursorPos] = useState<DrawVertex | null>(null);
  const [savedFeedback, setSavedFeedback] = useState<string | null>(null);
  const [savedPolygon, setSavedPolygon] = useState<[number, number][] | null>(null);
  // Mission area draw (controlled externally via missionDrawMode prop)
  const [missionVertices, setMissionVertices] = useState<DrawVertex[]>([]);
  // Entity trail
  const [trailCoords, setTrailCoords] = useState<[number, number][]>([]);
  // OSINT layer data
  const [satellites, setSatellites] = useState<SatellitePos[]>([]);
  const [jamZones, setJamZones] = useState<JamZone[]>([]);
  const [vessels, setVessels] = useState<Vessel[]>([]);
  const [noFlyZones, setNoFlyZones] = useState<NoFlyZone[]>([]);

  // Fly to location when prop changes
  useEffect(() => {
    if (!flyToLocation || !mapRef.current) return;
    mapRef.current.flyTo({
      center: [flyToLocation.lon, flyToLocation.lat],
      zoom: 13,
      duration: 800,
    });
  }, [flyToLocation]);

  // Clear mission vertices when missionDrawMode turns off
  useEffect(() => {
    if (!missionDrawMode) setMissionVertices([]);
  }, [missionDrawMode]);

  // Reset geofence draw state when mode turns off
  useEffect(() => {
    if (!geofenceDrawMode) {
      setVertices([]);
      setGeoName('');
      setCursorPos(null);
    }
  }, [geofenceDrawMode]);

  const finishPolygon = useCallback(async (verts: DrawVertex[], name: string) => {
    if (verts.length < 3) return;
    const coords = verts.map((v): [number, number] => [v.lon, v.lat]);
    coords.push(coords[0]);
    const finalName = name.trim() || `GEOFENCE-${Date.now().toString(36).toUpperCase()}`;
    try {
      await createGeofence({ name: finalName, type: 'exclusion', coordinates: coords.map(([lon, lat]) => ({ lat, lon })) });
      setSavedFeedback(`SAVED: ${finalName}`);
      setSavedPolygon(coords);
    } catch {
      setSavedFeedback('SAVE FAILED');
    }
    setVertices([]);
    setGeoName('');
    onGeofenceDrawEnd?.();
    setTimeout(() => { setSavedFeedback(null); setSavedPolygon(null); }, 3000);
  }, [onGeofenceDrawEnd]);

  const verticesRef = useRef(vertices);
  const geoNameRef  = useRef(geoName);
  useEffect(() => { verticesRef.current = vertices; }, [vertices]);
  useEffect(() => { geoNameRef.current  = geoName;  }, [geoName]);

  // ESC to cancel, Enter to finish geofence draw
  useEffect(() => {
    if (!geofenceDrawMode) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setVertices([]);
        setGeoName('');
        setCursorPos(null);
        onGeofenceDrawEnd?.();
      }
      if (e.key === 'Enter' && verticesRef.current.length >= 3) {
        finishPolygon(verticesRef.current, geoNameRef.current);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [geofenceDrawMode, onGeofenceDrawEnd, finishPolygon]);

  // OSINT layer fetchers — only poll when layer is visible
  useEffect(() => {
    if (!showSatellites) { setSatellites([]); return; }
    const doFetch = () => apiFetch('/v1/satellites').then(r => r.json()).then(d => setSatellites(d.satellites || [])).catch(() => {});
    doFetch();
    const t = setInterval(doFetch, 30000);
    return () => clearInterval(t);
  }, [showSatellites]);

  useEffect(() => {
    if (!showGpsJam) { setJamZones([]); return; }
    const doFetch = () => apiFetch('/v1/gpsjam').then(r => r.json()).then(d => setJamZones(d.zones || [])).catch(() => {});
    doFetch();
    const t = setInterval(doFetch, 30000);
    return () => clearInterval(t);
  }, [showGpsJam]);

  useEffect(() => {
    if (!showMaritime) { setVessels([]); return; }
    const doFetch = () => apiFetch('/v1/maritime').then(r => r.json()).then(d => setVessels(d.vessels || [])).catch(() => {});
    doFetch();
    const t = setInterval(doFetch, 30000);
    return () => clearInterval(t);
  }, [showMaritime]);

  useEffect(() => {
    if (!showNoFlyZones) { setNoFlyZones([]); return; }
    const doFetch = () => apiFetch('/v1/noflyzones').then(r => r.json()).then(d => setNoFlyZones(d.zones || [])).catch(() => {});
    doFetch();
    const t = setInterval(doFetch, 30000);
    return () => clearInterval(t);
  }, [showNoFlyZones]);

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
    if (!geofenceDrawMode) return;
    setVertices((prev) => [...prev, { lat: e.lngLat.lat, lon: e.lngLat.lng }]);
  }, [geofenceDrawMode, missionDrawMode]);

  const handleMapDblClick = useCallback((_evt: MapLayerMouseEvent) => {
    if (missionDrawMode && missionVertices.length >= 3) {
      onMissionArea?.(missionVertices.map((v) => ({ lat: v.lat, lon: v.lon })));
      setMissionVertices([]);
    }
  }, [missionDrawMode, missionVertices, onMissionArea]);

  const handleMouseMove = useCallback((e: MapLayerMouseEvent) => {
    if (geofenceDrawMode) {
      setCursorPos({ lat: e.lngLat.lat, lon: e.lngLat.lng });
    }
  }, [geofenceDrawMode]);

  const handleContextMenu = useCallback((e: MapLayerMouseEvent) => {
    if (geofenceDrawMode && verticesRef.current.length >= 3) {
      e.originalEvent.preventDefault();
      finishPolygon(verticesRef.current, geoNameRef.current);
    }
  }, [geofenceDrawMode, finishPolygon]);

  // Build GeoJSON for in-progress polygon (includes live cursor position)
  const drawPoints: [number, number][] = vertices.map((v): [number, number] => [v.lon, v.lat]);
  if (cursorPos) drawPoints.push([cursorPos.lon, cursorPos.lat]);
  if (drawPoints.length >= 3) drawPoints.push(drawPoints[0]); // close preview
  const drawGeoJSON = drawPoints.length >= 2 ? {
    type: 'Feature' as const,
    geometry: { type: 'LineString' as const, coordinates: drawPoints },
    properties: {},
  } : null;

  return (
    <div className="w-full h-full relative">
      <Map
        ref={mapRef}
        initialViewState={{ longitude: -98.5, latitude: 39.8, zoom: 4 }}
        style={{ width: '100%', height: '100%' }}
        mapStyle={DARK_STYLE}
        cursor={geofenceDrawMode || missionDrawMode ? 'crosshair' : 'grab'}
        doubleClickZoom={!geofenceDrawMode && !missionDrawMode}
        onClick={handleMapClick}
        onDblClick={handleMapDblClick}
        onMouseMove={handleMouseMove}
        onContextMenu={handleContextMenu}
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
        {geofenceDrawMode && vertices.map((v, i) => (
          <Marker key={`v-${i}`} longitude={v.lon} latitude={v.lat}>
            <div style={{
              width: '12px', height: '12px', borderRadius: '50%',
              background: i === 0 ? 'var(--warning)' : 'rgba(255,179,0,0.6)',
              border: '2px solid var(--warning)',
              boxShadow: '0 0 6px rgba(255,179,0,0.6)',
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

        {/* === OSINT LAYERS === */}

        {/* GPS Jamming — filled circles graded by intensity */}
        {showGpsJam && jamZones.length > 0 && (() => {
          const geojson = {
            type: 'FeatureCollection' as const,
            features: jamZones.map(z => ({
              type: 'Feature' as const,
              geometry: { type: 'Polygon' as const, coordinates: [circlePolygon(z.lat, z.lon, z.radius_km)] },
              properties: { intensity: z.intensity },
            })),
          };
          return (
            <Source id="gpsjam-zones" type="geojson" data={geojson}>
              <Layer id="gpsjam-fill" type="fill"
                paint={{ 'fill-color': '#FF3B3B', 'fill-opacity': ['interpolate', ['linear'], ['get', 'intensity'], 0.3, 0.08, 1, 0.35] as unknown as number }} />
              <Layer id="gpsjam-outline" type="line"
                paint={{ 'line-color': '#FF3B3B', 'line-width': 1, 'line-opacity': 0.7, 'line-dasharray': [3, 2] }} />
            </Source>
          );
        })()}

        {/* No-fly zones — polygon fills */}
        {showNoFlyZones && noFlyZones.length > 0 && (() => {
          const activeZones = noFlyZones.filter(z => z.active && z.coordinates && z.coordinates.length >= 3);
          if (!activeZones.length) return null;
          const geojson = {
            type: 'FeatureCollection' as const,
            features: activeZones.map(z => ({
              type: 'Feature' as const,
              geometry: {
                type: 'Polygon' as const,
                coordinates: [[
                  ...z.coordinates.map((c): [number, number] => [c.lon, c.lat]),
                  [z.coordinates[0].lon, z.coordinates[0].lat] as [number, number],
                ]],
              },
              properties: { severity: z.severity },
            })),
          };
          return (
            <Source id="noflyzones" type="geojson" data={geojson}>
              <Layer id="noflyzones-fill" type="fill"
                paint={{ 'fill-color': '#FF3B3B', 'fill-opacity': 0.12 }} />
              <Layer id="noflyzones-outline" type="line"
                paint={{ 'line-color': '#FF3B3B', 'line-width': 1.5, 'line-dasharray': [6, 3] }} />
            </Source>
          );
        })()}

        {/* Satellite positions — small dim dots */}
        {showSatellites && satellites.map(sat => (
          <Marker key={sat.entity_id} longitude={sat.position.lon} latitude={sat.position.lat}>
            <div
              title={`${sat.name} · ${sat.sat_type.toUpperCase()}`}
              style={{
                width: '6px', height: '6px', borderRadius: '50%',
                background: SAT_COLOR[sat.sat_type] || '#4FC3F7',
                border: `1px solid ${SAT_COLOR[sat.sat_type] || '#4FC3F7'}`,
                boxShadow: `0 0 5px ${SAT_COLOR[sat.sat_type] || '#4FC3F7'}50`,
                opacity: 0.75,
                cursor: 'default',
              }}
            />
          </Marker>
        ))}

        {/* Maritime vessels — heading triangles */}
        {showMaritime && vessels.map(v => (
          <Marker key={v.id} longitude={v.lon} latitude={v.lat}>
            <div
              title={`${v.name} · ${v.type.toUpperCase()} · ${v.speed_kts.toFixed(1)} kts`}
              style={{
                width: 0, height: 0,
                borderLeft: '4px solid transparent',
                borderRight: '4px solid transparent',
                borderBottom: `9px solid ${v.status === 'anchored' ? '#8A8A8A' : '#4FC3F7'}`,
                transform: `rotate(${v.heading}deg)`,
                opacity: 0.85,
                cursor: 'default',
                filter: `drop-shadow(0 0 3px ${v.status === 'anchored' ? '#8A8A8A60' : '#4FC3F760'})`,
              }}
            />
          </Marker>
        ))}

        {/* Saved geofence flash — visible for 3s after save */}
        {savedPolygon && savedPolygon.length >= 4 && (
          <Source id="saved-geofence" type="geojson" data={{ type: 'Feature', geometry: { type: 'Polygon', coordinates: [savedPolygon] }, properties: {} }}>
            <Layer id="saved-geofence-fill" type="fill" paint={{ 'fill-color': '#FFB300', 'fill-opacity': 0.18 }} />
            <Layer id="saved-geofence-outline" type="line" paint={{ 'line-color': '#FFB300', 'line-width': 2 }} />
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
                if (geofenceDrawMode) return;
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
                  cursor: geofenceDrawMode ? 'crosshair' : 'pointer',
                  boxShadow: `0 0 6px ${markerColor(entity)}80`,
                  opacity: markerOpacity(entity.last_seen),
                  transition: 'transform 0.1s, opacity 0.3s',
                }}
                onMouseEnter={(e) => { if (!geofenceDrawMode) (e.currentTarget as HTMLDivElement).style.transform = 'scale(1.6)'; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLDivElement).style.transform = 'scale(1)'; }}
              />
            </Marker>
          ) : null
        )}
      </Map>

      {/* Geofence draw mode — top banner so it's impossible to miss */}
      {geofenceDrawMode && (
        <div
          className="absolute top-0 left-0 right-0 flex items-center justify-between px-4 py-2"
          style={{
            background: 'color-mix(in srgb, var(--warning) 12%, var(--background-panel))',
            borderBottom: '1px solid color-mix(in srgb, var(--warning) 40%, transparent)',
            zIndex: 20,
          }}
        >
          <div className="flex items-center gap-3">
            <span
              className="text-[11px] font-bold tracking-widest"
              style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--warning)' }}
            >
              ⬡ DRAWING GEOFENCE
            </span>
            <span
              className="text-[10px]"
              style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'color-mix(in srgb, var(--warning) 70%, transparent)' }}
            >
              {vertices.length < 3
                ? `Click to add points — ${vertices.length}/3 minimum`
                : `${vertices.length} points — right-click or [Enter] to finish`}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={geoName}
              onChange={(e) => setGeoName(e.target.value)}
              placeholder="Name (optional)"
              className="text-[10px] px-2 py-1 outline-none"
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                background: 'rgba(0,0,0,0.4)',
                border: '1px solid color-mix(in srgb, var(--warning) 30%, transparent)',
                color: 'var(--warning)',
                width: '140px',
              }}
            />
            {vertices.length >= 3 && (
              <button
                onClick={() => finishPolygon(vertices, geoName)}
                className="text-[10px] px-3 py-1 tracking-wider"
                style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  color: 'var(--background)', background: 'var(--warning)', border: 'none', cursor: 'pointer',
                }}
              >
                SAVE
              </button>
            )}
            <button
              onClick={() => { setVertices([]); setGeoName(''); onGeofenceDrawEnd?.(); }}
              className="text-[10px] px-3 py-1 tracking-wider"
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                color: 'var(--critical)',
                background: 'transparent',
                border: '1px solid color-mix(in srgb, var(--critical) 40%, transparent)',
                cursor: 'pointer',
              }}
            >
              CANCEL [ESC]
            </button>
          </div>
        </div>
      )}

      {/* Save feedback */}
      {savedFeedback && (
        <div
          className="absolute bottom-8 right-3 text-[10px] px-2 py-1"
          style={{
            zIndex: 10,
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: savedFeedback.startsWith('SAVE FAILED') ? 'var(--critical)' : 'var(--accent)',
            background: 'var(--background-panel)',
            border: `1px solid ${savedFeedback.startsWith('SAVE FAILED') ? 'color-mix(in srgb, var(--critical) 40%, transparent)' : 'var(--accent-30)'}`,
          }}
        >
          {savedFeedback}
        </div>
      )}
    </div>
  );
}
