'use client';

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import MapGL, { Marker, NavigationControl, ScaleControl, MapRef, Source, Layer, MapLayerMouseEvent } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useEntityStream, EntityData } from '@/hooks/useEntityStream';
import { apiFetch, createGeofence } from '@/lib/api';

interface SatellitePos { entity_id: string; name: string; sat_type: string; position: { lat: number; lon: number; alt: number }; }
interface JamZone { id: string; name: string; lat: number; lon: number; radius_km: number; intensity: number; }
interface Vessel { id: string; name: string; type: string; lat: number; lon: number; heading: number; speed_kts: number; status?: string; }
interface NoFlyZone { id: string; name: string; severity: string; active: boolean; coordinates: { lat: number; lon: number }[]; }

const DARK_STYLE =
  process.env.NEXT_PUBLIC_TILE_URL ||
  'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

const SATELLITE_STYLE = {
  version: 8 as const,
  sources: {
    'esri-satellite': {
      type: 'raster' as const,
      tiles: ['https://server.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
      tileSize: 256,
      attribution: 'Esri, Maxar, GeoEye, Earthstar Geographics',
    },
    'esri-labels': {
      type: 'raster' as const,
      tiles: ['https://server.arcgisonline.com/arcgis/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}'],
      tileSize: 256,
    },
  },
  layers: [
    { id: 'satellite-layer', type: 'raster' as const, source: 'esri-satellite' },
    { id: 'labels-layer', type: 'raster' as const, source: 'esri-labels', paint: { 'raster-opacity': 0.6 } },
  ],
};

export interface RouteOverlay {
  entityId: string;
  targetLat: number;
  targetLon: number;
  taskType: string;
  distanceKm?: number;
  etaMinutes?: number;
}

interface OpsMapViewProps {
  onSelectEntity?: (entity: EntityData | null) => void;
  flyToLocation?: { lat: number; lon: number } | null;
  alertEntityIds?: Set<string>;
  selectedEntityId?: string | null;
  missionDrawMode?: boolean;
  onMissionArea?: (coords: { lat: number; lon: number }[]) => void;
  missionWaypoints?: { lat: number; lon: number; alt: number }[];
  geofenceDrawMode?: boolean;
  onGeofenceDrawEnd?: () => void;
  showSatellites?: boolean;
  showGpsJam?: boolean;
  showMaritime?: boolean;
  showNoFlyZones?: boolean;
  showGrid?: boolean;
  taskRoutes?: RouteOverlay[];
}

// Hex colors for GeoJSON expressions
function markerColorHex(e: EntityData): string {
  switch (e.entity_type) {
    case 'active':  return '#00FF9C';
    case 'alert':   return '#FF3B3B';
    case 'neutral': return '#4FC3F7';
    default:        return '#FFB300';
  }
}

function markerColor(e: EntityData): string {
  switch (e.entity_type) {
    case 'active':  return 'var(--accent)';
    case 'alert':   return 'var(--critical)';
    case 'neutral': return 'var(--text-dim)';
    default:        return 'var(--warning)';
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

// Asset type → icon shape
function AssetIcon({ assetType, color, heading, size = 16 }: {
  assetType: string; color: string; heading?: number | null; size?: number;
}) {
  const type = (assetType || '').toUpperCase();
  const hasHeading = heading != null && !isNaN(heading);

  return (
    <svg
      width={size} height={size}
      viewBox="-8 -8 16 16"
      style={{ overflow: 'visible', display: 'block' }}
    >
      {/* Heading line */}
      {hasHeading && (
        <line
          x1={0} y1={0} x2={0} y2={-14}
          stroke={color} strokeWidth={1.5} opacity={0.8}
          transform={`rotate(${heading})`}
          strokeLinecap="round"
        />
      )}
      {/* Shape by asset type */}
      {(type.includes('AIRCRAFT') || type.includes('DRONE') || type.includes('AUV')) ? (
        <polygon points="0,-5 4,2 0,0 -4,2" fill={color} opacity={0.95} />
      ) : (type.includes('VESSEL') || type.includes('SHIP') || type.includes('CARGO') || type.includes('TANKER')) ? (
        <polygon points="0,-5 5,4 0,2 -5,4" fill={color} opacity={0.95} />
      ) : (type.includes('UGV') || type.includes('VEHICLE') || type.includes('GROUND')) ? (
        <rect x={-4} y={-4} width={8} height={8} fill={color} opacity={0.95} />
      ) : (type.includes('SENSOR') || type.includes('TOWER') || type.includes('NODE')) ? (
        <>
          <circle cx={0} cy={0} r={4} fill="none" stroke={color} strokeWidth={1.5} opacity={0.9} />
          <line x1={-4} y1={0} x2={4} y2={0} stroke={color} strokeWidth={1} opacity={0.9} />
          <line x1={0} y1={-4} x2={0} y2={4} stroke={color} strokeWidth={1} opacity={0.9} />
        </>
      ) : (
        <circle cx={0} cy={0} r={4} fill={color} opacity={0.95} />
      )}
    </svg>
  );
}

const TRAIL_MAX = 60;

function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function buildGridGeoJSON(bounds: { minLon: number; maxLon: number; minLat: number; maxLat: number }, step: number) {
  const features: object[] = [];
  const lonStart = Math.floor(bounds.minLon / step) * step;
  const latStart = Math.floor(bounds.minLat / step) * step;
  for (let lon = lonStart; lon <= bounds.maxLon + step; lon += step) {
    features.push({ type: 'Feature', geometry: { type: 'LineString', coordinates: [[lon, bounds.minLat - 2], [lon, bounds.maxLat + 2]] }, properties: {} });
  }
  for (let lat = latStart; lat <= bounds.maxLat + step; lat += step) {
    features.push({ type: 'Feature', geometry: { type: 'LineString', coordinates: [[bounds.minLon - 2, lat], [bounds.maxLon + 2, lat]] }, properties: {} });
  }
  return { type: 'FeatureCollection', features };
}

export default function OpsMapView({
  onSelectEntity, flyToLocation, alertEntityIds, selectedEntityId,
  missionDrawMode, onMissionArea, missionWaypoints,
  geofenceDrawMode, onGeofenceDrawEnd,
  showSatellites, showGpsJam, showMaritime, showNoFlyZones,
  showGrid, taskRoutes,
}: OpsMapViewProps) {
  const { entityList } = useEntityStream();
  const mapRef = useRef<MapRef>(null);

  // Basemap
  const [basemap, setBasemap] = useState<'dark' | 'satellite'>('dark');

  // Entity type filters — all visible by default
  const ALL_ASSET_TYPES = ['DRONE', 'AIRCRAFT', 'UGV', 'VESSEL', 'SENSOR', 'TOWER'] as const;
  const [hiddenAssetTypes, setHiddenAssetTypes] = useState<Set<string>>(new Set());
  const toggleAssetType = (type: string) => {
    setHiddenAssetTypes(prev => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type); else next.add(type);
      return next;
    });
  };

  // Geofence draw
  const [vertices, setVertices] = useState<DrawVertex[]>([]);
  const [geoName, setGeoName] = useState('');
  const [cursorPos, setCursorPos] = useState<DrawVertex | null>(null);
  const [savedFeedback, setSavedFeedback] = useState<string | null>(null);
  const [savedPolygon, setSavedPolygon] = useState<[number, number][] | null>(null);

  // Mission draw
  const [missionVertices, setMissionVertices] = useState<DrawVertex[]>([]);

  // Cursor coordinates display
  const [mapCursor, setMapCursor] = useState<{ lat: number; lon: number } | null>(null);

  // Entity trails — rolling buffer, stored in ref to avoid re-renders, versioned for memo
  const entityTrailsRef = useRef<Map<string, [number, number][]>>(new Map());
  const [trailTick, setTrailTick] = useState(0);

  // OSINT layer data
  const [satellites, setSatellites] = useState<SatellitePos[]>([]);
  const [jamZones, setJamZones] = useState<JamZone[]>([]);
  const [vessels, setVessels] = useState<Vessel[]>([]);
  const [noFlyZones, setNoFlyZones] = useState<NoFlyZone[]>([]);

  // Selected entity trail (from API history)
  const [selectedTrailCoords, setSelectedTrailCoords] = useState<[number, number][]>([]);

  // Operational grid bounds (updated on map move)
  const [gridBounds, setGridBounds] = useState<{ minLon: number; maxLon: number; minLat: number; maxLat: number } | null>(null);

  const handleMapMoveEnd = useCallback(() => {
    if (!mapRef.current || !showGrid) return;
    const b = mapRef.current.getBounds();
    if (b) setGridBounds({ minLon: b.getWest(), maxLon: b.getEast(), minLat: b.getSouth(), maxLat: b.getNorth() });
  }, [showGrid]);

  // Initialise grid bounds when showGrid turns on
  useEffect(() => {
    if (showGrid && mapRef.current) {
      const b = mapRef.current.getBounds();
      if (b) setGridBounds({ minLon: b.getWest(), maxLon: b.getEast(), minLat: b.getSouth(), maxLat: b.getNorth() });
    } else if (!showGrid) {
      setGridBounds(null);
    }
  }, [showGrid]);

  // --- Update rolling trails on every entity position change ---
  useEffect(() => {
    let dirty = false;
    for (const e of entityList) {
      if (!e.position) continue;
      const pt: [number, number] = [e.position.lon, e.position.lat];
      const hist = entityTrailsRef.current.get(e.entity_id) ?? [];
      const prev = hist.at(-1);
      if (prev && Math.abs(prev[0] - pt[0]) < 1e-6 && Math.abs(prev[1] - pt[1]) < 1e-6) continue;
      entityTrailsRef.current.set(e.entity_id, [...hist.slice(-(TRAIL_MAX - 1)), pt]);
      dirty = true;
    }
    if (dirty) setTrailTick(t => t + 1);
  }, [entityList]);

  // Build GeoJSON FeatureCollection for all entity trails
  const allTrailsGeoJSON = useMemo(() => {
    const entityMap = new Map(entityList.map(e => [e.entity_id, e]));
    const features: object[] = [];
    for (const [entityId, coords] of entityTrailsRef.current) {
      if (coords.length < 2) continue;
      const entity = entityMap.get(entityId);
      const color = entity ? markerColorHex(entity) : '#4FC3F7';
      const isSelected = entityId === selectedEntityId;
      features.push({
        type: 'Feature',
        geometry: { type: 'LineString', coordinates: coords },
        properties: { color, width: isSelected ? 2.5 : 1.2, opacity: isSelected ? 0.85 : 0.4 },
      });
    }
    return { type: 'FeatureCollection', features };
  }, [trailTick, selectedEntityId, entityList]); // eslint-disable-line react-hooks/exhaustive-deps

  // Fly to location
  useEffect(() => {
    if (!flyToLocation || !mapRef.current) return;
    mapRef.current.flyTo({ center: [flyToLocation.lon, flyToLocation.lat], zoom: 13, duration: 800 });
  }, [flyToLocation]);

  // Clear mission vertices when draw mode off
  useEffect(() => {
    if (!missionDrawMode) setMissionVertices([]);
  }, [missionDrawMode]);

  // Reset geofence draw state when mode off
  useEffect(() => {
    if (!geofenceDrawMode) { setVertices([]); setGeoName(''); setCursorPos(null); }
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

  useEffect(() => {
    if (!geofenceDrawMode) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { setVertices([]); setGeoName(''); setCursorPos(null); onGeofenceDrawEnd?.(); }
      if (e.key === 'Enter' && verticesRef.current.length >= 3) finishPolygon(verticesRef.current, geoNameRef.current);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [geofenceDrawMode, onGeofenceDrawEnd, finishPolygon]);

  // OSINT fetchers
  useEffect(() => {
    if (!showSatellites) { setSatellites([]); return; }
    const fn = () => apiFetch('/v1/satellites').then(r => r.json()).then(d => setSatellites(d.satellites || [])).catch(() => {});
    fn(); const t = setInterval(fn, 30000); return () => clearInterval(t);
  }, [showSatellites]);

  useEffect(() => {
    if (!showGpsJam) { setJamZones([]); return; }
    const fn = () => apiFetch('/v1/gpsjam').then(r => r.json()).then(d => setJamZones(d.zones || [])).catch(() => {});
    fn(); const t = setInterval(fn, 30000); return () => clearInterval(t);
  }, [showGpsJam]);

  useEffect(() => {
    if (!showMaritime) { setVessels([]); return; }
    const fn = () => apiFetch('/v1/maritime').then(r => r.json()).then(d => setVessels(d.vessels || [])).catch(() => {});
    fn(); const t = setInterval(fn, 30000); return () => clearInterval(t);
  }, [showMaritime]);

  useEffect(() => {
    if (!showNoFlyZones) { setNoFlyZones([]); return; }
    const fn = () => apiFetch('/v1/noflyzones').then(r => r.json()).then(d => setNoFlyZones(d.zones || [])).catch(() => {});
    fn(); const t = setInterval(fn, 30000); return () => clearInterval(t);
  }, [showNoFlyZones]);

  // Fetch position trail for selected entity from API
  useEffect(() => {
    if (!selectedEntityId) { setSelectedTrailCoords([]); return; }
    let cancelled = false;
    apiFetch(`/v1/entities/${selectedEntityId}/trail`)
      .then(res => res.json())
      .then((data: { trail: { lat: number; lon: number }[] }) => {
        if (!cancelled && data?.trail) setSelectedTrailCoords(data.trail.map(p => [p.lon, p.lat]));
      })
      .catch(() => setSelectedTrailCoords([]));
    return () => { cancelled = true; };
  }, [selectedEntityId]);

  const handleMapClick = useCallback((e: MapLayerMouseEvent) => {
    if (missionDrawMode) { setMissionVertices(prev => [...prev, { lat: e.lngLat.lat, lon: e.lngLat.lng }]); return; }
    if (!geofenceDrawMode) return;
    setVertices(prev => [...prev, { lat: e.lngLat.lat, lon: e.lngLat.lng }]);
  }, [geofenceDrawMode, missionDrawMode]);

  const handleMapDblClick = useCallback((_evt: MapLayerMouseEvent) => {
    if (missionDrawMode && missionVertices.length >= 3) {
      onMissionArea?.(missionVertices.map(v => ({ lat: v.lat, lon: v.lon })));
      setMissionVertices([]);
    }
  }, [missionDrawMode, missionVertices, onMissionArea]);

  const handleMouseMove = useCallback((e: MapLayerMouseEvent) => {
    setMapCursor({ lat: e.lngLat.lat, lon: e.lngLat.lng });
    if (geofenceDrawMode) setCursorPos({ lat: e.lngLat.lat, lon: e.lngLat.lng });
  }, [geofenceDrawMode]);

  const handleContextMenu = useCallback((e: MapLayerMouseEvent) => {
    if (geofenceDrawMode && verticesRef.current.length >= 3) {
      e.originalEvent.preventDefault();
      finishPolygon(verticesRef.current, geoNameRef.current);
    }
  }, [geofenceDrawMode, finishPolygon]);

  // In-progress geofence polygon preview
  const drawPoints: [number, number][] = vertices.map((v): [number, number] => [v.lon, v.lat]);
  if (cursorPos) drawPoints.push([cursorPos.lon, cursorPos.lat]);
  if (drawPoints.length >= 3) drawPoints.push(drawPoints[0]);
  const drawGeoJSON = drawPoints.length >= 2 ? {
    type: 'Feature' as const,
    geometry: { type: 'LineString' as const, coordinates: drawPoints },
    properties: {},
  } : null;

  const mapStyle = basemap === 'satellite' ? SATELLITE_STYLE : DARK_STYLE;

  return (
    <div className="w-full h-full relative">
      <MapGL
        ref={mapRef}
        initialViewState={{ longitude: -98.5, latitude: 39.8, zoom: 4, pitch: 45, bearing: 0 }}
        style={{ width: '100%', height: '100%' }}
        mapStyle={mapStyle as string}
        cursor={geofenceDrawMode || missionDrawMode ? 'crosshair' : 'grab'}
        doubleClickZoom={!geofenceDrawMode && !missionDrawMode}
        onClick={handleMapClick}
        onDblClick={handleMapDblClick}
        onMouseMove={handleMouseMove}
        onContextMenu={handleContextMenu}
        onMoveEnd={handleMapMoveEnd}
      >
        <NavigationControl position="top-right" />
        <ScaleControl position="bottom-left" />

        {/* === All entity trails (rolling buffer) === */}
        {(allTrailsGeoJSON.features as object[]).length > 0 && (
          <Source id="entity-trails-all" type="geojson" data={allTrailsGeoJSON as GeoJSON.FeatureCollection}>
            <Layer
              id="entity-trails-all-layer"
              type="line"
              paint={{
                'line-color': ['get', 'color'],
                'line-width': ['get', 'width'],
                'line-opacity': ['get', 'opacity'],
              }}
              layout={{ 'line-cap': 'round', 'line-join': 'round' }}
            />
          </Source>
        )}

        {/* API history trail for selected entity */}
        {selectedTrailCoords.length >= 2 && (
          <Source id="entity-trail-history" type="geojson"
            data={{ type: 'Feature', geometry: { type: 'LineString', coordinates: selectedTrailCoords }, properties: {} }}
          >
            <Layer id="entity-trail-history-layer" type="line"
              paint={{ 'line-color': 'var(--accent)', 'line-width': 2, 'line-opacity': 0.5, 'line-dasharray': [4, 2] }}
            />
          </Source>
        )}

        {/* Mission area draw */}
        {missionDrawMode && missionVertices.length >= 2 && (() => {
          const coords = missionVertices.map((v): [number, number] => [v.lon, v.lat]);
          if (missionVertices.length >= 3) coords.push(coords[0]);
          return (
            <Source id="mission-draw-line" type="geojson"
              data={{ type: 'Feature', geometry: { type: 'LineString', coordinates: coords }, properties: {} }}>
              <Layer id="mission-draw-line-layer" type="line"
                paint={{ 'line-color': '#4FC3F7', 'line-width': 2, 'line-dasharray': [4, 2] }} />
            </Source>
          );
        })()}
        {missionDrawMode && missionVertices.map((v, i) => (
          <Marker key={`mv-${i}`} longitude={v.lon} latitude={v.lat}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%',
              background: i === 0 ? 'var(--color-active)' : 'color-mix(in srgb, var(--color-active) 50%, transparent)',
              border: '1px solid var(--color-active)' }} />
          </Marker>
        ))}

        {/* Mission waypoints */}
        {missionWaypoints && missionWaypoints.length > 0 && (
          <Source id="mission-waypoints" type="geojson" data={{
            type: 'FeatureCollection' as const,
            features: missionWaypoints.map(wp => ({
              type: 'Feature' as const,
              geometry: { type: 'Point' as const, coordinates: [wp.lon, wp.lat] },
              properties: { alt: wp.alt },
            })),
          }}>
            <Layer id="mission-waypoints-layer" type="circle"
              paint={{ 'circle-radius': 3, 'circle-color': '#4FC3F7', 'circle-opacity': 0.6 }} />
          </Source>
        )}

        {/* Geofence draw preview */}
        {drawGeoJSON && (
          <Source id="draw-line" type="geojson" data={drawGeoJSON}>
            <Layer id="draw-line-layer" type="line"
              paint={{ 'line-color': '#FFB300', 'line-width': 2, 'line-dasharray': [4, 2] }} />
          </Source>
        )}
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

        {/* GPS jamming zones */}
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

        {/* No-fly zones */}
        {showNoFlyZones && noFlyZones.length > 0 && (() => {
          const activeZones = noFlyZones.filter(z => z.active && z.coordinates?.length >= 3);
          if (!activeZones.length) return null;
          return (
            <Source id="noflyzones" type="geojson" data={{
              type: 'FeatureCollection' as const,
              features: activeZones.map(z => ({
                type: 'Feature' as const,
                geometry: { type: 'Polygon' as const, coordinates: [[
                  ...z.coordinates.map((c): [number, number] => [c.lon, c.lat]),
                  [z.coordinates[0].lon, z.coordinates[0].lat] as [number, number],
                ]] },
                properties: { severity: z.severity },
              })),
            }}>
              <Layer id="noflyzones-fill" type="fill" paint={{ 'fill-color': '#FF3B3B', 'fill-opacity': 0.12 }} />
              <Layer id="noflyzones-outline" type="line" paint={{ 'line-color': '#FF3B3B', 'line-width': 1.5, 'line-dasharray': [6, 3] }} />
            </Source>
          );
        })()}

        {/* Satellites */}
        {showSatellites && satellites.map(sat => (
          <Marker key={sat.entity_id} longitude={sat.position.lon} latitude={sat.position.lat}>
            <div title={`${sat.name} · ${sat.sat_type.toUpperCase()}`} style={{
              width: '6px', height: '6px', borderRadius: '50%',
              background: SAT_COLOR[sat.sat_type] || '#4FC3F7',
              border: `1px solid ${SAT_COLOR[sat.sat_type] || '#4FC3F7'}`,
              boxShadow: `0 0 5px ${SAT_COLOR[sat.sat_type] || '#4FC3F7'}50`,
              opacity: 0.75, cursor: 'default',
            }} />
          </Marker>
        ))}

        {/* Maritime vessels */}
        {showMaritime && vessels.map(v => (
          <Marker key={v.id} longitude={v.lon} latitude={v.lat}>
            <div title={`${v.name} · ${v.type.toUpperCase()} · ${v.speed_kts.toFixed(1)} kts`} style={{
              width: 0, height: 0,
              borderLeft: '4px solid transparent', borderRight: '4px solid transparent',
              borderBottom: `9px solid ${v.status === 'anchored' ? '#8A8A8A' : '#4FC3F7'}`,
              transform: `rotate(${v.heading}deg)`, opacity: 0.85, cursor: 'default',
              filter: `drop-shadow(0 0 3px ${v.status === 'anchored' ? '#8A8A8A60' : '#4FC3F760'})`,
            }} />
          </Marker>
        ))}

        {/* === Operational grid === */}
        {showGrid && gridBounds && (() => {
          const zoom = mapRef.current?.getZoom() ?? 5;
          const step = zoom < 5 ? 5 : zoom < 7 ? 2 : zoom < 10 ? 1 : zoom < 13 ? 0.25 : 0.05;
          const geo = buildGridGeoJSON(gridBounds, step);
          return (
            <Source id="op-grid" type="geojson" data={geo as GeoJSON.FeatureCollection}>
              <Layer id="op-grid-layer" type="line"
                paint={{ 'line-color': '#FF4444', 'line-width': 0.5, 'line-opacity': 0.35, 'line-dasharray': [4, 4] }}
              />
            </Source>
          );
        })()}

        {/* === Route ETA overlays === */}
        {taskRoutes && taskRoutes.length > 0 && (() => {
          const entityMap = new Map(entityList.map(e => [e.entity_id, e]));
          const lines: object[] = [];
          const labels: { lon: number; lat: number; text: string }[] = [];

          for (const route of taskRoutes) {
            const entity = entityMap.get(route.entityId);
            if (!entity?.position) continue;
            const { lat: eLat, lon: eLon } = entity.position;
            const midLat = (eLat + route.targetLat) / 2;
            const midLon = (eLon + route.targetLon) / 2;
            const distKm = route.distanceKm ?? haversineKm(eLat, eLon, route.targetLat, route.targetLon);
            const speedMs = entity.speed_mps || 15;
            const etaMins = route.etaMinutes ?? Math.round((distKm * 1000) / speedMs / 60);

            lines.push({
              type: 'Feature',
              geometry: { type: 'LineString', coordinates: [[eLon, eLat], [route.targetLon, route.targetLat]] },
              properties: {},
            });
            labels.push({
              lon: midLon, lat: midLat,
              text: `${distKm.toFixed(1)} km  ${etaMins} min`,
            });
          }

          return (
            <>
              <Source id="task-routes" type="geojson"
                data={{ type: 'FeatureCollection', features: lines } as GeoJSON.FeatureCollection}>
                <Layer id="task-routes-layer" type="line"
                  paint={{ 'line-color': '#FFB300', 'line-width': 1.5, 'line-opacity': 0.8, 'line-dasharray': [6, 3] }}
                />
              </Source>
              {labels.map((l, i) => (
                <Marker key={`eta-${i}`} longitude={l.lon} latitude={l.lat} anchor="center">
                  <div style={{
                    background: 'rgba(8,12,10,0.9)',
                    border: '1px solid #FFB30080',
                    color: '#FFB300',
                    fontFamily: 'var(--font-ibm-plex-mono), monospace',
                    fontSize: '9px',
                    padding: '2px 6px',
                    whiteSpace: 'nowrap',
                    letterSpacing: '0.06em',
                    pointerEvents: 'none',
                  }}>
                    {l.text}
                  </div>
                </Marker>
              ))}
            </>
          );
        })()}

        {/* Saved geofence flash */}
        {savedPolygon && savedPolygon.length >= 4 && (
          <Source id="saved-geofence" type="geojson"
            data={{ type: 'Feature', geometry: { type: 'Polygon', coordinates: [savedPolygon] }, properties: {} }}>
            <Layer id="saved-geofence-fill" type="fill" paint={{ 'fill-color': '#FFB300', 'fill-opacity': 0.18 }} />
            <Layer id="saved-geofence-outline" type="line" paint={{ 'line-color': '#FFB300', 'line-width': 2 }} />
          </Source>
        )}

        {/* === Entity markers — Lattice-style icons + labels === */}
        {entityList.filter(entity => {
          const atype = ((entity.properties?.asset_type as string) || '').toUpperCase();
          if (hiddenAssetTypes.size === 0) return true;
          // Map to one of the known filter categories
          const cat = ALL_ASSET_TYPES.find(t => atype.includes(t)) ?? 'OTHER';
          return !hiddenAssetTypes.has(cat);
        }).map((entity) => {
          if (!entity.position) return null;
          const color = markerColor(entity);
          const colorHex = markerColorHex(entity);
          const opacity = markerOpacity(entity.last_seen);
          const isSelected = entity.entity_id === selectedEntityId;
          const isPulsing = alertEntityIds?.has(entity.entity_id);
          const assetType = (entity.properties?.asset_type as string) || entity.entity_type;
          const heading = entity.position.heading_deg ?? (entity.properties?.heading as number | null) ?? null;
          const altM = entity.position.alt;
          const altFt = altM != null ? Math.round(altM * 3.28084) : null;
          const callsign = (entity.callsign || entity.entity_id.slice(0, 8)).toUpperCase();

          return (
            <Marker
              key={entity.entity_id}
              longitude={entity.position.lon}
              latitude={entity.position.lat}
              anchor="center"
              onClick={(e) => {
                if (geofenceDrawMode) return;
                e.originalEvent.stopPropagation();
                onSelectEntity?.(entity);
              }}
            >
              <div
                className={isPulsing ? 'alert-pulse' : undefined}
                style={{
                  position: 'relative',
                  opacity,
                  cursor: geofenceDrawMode ? 'crosshair' : 'pointer',
                  transition: 'opacity 0.3s',
                  filter: isSelected ? `drop-shadow(0 0 8px ${colorHex})` : `drop-shadow(0 0 4px ${colorHex}60)`,
                }}
              >
                {/* Icon */}
                <AssetIcon
                  assetType={assetType}
                  color={colorHex}
                  heading={heading}
                  size={isSelected ? 18 : 14}
                />

                {/* Selected ring */}
                {isSelected && (
                  <div style={{
                    position: 'absolute',
                    inset: '-6px',
                    borderRadius: '50%',
                    border: `1px solid ${colorHex}`,
                    opacity: 0.6,
                    pointerEvents: 'none',
                  }} />
                )}

                {/* Label: [CALLSIGN] ALTft */}
                <div style={{
                  position: 'absolute',
                  top: '100%',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  marginTop: '3px',
                  background: 'rgba(8,12,10,0.88)',
                  border: `1px solid ${colorHex}50`,
                  color,
                  fontSize: '8px',
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  whiteSpace: 'nowrap',
                  padding: '1px 4px',
                  lineHeight: 1.4,
                  letterSpacing: '0.05em',
                  pointerEvents: 'none',
                }}>
                  [{callsign}]{altFt != null ? ` ${altFt}ft` : ''}
                </div>
              </div>
            </Marker>
          );
        })}
      </MapGL>

      {/* === Basemap toggle + entity type filters — top-left === */}
      <div
        className="absolute"
        style={{ top: '10px', left: '10px', zIndex: 10, display: 'flex', flexDirection: 'column', gap: '4px' }}
      >
        {/* Basemap row */}
        <div style={{ display: 'flex', gap: '2px' }}>
          {(['dark', 'satellite'] as const).map(mode => (
            <button
              key={mode}
              onClick={() => setBasemap(mode)}
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                fontSize: '9px',
                letterSpacing: '0.08em',
                padding: '3px 8px',
                cursor: 'pointer',
                border: `1px solid ${basemap === mode ? 'var(--accent)' : 'var(--border)'}`,
                background: basemap === mode ? 'color-mix(in srgb, var(--accent) 15%, var(--background-panel))' : 'var(--background-panel)',
                color: basemap === mode ? 'var(--accent)' : 'var(--text-muted)',
                transition: 'all 0.15s',
              }}
            >
              {mode.toUpperCase()}
            </button>
          ))}
        </div>

        {/* Entity type filter chips */}
        <div style={{ display: 'flex', gap: '2px', flexWrap: 'wrap', maxWidth: '220px' }}>
          {ALL_ASSET_TYPES.map(type => {
            const hidden = hiddenAssetTypes.has(type);
            const count = entityList.filter(e => {
              const atype = ((e.properties?.asset_type as string) || '').toUpperCase();
              return ALL_ASSET_TYPES.find(t => atype.includes(t)) === type || (!ALL_ASSET_TYPES.find(t => atype.includes(t)) && type === 'OTHER' as never);
            }).length;
            const typeColor: Record<string, string> = {
              DRONE: '#00FF9C', AIRCRAFT: '#4FC3F7', UGV: '#00FF9C',
              VESSEL: '#4FC3F7', SENSOR: '#8A9A8A', TOWER: '#8A9A8A',
            };
            const color = typeColor[type] ?? '#8A8A8A';
            return (
              <button
                key={type}
                onClick={() => toggleAssetType(type)}
                title={`${hidden ? 'Show' : 'Hide'} ${type}`}
                style={{
                  fontFamily: 'var(--font-ibm-plex-mono), monospace',
                  fontSize: '8px',
                  letterSpacing: '0.08em',
                  padding: '2px 5px',
                  cursor: 'pointer',
                  border: `1px solid ${hidden ? 'var(--border)' : `${color}60`}`,
                  background: hidden ? 'rgba(8,12,10,0.75)' : `color-mix(in srgb, ${color} 10%, rgba(8,12,10,0.85))`,
                  color: hidden ? 'var(--text-muted)' : color,
                  transition: 'all 0.15s',
                  opacity: hidden ? 0.5 : 1,
                }}
              >
                {type} {count > 0 && <span style={{ opacity: 0.7 }}>{count}</span>}
              </button>
            );
          })}
          {hiddenAssetTypes.size > 0 && (
            <button
              onClick={() => setHiddenAssetTypes(new Set())}
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                fontSize: '8px',
                letterSpacing: '0.08em',
                padding: '2px 5px',
                cursor: 'pointer',
                border: '1px solid var(--warning)',
                background: 'transparent',
                color: 'var(--warning)',
              }}
            >
              ALL
            </button>
          )}
        </div>
      </div>

      {/* === Cursor coordinate display — bottom right === */}
      {mapCursor && (
        <div
          style={{
            position: 'absolute',
            bottom: '28px',
            right: '10px',
            zIndex: 10,
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            fontSize: '9px',
            color: 'var(--text-muted)',
            background: 'rgba(8,12,10,0.75)',
            border: '1px solid var(--border)',
            padding: '2px 8px',
            letterSpacing: '0.06em',
            pointerEvents: 'none',
          }}
        >
          {mapCursor.lat.toFixed(4)}, {mapCursor.lon.toFixed(4)}
        </div>
      )}

      {/* Geofence draw mode banner */}
      {geofenceDrawMode && (
        <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-4 py-2"
          style={{
            background: 'color-mix(in srgb, var(--warning) 12%, var(--background-panel))',
            borderBottom: '1px solid color-mix(in srgb, var(--warning) 40%, transparent)',
            zIndex: 20,
          }}
        >
          <div className="flex items-center gap-3">
            <span className="text-[11px] font-bold tracking-widest"
              style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--warning)' }}>
              ⬡ DRAWING GEOFENCE
            </span>
            <span className="text-[10px]"
              style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'color-mix(in srgb, var(--warning) 70%, transparent)' }}>
              {vertices.length < 3
                ? `Click to add points — ${vertices.length}/3 minimum`
                : `${vertices.length} points — right-click or [Enter] to finish`}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <input type="text" value={geoName} onChange={e => setGeoName(e.target.value)}
              placeholder="Name (optional)" className="text-[10px] px-2 py-1 outline-none"
              style={{
                fontFamily: 'var(--font-ibm-plex-mono), monospace',
                background: 'rgba(0,0,0,0.4)', border: '1px solid color-mix(in srgb, var(--warning) 30%, transparent)',
                color: 'var(--warning)', width: '140px',
              }} />
            {vertices.length >= 3 && (
              <button onClick={() => finishPolygon(vertices, geoName)}
                className="text-[10px] px-3 py-1 tracking-wider"
                style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--background)', background: 'var(--warning)', border: 'none', cursor: 'pointer' }}>
                SAVE
              </button>
            )}
            <button onClick={() => { setVertices([]); setGeoName(''); onGeofenceDrawEnd?.(); }}
              className="text-[10px] px-3 py-1 tracking-wider"
              style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--critical)', background: 'transparent', border: '1px solid color-mix(in srgb, var(--critical) 40%, transparent)', cursor: 'pointer' }}>
              CANCEL [ESC]
            </button>
          </div>
        </div>
      )}

      {/* Save feedback */}
      {savedFeedback && (
        <div className="absolute bottom-8 right-3 text-[10px] px-2 py-1"
          style={{
            zIndex: 10, fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: savedFeedback.startsWith('SAVE FAILED') ? 'var(--critical)' : 'var(--accent)',
            background: 'var(--background-panel)',
            border: `1px solid ${savedFeedback.startsWith('SAVE FAILED') ? 'color-mix(in srgb, var(--critical) 40%, transparent)' : 'var(--accent-30)'}`,
          }}>
          {savedFeedback}
        </div>
      )}
    </div>
  );
}
