'use client';

import React, { useState, useEffect, useRef } from 'react';
import Map, { Marker, NavigationControl, ScaleControl, MapRef } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useEntityStream, EntityData } from '@/hooks/useEntityStream';

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

export default function OpsMapView({ onSelectEntity, flyToLocation, alertEntityIds }: OpsMapViewProps) {
  const { entityList } = useEntityStream();
  const mapRef = useRef<MapRef>(null);

  // Fly to location when prop changes
  useEffect(() => {
    if (!flyToLocation || !mapRef.current) return;
    mapRef.current.flyTo({
      center: [flyToLocation.lon, flyToLocation.lat],
      zoom: 13,
      duration: 800,
    });
  }, [flyToLocation]);

  return (
    <div className="w-full h-full relative">
      <Map
        ref={mapRef}
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

        {entityList.map((entity) =>
          entity.position ? (
            <Marker
              key={entity.entity_id}
              longitude={entity.position.lon}
              latitude={entity.position.lat}
              onClick={(e) => {
                e.originalEvent.stopPropagation();
                onSelectEntity?.(entity);
              }}
            >
              <div
                title={entity.callsign || entity.entity_id}
                className={alertEntityIds?.has(entity.entity_id) ? 'alert-pulse' : undefined}
                style={{
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  background: markerColor(entity),
                  border: '2px solid rgba(8,12,10,0.8)',
                  cursor: 'pointer',
                  boxShadow: `0 0 6px ${markerColor(entity)}80`,
                  transition: 'transform 0.1s',
                }}
                onMouseEnter={(e) => ((e.currentTarget as HTMLDivElement).style.transform = 'scale(1.6)')}
                onMouseLeave={(e) => ((e.currentTarget as HTMLDivElement).style.transform = 'scale(1)')}
              />
            </Marker>
          ) : null
        )}
      </Map>
    </div>
  );
}
