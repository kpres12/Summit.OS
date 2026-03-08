'use client';

import React, { useState } from 'react';
import Map, { Marker, NavigationControl, ScaleControl } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useEntityStream, EntityData } from '@/hooks/useEntityStream';

const DARK_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

const LAYER_IDS = ['entities', 'tracks', 'geofences'];

interface LayerState {
  entities: boolean;
  tracks: boolean;
  geofences: boolean;
}

interface OpsMapViewProps {
  onSelectEntity?: (entity: EntityData | null) => void;
  showLayers?: LayerState;
}

function markerColor(e: EntityData): string {
  switch (e.entity_type) {
    case 'friendly': return '#00FF9C';
    case 'hostile': return '#FF3B3B';
    case 'neutral': return 'rgba(200,230,201,0.6)';
    default: return '#FFB300';
  }
}

export default function OpsMapView({ onSelectEntity, showLayers }: OpsMapViewProps) {
  const { entityList } = useEntityStream();
  const layers = showLayers ?? { entities: true, tracks: true, geofences: true };

  return (
    <div className="w-full h-full relative">
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

        {layers.entities && entityList.map((entity) =>
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
