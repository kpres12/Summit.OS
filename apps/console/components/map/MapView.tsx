'use client'

import { useEffect, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { MapControls } from './MapControls'
import { AlertMarker } from './AlertMarker'
import { DeviceMarker } from './DeviceMarker'
import { MissionLayer } from './MissionLayer'

interface MapViewProps {
  alerts: any[]
  telemetry: any[]
  missions: any[]
}

export function MapView({ alerts, telemetry, missions }: MapViewProps) {
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<maplibregl.Map | null>(null)
  const [mapLoaded, setMapLoaded] = useState(false)

  useEffect(() => {
    if (mapContainer.current && !map.current) {
      map.current = new maplibregl.Map({
        container: mapContainer.current,
        style: {
          version: 8,
          sources: {
            'raster-tiles': {
              type: 'raster',
              tiles: [
                'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
              ],
              tileSize: 256,
              attribution: '© Esri'
            }
          },
          layers: [
            {
              id: 'background',
              type: 'raster',
              source: 'raster-tiles'
            }
          ]
        },
        center: [-122.4194, 37.7749], // San Francisco
        zoom: 10,
        pitch: 0,
        bearing: 0
      })

      map.current.on('load', () => {
        setMapLoaded(true)
      })

      // Add navigation controls
      map.current.addControl(new maplibregl.NavigationControl(), 'top-right')
      map.current.addControl(new maplibregl.FullscreenControl(), 'top-right')
    }

    return () => {
      if (map.current) {
        map.current.remove()
        map.current = null
      }
    }
  }, [])

  return (
    <div className="relative w-full h-full">
      <div ref={mapContainer} className="w-full h-full" />
      
      {mapLoaded && map.current && (
        <>
          <MapControls map={map.current} />
          <AlertMarker map={map.current} alerts={alerts} />
          <DeviceMarker map={map.current} telemetry={telemetry} />
          <MissionLayer map={map.current} missions={missions} />
        </>
      )}
    </div>
  )
}
