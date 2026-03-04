'use client';

import React, { useEffect, useRef } from 'react';
import { EntityData } from '../../hooks/useEntityStream';

/**
 * CesiumMap — 3D globe component using CesiumJS.
 *
 * Renders entities as billboards at their real 3D positions:
 * - Aircraft at barometric altitude
 * - Satellites at orbital altitude with orbit path polylines
 * - Ground entities clamped to terrain
 *
 * Loaded dynamically (client-only) to avoid SSR issues with Cesium.
 */

interface CesiumMapProps {
  entities: EntityData[];
  onSelectEntity: (entity: EntityData) => void;
  showEntities: boolean;
  showOrbits: boolean;
}

export default function CesiumMap({ entities, onSelectEntity, showEntities, showOrbits }: CesiumMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const viewerRef = useRef<Record<string, any> | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const entitySourceRef = useRef<Record<string, any> | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const orbitSourceRef = useRef<Record<string, any> | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const cesiumRef = useRef<Record<string, any> | null>(null);

  // Initialize Cesium viewer
  useEffect(() => {
    let mounted = true;

    async function init() {
      if (!containerRef.current || viewerRef.current) return;

      // Dynamic import to avoid SSR
      const Cesium = await import('cesium');
      if (!mounted) return;

      cesiumRef.current = Cesium;

      // Set base URL for Cesium assets
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as Record<string, any>).CESIUM_BASE_URL = '/cesium';

      // Configure Ion token if available
      const token = process.env.NEXT_PUBLIC_CESIUM_TOKEN;
      if (token) {
        Cesium.Ion.defaultAccessToken = token;
      }

      const viewer = new Cesium.Viewer(containerRef.current!, {
        // Dark minimal UI
        animation: false,
        timeline: false,
        baseLayerPicker: false,
        fullscreenButton: false,
        vrButton: false,
        geocoder: false,
        homeButton: false,
        infoBox: false,
        selectionIndicator: false,
        navigationHelpButton: false,
        sceneModePicker: false,
        projectionPicker: false,
        // Dark basemap
        baseLayer: new Cesium.ImageryLayer(
          new Cesium.UrlTemplateImageryProvider({
            url: 'https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
            credit: new Cesium.Credit('© CartoDB © OpenStreetMap'),
            minimumLevel: 0,
            maximumLevel: 18,
          })
        ),
        // Terrain
        terrain: undefined,
        // Performance
        requestRenderMode: false,
        maximumRenderTimeChange: Infinity,
        msaaSamples: 1,
      });

      // Dark sky / atmosphere
      viewer.scene.backgroundColor = Cesium.Color.fromCssColorString('#0A0A0A');
      viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString('#0F0F0F');
      if (viewer.scene.moon) viewer.scene.moon.show = true;
      if (viewer.scene.sun) viewer.scene.sun.show = true;

      // Enable lighting for day/night
      viewer.scene.globe.enableLighting = true;

      // Atmosphere
      if (viewer.scene.skyAtmosphere) {
        viewer.scene.skyAtmosphere.show = true;
      }

      // Remove Cesium credit display clutter
      const creditContainer = viewer.cesiumWidget.creditContainer as HTMLElement;
      if (creditContainer) {
        creditContainer.style.display = 'none';
      }

      // Default camera: CONUS overview
      viewer.camera.setView({
        destination: Cesium.Cartesian3.fromDegrees(-98.5, 39.8, 15000000),
      });

      // Entity data sources
      const entitySource = new Cesium.CustomDataSource('entities');
      const orbitSource = new Cesium.CustomDataSource('orbits');
      viewer.dataSources.add(entitySource);
      viewer.dataSources.add(orbitSource);

      // Click handler
      const handler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      handler.setInputAction((click: Record<string, any>) => {
        const picked = viewer.scene.pick(click.position);
        if (Cesium.defined(picked) && picked.id && picked.id._summitEntity) {
          onSelectEntity(picked.id._summitEntity);
        }
      }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

      viewerRef.current = viewer;
      entitySourceRef.current = entitySource;
      orbitSourceRef.current = orbitSource;
    }

    init();

    return () => {
      mounted = false;
      if (viewerRef.current && !viewerRef.current.isDestroyed()) {
        viewerRef.current.destroy();
        viewerRef.current = null;
      }
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Update entity markers
  useEffect(() => {
    const Cesium = cesiumRef.current;
    const entitySource = entitySourceRef.current;
    if (!Cesium || !entitySource) return;

    entitySource.entities.removeAll();

    if (!showEntities) return;

    for (const entity of entities) {
      if (!entity.position || !entity.position.lat || !entity.position.lon) continue;

      const isSatellite = entity.classification === 'satellite';
      const isAircraft = entity.classification === 'aircraft';
      const alt = entity.position.alt || 0;

      // Color by type
      let color = Cesium.Color.fromCssColorString('#a1a1aa'); // neutral gray
      const label = entity.callsign || entity.entity_id;
      let pixelSize = 6;

      if (isSatellite) {
        color = Cesium.Color.fromCssColorString('#818cf8'); // indigo for satellites
        pixelSize = 4;
      } else if (isAircraft) {
        color = Cesium.Color.fromCssColorString('#34d399'); // emerald for aircraft
        pixelSize = 5;
      } else {
        switch (entity.entity_type) {
          case 'friendly': color = Cesium.Color.fromCssColorString('#34d399'); break;
          case 'hostile': color = Cesium.Color.fromCssColorString('#ef4444'); break;
          case 'unknown': color = Cesium.Color.fromCssColorString('#fbbf24'); break;
        }
      }

      const cesiumEntity = entitySource.entities.add({
        position: Cesium.Cartesian3.fromDegrees(
          entity.position.lon,
          entity.position.lat,
          alt
        ),
        point: {
          pixelSize,
          color,
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 1,
          heightReference: alt > 1000 ? Cesium.HeightReference.NONE : Cesium.HeightReference.CLAMP_TO_GROUND,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        },
        label: {
          text: label.length > 12 ? label.slice(0, 12) : label,
          font: '10px monospace',
          fillColor: color.withAlpha(0.8),
          outlineColor: Cesium.Color.BLACK,
          outlineWidth: 2,
          style: Cesium.LabelStyle.FILL_AND_OUTLINE,
          pixelOffset: new Cesium.Cartesian2(0, -14),
          scale: 0.8,
          showBackground: false,
          heightReference: alt > 1000 ? Cesium.HeightReference.NONE : Cesium.HeightReference.CLAMP_TO_GROUND,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
          show: isSatellite || isAircraft, // only label aircraft/satellites
        },
      });

      // Attach entity data for click handler
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (cesiumEntity as Record<string, any>)._summitEntity = entity;
    }
  }, [entities, showEntities]);

  // Update orbit paths for satellites
  useEffect(() => {
    const Cesium = cesiumRef.current;
    const orbitSource = orbitSourceRef.current;
    if (!Cesium || !orbitSource) return;

    orbitSource.entities.removeAll();

    if (!showOrbits) return;

    const satellites = entities.filter(e => e.classification === 'satellite');

    for (const sat of satellites) {
      if (!sat.position) continue;

      // Generate a simple great-circle orbit approximation
      // Use the satellite's current position and speed to project forward
      const speed = sat.speed_mps || 7500; // default orbital speed
      const points: number[] = [];
      const lat0 = sat.position.lat * (Math.PI / 180);
      const lon0 = sat.position.lon * (Math.PI / 180);
      const alt = sat.position.alt || 400000;
      const heading = (sat.position.heading_deg || 0) * (Math.PI / 180);

      // Project ~45 min of orbit path (one half-orbit for LEO)
      const R = 6371000 + alt;
      const angularVelocity = speed / R; // rad/s

      for (let t = 0; t < 2700; t += 60) {
        const angle = angularVelocity * t;
        // Approximate: project along heading from starting point
        const lat = Math.asin(
          Math.sin(lat0) * Math.cos(angle) +
          Math.cos(lat0) * Math.sin(angle) * Math.cos(heading)
        );
        const lon = lon0 + Math.atan2(
          Math.sin(heading) * Math.sin(angle) * Math.cos(lat0),
          Math.cos(angle) - Math.sin(lat0) * Math.sin(lat)
        );
        points.push(lon * (180 / Math.PI), lat * (180 / Math.PI), alt);
      }

      if (points.length >= 6) {
        orbitSource.entities.add({
          polyline: {
            positions: Cesium.Cartesian3.fromDegreesArrayHeights(points),
            width: 1,
            material: new Cesium.PolylineGlowMaterialProperty({
              glowPower: 0.2,
              color: Cesium.Color.fromCssColorString('#818cf8').withAlpha(0.3),
            }),
          },
        });
      }
    }
  }, [entities, showOrbits]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full"
      style={{ background: '#0A0A0A' }}
    />
  );
}
