'use client';

import { useState, useCallback } from 'react';
import type { WaypointPreview } from '@/lib/api';

export interface MissionDrawState {
  missionDrawMode: boolean;
  missionPolygon: { lat: number; lon: number }[] | null;
  missionWaypoints: WaypointPreview[];
  setMissionDrawMode: (on: boolean) => void;
  setMissionPolygon: (coords: { lat: number; lon: number }[] | null) => void;
  setMissionWaypoints: (wps: WaypointPreview[]) => void;
  resetMissionDraw: () => void;
}

export function useMissionDraw(): MissionDrawState {
  const [missionDrawMode, setMissionDrawMode] = useState(false);
  const [missionPolygon, setMissionPolygon] = useState<{ lat: number; lon: number }[] | null>(null);
  const [missionWaypoints, setMissionWaypoints] = useState<WaypointPreview[]>([]);

  const resetMissionDraw = useCallback(() => {
    setMissionDrawMode(false);
    setMissionPolygon(null);
    setMissionWaypoints([]);
  }, []);

  return {
    missionDrawMode,
    missionPolygon,
    missionWaypoints,
    setMissionDrawMode,
    setMissionPolygon,
    setMissionWaypoints,
    resetMissionDraw,
  };
}
