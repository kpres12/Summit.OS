'use client';

import { useState, useCallback } from 'react';
import type { EntityData } from '@/hooks/useEntityStream';
import type { AlertAPI } from '@/lib/api';

export interface InvestigationState {
  selectedEntity: EntityData | null;
  flyToLocation: { lat: number; lon: number } | null;
  alertEntityIds: Set<string>;
  setSelectedEntity: (entity: EntityData | null) => void;
  investigateAlert: (alert: AlertAPI) => void;
  investigateEntity: (callsign: string) => void;
}

/**
 * Core investigate flow — the one interaction that has to be perfect.
 * Finds the entity matching the alert source, zooms the map, opens entity detail.
 */
export function useInvestigation(entityList: EntityData[]): InvestigationState {
  const [selectedEntity, setSelectedEntity] = useState<EntityData | null>(null);
  const [flyToLocation, setFlyToLocation] = useState<{ lat: number; lon: number } | null>(null);
  const [alertEntityIds, setAlertEntityIds] = useState<Set<string>>(new Set());

  const investigateAlert = useCallback((alert: AlertAPI) => {
    const source = alert.source || '';
    const match = entityList.find((e) =>
      e.callsign === source ||
      e.entity_id === source ||
      e.entity_id.startsWith(source.slice(0, 8)) ||
      source.toLowerCase().includes(e.entity_id.slice(0, 8).toLowerCase()) ||
      source.toLowerCase().includes((e.callsign || '').toLowerCase())
    ) || (entityList.length > 0 ? entityList[0] : null);

    if (match) {
      setSelectedEntity(match);
      setFlyToLocation({ lat: match.position.lat, lon: match.position.lon });
      // Mark this entity for alert pulse on the map
      setAlertEntityIds((prev) => {
        const next = new Set(prev);
        next.add(match.entity_id);
        return next;
      });
      // Clear pulse after animation completes (3 pulses × 600ms ≈ 1.8s)
      setTimeout(() => {
        setAlertEntityIds((prev) => {
          const next = new Set(prev);
          next.delete(match.entity_id);
          return next;
        });
      }, 2000);
    }
  }, [entityList]);

  const investigateEntity = useCallback((callsign: string) => {
    const match = entityList.find((e) =>
      e.callsign === callsign || e.entity_id.startsWith(callsign.slice(0, 8))
    );
    if (match) {
      setSelectedEntity(match);
      setFlyToLocation({ lat: match.position.lat, lon: match.position.lon });
    }
  }, [entityList]);

  return {
    selectedEntity,
    flyToLocation,
    alertEntityIds,
    setSelectedEntity,
    investigateAlert,
    investigateEntity,
  };
}
