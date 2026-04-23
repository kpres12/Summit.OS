'use client';

import React, { createContext, useContext, useState, useCallback } from 'react';

export type ActionType =
  | 'DISPATCH'
  | 'HALT'
  | 'RTB'
  | 'CAMERA'
  | 'ENGAGE'
  | 'ALERT_ACK'
  | 'ALERT_DISMISS'
  | 'MISSION_CREATE'
  | 'GEOFENCE_CREATE'
  | 'GEOFENCE_DELETE';

export interface ActionEntry {
  id: string;
  timestamp: number;
  action: ActionType;
  target: string;
  detail?: string;
  status: 'success' | 'failed' | 'pending';
}

interface ActionLogContextType {
  actions: ActionEntry[];
  logAction: (entry: Omit<ActionEntry, 'id' | 'timestamp'>) => string;
  updateAction: (id: string, status: ActionEntry['status']) => void;
  clearLog: () => void;
}

const ActionLogContext = createContext<ActionLogContextType | undefined>(undefined);

export function ActionLogProvider({ children }: { children: React.ReactNode }) {
  const [actions, setActions] = useState<ActionEntry[]>([]);

  const logAction = useCallback((entry: Omit<ActionEntry, 'id' | 'timestamp'>): string => {
    const id = `act-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setActions(prev => [{ ...entry, id, timestamp: Date.now() }, ...prev].slice(0, 50));
    return id;
  }, []);

  const updateAction = useCallback((id: string, status: ActionEntry['status']) => {
    setActions(prev => prev.map(a => a.id === id ? { ...a, status } : a));
  }, []);

  const clearLog = useCallback(() => setActions([]), []);

  return (
    <ActionLogContext.Provider value={{ actions, logAction, updateAction, clearLog }}>
      {children}
    </ActionLogContext.Provider>
  );
}

export function useActionLog() {
  const ctx = useContext(ActionLogContext);
  if (!ctx) throw new Error('useActionLog must be used within ActionLogProvider');
  return ctx;
}
