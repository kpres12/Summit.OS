'use client';

import React, { createContext, useContext, useState, useCallback } from 'react';

export type ToastSeverity = 'info' | 'success' | 'warning' | 'critical';

export interface Toast {
  id: string;
  message: string;
  severity: ToastSeverity;
  persistent?: boolean;
}

interface ToastContextType {
  toasts: Toast[];
  addToast: (toast: Omit<Toast, 'id'>) => string;
  dismissToast: (id: string) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const dismissToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  const addToast = useCallback((toast: Omit<Toast, 'id'>): string => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setToasts(prev => [...prev.slice(-4), { ...toast, id }]);
    if (!toast.persistent) {
      const ttl = toast.severity === 'critical' ? 8000
                : toast.severity === 'warning'  ? 6000
                : 4000;
      setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), ttl);
    }
    return id;
  }, []);

  return (
    <ToastContext.Provider value={{ toasts, addToast, dismissToast }}>
      {children}
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used within ToastProvider');
  return ctx;
}
