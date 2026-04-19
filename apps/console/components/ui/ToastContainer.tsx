'use client';

import React from 'react';
import { useToast, type Toast } from '@/contexts/ToastContext';

function severityColor(s: Toast['severity']): string {
  if (s === 'critical') return 'var(--critical)';
  if (s === 'warning')  return 'var(--warning)';
  if (s === 'success')  return 'var(--accent)';
  return 'var(--text-dim)';
}

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: (id: string) => void }) {
  const color = severityColor(toast.severity);
  return (
    <div
      role="status"
      style={{
        pointerEvents: 'all',
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        padding: '8px 12px',
        background: 'var(--background-panel)',
        border: `1px solid color-mix(in srgb, ${color} 40%, transparent)`,
        borderLeft: `3px solid ${color}`,
        minWidth: '240px',
        maxWidth: '360px',
        boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
      }}
    >
      <span style={{
        fontFamily: 'var(--font-ibm-plex-mono), monospace',
        fontSize: '10px',
        color: 'var(--text-dim)',
        flex: 1,
        letterSpacing: '0.05em',
        lineHeight: 1.4,
      }}>
        {toast.message}
      </span>
      <button
        onClick={() => onDismiss(toast.id)}
        aria-label="Dismiss notification"
        style={{
          background: 'none',
          border: 'none',
          color: 'var(--text-muted)',
          cursor: 'pointer',
          fontFamily: 'var(--font-ibm-plex-mono), monospace',
          fontSize: '10px',
          padding: '0 2px',
          flexShrink: 0,
        }}
      >
        ✕
      </button>
    </div>
  );
}

export default function ToastContainer() {
  const { toasts, dismissToast } = useToast();
  if (toasts.length === 0) return null;

  return (
    <div
      aria-live="assertive"
      aria-atomic="false"
      style={{
        position: 'fixed',
        bottom: '56px',
        right: '16px',
        zIndex: 400,
        display: 'flex',
        flexDirection: 'column',
        gap: '6px',
        pointerEvents: 'none',
      }}
    >
      {toasts.map(t => (
        <ToastItem key={t.id} toast={t} onDismiss={dismissToast} />
      ))}
    </div>
  );
}
