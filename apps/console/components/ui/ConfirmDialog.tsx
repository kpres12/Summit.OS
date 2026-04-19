'use client';

import React, { useEffect } from 'react';

interface ConfirmDialogProps {
  title: string;
  message: string;
  confirmLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
  danger?: boolean;
}

export default function ConfirmDialog({
  title,
  message,
  confirmLabel = 'CONFIRM',
  onConfirm,
  onCancel,
  danger = false,
}: ConfirmDialogProps) {
  // Escape key cancels
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onCancel(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onCancel]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="confirm-dlg-title"
      aria-describedby="confirm-dlg-msg"
      className="fixed inset-0 flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,0.78)', zIndex: 500 }}
      onClick={(e) => { if (e.target === e.currentTarget) onCancel(); }}
    >
      <div
        style={{
          background: 'var(--background-panel)',
          border: `1px solid ${danger ? 'color-mix(in srgb, var(--critical) 50%, transparent)' : 'var(--border)'}`,
          padding: '24px',
          width: '340px',
          maxWidth: '90vw',
        }}
      >
        <div
          id="confirm-dlg-title"
          className="text-[11px] font-bold tracking-[0.2em] mb-3"
          style={{
            fontFamily: 'var(--font-ibm-plex-mono), monospace',
            color: danger ? 'var(--critical)' : 'var(--accent)',
          }}
        >
          {title}
        </div>
        <div
          id="confirm-dlg-msg"
          className="text-[11px] leading-relaxed mb-6"
          style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-dim)' }}
        >
          {message}
        </div>
        <div className="flex gap-2">
          <button
            onClick={onCancel}
            autoFocus
            className="summit-btn flex-1 text-[10px] py-2.5 tracking-widest"
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: 'var(--text-dim)',
              border: '1px solid var(--border)',
              background: 'transparent',
              cursor: 'pointer',
            }}
          >
            CANCEL
          </button>
          <button
            onClick={onConfirm}
            className="summit-btn flex-1 text-[10px] py-2.5 tracking-widest font-bold"
            style={{
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
              color: 'var(--background)',
              background: danger ? 'var(--critical)' : 'var(--accent)',
              border: 'none',
              cursor: 'pointer',
            }}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
