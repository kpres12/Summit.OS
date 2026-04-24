'use client';

import React from 'react';

interface PanelHeaderProps {
  title: string;
  subtitle?: string;
  count?: number;
  badgeVariant?: 'accent' | 'critical';
  onClose?: () => void;
}

export default function PanelHeader({ title, subtitle, count, badgeVariant = 'accent', onClose }: PanelHeaderProps) {
  const isAlert = badgeVariant === 'critical';
  return (
    <div
      className="flex-none flex items-center justify-between px-4 py-3"
      style={{ borderBottom: '1px solid var(--border)' }}
    >
      <div className="flex flex-col gap-0.5">
        <span
          className="text-[11px] font-bold tracking-[0.18em] uppercase"
          style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--accent-50)' }}
        >
          {title}
        </span>
        {subtitle && (
          <span
            className="text-[9px] tracking-[0.12em] uppercase"
            style={{ fontFamily: 'var(--font-ibm-plex-mono), monospace', color: 'var(--text-dim)' }}
          >
            {subtitle}
          </span>
        )}
      </div>
      <div className="flex items-center gap-2">
        {count !== undefined && (
          <span
            className="text-[10px] px-1.5 py-0.5"
            style={{
              background: isAlert ? 'rgba(255,59,59,0.15)' : 'var(--accent-10)',
              color: isAlert ? 'var(--critical)' : 'var(--accent)',
              border: `1px solid ${isAlert ? 'rgba(255,59,59,0.3)' : 'var(--accent-30)'}`,
              fontFamily: 'var(--font-ibm-plex-mono), monospace',
            }}
          >
            {count}
          </span>
        )}
        {onClose && (
          <button
            onClick={onClose}
            className="text-[10px] leading-none opacity-50 hover:opacity-100 transition-opacity"
            style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-ibm-plex-mono), monospace' }}
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
}
