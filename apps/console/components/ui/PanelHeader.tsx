'use client';

import React from 'react';

interface PanelHeaderProps {
  title: string;
  count?: number;
  /** Override badge color (defaults to accent for normal, critical for alerts). */
  badgeVariant?: 'accent' | 'critical';
}

export default function PanelHeader({ title, count, badgeVariant = 'accent' }: PanelHeaderProps) {
  const isAlert = badgeVariant === 'critical';
  return (
    <div
      className="flex-none flex items-center justify-between px-3 py-2"
      style={{ borderBottom: '1px solid var(--border)' }}
    >
      <span
        className="text-xs font-bold tracking-widest"
        style={{ fontFamily: 'var(--font-orbitron), Orbitron, sans-serif', color: 'var(--accent)' }}
      >
        {title}
      </span>
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
    </div>
  );
}
