'use client';

import React from 'react';

interface StatusBadgeProps {
  label: string;
  /** CSS color value or CSS variable reference. */
  color?: string;
  className?: string;
}

export default function StatusBadge({ label, color = 'var(--accent)', className = '' }: StatusBadgeProps) {
  return (
    <span
      className={`text-[9px] font-bold tracking-widest px-1.5 py-0.5 ${className}`}
      style={{
        color,
        border: `1px solid ${color}`,
        background: `${color}15`,
        fontFamily: 'var(--font-ibm-plex-mono), monospace',
      }}
    >
      {label}
    </span>
  );
}
