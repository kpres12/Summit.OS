'use client';

import React from 'react';

interface StatusDotProps {
  color?: string;
  /** Convenience presets: accent (green), critical (red), warning (amber), dim. */
  variant?: 'accent' | 'critical' | 'warning' | 'dim';
  glow?: boolean;
  size?: number;
}

const VARIANT_MAP: Record<string, string> = {
  accent: 'var(--accent)',
  critical: 'var(--critical)',
  warning: 'var(--warning)',
  dim: 'var(--text-muted)',
};

export default function StatusDot({ color, variant = 'accent', glow = false, size = 6 }: StatusDotProps) {
  const c = color || VARIANT_MAP[variant] || VARIANT_MAP.accent;
  return (
    <div
      aria-hidden="true"
      style={{
        width: size,
        height: size,
        borderRadius: '50%',
        background: c,
        boxShadow: glow ? `0 0 6px ${c}` : 'none',
        flexShrink: 0,
      }}
    />
  );
}
